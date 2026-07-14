"""Build the day ahead model matrix and guard it against using future data.

The builders are functions that take the processed tables and config and return a frame.
Only information known before day ahead close may enter the matrix. Reading the processed
tables and writing the matrix happen in the helper functions and in run_features.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from ..config import Config
from ..validate.build import read_processed
from ..validate.schema import INTERVAL as PROCESSED_INTERVAL
from ..validate.schema import PRICE as PROCESSED_PRICE
from ..validate.schema import SETTLEMENT_POINT as PROCESSED_SP
from .schema import (
    DA_DEMAND_FORECAST,
    DAY_OF_WEEK,
    DELIVERY_DATE,
    FORBIDDEN_FEATURES,
    HOLIDAY_FLAG,
    HOUR_OF_DAY,
    INTERVAL,
    META_COLUMNS,
    MONTH,
    REGIME,
    SETTLEMENT_POINT,
    TARGET,
    TEMP,
    enforce_matrix_schema,
    feature_names,
    price_lag_name,
)

FEATURES_NAME = "da_model_matrix"
FEATURE_NAMES_FILE = "feature_names.json"

_ONE_DAY = pd.Timedelta(1, "D")


class LeakageError(ValueError):
    """Raised when the matrix carries a feature that is not knowable before close."""


def _hub_frame(da_prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Rows for the primary hub with the local delivery day and hour attached.

    The delivery day is the ERCOT local calendar date and the hour is the local hour, so
    the autoregressive lags line up on the same local hour across days.
    """
    hub = cfg.market.market.primary_settlement_point
    display_tz = cfg.market.market.timezone_display
    rows = da_prices[da_prices[PROCESSED_SP] == hub].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"no day ahead prices for the primary settlement point {hub}")
    interval = rows[PROCESSED_INTERVAL]
    local = interval.dt.tz_convert(display_tz)
    return pd.DataFrame(
        {
            INTERVAL: interval,
            DELIVERY_DATE: local.dt.normalize().dt.tz_localize(None),
            HOUR_OF_DAY: local.dt.hour.astype("int64"),
            SETTLEMENT_POINT: hub,
            REGIME: rows[REGIME],
            TARGET: pd.to_numeric(rows[PROCESSED_PRICE]),
        }
    )


def _add_price_lags(matrix: pd.DataFrame, hub: pd.DataFrame, lag_days: list[int]) -> pd.DataFrame:
    """Attach the autoregressive price lags by joining the hub to its own past days."""
    out = matrix
    for lag in lag_days:
        source = (
            hub[[DELIVERY_DATE, HOUR_OF_DAY, TARGET]]
            # the autumn daylight saving change gives one local hour two rows, keep one so
            # the join stays one to one and a source day cannot multiply the target rows
            .drop_duplicates([DELIVERY_DATE, HOUR_OF_DAY], keep="first")
            .rename(columns={TARGET: price_lag_name(lag)})
        )
        # shift the source day forward so a past day lands on the day it is a lag for
        source[DELIVERY_DATE] = source[DELIVERY_DATE] + lag * _ONE_DAY
        out = out.merge(source, on=[DELIVERY_DATE, HOUR_OF_DAY], how="left")
    return out


def _holiday_flags(delivery_date: pd.Series) -> pd.Series:
    """One when the local delivery day is a US federal holiday, zero otherwise.

    US federal holidays are a standard offline stand in for the ERCOT public holiday
    calendar. They differ slightly from the NERC holiday set.
    """
    if delivery_date.empty:
        return pd.Series([], dtype="int64")
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start=delivery_date.min(), end=delivery_date.max())
    return delivery_date.isin(holidays).astype("int64")


def _add_calendar(matrix: pd.DataFrame) -> pd.DataFrame:
    """Attach the local calendar features, hour already present from the hub frame."""
    day = matrix[DELIVERY_DATE]
    matrix[DAY_OF_WEEK] = day.dt.dayofweek.astype("int64")
    matrix[MONTH] = day.dt.month.astype("int64")
    matrix[HOLIDAY_FLAG] = _holiday_flags(day)
    return matrix


def _add_exogenous(matrix: pd.DataFrame, load: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach the pre close exogenous forecasts on the delivery hour timestamp.

    The day ahead demand forecast and the temperature forecast are values for the delivery
    day that are published before day ahead close, so they are exogenous inputs not leaks.
    """
    demand = load[[PROCESSED_INTERVAL, DA_DEMAND_FORECAST]].rename(
        columns={PROCESSED_INTERVAL: INTERVAL}
    )
    temp = weather[[PROCESSED_INTERVAL, TEMP]].rename(columns={PROCESSED_INTERVAL: INTERVAL})
    return matrix.merge(demand, on=INTERVAL, how="left").merge(temp, on=INTERVAL, how="left")


def build_model_matrix(
    da_prices: pd.DataFrame, load: pd.DataFrame, weather: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Assemble the day ahead model matrix for the primary hub, one row per delivery hour.

    Warm up rows that lack a full set of lags are dropped, and so is any day that is not a
    full set of hourly deliveries. So the result has no nulls and a whole number of days.
    """
    model = cfg.model
    hub = _hub_frame(da_prices, cfg)
    matrix = _add_price_lags(hub.copy(), hub, model.features.price_lag_days)
    matrix = _add_calendar(matrix)
    matrix = _add_exogenous(matrix, load, weather)

    features = feature_names(model)
    matrix = matrix.dropna(subset=[*features, TARGET])
    hours_per_day = model.forecast.horizon_hours
    sizes = matrix.groupby(DELIVERY_DATE)[INTERVAL].transform("size")
    matrix = matrix[sizes == hours_per_day]
    matrix = matrix.sort_values(INTERVAL, kind="stable").reset_index(drop=True)
    return enforce_matrix_schema(matrix, model)


def check_no_leakage(matrix: pd.DataFrame, cfg: Config) -> None:
    """Assert the matrix carries exactly the allowed pre close features and no leaks.

    A simple check on source timestamps would wrongly reject the exogenous forecasts, because
    they carry delivery day timestamps even though they are known before close. So the guard is
    an allowlist of column names instead. It raises on any column outside the allowed set and on
    any known realised series, and passes only on the exact feature list from the config.
    """
    allowed = set(feature_names(cfg.model))
    present = set(matrix.columns) - set(META_COLUMNS) - {TARGET}

    forbidden = present & FORBIDDEN_FEATURES
    if forbidden:
        raise LeakageError(f"matrix carries forbidden realised series {sorted(forbidden)}")
    unexpected = present - allowed
    if unexpected:
        raise LeakageError(f"matrix has columns not knowable before close {sorted(unexpected)}")
    missing = allowed - present
    if missing:
        raise LeakageError(f"matrix is missing allowed features {sorted(missing)}")


def read_inputs(cfg: Config, repo_root: Path) -> dict[str, pd.DataFrame]:
    """Read the processed tables the matrix depends on."""
    processed_root = repo_root / cfg.data.paths.processed
    return {
        "da_prices": read_processed(processed_root, "da_prices"),
        "load": read_processed(processed_root, "load"),
        "weather": read_processed(processed_root, "weather"),
    }


def write_features(matrix: pd.DataFrame, features_root: Path) -> Path:
    """Write the model matrix as a single parquet file and return its path."""
    path = Path(features_root) / f"{FEATURES_NAME}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(path, index=False)
    return path


def read_features(features_root: Path, cfg: Config) -> pd.DataFrame:
    """Read the model matrix and re-apply the schema so dtypes survive the parquet round trip."""
    path = Path(features_root) / f"{FEATURES_NAME}.parquet"
    frame = pd.read_parquet(path)
    return enforce_matrix_schema(frame, cfg.model)


def write_feature_names(cfg: Config, features_root: Path) -> Path:
    """Write the ordered sanctioned feature list next to the matrix and return its path."""
    path = Path(features_root) / FEATURE_NAMES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(feature_names(cfg.model), indent=2))
    return path


def read_feature_names(features_root: Path) -> list[str]:
    """Read the saved feature name list."""
    path = Path(features_root) / FEATURE_NAMES_FILE
    return json.loads(path.read_text())


def run_features(cfg: Config, repo_root: Path, *, write: bool = True) -> pd.DataFrame:
    """Build the day ahead model matrix from the processed tables and optionally write it."""
    inputs = read_inputs(cfg, repo_root)
    matrix = build_model_matrix(inputs["da_prices"], inputs["load"], inputs["weather"], cfg)
    check_no_leakage(matrix, cfg)
    if write:
        features_root = repo_root / cfg.data.paths.features
        write_features(matrix, features_root)
        write_feature_names(cfg, features_root)
    return matrix
