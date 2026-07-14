"""Schema for the day ahead model matrix and the allowed feature list.

The matrix has three kinds of column. Meta columns identify the delivery hour, the target
is the day ahead price being forecast, and the features are the only inputs a model may see.
The feature names come from model.yaml so the config stays the single source of truth.
"""

from __future__ import annotations

import pandas as pd

from ..config import ModelConfig

DELIVERY_DATE = "delivery_date"
INTERVAL = "interval_start_utc"
SETTLEMENT_POINT = "settlement_point"
REGIME = "regime"
PRICE = "price_usd_per_mwh"

HOUR_OF_DAY = "hour_of_day"
DAY_OF_WEEK = "day_of_week"
MONTH = "month"
HOLIDAY_FLAG = "holiday_flag"
DA_DEMAND_FORECAST = "da_demand_forecast_mw"
TEMP = "temp_c_ercot"

META_COLUMNS = [DELIVERY_DATE, INTERVAL, SETTLEMENT_POINT, REGIME]
TARGET = PRICE

# realised series that are only known on or after the delivery day, forbidden as features
FORBIDDEN_FEATURES = frozenset({"load_mw", "price_same_day", "temp_realised_c"})

_UTC = "datetime64[ns, UTC]"

# the delivery day is the local midnight timestamp so it survives parquet and date maths
# dtypes for the non feature columns, features are validated as float or int separately
_META_DTYPES = {
    DELIVERY_DATE: "datetime64[ns]",
    INTERVAL: _UTC,
    SETTLEMENT_POINT: "string",
    REGIME: "string",
    TARGET: "float64",
}


def price_lag_name(lag_days: int) -> str:
    """Column name for the autoregressive price lag at the given day offset."""
    return f"price_lag_{lag_days}d"


def feature_names(model: ModelConfig) -> list[str]:
    """The ordered list of allowed feature columns derived from model.yaml.

    Order is autoregressive price lags, then calendar features, then exogenous features,
    using the names in the config.
    """
    lags = [price_lag_name(lag) for lag in model.features.price_lag_days]
    return [*lags, *model.features.calendar, *model.features.exogenous]


def matrix_columns(model: ModelConfig) -> list[str]:
    """Full ordered column list for the matrix, meta then target then features."""
    return [*META_COLUMNS, TARGET, *feature_names(model)]


def enforce_matrix_schema(frame: pd.DataFrame, model: ModelConfig) -> pd.DataFrame:
    """Return the frame reduced to the matrix columns in order with meta dtypes applied.

    Feature dtypes are left as built since a mix of integer calendar flags and float
    prices is expected. The meta and target dtypes are pinned so the parquet round trip
    is stable.
    """
    columns = matrix_columns(model)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"matrix is missing required columns {missing}")
    shaped = frame[columns].copy()
    for column, dtype in _META_DTYPES.items():
        shaped[column] = shaped[column].astype(dtype)
    return shaped
