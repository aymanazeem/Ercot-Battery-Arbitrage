"""Turn the raw partitions into the clean processed tables.

The builders are functions that take a raw frame and config and return a clean frame.
Reading raw partitions and writing processed tables happen in the helper functions and in
run_build.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import Config, Regime
from ..ingest.raw_store import RETRIEVED_AT
from .clean import assign_regime, sort_and_dedup, spacing_gaps, to_utc
from .schema import (
    DA_DEMAND_FORECAST,
    DA_PRICES_SCHEMA,
    INTERVAL,
    LOAD_SCHEMA,
    NAME_TO_SCHEMA,
    PRICE,
    REGIME,
    RT_PRICES_SCHEMA,
    SETTLEMENT_POINT,
    TEMP,
    WEATHER_SCHEMA,
    enforce_schema,
)

HOURLY = pd.Timedelta(1, "h")
QUARTER_HOURLY = pd.Timedelta(15, "min")

# the hosted column names the clean tables are built from
_HOSTED_INTERVAL = "interval_start_utc"
_HOSTED_LOCATION = "location"
_HOSTED_PRICE = "spp"
_HOSTED_PUBLISH = "publish_time_utc"
# the hosted day ahead market load forecast column for the whole system
_DEMAND_FORECAST_COLUMN = "system_total"

# hosted temperature columns that are not a weather zone temperature reading
_WEATHER_META = {"interval_start_utc", "interval_end_utc", "publish_time_utc", RETRIEVED_AT}


def _build_prices(raw: pd.DataFrame, cfg: Config, schema: dict[str, str]) -> pd.DataFrame:
    market = cfg.market
    frame = pd.DataFrame(
        {
            INTERVAL: to_utc(raw[_HOSTED_INTERVAL]),
            SETTLEMENT_POINT: raw[_HOSTED_LOCATION].astype("string"),
            PRICE: pd.to_numeric(raw[_HOSTED_PRICE], errors="coerce"),
        }
    )
    # keep only the configured hub, so a raw layer that cached other locations does not
    # carry them into the processed tables the rest of the pipeline reads
    frame = frame[frame[SETTLEMENT_POINT] == market.market.primary_settlement_point]
    frame = sort_and_dedup(frame, [INTERVAL, SETTLEMENT_POINT])
    frame[REGIME] = assign_regime(frame[INTERVAL], market.regimes, market.market.timezone_display)
    return enforce_schema(frame, schema)


def build_da_prices(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Clean day ahead hourly settlement point prices tagged by regime."""
    return _build_prices(raw, cfg, DA_PRICES_SCHEMA)


def build_rt_prices(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Clean real time fifteen minute settlement point prices tagged by regime."""
    return _build_prices(raw, cfg, RT_PRICES_SCHEMA)


def build_load(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Clean hourly day ahead demand forecast for the whole ERCOT system.

    This is ERCOT's own day ahead market system load forecast, the exogenous demand signal
    published before day ahead close. The table carries the forecast alone, since the model
    uses no realised load and a realised load series would be future data if it leaked in.
    """
    frame = pd.DataFrame(
        {
            INTERVAL: to_utc(raw[_HOSTED_INTERVAL]),
            DA_DEMAND_FORECAST: pd.to_numeric(raw[_DEMAND_FORECAST_COLUMN], errors="coerce"),
        }
    )
    frame = sort_and_dedup(frame, [INTERVAL])
    return enforce_schema(frame, LOAD_SCHEMA)


def _fahrenheit_to_celsius(temp_f: pd.Series) -> pd.Series:
    return (temp_f - 32.0) * 5.0 / 9.0


def build_weather(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Clean hourly ERCOT wide temperature averaged across the weather zones.

    ERCOT publishes weather zone temperature in fahrenheit, so it is converted to celsius.
    The zones are combined with an equal weight mean since no zone weights are configured.
    The feed reforecasts each interval many times, so only forecasts published before the
    delivery day are kept and the most recent of those is used, which keeps the temperature
    knowable day ahead and stops a later revision of the realised weather leaking backwards.
    """
    display_tz = cfg.market.market.timezone_display
    zone_columns = [column for column in raw.columns if column not in _WEATHER_META]
    zones = raw[zone_columns].apply(pd.to_numeric, errors="coerce")
    interval = to_utc(raw[_HOSTED_INTERVAL])
    if _HOSTED_PUBLISH in raw.columns:
        publish = to_utc(raw[_HOSTED_PUBLISH])
    else:
        publish = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")
    frame = pd.DataFrame(
        {
            INTERVAL: interval,
            "publish": publish,
            TEMP: _fahrenheit_to_celsius(zones.mean(axis=1)),
        }
    )
    delivery_midnight = frame[INTERVAL].dt.tz_convert(display_tz).dt.normalize().dt.tz_convert("UTC")
    known_day_ahead = frame["publish"].isna() | (frame["publish"] <= delivery_midnight)
    frame = frame[known_day_ahead]
    # among the forecasts published before the delivery day keep the most recent one
    frame = frame.sort_values("publish", kind="stable")
    frame = sort_and_dedup(frame, [INTERVAL], keep="last")
    return enforce_schema(frame, WEATHER_SCHEMA)


def read_raw(raw_root: Path, dataset: str) -> pd.DataFrame:
    """Read every dated partition for one raw dataset into a single frame."""
    partitions = sorted((Path(raw_root) / dataset).glob("date=*/data.parquet"))
    if not partitions:
        raise FileNotFoundError(f"no raw partitions found for {dataset} under {raw_root}")
    frames = [pd.read_parquet(path) for path in partitions]
    return pd.concat(frames, ignore_index=True)


def write_processed(frame: pd.DataFrame, processed_root: Path, name: str) -> Path:
    """Write one clean table as a single parquet file and return its path."""
    path = Path(processed_root) / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def read_processed(processed_root: Path, name: str) -> pd.DataFrame:
    """Read one clean table and re-apply its schema so dtypes survive the parquet round trip."""
    path = Path(processed_root) / f"{name}.parquet"
    frame = pd.read_parquet(path)
    return enforce_schema(frame, NAME_TO_SCHEMA[name])


def _regime_map(cfg: Config) -> dict[str, Regime]:
    return {regime.name: regime for regime in cfg.market.regimes}


def _price_gaps(frame: pd.DataFrame, spacing: pd.Timedelta) -> int:
    total = 0
    for _, group in frame.groupby(SETTLEMENT_POINT):
        total += spacing_gaps(group[INTERVAL], spacing)
    return total


def _price_stats(
    frame: pd.DataFrame, spacing: pd.Timedelta, cap_attr: str, regimes: dict[str, Regime]
) -> dict:
    stats = {
        "rows": int(len(frame)),
        "gaps": _price_gaps(frame, spacing),
        "negative_share": float((frame[PRICE] < 0).mean()) if len(frame) else 0.0,
        "by_regime": {},
    }
    for name, group in frame.groupby(REGIME):
        cap = getattr(regimes[name], cap_attr)
        stats["by_regime"][str(name)] = {
            "rows": int(len(group)),
            "min_price": float(group[PRICE].min()),
            "max_price": float(group[PRICE].max()),
            "above_cap": int((group[PRICE] > cap).sum()),
        }
    return stats


def _series_stats(frame: pd.DataFrame, column: str) -> dict:
    return {
        "rows": int(len(frame)),
        "gaps": spacing_gaps(frame[INTERVAL], HOURLY),
        "min": float(frame[column].min()),
        "max": float(frame[column].max()),
    }


def quality_report(tables: dict[str, pd.DataFrame], cfg: Config) -> dict:
    """Summarise counts, gaps, negative price share, and range by regime for the tables."""
    regimes = _regime_map(cfg)
    return {
        "da_prices": _price_stats(
            tables["da_prices"], HOURLY, "offer_cap_da_usd_per_mwh", regimes
        ),
        "rt_prices": _price_stats(
            tables["rt_prices"], QUARTER_HOURLY, "offer_cap_rt_usd_per_mwh", regimes
        ),
        "load": _series_stats(tables["load"], DA_DEMAND_FORECAST),
        "weather": _series_stats(tables["weather"], TEMP),
    }


def write_report(report: dict, results_root: Path) -> Path:
    """Write the data quality report as json and return its path."""
    path = Path(results_root) / "data_quality.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path


def run_build(cfg: Config, repo_root: Path, *, write: bool = True) -> dict[str, pd.DataFrame]:
    """Build the four clean tables from raw and optionally write them with a report."""
    from ..ingest.ercot import DATASET_DA, DATASET_DEMAND, DATASET_RT
    from ..ingest.weather import DATASET as WEATHER_DATASET

    raw_root = repo_root / cfg.data.paths.raw
    processed_root = repo_root / cfg.data.paths.processed
    results_root = repo_root / cfg.data.paths.results

    tables = {
        "da_prices": build_da_prices(read_raw(raw_root, DATASET_DA), cfg),
        "rt_prices": build_rt_prices(read_raw(raw_root, DATASET_RT), cfg),
        "load": build_load(read_raw(raw_root, DATASET_DEMAND), cfg),
        "weather": build_weather(read_raw(raw_root, WEATHER_DATASET), cfg),
    }
    # building the report also validates interval spacing so it runs even without a write
    report = quality_report(tables, cfg)
    if write:
        for name, frame in tables.items():
            write_processed(frame, processed_root, name)
        write_report(report, results_root)
    return tables
