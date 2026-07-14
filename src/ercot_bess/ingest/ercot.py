"""ERCOT settlement point prices and the day ahead demand forecast, from gridstatus.io.

Prices are stored as settlement point prices, not raw LMP components, because ERCOT does
not publish LMP components the way other markets do. The demand forecast is ERCOT's own
day ahead market system load forecast, which is the exogenous demand signal known before
day ahead close, so it stands in for a separate demand forecast feed.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..config import Config
from .gridstatus_io import GridStatusClient
from .raw_store import cache_range

# the dated raw layer folders, kept stable so the rest of the pipeline reads the same paths
DATASET_DA = "ercot_spp_da"
DATASET_RT = "ercot_spp_rt"
DATASET_DEMAND = "ercot_load_forecast_dam"

# the gridstatus.io dataset ids the folders above are filled from
_HOSTED_DA = "ercot_spp_day_ahead_hourly"
_HOSTED_RT = "ercot_spp_real_time_15_min"
_HOSTED_DEMAND = "ercot_load_forecast_dam"

_LOCATION = "location"


def settlement_points(cfg: Config) -> list[str]:
    """The primary and secondary hubs from market.yaml, in that order."""
    market = cfg.market.market
    return [market.primary_settlement_point, market.secondary_settlement_point]


def fetch_da_spp(
    cfg: Config,
    start: date,
    end: date,
    raw_root: Path,
    *,
    force: bool = False,
    client: GridStatusClient | None = None,
    now_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Cache day ahead hourly settlement point prices for the configured hubs."""
    client = client or GridStatusClient.from_config(cfg)
    points = settlement_points(cfg)

    def fetch_range(range_start: date, range_end: date) -> pd.DataFrame:
        return client.query(
            _HOSTED_DA,
            range_start,
            range_end,
            filter_column=_LOCATION,
            filter_values=points,
        )

    return cache_range(DATASET_DA, start, end, fetch_range, raw_root, force=force, now_utc=now_utc)


def fetch_rt_spp(
    cfg: Config,
    start: date,
    end: date,
    raw_root: Path,
    *,
    force: bool = False,
    client: GridStatusClient | None = None,
    now_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Cache real time fifteen minute settlement point prices for the configured hubs."""
    client = client or GridStatusClient.from_config(cfg)
    points = settlement_points(cfg)

    def fetch_range(range_start: date, range_end: date) -> pd.DataFrame:
        return client.query(
            _HOSTED_RT,
            range_start,
            range_end,
            filter_column=_LOCATION,
            filter_values=points,
        )

    return cache_range(DATASET_RT, start, end, fetch_range, raw_root, force=force, now_utc=now_utc)


def fetch_demand_forecast(
    cfg: Config,
    start: date,
    end: date,
    raw_root: Path,
    *,
    force: bool = False,
    client: GridStatusClient | None = None,
    now_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Cache ERCOT's hourly day ahead market load forecast for the whole system."""
    client = client or GridStatusClient.from_config(cfg)

    def fetch_range(range_start: date, range_end: date) -> pd.DataFrame:
        return client.query(_HOSTED_DEMAND, range_start, range_end)

    return cache_range(
        DATASET_DEMAND, start, end, fetch_range, raw_root, force=force, now_utc=now_utc
    )
