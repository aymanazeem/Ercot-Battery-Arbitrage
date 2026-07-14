"""Weather ingestion, ERCOT weather zone temperature by default.

By default this uses ERCOT published weather zone temperature through gridstatus.io, set in
data.yaml, so no Copernicus account is needed. The ERA5 path is the richer option. It is
guarded so a missing token fails clearly instead of hanging.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..config import Config
from .gridstatus_io import GridStatusClient
from .raw_store import cache_range

DATASET = "weather"
SOURCE_ERCOT = "ercot_weather_zone"
SOURCE_ERA5 = "era5"

_HOSTED_TEMPERATURE = "ercot_temperature_forecast_by_weather_zone"


def _require_era5_token(home: Path | None = None) -> Path:
    home = home if home is not None else Path.home()
    token = home / ".cdsapirc"
    if not token.exists():
        raise RuntimeError(
            "ERA5 weather selected but no Copernicus token was found at ~/.cdsapirc. "
            "Create an ECMWF account, accept the licence, and write the token, or set "
            "weather source to ercot_weather_zone in data.yaml."
        )
    return token


def _fetch_era5(cfg: Config, start: date, end: date, raw_root: Path, **kwargs) -> pd.DataFrame:
    try:
        import cdsapi  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "ERA5 ingestion needs the cdsapi client, which is not installed by default. "
            "Use ercot_weather_zone in data.yaml or add cdsapi to the environment."
        ) from exc
    raise NotImplementedError("ERA5 ingestion is not enabled, use ercot_weather_zone.")


def fetch_weather(
    cfg: Config,
    start: date,
    end: date,
    raw_root: Path,
    *,
    force: bool = False,
    client: GridStatusClient | None = None,
    home: Path | None = None,
    now_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Cache hourly weather zone temperature for the configured weather source."""
    source = cfg.data.sources.weather.source
    if source == SOURCE_ERA5:
        _require_era5_token(home)
        return _fetch_era5(cfg, start, end, raw_root, force=force, now_utc=now_utc)
    if source != SOURCE_ERCOT:
        raise ValueError(f"unknown weather source {source}")

    client = client or GridStatusClient.from_config(cfg)

    def fetch_range(range_start: date, range_end: date) -> pd.DataFrame:
        return client.query(_HOSTED_TEMPERATURE, range_start, range_end)

    return cache_range(DATASET, start, end, fetch_range, raw_root, force=force, now_utc=now_utc)
