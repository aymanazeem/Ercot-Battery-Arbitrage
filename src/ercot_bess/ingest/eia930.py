"""EIA 930 hourly grid monitor pulled from the v2 REST API.

Latency differs by field. Demand is near real time at about one hour, but net
generation, generation by fuel, and interchange lag by one to two days, so a live
pull must not assume same hour availability of generation. Interchange follows the
EIA sign convention where negative is net inflow and positive is net outflow.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from ..config import Config
from .raw_store import cache_days

DATASET = "eia930"
EIA_BASE = "https://api.eia.gov/v2"
_TIMEOUT_SECONDS = 60

# EIA region series codes mapped to the names this project uses
_TYPE_TO_SERIES = {
    "D": "demand",
    "DF": "da_demand_forecast",
    "NG": "net_generation",
    "TI": "interchange",
}


def _require_key(env_name: str) -> str:
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(
            f"EIA 930 needs an api key. Set {env_name} in your .env before ingesting."
        )
    return key


def _get_data_rows(session: requests.Session, url: str, params: dict) -> list[dict]:
    response = session.get(url, params=params, timeout=_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise RuntimeError(f"EIA request to {url} failed with status {response.status_code}")
    payload = response.json()
    return payload.get("response", {}).get("data", [])


def _region_params(key: str, respondent: str, day: date) -> dict:
    return {
        "api_key": key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "facets[type][]": list(_TYPE_TO_SERIES),
        "start": f"{day.isoformat()}T00",
        "end": f"{day.isoformat()}T23",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }


def _fuel_params(key: str, respondent: str, day: date) -> dict:
    return {
        "api_key": key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "start": f"{day.isoformat()}T00",
        "end": f"{day.isoformat()}T23",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }


def _to_long(rows: list[dict], series_of: callable) -> pd.DataFrame:
    records = []
    for row in rows:
        series, fuel_type = series_of(row)
        records.append(
            {
                "interval_start_utc": pd.to_datetime(row["period"], utc=True),
                "respondent": row.get("respondent"),
                "series": series,
                "fuel_type": fuel_type,
                "value": pd.to_numeric(row.get("value"), errors="coerce"),
                "unit": row.get("value-units"),
            }
        )
    columns = ["interval_start_utc", "respondent", "series", "fuel_type", "value", "unit"]
    return pd.DataFrame(records, columns=columns)


def _region_series(row: dict) -> tuple[str, None]:
    return _TYPE_TO_SERIES.get(row.get("type"), row.get("type")), None


def _fuel_series(row: dict) -> tuple[str, str]:
    return "net_generation_by_fuel", row.get("fueltype")


def fetch_eia930(
    cfg: Config,
    start: date,
    end: date,
    raw_root: Path,
    *,
    force: bool = False,
    session: requests.Session | None = None,
    now_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Cache demand, day ahead demand forecast, net generation, interchange, and fuel mix.

    Stored as one long table keyed by interval and series so both the region series and
    the by fuel series share a partition. Shaping into the processed tables happens in the
    build step.
    """
    source = cfg.data.sources.eia930
    key = _require_key(source.api_key_env)
    respondent = source.balancing_authority
    session = session or requests.Session()
    api_path = source.api_path
    region_url = f"{EIA_BASE}/{api_path}/region-data/data/"
    fuel_url = f"{EIA_BASE}/{api_path}/fuel-type-data/data/"

    def fetch_day(day: date) -> pd.DataFrame:
        region_rows = _get_data_rows(session, region_url, _region_params(key, respondent, day))
        fuel_rows = _get_data_rows(session, fuel_url, _fuel_params(key, respondent, day))
        region = _to_long(region_rows, _region_series)
        fuel = _to_long(fuel_rows, _fuel_series)
        return pd.concat([region, fuel], ignore_index=True)

    return cache_days(DATASET, start, end, fetch_day, raw_root, force=force, now_utc=now_utc)
