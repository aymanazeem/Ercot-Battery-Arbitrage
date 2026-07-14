"""Tests for the raw store helpers and each source parser."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from ercot_bess.config import load_config
from ercot_bess.ingest import eia930, weather
from ercot_bess.ingest.raw_store import (
    RETRIEVED_AT,
    add_retrieved_at,
    cache_days,
    date_range,
    partition_path,
)

pytestmark = pytest.mark.ingest

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_NOW = pd.Timestamp("2026-07-07T12:00:00Z")


def _fixture(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text())


class _Response:
    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200

    def json(self) -> dict:
        return self._payload


class _Session:
    def __init__(self, region: dict, fuel: dict):
        self._region = region
        self._fuel = fuel

    def get(self, url, params=None, timeout=None):
        return _Response(self._region if "region-data" in url else self._fuel)


def test_date_range_is_inclusive_of_both_ends():
    days = date_range(date(2024, 1, 1), date(2024, 1, 3))
    assert days == [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]


def test_date_range_rejects_backwards_window():
    with pytest.raises(ValueError):
        date_range(date(2024, 1, 3), date(2024, 1, 1))


def test_partition_path_is_dated_and_dataset_scoped(tmp_path):
    path = partition_path(tmp_path, "ercot_spp_da", date(2024, 1, 1))
    assert path == tmp_path / "ercot_spp_da" / "date=2024-01-01" / "data.parquet"


def test_add_retrieved_at_stamps_every_row():
    frame = pd.DataFrame({"price": [10.0, 20.0, 30.0]})
    stamped = add_retrieved_at(frame, _NOW)
    assert (stamped[RETRIEVED_AT] == _NOW).all()
    assert list(frame.columns) == ["price"]


def test_cache_days_writes_one_partition_per_day(tmp_path):
    def fetch_day(day):
        return pd.DataFrame({"value": [1.0], "day": [day.isoformat()]})

    frame = cache_days("demo", date(2024, 1, 1), date(2024, 1, 2), fetch_day, tmp_path, now_utc=_NOW)

    assert len(frame) == 2
    assert (frame[RETRIEVED_AT] == _NOW).all()
    assert partition_path(tmp_path, "demo", date(2024, 1, 1)).exists()
    assert partition_path(tmp_path, "demo", date(2024, 1, 2)).exists()


def test_cache_days_skips_existing_and_force_refetches(tmp_path):
    calls = []

    def fetch_day(day):
        calls.append(day)
        return pd.DataFrame({"value": [1.0]})

    window = ("demo", date(2024, 1, 1), date(2024, 1, 2), fetch_day, tmp_path)
    cache_days(*window, now_utc=_NOW)
    assert calls == [date(2024, 1, 1), date(2024, 1, 2)]

    calls.clear()
    cache_days(*window, now_utc=_NOW)
    assert calls == []

    partitions = list((tmp_path / "demo").glob("date=*/data.parquet"))
    assert len(partitions) == 2

    cache_days(*window, force=True, now_utc=_NOW)
    assert calls == [date(2024, 1, 1), date(2024, 1, 2)]
    assert len(list((tmp_path / "demo").glob("date=*/data.parquet"))) == 2


def test_eia930_parses_region_and_fuel_series(tmp_path, monkeypatch):
    monkeypatch.setenv("EIA_API_KEY", "dummy")
    cfg = load_config()
    session = _Session(_fixture("eia930_region.json"), _fixture("eia930_fuel.json"))

    frame = eia930.fetch_eia930(
        cfg, date(2024, 1, 1), date(2024, 1, 1), tmp_path, session=session, now_utc=_NOW
    )

    series = set(frame["series"])
    assert {"demand", "da_demand_forecast", "net_generation", "interchange"} <= series
    assert "net_generation_by_fuel" in series

    interchange = frame.loc[frame["series"] == "interchange", "value"].iloc[0]
    assert interchange == -500.0

    fuels = set(frame.loc[frame["series"] == "net_generation_by_fuel", "fuel_type"])
    assert fuels == {"NG", "WND"}

    assert str(frame["interval_start_utc"].dt.tz) == "UTC"
    assert (frame[RETRIEVED_AT] == _NOW).all()
    assert partition_path(tmp_path, "eia930", date(2024, 1, 1)).exists()


def test_eia930_missing_key_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    cfg = load_config()

    with pytest.raises(RuntimeError, match="EIA_API_KEY"):
        eia930.fetch_eia930(cfg, date(2024, 1, 1), date(2024, 1, 1), tmp_path)


def test_weather_missing_era5_token_raises_clear_error(tmp_path):
    cfg = load_config()
    cfg.data.sources.weather.source = "era5"

    with pytest.raises(RuntimeError, match="cdsapirc"):
        weather.fetch_weather(
            cfg, date(2024, 1, 1), date(2024, 1, 1), tmp_path, home=tmp_path
        )
