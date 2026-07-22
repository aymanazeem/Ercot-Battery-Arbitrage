"""Tests for the raw store helpers and each source parser."""

from datetime import date

import pandas as pd
import pytest

from ercot_bess.config import load_config
from ercot_bess.ingest import weather
from ercot_bess.ingest.raw_store import (
    RETRIEVED_AT,
    add_retrieved_at,
    cache_days,
    date_range,
    partition_path,
)

pytestmark = pytest.mark.ingest

_NOW = pd.Timestamp("2026-07-07T12:00:00Z")


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


def test_weather_missing_era5_token_raises_clear_error(tmp_path):
    cfg = load_config()
    cfg.data.sources.weather.source = "era5"

    with pytest.raises(RuntimeError, match="cdsapirc"):
        weather.fetch_weather(
            cfg, date(2024, 1, 1), date(2024, 1, 1), tmp_path, home=tmp_path
        )
