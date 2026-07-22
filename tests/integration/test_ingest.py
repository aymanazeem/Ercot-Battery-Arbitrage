"""Integration tests for the fetchers against a fake gridstatus.io client, no network."""

from datetime import date

import pandas as pd
import pytest

from ercot_bess.config import load_config
from ercot_bess.ingest import __main__ as cli
from ercot_bess.ingest.ercot import fetch_da_spp, fetch_demand_forecast
from ercot_bess.ingest.weather import fetch_weather

pytestmark = pytest.mark.ingest

_NOW = pd.Timestamp("2026-07-07T12:00:00Z")
_START = date(2024, 1, 1)
_END = date(2024, 1, 2)

_HOSTED_DA = "ercot_spp_day_ahead_hourly"
_HOSTED_RT = "ercot_spp_real_time_15_min"
_HOSTED_DEMAND = "ercot_load_forecast_dam"
_HOSTED_TEMPERATURE = "ercot_temperature_forecast_by_weather_zone"
_ZONES = ["coast", "east", "far_west", "north", "north_central", "south_central", "southern", "west"]


class FakeClient:
    """Stand in for GridStatusClient that returns tiny ranged frames and logs each query.

    The real client pulls a whole date range in one paginated call, so the fake returns one
    row per day in the range, in the hosted snake case shape, and records every call it gets.
    """

    def __init__(self):
        self.calls = []

    def query(self, dataset, start, end, *, filter_column=None, filter_values=None, page_size=None):
        self.calls.append((dataset, start, end, tuple(filter_values) if filter_values else None))
        days = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="D")
        if dataset in (_HOSTED_DA, _HOSTED_RT):
            points = filter_values or ["HB_HUBAVG"]
            frames = [
                pd.DataFrame({"interval_start_utc": days, "location": point, "spp": 30.0 + offset})
                for offset, point in enumerate(points)
            ]
            return pd.concat(frames, ignore_index=True)
        if dataset == _HOSTED_DEMAND:
            return pd.DataFrame({"interval_start_utc": days, "system_total": 45000.0})
        if dataset == _HOSTED_TEMPERATURE:
            columns = {"interval_start_utc": days, "publish_time_utc": days - pd.Timedelta(1, "D")}
            columns.update({zone: 50.0 for zone in _ZONES})
            return pd.DataFrame(columns)
        raise AssertionError(f"unexpected dataset {dataset}")


def test_fetch_da_spp_two_day_window_has_expected_columns_and_stamp(tmp_path):
    cfg = load_config()
    fake = FakeClient()

    frame = fetch_da_spp(cfg, _START, _END, tmp_path, client=fake, now_utc=_NOW)

    assert not frame.empty
    expected = {"interval_start_utc", "location", "spp", "retrieved_at_utc"}
    assert expected <= set(frame.columns)
    assert (frame["retrieved_at_utc"] == _NOW).all()
    # one ranged query covering the whole window, filtered to the configured hub
    assert len(fake.calls) == 1
    dataset, start, end, points = fake.calls[0]
    assert (dataset, start, end) == (_HOSTED_DA, _START, _END)
    assert points == ("HB_HUBAVG",)

    one_hub_two_days = 2
    assert len(frame) == one_hub_two_days
    assert (tmp_path / "ercot_spp_da" / "date=2024-01-01" / "data.parquet").exists()
    assert (tmp_path / "ercot_spp_da" / "date=2024-01-02" / "data.parquet").exists()


def test_rerun_does_not_refetch_or_duplicate_partitions(tmp_path):
    cfg = load_config()
    fake = FakeClient()

    fetch_da_spp(cfg, _START, _END, tmp_path, client=fake, now_utc=_NOW)
    fake.calls.clear()

    frame = fetch_da_spp(cfg, _START, _END, tmp_path, client=fake, now_utc=_NOW)

    assert fake.calls == []
    assert len(frame) == 2
    partitions = list((tmp_path / "ercot_spp_da").glob("date=*/data.parquet"))
    assert len(partitions) == 2


def test_force_refetches_without_duplicating(tmp_path):
    cfg = load_config()
    fake = FakeClient()

    fetch_da_spp(cfg, _START, _END, tmp_path, client=fake, now_utc=_NOW)
    fake.calls.clear()

    fetch_da_spp(cfg, _START, _END, tmp_path, client=fake, force=True, now_utc=_NOW)

    assert len(fake.calls) == 1
    partitions = list((tmp_path / "ercot_spp_da").glob("date=*/data.parquet"))
    assert len(partitions) == 2


def test_fetch_demand_forecast_shape(tmp_path):
    cfg = load_config()
    fake = FakeClient()

    frame = fetch_demand_forecast(cfg, _START, _END, tmp_path, client=fake, now_utc=_NOW)

    assert {"system_total", "retrieved_at_utc"} <= set(frame.columns)
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == _HOSTED_DEMAND
    assert (tmp_path / "ercot_load_forecast_dam" / "date=2024-01-01" / "data.parquet").exists()


def test_fetch_weather_default_ercot_path_shape(tmp_path):
    cfg = load_config()
    fake = FakeClient()

    frame = fetch_weather(cfg, _START, _END, tmp_path, client=fake, now_utc=_NOW)

    assert {"coast", "west", "retrieved_at_utc"} <= set(frame.columns)
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == _HOSTED_TEMPERATURE
    assert (tmp_path / "weather" / "date=2024-01-01" / "data.parquet").exists()


def test_cli_routes_source_start_end_to_the_fetcher(tmp_path, monkeypatch):
    captured = {}

    def stub(cfg, start, end, raw_root, *, force=False):
        captured["args"] = (start, end, force)
        return pd.DataFrame({"value": [1.0]})

    monkeypatch.setitem(cli._FETCHERS, "ercot_demand", stub)

    cli.main(["--source", "ercot_demand", "--start", "2024-01-01", "--end", "2024-01-02"])

    assert captured["args"] == (date(2024, 1, 1), date(2024, 1, 2), False)
