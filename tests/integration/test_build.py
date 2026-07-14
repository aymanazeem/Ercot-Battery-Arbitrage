"""Integration tests for the processed build, raw partitions through to the tables, no network."""

import json
from datetime import date

import pandas as pd
import pytest

from ercot_bess.config import load_config
from ercot_bess.ingest.raw_store import add_retrieved_at, partition_path, write_partition
from ercot_bess.validate.build import read_processed, run_build
from ercot_bess.validate.schema import (
    DA_PRICES_SCHEMA,
    INTERVAL,
    LOAD_SCHEMA,
    RT_PRICES_SCHEMA,
    SETTLEMENT_POINT,
    WEATHER_SCHEMA,
)

pytestmark = pytest.mark.validate

_NOW = pd.Timestamp("2026-07-07T12:00:00Z")
_DAYS = [date(2024, 1, 1), date(2024, 1, 2)]
_HUBS = ["HB_HUBAVG", "HB_WEST"]
_ZONES = ["coast", "east", "far_west", "north", "north_central", "south_central", "southern", "west"]


def _write(raw_root, dataset, day, frame):
    write_partition(partition_path(raw_root, dataset, day), add_retrieved_at(frame, _NOW))


def _da_frame(day, specials=None):
    start = pd.Timestamp(day, tz="UTC")
    intervals = pd.date_range(start, periods=24, freq="h")
    rows = []
    for offset, hub in enumerate(_HUBS):
        for interval in intervals:
            rows.append({"interval_start_utc": interval, "location": hub, "spp": 25.0 + offset})
    frame = pd.DataFrame(rows)
    for hub, hour, price in specials or []:
        mask = (frame["location"] == hub) & (frame["interval_start_utc"] == intervals[hour])
        frame.loc[mask, "spp"] = price
    return frame


def _rt_frame(day):
    start = pd.Timestamp(day, tz="UTC")
    intervals = pd.date_range(start, periods=96, freq="15min")
    rows = [
        {"interval_start_utc": interval, "location": hub, "spp": 20.0 + offset}
        for offset, hub in enumerate(_HUBS)
        for interval in intervals
    ]
    return pd.DataFrame(rows)


def _demand_frame(day):
    start = pd.Timestamp(day, tz="UTC")
    intervals = pd.date_range(start, periods=24, freq="h")
    rows = [{"interval_start_utc": interval, "system_total": 41000.0} for interval in intervals]
    return pd.DataFrame(rows)


def _weather_frame(day):
    start = pd.Timestamp(day, tz="UTC")
    intervals = pd.date_range(start, periods=24, freq="h")
    # publish a day before delivery so the day ahead cutoff keeps every interval
    publish = start - pd.Timedelta(1, "D")
    rows = []
    for interval in intervals:
        row = {"interval_start_utc": interval, "publish_time_utc": publish}
        row.update({zone: 50.0 for zone in _ZONES})
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def repo(tmp_path):
    raw_root = tmp_path / "data" / "raw"
    for day in _DAYS:
        specials = [("HB_HUBAVG", 0, -10.0), ("HB_HUBAVG", 1, 6000.0)] if day == _DAYS[0] else None
        da = _da_frame(day, specials)
        if day == _DAYS[0]:
            # a duplicate two a.m. interval that must collapse to one row
            duplicate = da[(da["location"] == "HB_HUBAVG")].iloc[[2]].copy()
            duplicate["spp"] = 999.0
            da = pd.concat([da, duplicate], ignore_index=True)
        _write(raw_root, "ercot_spp_da", day, da)
        _write(raw_root, "ercot_spp_rt", day, _rt_frame(day))
        _write(raw_root, "ercot_load_forecast_dam", day, _demand_frame(day))
        _write(raw_root, "weather", day, _weather_frame(day))
    return tmp_path


def test_run_build_writes_all_processed_tables_and_report(repo):
    cfg = load_config()
    run_build(cfg, repo)

    processed = repo / "data" / "processed"
    for name in ("da_prices", "rt_prices", "load", "weather"):
        assert (processed / f"{name}.parquet").exists()
    assert (repo / "data" / "results" / "data_quality.json").exists()


def test_processed_tables_match_the_schema_contracts(repo):
    cfg = load_config()
    run_build(cfg, repo)
    processed = repo / "data" / "processed"

    expected = {
        "da_prices": DA_PRICES_SCHEMA,
        "rt_prices": RT_PRICES_SCHEMA,
        "load": LOAD_SCHEMA,
        "weather": WEATHER_SCHEMA,
    }
    for name, schema in expected.items():
        frame = read_processed(processed, name)
        assert list(frame.columns) == list(schema)
        for column, dtype in schema.items():
            assert str(frame[column].dtype) == dtype


def test_no_duplicated_primary_keys_and_valid_regimes(repo):
    cfg = load_config()
    run_build(cfg, repo)
    processed = repo / "data" / "processed"

    valid = {regime.name for regime in cfg.market.regimes}
    for name in ("da_prices", "rt_prices"):
        frame = read_processed(processed, name)
        assert not frame.duplicated(subset=[INTERVAL, SETTLEMENT_POINT]).any()
        assert set(frame["regime"]).issubset(valid)
        assert frame["regime"].notna().all()

    for name in ("load", "weather"):
        frame = read_processed(processed, name)
        assert not frame.duplicated(subset=[INTERVAL]).any()


def test_duplicate_two_am_interval_collapses_and_keeps_first(repo):
    cfg = load_config()
    run_build(cfg, repo)
    da = read_processed(repo / "data" / "processed", "da_prices")

    hubavg = da[da[SETTLEMENT_POINT] == "HB_HUBAVG"]
    assert not hubavg.duplicated(subset=[INTERVAL]).any()
    # the injected duplicate carried price 999, the first occurrence must win
    assert (da["price_usd_per_mwh"] == 999.0).sum() == 0


def test_negative_and_above_cap_prices_survive_cleaning(repo):
    cfg = load_config()
    run_build(cfg, repo)
    da = read_processed(repo / "data" / "processed", "da_prices")

    assert (da["price_usd_per_mwh"] == -10.0).sum() == 1
    assert (da["price_usd_per_mwh"] == 6000.0).sum() == 1


def test_load_carries_the_day_ahead_demand_forecast(repo):
    cfg = load_config()
    run_build(cfg, repo)
    load = read_processed(repo / "data" / "processed", "load")

    # the table carries ERCOT's day ahead market system load forecast alone
    assert (load["da_demand_forecast_mw"] == 41000.0).all()


def test_quality_report_has_expected_shape(repo):
    cfg = load_config()
    run_build(cfg, repo)
    report = json.loads((repo / "data" / "results" / "data_quality.json").read_text())

    assert report["da_prices"]["by_regime"]["swcap5000"]["above_cap"] == 1
    assert report["da_prices"]["negative_share"] > 0.0
    # the fixtures are on a clean grid so there are no gaps
    assert report["da_prices"]["gaps"] == 0
    assert report["rt_prices"]["gaps"] == 0
    assert report["load"]["gaps"] == 0
