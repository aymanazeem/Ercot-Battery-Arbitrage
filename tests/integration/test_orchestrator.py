"""Integration tests for the orchestrator, the daily pipeline end to end with no network.

Synthetic per day fetchers write raw partitions through the real raw store, then the pipeline
builds the processed tables, the feature matrix, the forecasts, and the backtest. A second test
takes one source offline after the raw history exists and checks the run is logged, skipped, and
still produces the downstream tables. The window is a summer month so there is no daylight saving
change to break the interval spacing checks, and the calibration is shrunk so the run stays quick.
"""

import logging
from datetime import date

import numpy as np
import pandas as pd
import pytest

from ercot_bess.api.orchestrator import ingest_sources, run_pipeline
from ercot_bess.config import load_config
from ercot_bess.ingest.ercot import DATASET_DA, DATASET_DEMAND, DATASET_RT
from ercot_bess.ingest.raw_store import cache_days
from ercot_bess.ingest.weather import DATASET as WEATHER_DATASET

pytestmark = pytest.mark.api

_HUBS = ["HB_HUBAVG", "HB_WEST"]
_ZONES = ["coast", "east", "far_west", "north", "north_central", "south_central", "southern", "west"]
_START = date(2024, 6, 1)
_END = date(2024, 7, 20)


def _price_at(interval_utc: pd.DatetimeIndex, hub_offset: float) -> np.ndarray:
    """A diurnal and weekly price shape on the local clock so the forecast has real signal."""
    local = interval_utc.tz_convert("America/Chicago")
    diurnal = 20.0 * np.sin(2 * np.pi * (local.hour - 3) / 24)
    weekly = 2.0 * local.dayofweek
    return 30.0 + diurnal + weekly + hub_offset


def _hub_rows(idx: pd.DatetimeIndex) -> pd.DataFrame:
    frames = []
    for offset, hub in enumerate(_HUBS):
        frames.append(
            pd.DataFrame({"interval_start_utc": idx, "location": hub, "spp": _price_at(idx, 5.0 * offset)})
        )
    return pd.concat(frames, ignore_index=True)


def _da_day(day: date) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(day, tz="UTC"), periods=24, freq="h")
    return _hub_rows(idx)


def _rt_day(day: date) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(day, tz="UTC"), periods=96, freq="15min")
    return _hub_rows(idx)


def _demand_day(day: date) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(day, tz="UTC"), periods=24, freq="h")
    return pd.DataFrame({"interval_start_utc": idx, "system_total": 40000.0 + 100.0 * idx.hour})


def _weather_day(day: date) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(day, tz="UTC"), periods=24, freq="h")
    temp_f = 70.0 + 10.0 * np.sin(2 * np.pi * idx.hour / 24)
    # publish the forecast a day before delivery so the day ahead cutoff keeps every interval
    publish = pd.Timestamp(day, tz="UTC") - pd.Timedelta(1, "D")
    columns = {"interval_start_utc": idx, "publish_time_utc": publish}
    for zone in _ZONES:
        columns[zone] = temp_f
    return pd.DataFrame(columns)


def _fetcher(dataset: str, day_builder):
    """Wrap a per day builder to look like a real fetcher, writing raw through the cache layer."""

    def fetch(cfg, start, end, raw_root, *, force=False):
        return cache_days(dataset, start, end, day_builder, raw_root, force=force)

    return fetch


def _fakes() -> dict:
    return {
        DATASET_DA: _fetcher(DATASET_DA, _da_day),
        DATASET_RT: _fetcher(DATASET_RT, _rt_day),
        DATASET_DEMAND: _fetcher(DATASET_DEMAND, _demand_day),
        WEATHER_DATASET: _fetcher(WEATHER_DATASET, _weather_day),
    }


def _small_config():
    # LEAR fits one LassoLarsIC per delivery hour and needs more training days than features,
    # so the window matches the forecast integration test, the known good small size.
    cfg = load_config()
    cfg.model.calibration.min_train_days = 35
    cfg.model.calibration.window_days = 35
    cfg.model.calibration.recalibrate_every_days = 1
    return cfg


def test_pipeline_runs_end_to_end_and_writes_every_downstream_table(tmp_path):
    cfg = _small_config()
    skipped = run_pipeline(cfg, tmp_path, _START, _END, fetchers=_fakes(), sensitivities=False)

    assert skipped == []
    processed = tmp_path / cfg.data.paths.processed
    results = tmp_path / cfg.data.paths.results
    for name in ("da_prices", "rt_prices", "load", "weather"):
        assert (processed / f"{name}.parquet").exists()
    for name in ("forecasts", "backtest", "backtest_annual", "backtest_annualised_per_kw_year"):
        assert (results / f"{name}.parquet").exists()
    assert not pd.read_parquet(results / "backtest.parquet").empty


def test_single_source_outage_is_skipped_and_the_run_still_completes(tmp_path, caplog):
    cfg = _small_config()
    # seed the raw history for every source first, without the downstream build
    ingest_sources(cfg, tmp_path, _START, _END, fetchers=_fakes())

    fakes = _fakes()

    def weather_down(cfg, start, end, raw_root, *, force=False):
        raise RuntimeError("weather feed returned no data")

    fakes[WEATHER_DATASET] = weather_down

    with caplog.at_level(logging.WARNING, logger="ercot_bess.pipeline"):
        skipped = run_pipeline(cfg, tmp_path, _START, _END, fetchers=fakes, sensitivities=False)

    assert skipped == [WEATHER_DATASET]
    # the raw weather partitions from the seed run keep the downstream build fed
    assert (tmp_path / cfg.data.paths.results / "backtest.parquet").exists()
    assert any("skipping source weather" in message for message in caplog.messages)
