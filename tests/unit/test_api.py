"""Tests for the serialisers, the disk edge, and the pipeline helpers.

No network and no FastAPI here. The serialisers run on tiny frames, the store is checked against
a seeded temp directory, and the orchestrator helpers are checked on their own. This includes
that a single source failure is caught and skipped rather than raised.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import requests

from ercot_bess.api import serialise
from ercot_bess.api.orchestrator import ingest_sources, resolve_window
from ercot_bess.api.serialise import ANNUAL, ANNUALISED, HOURS, INTERVALS, LOCAL_DAY, MARKET
from ercot_bess.api.store import ArtifactMissing, Store
from ercot_bess.backtest.aggregate import N_DAYS, PERIOD, USD_PER_KW_YEAR
from ercot_bess.backtest.build import ANNUAL_NAME, ANNUALISED_NAME, write_summary
from ercot_bess.backtest.schema import (
    CAPTURE_RATE,
    EQUIV_FULL_CYCLES,
    PROFIT,
    SCENARIO,
    SCENARIO_CEILING,
    SCENARIO_FORECAST_DRIVEN,
)
from ercot_bess.config import load_config
from ercot_bess.forecast.build import write_forecasts
from ercot_bess.forecast.schema import (
    DELIVERY_DATE,
    FOLD_INDEX,
    MODEL,
    MODEL_LIGHTGBM,
    PREDICTED,
    REALISED,
    enforce_forecasts_schema,
)
from ercot_bess.validate.build import write_processed
from ercot_bess.validate.schema import DA_PRICES_SCHEMA, INTERVAL, PRICE, REGIME, SETTLEMENT_POINT
from ercot_bess.validate.schema import enforce_schema

pytestmark = pytest.mark.api

_PRIMARY = "HB_HUBAVG"
_REGIME = "swcap5000"
_TZ = "America/Chicago"


def _two_day_prices() -> pd.DataFrame:
    """One point across two local days, the later day carrying a missing price."""
    day1 = pd.date_range(pd.Timestamp("2024-06-01", tz=_TZ), periods=2, freq="h").tz_convert("UTC")
    day2 = pd.date_range(pd.Timestamp("2024-06-02", tz=_TZ), periods=2, freq="h").tz_convert("UTC")
    frame = pd.DataFrame(
        {
            INTERVAL: [*day1, *day2],
            SETTLEMENT_POINT: [_PRIMARY] * 4,
            PRICE: [10.0, 11.0, 20.0, np.nan],
            REGIME: [_REGIME] * 4,
        }
    )
    return enforce_schema(frame, DA_PRICES_SCHEMA)


def test_latest_prices_keeps_only_the_latest_local_day_and_maps_nan_to_null():
    payload = serialise.latest_prices(_two_day_prices(), _PRIMARY, "da", _TZ)

    assert payload[SETTLEMENT_POINT] == _PRIMARY
    assert payload[MARKET] == "da"
    assert payload[LOCAL_DAY] == "2024-06-02"
    intervals = payload[INTERVALS]
    assert len(intervals) == 2
    assert intervals[0][PRICE] == 20.0
    assert intervals[1][PRICE] is None
    # timestamps serialise as ISO 8601 in UTC
    assert intervals[0][INTERVAL].endswith("+00:00")
    assert isinstance(intervals[0][INTERVAL], str)


def test_latest_prices_unknown_point_is_empty_not_an_error():
    payload = serialise.latest_prices(_two_day_prices(), "NOT_A_HUB", "da", _TZ)
    assert payload[LOCAL_DAY] is None
    assert payload[INTERVALS] == []


def _forecast_frame() -> pd.DataFrame:
    """Two delivery days for one model with the hours deliberately out of order."""
    rows = []
    for day in ("2024-06-01", "2024-06-02"):
        delivery = pd.Timestamp(day)
        start = pd.Timestamp(day, tz="UTC")
        for hour in (2, 0, 1):
            rows.append(
                {
                    DELIVERY_DATE: delivery,
                    "hour_of_day": hour,
                    INTERVAL: start + pd.Timedelta(hour, "h"),
                    SETTLEMENT_POINT: _PRIMARY,
                    REGIME: _REGIME,
                    MODEL: MODEL_LIGHTGBM,
                    PREDICTED: 30.0 + hour,
                    REALISED: 31.0 + hour,
                    FOLD_INDEX: 0,
                }
            )
    return enforce_forecasts_schema(pd.DataFrame(rows))


def test_latest_forecast_picks_the_latest_day_and_sorts_by_hour():
    payload = serialise.latest_forecast(_forecast_frame(), _PRIMARY, MODEL_LIGHTGBM)

    assert payload[SETTLEMENT_POINT] == _PRIMARY
    assert payload[MODEL] == MODEL_LIGHTGBM
    assert payload["delivery_date"] == "2024-06-02"
    hours = [row["hour_of_day"] for row in payload[HOURS]]
    assert hours == [0, 1, 2]


def _annual() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SETTLEMENT_POINT: [_PRIMARY, _PRIMARY],
            SCENARIO: [SCENARIO_CEILING, SCENARIO_FORECAST_DRIVEN],
            PERIOD: [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            PROFIT: [1000.0, 800.0],
            EQUIV_FULL_CYCLES: [300.0, 260.0],
            # the ceiling has no defined capture rate, it must serialise to null
            CAPTURE_RATE: [np.nan, 0.8],
            N_DAYS: [365, 365],
        }
    )


def _annualised() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SETTLEMENT_POINT: [_PRIMARY, _PRIMARY],
            SCENARIO: [SCENARIO_CEILING, SCENARIO_FORECAST_DRIVEN],
            N_DAYS: [365, 365],
            USD_PER_KW_YEAR: [55.0, 44.0],
        }
    )


def test_backtest_summary_shape_and_ceiling_capture_rate_is_null():
    payload = serialise.backtest_summary(_annual(), _annualised())

    assert set(payload) == {ANNUAL, ANNUALISED}
    ceiling = next(row for row in payload[ANNUAL] if row[SCENARIO] == SCENARIO_CEILING)
    assert ceiling[CAPTURE_RATE] is None
    assert ceiling[PERIOD] == "2024-01-01"
    per_kw = payload[ANNUALISED]
    assert {row[SCENARIO] for row in per_kw} == {SCENARIO_CEILING, SCENARIO_FORECAST_DRIVEN}


def test_store_reads_seeded_tables_and_flags_a_missing_artifact(tmp_path):
    cfg = load_config()
    store = Store(cfg, tmp_path)

    # nothing has been produced yet, so a read is a clean ArtifactMissing not a bare error
    with pytest.raises(ArtifactMissing):
        store.prices("da")
    with pytest.raises(ArtifactMissing):
        store.annual()

    processed_root = tmp_path / cfg.data.paths.processed
    results_root = tmp_path / cfg.data.paths.results
    write_processed(_two_day_prices(), processed_root, "da_prices")
    write_forecasts(_forecast_frame(), results_root)
    write_summary(_annual(), results_root, ANNUAL_NAME)
    write_summary(_annualised(), results_root, ANNUALISED_NAME)

    assert not store.prices("da").empty
    assert not store.forecasts().empty
    assert not store.annual().empty
    assert not store.annualised().empty


def test_resolve_window_defaults_respect_field_latency():
    # the window ends two days back so a live pull never assumes same hour generation by fuel
    start, end = resolve_window(None, None, today=date(2026, 7, 9))
    assert end == date(2026, 7, 7)
    assert start == date(2026, 7, 6)
    # explicit values pass straight through
    assert resolve_window(date(2024, 6, 1), date(2024, 6, 24)) == (date(2024, 6, 1), date(2024, 6, 24))


@pytest.mark.parametrize(
    "error",
    [
        requests.exceptions.ConnectionError("network down"),
        RuntimeError("hosted feed bad status"),
    ],
)
def test_ingest_sources_skips_a_failing_source_and_keeps_going(tmp_path, error):
    cfg = load_config()
    called = []

    def good(cfg, start, end, raw_root, *, force=False):
        called.append("good")
        return pd.DataFrame({"value": [1.0]})

    def down(cfg, start, end, raw_root, *, force=False):
        raise error

    fetchers = {"good_before": good, "down": down, "good_after": good}
    skipped = ingest_sources(cfg, tmp_path, date(2024, 6, 1), date(2024, 6, 1), fetchers=fetchers)

    assert skipped == ["down"]
    assert called == ["good", "good"]
