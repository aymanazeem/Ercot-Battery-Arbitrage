"""Integration tests for the API, endpoints driven in process against seeded artifacts.

Tiny frames are written to a temp results directory, the app is built against it, and the
FastAPI test client drives the endpoints. No network and no computation, the service only reads
the precomputed tables. A directory with nothing in it exercises the clean not found path.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ercot_bess.api.app import create_app
from ercot_bess.api.serialise import ANNUAL, ANNUALISED, HOURS, INTERVALS, LOCAL_DAY, MARKET
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


def _prices(marker: float) -> pd.DataFrame:
    """Two local days for the primary hub, offset by a marker so da and rt are distinguishable."""
    day1 = pd.date_range(pd.Timestamp("2024-06-01", tz=_TZ), periods=2, freq="h").tz_convert("UTC")
    day2 = pd.date_range(pd.Timestamp("2024-06-02", tz=_TZ), periods=2, freq="h").tz_convert("UTC")
    frame = pd.DataFrame(
        {
            INTERVAL: [*day1, *day2],
            SETTLEMENT_POINT: [_PRIMARY] * 4,
            PRICE: [10.0 + marker, 11.0 + marker, 20.0 + marker, np.nan],
            REGIME: [_REGIME] * 4,
        }
    )
    return enforce_schema(frame, DA_PRICES_SCHEMA)


def _forecasts() -> pd.DataFrame:
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


def _annual() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SETTLEMENT_POINT: [_PRIMARY, _PRIMARY],
            SCENARIO: [SCENARIO_CEILING, SCENARIO_FORECAST_DRIVEN],
            PERIOD: [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            PROFIT: [1000.0, 800.0],
            EQUIV_FULL_CYCLES: [300.0, 260.0],
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


@pytest.fixture
def client(tmp_path) -> TestClient:
    cfg = load_config()
    processed_root = tmp_path / cfg.data.paths.processed
    results_root = tmp_path / cfg.data.paths.results
    write_processed(_prices(0.0), processed_root, "da_prices")
    write_processed(_prices(100.0), processed_root, "rt_prices")
    write_forecasts(_forecasts(), results_root)
    write_summary(_annual(), results_root, ANNUAL_NAME)
    write_summary(_annualised(), results_root, ANNUALISED_NAME)
    return TestClient(create_app(cfg, tmp_path))


def test_health_is_live_without_touching_disk(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prices_latest_returns_the_latest_local_day_with_utc_and_null(client):
    response = client.get("/prices/latest")
    assert response.status_code == 200
    body = response.json()
    assert body[SETTLEMENT_POINT] == _PRIMARY
    assert body[MARKET] == "da"
    assert body[LOCAL_DAY] == "2024-06-02"
    intervals = body[INTERVALS]
    assert len(intervals) == 2
    assert intervals[0][INTERVAL].endswith("+00:00")
    assert intervals[1][PRICE] is None


def test_prices_latest_selects_the_market(client):
    body = client.get("/prices/latest", params={"market": "rt"}).json()
    assert body[MARKET] == "rt"
    # the rt seed is offset by one hundred, so the market switch is observable
    assert body[INTERVALS][0][PRICE] == 120.0


def test_prices_latest_rejects_an_unknown_market(client):
    assert client.get("/prices/latest", params={"market": "spot"}).status_code == 422


def test_prices_latest_unknown_point_is_empty(client):
    body = client.get("/prices/latest", params={"settlement_point": "NOPE"}).json()
    assert body[LOCAL_DAY] is None
    assert body[INTERVALS] == []


def test_forecast_latest_returns_hours_sorted_for_the_latest_day(client):
    response = client.get("/forecast/latest")
    assert response.status_code == 200
    body = response.json()
    assert body[MODEL] == MODEL_LIGHTGBM
    assert body["delivery_date"] == "2024-06-02"
    assert [row["hour_of_day"] for row in body[HOURS]] == [0, 1, 2]


def test_backtest_summary_carries_both_tables_and_a_null_ceiling_capture(client):
    response = client.get("/backtest/summary")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {ANNUAL, ANNUALISED}
    ceiling = next(row for row in body[ANNUAL] if row[SCENARIO] == SCENARIO_CEILING)
    assert ceiling[CAPTURE_RATE] is None
    assert len(body[ANNUALISED]) == 2


def test_missing_artifact_is_a_clean_not_found(tmp_path):
    cfg = load_config()
    empty = TestClient(create_app(cfg, tmp_path))
    assert empty.get("/prices/latest").status_code == 404
    assert empty.get("/backtest/summary").status_code == 404
    # liveness still answers even with no artifacts on disk
    assert empty.get("/health").status_code == 200
