"""Integration test for the dashboard, the assembled Streamlit app run headless.

Tiny frozen tables are seeded into a temp data root, the environment points the app there, and
the app runs through Streamlit's AppTest with no network. The app only reads the precomputed
tables, so the assertions pin the displayed numbers to the seeded values. The dispatch schedule
is the one thing the app solves, and its profit and capture are still read from the stored
backtest, so this test also checks that nothing shown drifts from the results. An empty root
exercises the clean note the app shows instead of a stack trace.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ercot_bess.backtest.aggregate import CUMULATIVE_SHARE, DAY_SHARE, N_DAYS, USD_PER_KW_YEAR
from ercot_bess.backtest.build import (
    CONCENTRATION_NAME,
    SENSITIVITIES_NAME,
    write_backtest,
    write_summary,
)
from ercot_bess.backtest.schema import (
    CAPTURE_RATE,
    EQUIV_FULL_CYCLES,
    PROFIT,
    SCENARIO,
    SCENARIO_CEILING,
    SCENARIO_FORECAST_DRIVEN,
    enforce_backtest_schema,
)
from ercot_bess.config import load_config
from ercot_bess.dashboard import data as dashboard_data
from ercot_bess.dashboard import views
from ercot_bess.features.schema import DELIVERY_DATE, HOUR_OF_DAY
from ercot_bess.forecast.build import write_forecasts, write_metrics
from ercot_bess.forecast.schema import (
    ALL_HOURS,
    FOLD_INDEX,
    MAE,
    MODEL,
    MODEL_LEAR,
    MODEL_LIGHTGBM,
    N_OBS,
    PREDICTED,
    REALISED,
    REL_MAE,
    SMAPE,
    enforce_forecasts_schema,
    enforce_metrics_schema,
)
from ercot_bess.validate.build import write_processed
from ercot_bess.validate.schema import (
    DA_PRICES_SCHEMA,
    INTERVAL,
    PRICE,
    REGIME,
    SETTLEMENT_POINT,
    enforce_schema,
)

pytestmark = pytest.mark.dashboard

_POINT = "HB_HUBAVG"
_REGIME = "swcap5000"
_DAY = pd.Timestamp("2024-06-01")
_APP_PATH = Path(dashboard_data.__file__).with_name("app.py")

# the stored numbers the views must surface unchanged, checked against the metrics below
_DA_LATEST = 30.0
_RT_LATEST = 60.0
_PROFIT = 137.77
_CAPTURE = 0.6321
_MAE = 7.5
_SMAPE = 21.0
_REL_MAE = 0.83


def _prices(last: float) -> pd.DataFrame:
    """Three ascending hourly prices for the primary hub so the last one is the latest price."""
    start = pd.Timestamp("2024-06-01 05:00", tz="UTC")
    interval = pd.date_range(start, periods=3, freq="h")
    frame = pd.DataFrame(
        {
            INTERVAL: interval,
            SETTLEMENT_POINT: [_POINT] * 3,
            PRICE: [last - 20.0, last - 10.0, last],
            REGIME: [_REGIME] * 3,
        }
    )
    return enforce_schema(frame, DA_PRICES_SCHEMA)


def _forecasts() -> pd.DataFrame:
    """One delivery day for the primary hub, cheap early and dear late so a schedule reproduces."""
    start = pd.Timestamp("2024-06-01 06:00", tz="UTC")
    predicted = [8.0, 6.0, 55.0, 60.0]
    realised = [10.0, 7.0, 52.0, 58.0]
    rows = pd.DataFrame(
        {
            DELIVERY_DATE: [_DAY] * 4,
            HOUR_OF_DAY: list(range(4)),
            INTERVAL: [start + pd.Timedelta(hour, "h") for hour in range(4)],
            SETTLEMENT_POINT: [_POINT] * 4,
            REGIME: [_REGIME] * 4,
            MODEL: [MODEL_LIGHTGBM] * 4,
            PREDICTED: predicted,
            REALISED: realised,
            FOLD_INDEX: [0] * 4,
        }
    )
    return enforce_forecasts_schema(rows)


def _forecasts_multi() -> pd.DataFrame:
    """The lightgbm rows beside a second model priced in reverse, so the two would solve apart."""
    lear = _forecasts().copy()
    lear[MODEL] = MODEL_LEAR
    lear[PREDICTED] = lear[PREDICTED].to_numpy()[::-1]
    return enforce_forecasts_schema(pd.concat([_forecasts(), lear], ignore_index=True))


def _metrics() -> pd.DataFrame:
    """The pooled all hours row plus one hourly row, the view reads only the pooled one."""
    rows = pd.DataFrame(
        {
            MODEL: [MODEL_LIGHTGBM, MODEL_LIGHTGBM],
            SETTLEMENT_POINT: [_POINT, _POINT],
            REGIME: [_REGIME, _REGIME],
            HOUR_OF_DAY: [ALL_HOURS, 0],
            N_OBS: [48, 2],
            MAE: [_MAE, 9.0],
            SMAPE: [_SMAPE, 25.0],
            REL_MAE: [_REL_MAE, 0.90],
        }
    )
    return enforce_metrics_schema(rows)


def _backtest() -> pd.DataFrame:
    """Both scenarios for the one day, the forecast driven row carrying the shown profit."""
    rows = pd.DataFrame(
        {
            DELIVERY_DATE: [_DAY, _DAY],
            SCENARIO: [SCENARIO_CEILING, SCENARIO_FORECAST_DRIVEN],
            SETTLEMENT_POINT: [_POINT, _POINT],
            PROFIT: [200.0, _PROFIT],
            EQUIV_FULL_CYCLES: [1.2, 1.0],
            CAPTURE_RATE: [np.nan, _CAPTURE],
        }
    )
    return enforce_backtest_schema(rows)


def _sensitivities() -> pd.DataFrame:
    records = []
    for duration in (1.0, 2.0, 4.0):
        records.append(
            {
                SETTLEMENT_POINT: _POINT,
                views.DURATION_H: duration,
                views.CYCLING_COST: 0.0,
                SCENARIO: SCENARIO_FORECAST_DRIVEN,
                N_DAYS: 30,
                USD_PER_KW_YEAR: duration * 10.0,
            }
        )
    return pd.DataFrame.from_records(records)


def _concentration() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SCENARIO: [SCENARIO_FORECAST_DRIVEN] * 3,
            SETTLEMENT_POINT: [_POINT] * 3,
            DAY_SHARE: [0.33, 0.66, 1.0],
            CUMULATIVE_SHARE: [0.4, 0.7, 1.0],
        }
    )


def _seed(root: Path, forecasts: pd.DataFrame | None = None) -> None:
    """Write the processed and results tables the four views read under one temp root."""
    cfg = load_config()
    processed_root = root / cfg.data.paths.processed
    results_root = root / cfg.data.paths.results
    write_processed(_prices(_DA_LATEST), processed_root, "da_prices")
    write_processed(_prices(_RT_LATEST), processed_root, "rt_prices")
    write_forecasts(_forecasts() if forecasts is None else forecasts, results_root)
    write_metrics(_metrics(), results_root)
    write_backtest(_backtest(), results_root)
    write_summary(_sensitivities(), results_root, SENSITIVITIES_NAME)
    write_summary(_concentration(), results_root, CONCENTRATION_NAME)


def _run(root: Path, monkeypatch) -> AppTest:
    monkeypatch.setenv(dashboard_data.DATA_ROOT_ENV, str(root))
    return AppTest.from_file(str(_APP_PATH), default_timeout=60).run()


def _selectbox(app: AppTest, label: str):
    """The selectbox carrying one label so a test can drive the sidebar model choice."""
    return next(box for box in app.selectbox if box.label == label)


def test_the_app_renders_every_view_against_the_seeded_tables(tmp_path, monkeypatch):
    _seed(tmp_path)
    app = _run(tmp_path, monkeypatch)
    assert not app.exception
    assert app.title[0].value == "ERCOT battery arbitrage dashboard"
    assert len(app.tabs) == 4


def test_the_displayed_numbers_equal_the_stored_values(tmp_path, monkeypatch):
    _seed(tmp_path)
    app = _run(tmp_path, monkeypatch)
    shown = {metric.label: metric.value for metric in app.metric}
    # the day profit and capture are read from the backtest, never recomputed in the ui
    assert shown["Day profit USD"] == "137.77"
    assert shown["Capture rate percent"] == "63.21"
    # the forecast error is the pooled all hours row, shown honestly beside the naive ratio
    assert shown["MAE USD per MWh"] == "7.50"
    assert shown["sMAPE percent"] == "21.00"
    assert shown["MAE relative to naive"] == "0.830"
    # the latest price is the last non missing value for each market
    assert shown["Day ahead latest price USD per MWh"] == "30.00"
    assert shown["Real time latest price USD per MWh"] == "60.00"


def test_a_second_run_returns_the_same_numbers_from_the_cache(tmp_path, monkeypatch):
    _seed(tmp_path)
    app = _run(tmp_path, monkeypatch)
    assert not app.exception
    first = {metric.label: metric.value for metric in app.metric}
    app.run()
    assert not app.exception
    second = {metric.label: metric.value for metric in app.metric}
    # the cached reads and the cached dispatch solve must return identical data on a rerun, so no
    # displayed number moves between the first run and the second
    assert first == second
    assert first["Day profit USD"] == "137.77"


def test_an_empty_root_shows_a_clean_note_not_a_stack_trace(tmp_path, monkeypatch):
    app = _run(tmp_path, monkeypatch)
    assert not app.exception
    assert any("no processed prices" in error.value for error in app.error)


def test_dispatch_profit_and_capture_hold_when_a_non_lightgbm_model_is_selected(
    tmp_path, monkeypatch
):
    _seed(tmp_path, _forecasts_multi())
    app = _run(tmp_path, monkeypatch)
    assert not app.exception
    _selectbox(app, "Forecast model").set_value(MODEL_LEAR).run()
    assert not app.exception
    # the dispatch schedule pins to the lightgbm operator, so its profit and capture stay put
    shown = {metric.label: metric.value for metric in app.metric}
    assert shown["Day profit USD"] == "137.77"
    assert shown["Capture rate percent"] == "63.21"
