"""Tests for the dashboard shaping functions, run on tiny frames with no Streamlit.

These cover the view functions only. The disk edge and the assembled app are covered by the
integration test. The dispatch check pins the view to the optimiser so the reproduced schedule
matches the one the backtest ran. The summary check pins the shown profit and capture to the
stored backtest so nothing is recomputed differently in the UI.
"""

import numpy as np
import pandas as pd
import pytest

from ercot_bess.backtest.aggregate import CUMULATIVE_SHARE, DAY_SHARE, USD_PER_KW_YEAR
from ercot_bess.backtest.engine import DA_INTERVAL_HOURS
from ercot_bess.backtest.schema import (
    CAPTURE_RATE,
    DELIVERY_DATE,
    PROFIT,
    SCENARIO,
    SCENARIO_CEILING,
    SCENARIO_FORECAST_DRIVEN,
    SETTLEMENT_POINT,
)
from ercot_bess.config import load_config
from ercot_bess.dashboard import views
from ercot_bess.forecast.schema import (
    ALL_HOURS,
    MAE,
    MODEL,
    MODEL_LEAR,
    MODEL_LIGHTGBM,
    N_OBS,
    PREDICTED,
    REALISED,
    REL_MAE,
    SMAPE,
)
from ercot_bess.optimise import optimise_dispatch
from ercot_bess.validate.schema import INTERVAL, PRICE, REGIME
from ercot_bess.features.schema import HOUR_OF_DAY

pytestmark = pytest.mark.dashboard

_POINT = "HB_HUBAVG"
_TZ = "America/Chicago"
_DAY = pd.Timestamp("2024-06-01")


def _prices() -> pd.DataFrame:
    """Three hourly day ahead prices for the hub, ending on a missing value."""
    start = pd.Timestamp("2024-06-01 05:00", tz="UTC")
    interval = pd.date_range(start, periods=3, freq="h")
    return pd.DataFrame(
        {
            INTERVAL: interval,
            SETTLEMENT_POINT: [_POINT] * 3,
            PRICE: [10.0, 20.0, np.nan],
            REGIME: ["swcap5000"] * 3,
        }
    )


def _day_rows() -> pd.DataFrame:
    """One delivery day of forecast rows for the primary hub, deliberately out of interval order."""
    start = pd.Timestamp("2024-06-01", tz="UTC")
    predicted = [50.0, 10.0, 60.0, 5.0, 55.0, 8.0]
    realised = [48.0, 12.0, 58.0, 6.0, 52.0, 9.0]
    rows = pd.DataFrame(
        {
            DELIVERY_DATE: [_DAY] * 6,
            HOUR_OF_DAY: list(range(6)),
            INTERVAL: [start + pd.Timedelta(hour, "h") for hour in range(6)],
            SETTLEMENT_POINT: [_POINT] * 6,
            MODEL: [MODEL_LIGHTGBM] * 6,
            PREDICTED: predicted,
            REALISED: realised,
        }
    )
    return rows.sample(frac=1.0, random_state=0)


def test_price_history_filters_sorts_and_converts_to_local_wall_clock():
    history = views.price_history(_prices(), _POINT, _TZ)
    assert list(history[PRICE].to_numpy()[:2]) == [10.0, 20.0]
    # five in the morning utc is midnight in central daylight time
    assert history[views.LOCAL_TIME].iloc[0] == pd.Timestamp("2024-06-01 00:00:00")


def _rt_history() -> pd.DataFrame:
    """Three local days of fifteen minute prices for one hub, with one flat topped peak hour.

    The window starts and ends on a local hour boundary so hourly bucketing keeps the first and
    last timestamps, and the peak fills a whole hour so its hourly mean stays the global maximum.
    """
    start = pd.Timestamp("2024-06-01 05:00", tz="UTC")  # local midnight in central daylight time
    periods = 3 * 96 + 1  # through local midnight three days on, so the final hour is a lone point
    interval = pd.date_range(start, periods=periods, freq="15min")
    price = 20.0 + np.linspace(0.0, 5.0, periods)
    price[40:44] = 500.0  # a full local hour at the peak, so its hourly mean is the global max
    return pd.DataFrame(
        {INTERVAL: interval, SETTLEMENT_POINT: [_POINT] * periods, PRICE: price}
    )


def test_price_history_for_chart_draws_fewer_points_but_keeps_the_shape():
    prices = _rt_history()
    full = views.price_history(prices, _POINT, _TZ)
    chart = views.price_history_for_chart(
        prices, _POINT, _TZ, recent_days=30, resample_hourly=True
    )
    # the hourly resample hands the browser far fewer points than the fifteen minute table
    assert len(chart) < len(full)
    # yet the window the eye reads stays faithful, both endpoints and the peak are unmoved
    assert chart[views.LOCAL_TIME].iloc[0] == full[views.LOCAL_TIME].iloc[0]
    assert chart[views.LOCAL_TIME].iloc[-1] == full[views.LOCAL_TIME].iloc[-1]
    assert chart[PRICE].max() == full[PRICE].max()


def test_price_history_for_chart_trims_day_ahead_to_the_recent_window():
    prices = _rt_history()
    full = views.price_history(prices, _POINT, _TZ)
    chart = views.price_history_for_chart(prices, _POINT, _TZ, recent_days=1)
    # day ahead is hourly already so it is only trimmed, never resampled, keeping full resolution
    assert len(chart) < len(full)
    assert chart[views.LOCAL_TIME].iloc[-1] == full[views.LOCAL_TIME].iloc[-1]
    span = chart[views.LOCAL_TIME].iloc[-1] - chart[views.LOCAL_TIME].iloc[0]
    assert span <= pd.Timedelta(1, "D")


def test_latest_price_skips_the_trailing_missing_value():
    assert views.latest_price(_prices(), _POINT) == 20.0


def test_latest_price_is_none_for_an_unknown_point():
    assert views.latest_price(_prices(), "NOPE") is None


def test_forecast_day_returns_the_latest_day_sorted_by_hour():
    start = pd.Timestamp("2024-06-01", tz="UTC")
    rows = []
    for day in ("2024-06-01", "2024-06-02"):
        for hour in (2, 0, 1):
            rows.append(
                {
                    DELIVERY_DATE: pd.Timestamp(day),
                    HOUR_OF_DAY: hour,
                    INTERVAL: start + pd.Timedelta(hour, "h"),
                    SETTLEMENT_POINT: _POINT,
                    MODEL: MODEL_LIGHTGBM,
                    PREDICTED: 30.0 + hour,
                    REALISED: 31.0 + hour,
                }
            )
    day = views.forecast_day(pd.DataFrame(rows), _POINT, MODEL_LIGHTGBM)
    assert day.delivery_date == pd.Timestamp("2024-06-02")
    assert list(day.curve[HOUR_OF_DAY].to_numpy()) == [0, 1, 2]


def test_forecast_day_is_empty_for_an_unknown_point():
    day = views.forecast_day(_day_rows(), "NOPE", MODEL_LIGHTGBM)
    assert day.delivery_date is None
    assert day.curve.empty


def _metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            MODEL: [MODEL_LIGHTGBM, MODEL_LIGHTGBM],
            SETTLEMENT_POINT: [_POINT, _POINT],
            REGIME: ["swcap5000", "swcap5000"],
            HOUR_OF_DAY: [ALL_HOURS, 0],
            N_OBS: [48, 2],
            MAE: [7.5, 9.0],
            SMAPE: [21.0, 25.0],
            REL_MAE: [0.83, 0.90],
        }
    )


def test_forecast_error_reads_the_pooled_all_hours_row():
    error = views.forecast_error(_metrics(), _POINT, MODEL_LIGHTGBM)
    assert error[MAE] == 7.5
    assert error[SMAPE] == 21.0
    assert error[REL_MAE] == 0.83
    assert error[N_OBS] == 48


def test_forecast_error_is_none_when_the_cell_is_missing():
    error = views.forecast_error(_metrics(), "NOPE", MODEL_LIGHTGBM)
    assert error[MAE] is None
    assert error[REL_MAE] is None


def test_dispatch_day_reproduces_the_schedule_exactly():
    cfg = load_config()
    rows = _day_rows()
    ordered = rows.sort_values(INTERVAL, kind="stable")
    reference = optimise_dispatch(
        ordered[PREDICTED].to_numpy(dtype=float), cfg.battery, DA_INTERVAL_HOURS
    )
    schedule = views.dispatch_day(rows, cfg.battery).schedule
    np.testing.assert_allclose(schedule[views.CHARGE_MW].to_numpy(), reference.charge_mw)
    np.testing.assert_allclose(schedule[views.DISCHARGE_MW].to_numpy(), reference.discharge_mw)
    np.testing.assert_allclose(schedule[views.SOE_MWH].to_numpy(), reference.soe_mwh)
    # the price line is the realised price the schedule settles against, in interval order
    np.testing.assert_allclose(
        schedule[PRICE].to_numpy(), ordered[REALISED].to_numpy(dtype=float)
    )
    assert list(schedule[views.HOUR].to_numpy()) == list(range(6))


def test_dispatch_rows_for_day_pins_to_lightgbm_ignoring_the_selected_model():
    cfg = load_config()
    lightgbm_rows = _day_rows()
    # a second model on the same day and point, priced in the reverse hour order so its schedule
    # would differ if the view mistakenly followed the selected model rather than the operator
    other = lightgbm_rows.sort_values(INTERVAL, kind="stable").copy()
    other[MODEL] = MODEL_LEAR
    other[PREDICTED] = other[PREDICTED].to_numpy()[::-1]
    forecasts = pd.concat([lightgbm_rows, other], ignore_index=True)

    pinned = views.dispatch_rows_for_day(forecasts, _POINT, _DAY)
    assert list(pinned[MODEL].unique()) == [MODEL_LIGHTGBM]

    pinned_schedule = views.dispatch_day(pinned, cfg.battery).schedule
    lightgbm_schedule = views.dispatch_day(
        views.dispatch_rows_for_day(lightgbm_rows, _POINT, _DAY), cfg.battery
    ).schedule
    other_schedule = views.dispatch_day(other, cfg.battery).schedule
    # the pinned schedule is the lightgbm one, and the other model would have dispatched differently
    np.testing.assert_allclose(
        pinned_schedule[views.CHARGE_MW].to_numpy(), lightgbm_schedule[views.CHARGE_MW].to_numpy()
    )
    assert not np.allclose(
        pinned_schedule[views.DISCHARGE_MW].to_numpy(), other_schedule[views.DISCHARGE_MW].to_numpy()
    )


def _backtest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            DELIVERY_DATE: [_DAY, _DAY],
            SCENARIO: [SCENARIO_CEILING, SCENARIO_FORECAST_DRIVEN],
            SETTLEMENT_POINT: [_POINT, _POINT],
            PROFIT: [200.0, 137.77],
            "equiv_full_cycles": [1.2, 1.0],
            CAPTURE_RATE: [np.nan, 0.6321],
        }
    )


def test_dispatch_summary_reads_the_stored_profit_and_capture():
    summary = views.dispatch_summary(_backtest(), _POINT, _DAY)
    assert summary[PROFIT] == 137.77
    assert summary[CAPTURE_RATE] == 0.6321


def test_dispatch_summary_is_none_for_a_day_without_a_row():
    summary = views.dispatch_summary(_backtest(), _POINT, pd.Timestamp("2024-07-04"))
    assert summary[PROFIT] is None
    assert summary[CAPTURE_RATE] is None


def test_available_days_are_most_recent_first():
    frame = _backtest()
    extra = frame.copy()
    extra[DELIVERY_DATE] = pd.Timestamp("2024-06-03")
    days = views.available_days(pd.concat([frame, extra], ignore_index=True), _POINT)
    assert days == [pd.Timestamp("2024-06-03"), _DAY]


def _sensitivities() -> pd.DataFrame:
    records = [
        {
            SETTLEMENT_POINT: _POINT,
            views.DURATION_H: duration,
            views.CYCLING_COST: 0.0,
            SCENARIO: SCENARIO_CEILING,
            "n_days": 30,
            USD_PER_KW_YEAR: duration * 10.0,
        }
        for duration in (1.0, 2.0, 4.0)
    ]
    return pd.DataFrame(records)


def test_annualised_by_duration_keeps_only_the_point_and_the_stored_values():
    table = views.annualised_by_duration(_sensitivities(), _POINT)
    assert list(table[views.DURATION_H].to_numpy()) == [1.0, 2.0, 4.0]
    assert list(table[USD_PER_KW_YEAR].to_numpy()) == [10.0, 20.0, 40.0]


def _concentration() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SCENARIO: [SCENARIO_CEILING] * 3 + [SCENARIO_FORECAST_DRIVEN] * 3,
            SETTLEMENT_POINT: [_POINT] * 6,
            DAY_SHARE: [0.33, 0.66, 1.0] * 2,
            CUMULATIVE_SHARE: [0.5, 0.8, 1.0, 0.4, 0.7, 1.0],
        }
    )


def test_concentration_curve_filters_the_point_and_scenario_ordered_by_day_share():
    curve = views.concentration_curve(_concentration(), _POINT, SCENARIO_FORECAST_DRIVEN)
    assert list(curve[DAY_SHARE].to_numpy()) == [0.33, 0.66, 1.0]
    assert list(curve[CUMULATIVE_SHARE].to_numpy()) == [0.4, 0.7, 1.0]
