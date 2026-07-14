"""Tests for the backtest core that uses no future data.

The profit, settlement, capture, and equivalent full cycle maths run on tiny synthetic frames.
Two rules are checked directly. The forecast driven profit never beats the ceiling, and no
dispatch decision ever sees a delivery day price.
"""

import math

import numpy as np
import pandas as pd
import pytest

from ercot_bess.backtest.engine import (
    RegimeError,
    capture_rate,
    equiv_full_cycles,
    run_backtest,
    run_day,
    settle_schedule,
)
from ercot_bess.backtest.schema import (
    CAPTURE_RATE,
    SCENARIO,
    SCENARIO_CEILING,
    SCENARIO_FORECAST_DRIVEN,
)
from ercot_bess.config import BatteryConfig, BatterySpec, CostSpec, load_config
from ercot_bess.features.schema import (
    DELIVERY_DATE,
    HOUR_OF_DAY,
    INTERVAL,
    REGIME,
    SETTLEMENT_POINT,
)
from ercot_bess.forecast.schema import (
    FOLD_INDEX,
    MODEL,
    MODEL_LEAR,
    MODEL_LIGHTGBM,
    PREDICTED,
    REALISED,
    enforce_forecasts_schema,
)
from ercot_bess.optimise import DispatchSchedule, optimise_dispatch
from ercot_bess.validate.schema import DA_PRICES_SCHEMA, PRICE, enforce_schema

pytestmark = pytest.mark.backtest

TOL = 1e-6

_PRIMARY = "HB_HUBAVG"
_SECONDARY = "HB_WEST"
_REGIME = "swcap5000"


def _battery(
    duration_h: float = 2.0,
    rte: float = 0.85,
    soc: float = 0.5,
    cyc: float = 0.0,
) -> BatteryConfig:
    return BatteryConfig(
        battery=BatterySpec(
            power_mw=1.0,
            duration_h=duration_h,
            round_trip_efficiency=rte,
            initial_soc_frac=soc,
            cycles_per_day_cap=None,
        ),
        cost=CostSpec(
            cycling_cost_per_mwh=cyc,
            pack_cost_per_kwh=200.0,
            cycle_life_at_80pct_dod=1000,
        ),
    )


def _cfg(battery: BatteryConfig):
    cfg = load_config()
    return cfg.model_copy(update={"battery": battery})


def _build_frames(day_specs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A forecasts frame and a matching day ahead prices frame from small day specs.

    Each spec names a local delivery day, a settlement point, and the predicted and realised
    hourly prices, and the two frames share the same interval start so they join cleanly.
    """
    forecast_rows = []
    price_rows = []
    for spec in day_specs:
        point = spec["point"]
        regime = spec.get("regime", _REGIME)
        predicted = spec["predicted"]
        realised = spec["realised"]
        local = pd.date_range(
            pd.Timestamp(spec["date"], tz="America/Chicago"), periods=len(predicted), freq="h"
        )
        utc = local.tz_convert("UTC")
        delivery_date = pd.Timestamp(spec["date"])
        for hour in range(len(predicted)):
            forecast_rows.append(
                {
                    DELIVERY_DATE: delivery_date,
                    HOUR_OF_DAY: hour,
                    INTERVAL: utc[hour],
                    SETTLEMENT_POINT: point,
                    REGIME: regime,
                    MODEL: spec.get("model", MODEL_LIGHTGBM),
                    PREDICTED: predicted[hour],
                    REALISED: realised[hour],
                    FOLD_INDEX: 0,
                }
            )
            price_rows.append(
                {
                    INTERVAL: utc[hour],
                    SETTLEMENT_POINT: point,
                    PRICE: realised[hour],
                    REGIME: regime,
                }
            )
    forecasts = enforce_forecasts_schema(pd.DataFrame(forecast_rows))
    da_prices = enforce_schema(pd.DataFrame(price_rows), DA_PRICES_SCHEMA)
    return forecasts, da_prices


def test_settle_schedule_matches_the_hand_computation():
    # net delivered is discharge minus charge, revenue values it at the realised prices
    schedule = DispatchSchedule(
        charge_mw=np.array([1.0, 0.0]),
        discharge_mw=np.array([0.0, 0.8]),
        soe_mwh=np.array([1.0, 0.5]),
        profit_usd=0.0,
        throughput_mwh=1.8,
        status="Optimal",
    )
    assert settle_schedule([10.0, 50.0], schedule, 0.0) == pytest.approx(30.0)
    # a cycling cost falls on delivered energy, here two dollars on the discharged mwh
    assert settle_schedule([10.0, 50.0], schedule, 2.0) == pytest.approx(30.0 - 1.6)


def test_equiv_full_cycles_is_throughput_over_two_times_capacity():
    schedule = DispatchSchedule(
        charge_mw=np.array([0.0]),
        discharge_mw=np.array([0.0]),
        soe_mwh=np.array([1.0]),
        profit_usd=0.0,
        throughput_mwh=3.4,
        status="Optimal",
    )
    # two hour one mw battery has four mwh of throughput per full cycle
    assert equiv_full_cycles(schedule, _battery()) == pytest.approx(3.4 / 4.0)


def test_capture_rate_undefined_when_the_ceiling_is_not_positive():
    assert capture_rate(5.0, 10.0) == pytest.approx(0.5)
    assert math.isnan(capture_rate(5.0, 0.0))
    assert math.isnan(capture_rate(-3.0, -1.0))


def test_perfect_forecast_captures_the_whole_ceiling():
    # when the prediction equals the realised prices the operator matches the ceiling exactly
    battery = _battery()
    realised = [10.0, 50.0, 12.0, 48.0]
    outcome = run_day(realised, realised, battery)
    assert outcome.forecast_driven.profit_usd == pytest.approx(outcome.ceiling.profit_usd)
    assert outcome.capture_rate == pytest.approx(1.0)


def test_forecast_driven_never_beats_the_ceiling():
    # an inverted forecast dispatches against the day, so it must earn less than the ceiling
    battery = _battery()
    realised = [10.0, 50.0, 12.0, 48.0]
    predicted = [50.0, 10.0, 48.0, 12.0]
    outcome = run_day(predicted, realised, battery)
    assert outcome.forecast_driven.profit_usd <= outcome.ceiling.profit_usd + TOL


def test_forecast_driven_settles_the_predicted_schedule_without_future_data():
    # the schedule is fixed from the predicted prices, so it is identical no matter what the
    # realised prices turn out to be, only the settled cash differs
    battery = _battery()
    predicted = [50.0, 10.0, 48.0, 12.0]
    realised_a = [10.0, 50.0, 12.0, 48.0]
    realised_b = [20.0, 40.0, 22.0, 38.0]

    out_a = run_day(predicted, realised_a, battery)
    out_b = run_day(predicted, realised_b, battery)
    assert out_a.forecast_driven.equiv_full_cycles == pytest.approx(
        out_b.forecast_driven.equiv_full_cycles
    )

    # the forecast driven profit equals settling the predicted schedule against realised
    fixed = optimise_dispatch(predicted, battery)
    assert out_a.forecast_driven.profit_usd == pytest.approx(settle_schedule(realised_a, fixed, 0.0))
    assert out_b.forecast_driven.profit_usd == pytest.approx(settle_schedule(realised_b, fixed, 0.0))


def test_flat_day_yields_nan_capture_without_crashing():
    battery = _battery()
    outcome = run_day([25.0] * 6, [25.0] * 6, battery)
    assert outcome.ceiling.profit_usd == pytest.approx(0.0, abs=TOL)
    assert math.isnan(outcome.capture_rate)


def test_run_backtest_emits_two_scenarios_per_day():
    peak = [12.0, 60.0, 15.0, 55.0]
    forecasts, da_prices = _build_frames(
        [
            {"date": "2024-06-01", "point": _PRIMARY, "predicted": peak, "realised": peak},
            {"date": "2024-06-02", "point": _PRIMARY, "predicted": peak, "realised": peak},
        ]
    )
    result = run_backtest(forecasts, da_prices, _cfg(_battery()))
    assert len(result) == 4
    assert set(result[SCENARIO]) == {SCENARIO_CEILING, SCENARIO_FORECAST_DRIVEN}
    # the ceiling row leaves capture undefined since capture describes the operator only
    ceiling = result[result[SCENARIO] == SCENARIO_CEILING]
    assert ceiling[CAPTURE_RATE].isna().all()


def test_run_backtest_keeps_the_ceiling_above_the_operator_every_day():
    rng = np.random.default_rng(7)
    specs = []
    for day in range(6):
        realised = list(30.0 + 20.0 * np.sin(2 * np.pi * (np.arange(24) - 3) / 24))
        predicted = list(np.asarray(realised) + rng.normal(0.0, 8.0, 24))
        specs.append(
            {
                "date": f"2024-06-{day + 1:02d}",
                "point": _PRIMARY,
                "predicted": predicted,
                "realised": realised,
            }
        )
    forecasts, da_prices = _build_frames(specs)
    result = run_backtest(forecasts, da_prices, _cfg(_battery()))

    ceiling = result[result[SCENARIO] == SCENARIO_CEILING].set_index(DELIVERY_DATE)["profit_usd"]
    operator = result[result[SCENARIO] == SCENARIO_FORECAST_DRIVEN].set_index(DELIVERY_DATE)[
        "profit_usd"
    ]
    assert (operator <= ceiling + TOL).all()


def test_run_backtest_handles_two_settlement_points_independently():
    peak = [12.0, 60.0, 15.0, 55.0]
    flat = [25.0, 25.0, 25.0, 25.0]
    forecasts, da_prices = _build_frames(
        [
            {"date": "2024-06-01", "point": _PRIMARY, "predicted": peak, "realised": peak},
            {"date": "2024-06-01", "point": _SECONDARY, "predicted": flat, "realised": flat},
        ]
    )
    result = run_backtest(forecasts, da_prices, _cfg(_battery()))
    assert set(result[SETTLEMENT_POINT]) == {_PRIMARY, _SECONDARY}

    # the flat point has no price spread so its operator capture is undefined
    west_operator = result[
        (result[SETTLEMENT_POINT] == _SECONDARY) & (result[SCENARIO] == SCENARIO_FORECAST_DRIVEN)
    ]
    assert west_operator[CAPTURE_RATE].isna().all()


def test_run_backtest_rejects_more_than_one_regime():
    peak = [12.0, 60.0, 15.0, 55.0]
    forecasts, da_prices = _build_frames(
        [
            {"date": "2024-06-01", "point": _PRIMARY, "predicted": peak, "realised": peak},
            {
                "date": "2025-12-06",
                "point": _PRIMARY,
                "predicted": peak,
                "realised": peak,
                "regime": "rtcb",
            },
        ]
    )
    with pytest.raises(RegimeError):
        run_backtest(forecasts, da_prices, _cfg(_battery()))


def test_run_backtest_requires_a_realised_price_for_every_forecast_hour():
    peak = [12.0, 60.0, 15.0, 55.0]
    forecasts, da_prices = _build_frames(
        [{"date": "2024-06-01", "point": _PRIMARY, "predicted": peak, "realised": peak}]
    )
    # drop a realised price so one forecast hour has nothing to settle against
    da_prices = da_prices.iloc[1:].reset_index(drop=True)
    with pytest.raises(ValueError, match="realised day ahead price"):
        run_backtest(forecasts, da_prices, _cfg(_battery()))


def test_run_backtest_needs_rows_for_the_chosen_model():
    peak = [12.0, 60.0, 15.0, 55.0]
    forecasts, da_prices = _build_frames(
        [{"date": "2024-06-01", "point": _PRIMARY, "predicted": peak, "realised": peak}]
    )
    with pytest.raises(ValueError, match="no rows for model"):
        run_backtest(forecasts, da_prices, _cfg(_battery()), model=MODEL_LEAR)
