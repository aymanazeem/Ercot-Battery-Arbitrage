"""The backtest core, it uses no future data and turns prices into daily profit and capture.

The two scenarios are kept apart. The ceiling runs the optimiser on the realised day ahead
prices, so it is the best the day could have done. The forecast driven operator fixes its
schedule from the predicted prices, then settles that fixed schedule against realised prices,
so no realised price on the delivery day ever feeds a dispatch choice. These functions take
frames and config and return frames, disk and the command line live in build and __main__.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import BatteryConfig, Config
from ..features.schema import DELIVERY_DATE, INTERVAL, REGIME, SETTLEMENT_POINT
from ..forecast.schema import MODEL, MODEL_LIGHTGBM, PREDICTED
from ..optimise import DispatchSchedule, optimise_dispatch
from ..validate.schema import PRICE as REALISED_PRICE
from .schema import (
    CAPTURE_RATE,
    DELIVERY_DATE as OUT_DELIVERY_DATE,
    EQUIV_FULL_CYCLES,
    PROFIT,
    SCENARIO,
    SCENARIO_CEILING,
    SCENARIO_FORECAST_DRIVEN,
    SETTLEMENT_POINT as OUT_SETTLEMENT_POINT,
    enforce_backtest_schema,
)

# day ahead prices clear hourly so each interval is one hour long
DA_INTERVAL_HOURS = 1.0

# below this a day's ceiling is treated as flat, so the capture ratio is left undefined
_CEILING_MIN = 1e-9


class RegimeError(ValueError):
    """Raised when the backtest is asked to span more than one offer cap regime."""


@dataclass(frozen=True)
class ScenarioOutcome:
    """One scenario's daily profit and its throughput expressed in whole battery cycles."""

    profit_usd: float
    equiv_full_cycles: float


@dataclass(frozen=True)
class DayOutcome:
    """Both scenarios for one delivery day and the forecast driven capture of the ceiling."""

    ceiling: ScenarioOutcome
    forecast_driven: ScenarioOutcome
    capture_rate: float


def settle_schedule(
    realised_prices: Sequence[float],
    schedule: DispatchSchedule,
    cycling_cost_per_mwh: float,
    interval_hours: float = DA_INTERVAL_HOURS,
) -> float:
    """Cash a fixed schedule earns when settled against realised prices, after cycling cost.

    The charge and discharge amounts are frozen from the forecast solve and only the prices
    are the realised ones, this is what keeps the forecast driven profit fair.
    """
    realised = np.asarray(realised_prices, dtype=float)
    net_delivered = schedule.discharge_mw - schedule.charge_mw
    revenue = float(np.sum(realised * net_delivered * interval_hours))
    wear = cycling_cost_per_mwh * float(np.sum(schedule.discharge_mw * interval_hours))
    return revenue - wear


def equiv_full_cycles(schedule: DispatchSchedule, battery: BatteryConfig) -> float:
    """Throughput measured in whole cycles, energy over two times the battery capacity."""
    spec = battery.battery
    return schedule.throughput_mwh / (2.0 * spec.power_mw * spec.duration_h)


def capture_rate(forecast_profit: float, ceiling_profit: float) -> float:
    """Forecast driven profit over the ceiling, undefined when the ceiling is not positive.

    A flat day has a ceiling at or below zero, so the ratio is left as NaN instead of
    dividing by zero, and the aggregates skip those days.
    """
    if ceiling_profit > _CEILING_MIN:
        return forecast_profit / ceiling_profit
    return float("nan")


def run_day(
    predicted_prices: Sequence[float],
    realised_prices: Sequence[float],
    battery: BatteryConfig,
    interval_hours: float = DA_INTERVAL_HOURS,
) -> DayOutcome:
    """Solve the ceiling on realised prices and settle the forecast schedule for one day."""
    realised = np.asarray(realised_prices, dtype=float)
    ceiling_schedule = optimise_dispatch(realised, battery, interval_hours)
    forecast_schedule = optimise_dispatch(predicted_prices, battery, interval_hours)

    ceiling_profit = ceiling_schedule.profit_usd
    settled_profit = settle_schedule(
        realised, forecast_schedule, battery.cost.cycling_cost_per_mwh, interval_hours
    )
    return DayOutcome(
        ceiling=ScenarioOutcome(ceiling_profit, equiv_full_cycles(ceiling_schedule, battery)),
        forecast_driven=ScenarioOutcome(
            settled_profit, equiv_full_cycles(forecast_schedule, battery)
        ),
        capture_rate=capture_rate(settled_profit, ceiling_profit),
    )


def check_single_regime(forecasts: pd.DataFrame) -> None:
    """Assert the forecasts carry one regime so no day blends two offer cap distributions."""
    regimes = sorted(forecasts[REGIME].dropna().unique())
    if len(regimes) != 1:
        raise RegimeError(f"the backtest needs a single regime, found {regimes}")


def _scenario_row(
    day: pd.Timestamp,
    point: str,
    scenario: str,
    outcome: ScenarioOutcome,
    capture: float,
) -> dict:
    return {
        OUT_DELIVERY_DATE: day,
        SCENARIO: scenario,
        OUT_SETTLEMENT_POINT: point,
        PROFIT: outcome.profit_usd,
        EQUIV_FULL_CYCLES: outcome.equiv_full_cycles,
        CAPTURE_RATE: capture,
    }


def run_backtest(
    forecasts: pd.DataFrame,
    da_prices: pd.DataFrame,
    cfg: Config,
    model: str = MODEL_LIGHTGBM,
) -> pd.DataFrame:
    """Daily ceiling and forecast driven profit for every day the chosen model forecast.

    Predicted prices come from the chosen model's forecasts and realised prices for
    settlement come from the clean day ahead prices, joined on the delivery hour and the
    settlement point. The result is the backtest results table, two rows per day, the
    ceiling and the forecast driven operator.
    """
    chosen = forecasts[forecasts[MODEL] == model]
    if chosen.empty:
        raise ValueError(f"the forecasts frame has no rows for model {model}")
    check_single_regime(chosen)

    realised = da_prices[[INTERVAL, SETTLEMENT_POINT, REALISED_PRICE]]
    joined = chosen.merge(realised, on=[INTERVAL, SETTLEMENT_POINT], how="left")
    missing = int(joined[REALISED_PRICE].isna().sum())
    if missing:
        raise ValueError(
            f"{missing} forecast hours have no realised day ahead price to settle against"
        )

    battery = cfg.battery
    ordered = joined.sort_values(INTERVAL, kind="stable")
    rows = []
    for (point, day), group in ordered.groupby([SETTLEMENT_POINT, DELIVERY_DATE], sort=True):
        outcome = run_day(
            group[PREDICTED].to_numpy(dtype=float),
            group[REALISED_PRICE].to_numpy(dtype=float),
            battery,
        )
        rows.append(_scenario_row(day, point, SCENARIO_CEILING, outcome.ceiling, float("nan")))
        rows.append(
            _scenario_row(
                day,
                point,
                SCENARIO_FORECAST_DRIVEN,
                outcome.forecast_driven,
                outcome.capture_rate,
            )
        )

    frame = pd.DataFrame.from_records(rows)
    frame = frame.sort_values(
        [OUT_SETTLEMENT_POINT, OUT_DELIVERY_DATE, SCENARIO], kind="stable"
    ).reset_index(drop=True)
    return enforce_backtest_schema(frame)
