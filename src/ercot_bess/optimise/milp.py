"""The battery arbitrage optimiser as an exact mixed integer linear program.

Given a price vector and the battery config this returns the profit maximising charge and
discharge schedule for one horizon. It is a plain function that uses no disk or network, so
the backtest can import and call it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pulp

from ..config import BatteryConfig

# the cycle cap is quoted per day so the horizon has to be measured in days against it
_HOURS_PER_DAY = 24.0


@dataclass(frozen=True)
class DispatchSchedule:
    """The optimised dispatch, charge and discharge power, state of energy, and net profit."""

    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soe_mwh: np.ndarray
    profit_usd: float
    throughput_mwh: float
    status: str


def optimise_dispatch(
    prices_usd_per_mwh: Sequence[float],
    battery: BatteryConfig,
    interval_hours: float = 1.0,
) -> DispatchSchedule:
    """Return the profit maximising charge and discharge schedule for the given prices.

    The battery ends the horizon with the same stored energy it started with, so a day cannot
    make fake profit by selling off its starting charge.
    """
    prices = [float(price) for price in prices_usd_per_mwh]
    intervals = len(prices)
    if intervals == 0:
        raise ValueError("the price vector is empty, there is nothing to schedule")

    spec = battery.battery
    power = spec.power_mw
    capacity = spec.power_mw * spec.duration_h
    one_way = sqrt(spec.round_trip_efficiency)
    initial_soe = spec.initial_soc_frac * capacity
    cycling_cost = battery.cost.cycling_cost_per_mwh
    step = interval_hours

    prob = pulp.LpProblem("battery_arbitrage", pulp.LpMaximize)
    charge = [pulp.LpVariable(f"charge_{t}", lowBound=0, upBound=power) for t in range(intervals)]
    discharge = [
        pulp.LpVariable(f"discharge_{t}", lowBound=0, upBound=power) for t in range(intervals)
    ]
    soe = [pulp.LpVariable(f"soe_{t}", lowBound=0, upBound=capacity) for t in range(intervals)]
    is_charging = [pulp.LpVariable(f"is_charging_{t}", cat="Binary") for t in range(intervals)]

    revenue = pulp.lpSum(prices[t] * discharge[t] * step for t in range(intervals))
    charge_cost = pulp.lpSum(prices[t] * charge[t] * step for t in range(intervals))
    # cycling cost is charged on energy sent out, it stands for battery wear
    wear_cost = pulp.lpSum(cycling_cost * discharge[t] * step for t in range(intervals))
    prob += revenue - charge_cost - wear_cost

    for t in range(intervals):
        previous = soe[t - 1] if t > 0 else initial_soe
        # one way efficiencies on each leg multiply to the round trip
        prob += soe[t] == previous + one_way * charge[t] * step - discharge[t] * step / one_way
        # the on off flag lets only charge or discharge run in an interval, never both
        prob += charge[t] <= power * is_charging[t]
        prob += discharge[t] <= power * (1 - is_charging[t])

    # end the horizon at the same stored energy it started from
    prob += soe[intervals - 1] == initial_soe

    if spec.cycles_per_day_cap is not None:
        horizon_days = intervals * step / _HOURS_PER_DAY
        # one full cycle moves twice the capacity, once in and once out
        throughput_limit = spec.cycles_per_day_cap * horizon_days * 2.0 * capacity
        moved = pulp.lpSum((charge[t] + discharge[t]) * step for t in range(intervals))
        prob += moved <= throughput_limit

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(f"the optimiser did not converge, solver status {status}")

    charge_mw = np.array([charge[t].value() for t in range(intervals)], dtype=float)
    discharge_mw = np.array([discharge[t].value() for t in range(intervals)], dtype=float)
    soe_mwh = np.array([soe[t].value() for t in range(intervals)], dtype=float)
    throughput_mwh = float(np.sum((charge_mw + discharge_mw) * step))
    profit_usd = float(pulp.value(prob.objective))

    return DispatchSchedule(
        charge_mw=charge_mw,
        discharge_mw=discharge_mw,
        soe_mwh=soe_mwh,
        profit_usd=profit_usd,
        throughput_mwh=throughput_mwh,
        status=status,
    )
