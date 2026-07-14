"""Tests for the battery arbitrage mixed integer linear program."""

import math

import numpy as np
import pytest

from ercot_bess.config import BatteryConfig, BatterySpec, CostSpec
from ercot_bess.optimise import optimise_dispatch

pytestmark = pytest.mark.optimise

TOL = 1e-6


def _battery(
    power_mw: float = 1.0,
    duration_h: float = 2.0,
    rte: float = 0.85,
    soc: float = 0.5,
    cyc: float = 0.0,
    cycles_cap: float | None = None,
) -> BatteryConfig:
    """A battery config built from chosen values so each test states its own physics."""
    return BatteryConfig(
        battery=BatterySpec(
            power_mw=power_mw,
            duration_h=duration_h,
            round_trip_efficiency=rte,
            initial_soc_frac=soc,
            cycles_per_day_cap=cycles_cap,
        ),
        cost=CostSpec(
            cycling_cost_per_mwh=cyc,
            pack_cost_per_kwh=200.0,
            cycle_life_at_80pct_dod=1000,
        ),
    )


def test_two_price_toy_returns_the_analytic_optimum():
    # buy a full power hour cheap, sell back what survives the round trip when dear, the
    # charge leg is limited by power not capacity so the optimum is closed form
    battery = _battery()
    result = optimise_dispatch([10.0, 50.0], battery)

    rte = 0.85
    expected_charge = 1.0
    expected_discharge = rte * expected_charge
    expected_profit = 50.0 * expected_discharge - 10.0 * expected_charge

    assert result.profit_usd == pytest.approx(expected_profit)
    assert result.charge_mw == pytest.approx([expected_charge, 0.0], abs=TOL)
    assert result.discharge_mw == pytest.approx([0.0, expected_discharge], abs=TOL)


def test_energy_balance_holds_every_interval():
    battery = _battery()
    prices = [30.0, 12.0, 45.0, 20.0, 60.0, 15.0, 55.0, 25.0]
    result = optimise_dispatch(prices, battery)

    one_way = math.sqrt(0.85)
    capacity = 2.0
    initial = 0.5 * capacity

    previous = initial
    for t in range(len(prices)):
        expected = previous + one_way * result.charge_mw[t] - result.discharge_mw[t] / one_way
        assert result.soe_mwh[t] == pytest.approx(expected, abs=TOL)
        previous = result.soe_mwh[t]

    # the horizon returns to its starting state of energy, the energy neutral boundary
    assert result.soe_mwh[-1] == pytest.approx(initial, abs=TOL)
    assert np.all(result.soe_mwh >= -TOL)
    assert np.all(result.soe_mwh <= capacity + TOL)


def test_binary_forbids_simultaneous_charge_and_discharge():
    # one interval at a deeply negative price, the battery is paid to consume, and the only
    # way to earn is to draw net power by running both legs at once, which the binary bans
    battery = _battery()
    result = optimise_dispatch([-100.0], battery)

    assert np.all(np.minimum(result.charge_mw, result.discharge_mw) < TOL)
    assert result.profit_usd == pytest.approx(0.0, abs=TOL)

    # a simultaneous point meets every other constraint and would earn a positive amount, so
    # the binary is exactly what removes this gaming
    rte = 0.85
    naive_charge = 1.0
    naive_discharge = rte * naive_charge
    naive_cash = -100.0 * (naive_discharge - naive_charge)
    assert naive_cash > result.profit_usd + TOL


def test_large_cycling_cost_drives_throughput_to_zero():
    prices = [20.0, 80.0, 15.0, 90.0, 10.0, 75.0]
    free = optimise_dispatch(prices, _battery(cyc=0.0))
    assert free.throughput_mwh > TOL

    penalised = optimise_dispatch(prices, _battery(cyc=1_000_000.0))
    assert penalised.throughput_mwh == pytest.approx(0.0, abs=TOL)
    assert penalised.profit_usd == pytest.approx(0.0, abs=TOL)


def test_flat_prices_leave_the_battery_idle():
    # at a single flat price every cycle only loses to efficiency so idle is optimal
    result = optimise_dispatch([25.0] * 6, _battery())
    assert result.throughput_mwh == pytest.approx(0.0, abs=TOL)
    assert result.profit_usd == pytest.approx(0.0, abs=TOL)


def test_cycle_cap_limits_throughput():
    prices = [20.0, 80.0, 15.0, 90.0, 10.0, 75.0]
    uncapped = optimise_dispatch(prices, _battery(cycles_cap=None))
    capped = optimise_dispatch(prices, _battery(cycles_cap=0.5))

    capacity = 2.0
    horizon_days = len(prices) / 24.0
    limit = 0.5 * horizon_days * 2.0 * capacity
    assert capped.throughput_mwh <= limit + TOL
    assert capped.throughput_mwh <= uncapped.throughput_mwh + TOL


def test_repeated_solves_are_identical():
    prices = [30.0, 12.0, 45.0, 20.0, 60.0, 15.0]
    battery = _battery()
    first = optimise_dispatch(prices, battery)
    second = optimise_dispatch(prices, battery)
    assert first.profit_usd == pytest.approx(second.profit_usd)
    assert np.array_equal(first.charge_mw, second.charge_mw)
    assert np.array_equal(first.discharge_mw, second.discharge_mw)
