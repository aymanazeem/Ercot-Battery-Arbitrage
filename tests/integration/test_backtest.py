"""Integration tests for the backtest, from disk inputs through to the results tables.

Tiny frozen frames are written to disk, the backtest runs end to end, and the tables are read
back. The check that the annualised ceiling lands in a sane range lives here, on a small fixture
with no network access.
"""

import numpy as np
import pandas as pd
import pytest

from ercot_bess.backtest.aggregate import USD_PER_KW_YEAR
from ercot_bess.backtest.build import (
    ANNUAL_NAME,
    ANNUALISED_NAME,
    BACKTEST_NAME,
    CONCENTRATION_NAME,
    DA_PRICES_NAME,
    MONTHLY_NAME,
    SENSITIVITIES_NAME,
    read_backtest,
    run_backtest_to_disk,
)
from ercot_bess.backtest.schema import (
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
from ercot_bess.forecast.build import write_forecasts
from ercot_bess.forecast.schema import (
    FOLD_INDEX,
    MODEL,
    MODEL_LIGHTGBM,
    PREDICTED,
    REALISED,
    enforce_forecasts_schema,
)
from ercot_bess.validate.build import write_processed
from ercot_bess.validate.schema import DA_PRICES_SCHEMA, PRICE, enforce_schema

pytestmark = pytest.mark.backtest

TOL = 1e-6

_PRIMARY = "HB_HUBAVG"
_REGIME = "swcap5000"
_START = "2024-06-01"


def _battery() -> BatteryConfig:
    return BatteryConfig(
        battery=BatterySpec(
            power_mw=1.0,
            duration_h=2.0,
            round_trip_efficiency=0.85,
            initial_soc_frac=0.5,
            cycles_per_day_cap=None,
        ),
        cost=CostSpec(
            cycling_cost_per_mwh=0.0,
            pack_cost_per_kwh=200.0,
            cycle_life_at_80pct_dod=1000,
        ),
    )


def _diurnal(n_days: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A forecasts frame and matching day ahead prices with a clear daily peak and trough.

    The realised prices carry a strong diurnal shape so the ceiling is comfortably positive and
    the predicted prices add a little noise so the operator trails but never beats the ceiling.
    """
    rng = np.random.default_rng(seed)
    forecast_rows = []
    price_rows = []
    for day in range(n_days):
        date = pd.Timestamp(_START) + pd.Timedelta(day, "D")
        local = pd.date_range(date.tz_localize("America/Chicago"), periods=24, freq="h")
        utc = local.tz_convert("UTC")
        realised = 30.0 + 20.0 * np.sin(2 * np.pi * (np.arange(24) - 3) / 24)
        predicted = realised + rng.normal(0.0, 5.0, 24)
        delivery_date = pd.Timestamp(date.date())
        for hour in range(24):
            forecast_rows.append(
                {
                    DELIVERY_DATE: delivery_date,
                    HOUR_OF_DAY: hour,
                    INTERVAL: utc[hour],
                    SETTLEMENT_POINT: _PRIMARY,
                    REGIME: _REGIME,
                    MODEL: MODEL_LIGHTGBM,
                    PREDICTED: predicted[hour],
                    REALISED: realised[hour],
                    FOLD_INDEX: 0,
                }
            )
            price_rows.append(
                {
                    INTERVAL: utc[hour],
                    SETTLEMENT_POINT: _PRIMARY,
                    PRICE: realised[hour],
                    REGIME: _REGIME,
                }
            )
    forecasts = enforce_forecasts_schema(pd.DataFrame(forecast_rows))
    da_prices = enforce_schema(pd.DataFrame(price_rows), DA_PRICES_SCHEMA)
    return forecasts, da_prices


@pytest.fixture
def repo(tmp_path):
    cfg = load_config().model_copy(update={"battery": _battery()})
    forecasts, da_prices = _diurnal(14, 0)
    write_forecasts(forecasts, tmp_path / cfg.data.paths.results)
    write_processed(da_prices, tmp_path / cfg.data.paths.processed, DA_PRICES_NAME)
    return tmp_path, cfg


def test_run_backtest_to_disk_writes_every_table(repo):
    root, cfg = repo
    run_backtest_to_disk(cfg, root)
    results_root = root / cfg.data.paths.results
    for name in (
        BACKTEST_NAME,
        MONTHLY_NAME,
        ANNUAL_NAME,
        ANNUALISED_NAME,
        CONCENTRATION_NAME,
        SENSITIVITIES_NAME,
    ):
        assert (results_root / f"{name}.parquet").exists()


def test_backtest_table_survives_the_parquet_round_trip(repo):
    root, cfg = repo
    backtest, _, _ = run_backtest_to_disk(cfg, root)
    results_root = root / cfg.data.paths.results
    pd.testing.assert_frame_equal(read_backtest(results_root), backtest)


def test_disk_backtest_keeps_the_ceiling_above_the_operator(repo):
    root, cfg = repo
    backtest, _, _ = run_backtest_to_disk(cfg, root, sensitivities=False)
    ceiling = backtest[backtest[SCENARIO] == SCENARIO_CEILING].set_index(DELIVERY_DATE)[
        "profit_usd"
    ]
    operator = backtest[backtest[SCENARIO] == SCENARIO_FORECAST_DRIVEN].set_index(DELIVERY_DATE)[
        "profit_usd"
    ]
    assert (operator <= ceiling + TOL).all()


def test_annualised_ceiling_lands_in_a_sane_range(repo):
    # the ceiling arbitrage value should be positive and nowhere near the hundreds
    # of dollars per kW year that would signal a units or future data bug, so bound it coarsely
    root, cfg = repo
    _, summaries, _ = run_backtest_to_disk(cfg, root, write=False, sensitivities=False)
    annualised = summaries[ANNUALISED_NAME]
    ceiling = annualised[annualised[SCENARIO] == SCENARIO_CEILING][USD_PER_KW_YEAR]
    assert (ceiling > 0.0).all()
    assert (ceiling < 100.0).all()
