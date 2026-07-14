"""Read the forecasts and prices, run the backtest, and write the results and summary tables.

The main work lives in engine and aggregate. Here the inputs are read from disk, the daily
backtest and its summaries are built, and the tables are written. Reading and writing are the
only side effects and they happen here, like the forecast module.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import BatteryConfig, Config
from ..forecast.build import read_forecasts as _read_forecasts_table
from ..forecast.schema import MODEL_LIGHTGBM
from ..validate.build import read_processed as _read_processed
from .aggregate import (
    N_DAYS,
    USD_PER_KW_YEAR,
    annual_summary,
    annualise_per_kw_year,
    concentration_curve,
    monthly_summary,
)
from .engine import run_backtest
from .schema import (
    SCENARIO,
    SCENARIO_CEILING,
    SCENARIO_FORECAST_DRIVEN,
    SETTLEMENT_POINT,
    enforce_backtest_schema,
)

BACKTEST_NAME = "backtest"
MONTHLY_NAME = "backtest_monthly"
ANNUAL_NAME = "backtest_annual"
ANNUALISED_NAME = "backtest_annualised_per_kw_year"
CONCENTRATION_NAME = "backtest_concentration"
SENSITIVITIES_NAME = "backtest_sensitivities"

DA_PRICES_NAME = "da_prices"

# the duration sweep, the two hour default flanked by one and four hours
_DURATIONS_H = (1.0, 2.0, 4.0)


def read_backtest_inputs(cfg: Config, repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the forecasts and the clean day ahead prices the backtest consumes."""
    results_root = repo_root / cfg.data.paths.results
    processed_root = repo_root / cfg.data.paths.processed
    forecasts = _read_forecasts_table(results_root)
    da_prices = _read_processed(processed_root, DA_PRICES_NAME)
    return forecasts, da_prices


def write_backtest(frame: pd.DataFrame, results_root: Path) -> Path:
    """Write the daily backtest table as a single parquet file and return its path."""
    path = Path(results_root) / f"{BACKTEST_NAME}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def read_backtest(results_root: Path) -> pd.DataFrame:
    """Read the backtest table and re-apply the schema after the parquet round trip."""
    path = Path(results_root) / f"{BACKTEST_NAME}.parquet"
    return enforce_backtest_schema(pd.read_parquet(path))


def write_summary(frame: pd.DataFrame, results_root: Path, name: str) -> Path:
    """Write one secondary summary table as parquet and return its path."""
    path = Path(results_root) / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _concentration_both(backtest: pd.DataFrame) -> pd.DataFrame:
    """The profit concentration curve for both scenarios, each tagged by scenario."""
    frames = []
    for scenario in (SCENARIO_CEILING, SCENARIO_FORECAST_DRIVEN):
        curve = concentration_curve(backtest, scenario)
        curve.insert(0, SCENARIO, scenario)
        frames.append(curve)
    return pd.concat(frames, ignore_index=True)


def build_summaries(backtest: pd.DataFrame, battery: BatteryConfig) -> dict[str, pd.DataFrame]:
    """Monthly and annual totals, per kW annualisation, and the concentration curve."""
    annualised = pd.concat(
        [
            annualise_per_kw_year(backtest, battery, SCENARIO_CEILING),
            annualise_per_kw_year(backtest, battery, SCENARIO_FORECAST_DRIVEN),
        ],
        ignore_index=True,
    )
    return {
        MONTHLY_NAME: monthly_summary(backtest),
        ANNUAL_NAME: annual_summary(backtest),
        ANNUALISED_NAME: annualised,
        CONCENTRATION_NAME: _concentration_both(backtest),
    }


def derived_cycling_cost_per_mwh(cost) -> float:
    """A rough cycling cost per MWh worked out from pack cost and cycle life.

    A full cycle is valued at pack cost divided by cycle life. This turns that into a cost
    per MWh discharged so it enters the same objective the optimiser and settlement use.
    """
    return cost.pack_cost_per_kwh * 1000.0 / cost.cycle_life_at_80pct_dod


def _battery_variant(
    battery: BatteryConfig, duration_h: float, cycling_cost: float
) -> BatteryConfig:
    spec = battery.battery.model_copy(update={"duration_h": duration_h})
    cost = battery.cost.model_copy(update={"cycling_cost_per_mwh": cycling_cost})
    return battery.model_copy(update={"battery": spec, "cost": cost})


def sensitivity_summary(
    forecasts: pd.DataFrame,
    da_prices: pd.DataFrame,
    cfg: Config,
    model: str = MODEL_LIGHTGBM,
) -> pd.DataFrame:
    """Annualised profit across the duration sweep with and without a cycling cost.

    This is the parameter sensitivity sweep, so the value of duration and the effect of the
    cycling cost are both visible in one table.
    """
    derived = derived_cycling_cost_per_mwh(cfg.battery.cost)
    records = []
    for duration_h in _DURATIONS_H:
        for cycling_cost in (0.0, derived):
            variant = _battery_variant(cfg.battery, duration_h, cycling_cost)
            variant_cfg = cfg.model_copy(update={"battery": variant})
            result = run_backtest(forecasts, da_prices, variant_cfg, model)
            for scenario in (SCENARIO_CEILING, SCENARIO_FORECAST_DRIVEN):
                per_kw = annualise_per_kw_year(result, variant, scenario)
                for _, row in per_kw.iterrows():
                    records.append(
                        {
                            SETTLEMENT_POINT: row[SETTLEMENT_POINT],
                            "duration_h": duration_h,
                            "cycling_cost_per_mwh": cycling_cost,
                            SCENARIO: scenario,
                            N_DAYS: int(row[N_DAYS]),
                            USD_PER_KW_YEAR: float(row[USD_PER_KW_YEAR]),
                        }
                    )
    return pd.DataFrame.from_records(records)


def run_backtest_to_disk(
    cfg: Config,
    repo_root: Path,
    *,
    model: str = MODEL_LIGHTGBM,
    write: bool = True,
    sensitivities: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Read the inputs, run the daily backtest and its summaries, and optionally write them."""
    forecasts, da_prices = read_backtest_inputs(cfg, repo_root)
    backtest = run_backtest(forecasts, da_prices, cfg, model)
    summaries = build_summaries(backtest, cfg.battery)
    sens = sensitivity_summary(forecasts, da_prices, cfg, model) if sensitivities else None
    if write:
        results_root = repo_root / cfg.data.paths.results
        write_backtest(backtest, results_root)
        for name, frame in summaries.items():
            write_summary(frame, results_root, name)
        if sens is not None:
            write_summary(sens, results_root, SENSITIVITIES_NAME)
    return backtest, summaries, sens
