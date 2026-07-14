"""Summaries over the daily backtest, monthly and annual totals, per kW annualisation, and the
profit concentration curve.

These are functions that take the daily backtest frame and return summary frames. They are
secondary to the daily table so they define their own summary columns rather than extending
it. Reading and writing happen in build.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BatteryConfig
from .schema import (
    CAPTURE_RATE,
    DELIVERY_DATE,
    EQUIV_FULL_CYCLES,
    PROFIT,
    SCENARIO,
    SETTLEMENT_POINT,
)

# a mean solar and calendar year, used to scale a mean daily profit to an annual figure
_DAYS_PER_YEAR = 365.25

PERIOD = "period"
N_DAYS = "n_days"
USD_PER_KW_YEAR = "usd_per_kw_year"
DAY_RANK = "day_rank"
DAY_SHARE = "day_share"
CUMULATIVE_PROFIT = "cumulative_profit_usd"
CUMULATIVE_SHARE = "cumulative_share"


def summarise_by_period(backtest: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Total profit and cycles and mean capture per settlement point, scenario, and period.

    The capture mean skips the undefined days so a flat day does not drag the average, and
    the ceiling rows aggregate to a missing capture rather than a made up number.
    """
    frame = backtest.copy()
    frame[PERIOD] = frame[DELIVERY_DATE].dt.to_period(freq).dt.to_timestamp()
    grouped = frame.groupby([SETTLEMENT_POINT, SCENARIO, PERIOD], sort=True)
    summary = grouped.agg(
        profit_usd=(PROFIT, "sum"),
        equiv_full_cycles=(EQUIV_FULL_CYCLES, "sum"),
        capture_rate=(CAPTURE_RATE, "mean"),
        n_days=(PROFIT, "size"),
    ).reset_index()
    return summary


def monthly_summary(backtest: pd.DataFrame) -> pd.DataFrame:
    """Monthly totals per settlement point and scenario."""
    return summarise_by_period(backtest, "M")


def annual_summary(backtest: pd.DataFrame) -> pd.DataFrame:
    """Annual totals per settlement point and scenario."""
    return summarise_by_period(backtest, "Y")


def annualise_per_kw_year(
    backtest: pd.DataFrame, battery: BatteryConfig, scenario: str
) -> pd.DataFrame:
    """Annualised profit per kW for one scenario so it compares to published benchmarks.

    The mean daily profit is scaled to a full year then divided by the rated power in kW,
    the unit the industry quotes battery revenue in.
    """
    rows = backtest[backtest[SCENARIO] == scenario]
    power_kw = battery.battery.power_mw * 1000.0
    records = []
    for point, group in rows.groupby(SETTLEMENT_POINT, sort=True):
        mean_daily = float(group[PROFIT].mean())
        records.append(
            {
                SETTLEMENT_POINT: point,
                SCENARIO: scenario,
                N_DAYS: int(len(group)),
                USD_PER_KW_YEAR: mean_daily * _DAYS_PER_YEAR / power_kw,
            }
        )
    return pd.DataFrame.from_records(records)


def concentration_curve(backtest: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Cumulative share of profit against days ranked from most to least profitable.

    This shows the well known pattern that a few days carry most of the profit. The curve
    is built per settlement point so ranking never mixes two locations.
    """
    rows = backtest[backtest[SCENARIO] == scenario]
    curves = []
    for point, group in rows.groupby(SETTLEMENT_POINT, sort=True):
        ordered = group.sort_values(PROFIT, ascending=False, kind="stable").reset_index(drop=True)
        total = float(ordered[PROFIT].sum())
        count = len(ordered)
        ordered[DAY_RANK] = np.arange(1, count + 1)
        ordered[CUMULATIVE_PROFIT] = ordered[PROFIT].cumsum()
        ordered[CUMULATIVE_SHARE] = (
            ordered[CUMULATIVE_PROFIT] / total if total > 0 else np.nan
        )
        ordered[DAY_SHARE] = ordered[DAY_RANK] / count
        curves.append(ordered)
    result = pd.concat(curves, ignore_index=True)
    return result[
        [
            SETTLEMENT_POINT,
            DAY_RANK,
            DAY_SHARE,
            DELIVERY_DATE,
            PROFIT,
            CUMULATIVE_PROFIT,
            CUMULATIVE_SHARE,
        ]
    ]
