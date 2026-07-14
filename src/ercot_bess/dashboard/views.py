"""Display shaping for the dashboard, frames and config in, plottable structures out.

No Streamlit and no disk happen here so every function is unit testable on tiny frames. The
disk reads live in data and the Streamlit wiring in app. A view never recomputes a stored
quantity, it only selects and reshapes, so the numbers a view shows trace straight back to the
table that produced them. The one reshaping that solves rather than selects is the dispatch
schedule, which is not stored, so it is reproduced with the optimiser on the day's stored
prices, the same deterministic solve the backtest dispatched.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..backtest.aggregate import CUMULATIVE_SHARE, DAY_SHARE, USD_PER_KW_YEAR
from ..backtest.engine import DA_INTERVAL_HOURS
from ..backtest.schema import CAPTURE_RATE, PROFIT, SCENARIO, SCENARIO_FORECAST_DRIVEN
from ..config import BatteryConfig
from ..features.schema import DELIVERY_DATE, HOUR_OF_DAY
from ..forecast.schema import (
    ALL_HOURS,
    MAE,
    MODEL,
    MODEL_LIGHTGBM,
    N_OBS,
    PREDICTED,
    REALISED,
    REL_MAE,
    SMAPE,
)
from ..optimise import optimise_dispatch
from ..validate.schema import INTERVAL, PRICE, SETTLEMENT_POINT

# display column names the charts key on, kept here so the app and the tests share one source
LOCAL_TIME = "local_time"
HOUR = "hour"
CHARGE_MW = "charge_mw"
DISCHARGE_MW = "discharge_mw"
SOE_MWH = "soe_mwh"

# columns of the stored sensitivities table the backtest wrote, referenced not redefined
DURATION_H = "duration_h"
CYCLING_COST = "cycling_cost_per_mwh"


def available_settlement_points(prices: pd.DataFrame) -> list[str]:
    """The settlement points present in a price table, sorted for a stable selector."""
    return sorted(prices[SETTLEMENT_POINT].dropna().unique().tolist())


def available_models(forecasts: pd.DataFrame) -> list[str]:
    """The models present in the forecasts table, sorted for a stable selector."""
    return sorted(forecasts[MODEL].dropna().unique().tolist())


def available_days(backtest: pd.DataFrame, settlement_point: str) -> list[pd.Timestamp]:
    """The delivery days the backtest scored for one settlement point, most recent first."""
    rows = backtest[backtest[SETTLEMENT_POINT] == settlement_point]
    return sorted(rows[DELIVERY_DATE].dropna().unique().tolist(), reverse=True)


def representative_day(backtest: pd.DataFrame, settlement_point: str) -> pd.Timestamp | None:
    """The delivery day whose forecast driven profit is the median for one settlement point.

    The dispatch view opens on this day so the first schedule a reader sees is a typical one,
    not the most recent, which may be one of the rare loss days and misread as the norm. The
    whole day range stays selectable and every number shown still comes from the backtest table.
    Ties break on the earlier date so the choice is deterministic.
    """
    rows = backtest[
        (backtest[SETTLEMENT_POINT] == settlement_point)
        & (backtest[SCENARIO] == SCENARIO_FORECAST_DRIVEN)
    ].dropna(subset=[PROFIT])
    if rows.empty:
        return None
    ordered = rows.sort_values([PROFIT, DELIVERY_DATE], kind="stable").reset_index(drop=True)
    median_position = (len(ordered) - 1) // 2
    return pd.Timestamp(ordered[DELIVERY_DATE].iloc[median_position])


def price_history(prices: pd.DataFrame, settlement_point: str, display_tz: str) -> pd.DataFrame:
    """Prices for one settlement point indexed by local wall clock time for a line chart.

    The stored UTC interval is converted to the display timezone so the chart reads in ERCOT
    local time, matching how the API frames a delivery day.
    """
    rows = prices[prices[SETTLEMENT_POINT] == settlement_point].sort_values(
        INTERVAL, kind="stable"
    )
    local = rows[INTERVAL].dt.tz_convert(display_tz).dt.tz_localize(None)
    return pd.DataFrame({LOCAL_TIME: local.to_numpy(), PRICE: rows[PRICE].to_numpy()})


# the default number of trailing days a price line draws. The browser plots every point client
# side so a full year of fifteen minute real time prices lags, while the numbers behind every
# metric keep reading the full resolution table
DEFAULT_CHART_DAYS = 30


def price_history_for_chart(
    prices: pd.DataFrame,
    settlement_point: str,
    display_tz: str,
    recent_days: int = DEFAULT_CHART_DAYS,
    resample_hourly: bool = False,
) -> pd.DataFrame:
    """A plotting only view of price_history, fewer points drawn but the shape kept faithful.

    Real time is resampled to hourly means so the browser draws hours not quarter hours, and both
    markets are trimmed to the most recent recent_days relative to the last interval. Only the
    line's visual density drops, latest_price and every metric still read the full resolution
    table, so no displayed number moves. Day ahead is hourly already so it is only trimmed.
    """
    history = price_history(prices, settlement_point, display_tz)
    if history.empty:
        return history
    if resample_hourly:
        hourly = history.set_index(LOCAL_TIME)[PRICE].resample("h").mean().dropna()
        history = pd.DataFrame(
            {LOCAL_TIME: hourly.index.to_numpy(), PRICE: hourly.to_numpy()}
        )
    cutoff = history[LOCAL_TIME].iloc[-1] - pd.Timedelta(recent_days, "D")
    return history[history[LOCAL_TIME] >= cutoff].reset_index(drop=True)


def latest_price(prices: pd.DataFrame, settlement_point: str) -> float | None:
    """The most recent non missing price for one settlement point, or None when there is none."""
    rows = prices[prices[SETTLEMENT_POINT] == settlement_point].sort_values(
        INTERVAL, kind="stable"
    )
    valid = rows[rows[PRICE].notna()]
    if valid.empty:
        return None
    return float(valid[PRICE].iloc[-1])


@dataclass(frozen=True)
class ForecastDay:
    """The latest delivery day of predicted against realised prices for one model."""

    delivery_date: pd.Timestamp | None
    curve: pd.DataFrame


def forecast_day(forecasts: pd.DataFrame, settlement_point: str, model: str) -> ForecastDay:
    """Predicted against realised prices for the most recent delivery day of one model."""
    rows = forecasts[
        (forecasts[SETTLEMENT_POINT] == settlement_point) & (forecasts[MODEL] == model)
    ]
    empty = pd.DataFrame({HOUR_OF_DAY: [], PREDICTED: [], REALISED: []})
    if rows.empty:
        return ForecastDay(None, empty)
    latest = rows[DELIVERY_DATE].max()
    kept = rows[rows[DELIVERY_DATE] == latest].sort_values(HOUR_OF_DAY, kind="stable")
    curve = pd.DataFrame(
        {
            HOUR_OF_DAY: kept[HOUR_OF_DAY].to_numpy(),
            PREDICTED: kept[PREDICTED].to_numpy(),
            REALISED: kept[REALISED].to_numpy(),
        }
    )
    return ForecastDay(latest, curve)


def forecast_error(metrics: pd.DataFrame, settlement_point: str, model: str) -> dict:
    """The pooled all hours error for one model, mae, smape, and the ratio to the naive baseline.

    The all hours row carries an hour of minus one, so one number stands for the whole model
    rather than a single hour that looks better or worse than the rest.
    """
    rows = metrics[
        (metrics[SETTLEMENT_POINT] == settlement_point)
        & (metrics[MODEL] == model)
        & (metrics[HOUR_OF_DAY] == ALL_HOURS)
    ]
    if rows.empty:
        return {MAE: None, SMAPE: None, REL_MAE: None, N_OBS: None}
    row = rows.iloc[0]
    return {
        MAE: float(row[MAE]),
        SMAPE: float(row[SMAPE]),
        REL_MAE: None if pd.isna(row[REL_MAE]) else float(row[REL_MAE]),
        N_OBS: int(row[N_OBS]),
    }


def dispatch_rows_for_day(
    forecasts: pd.DataFrame, settlement_point: str, delivery_date: pd.Timestamp
) -> pd.DataFrame:
    """The lightgbm forecast rows for one day and point, ordered as the backtest dispatched them.

    The backtest scored the deployed lightgbm operator, so the dispatch schedule is reproduced
    from its rows and never from the model the sidebar selects for the forecast view.
    """
    rows = forecasts[
        (forecasts[SETTLEMENT_POINT] == settlement_point)
        & (forecasts[MODEL] == MODEL_LIGHTGBM)
        & (forecasts[DELIVERY_DATE] == delivery_date)
    ]
    return rows.sort_values(INTERVAL, kind="stable")


@dataclass(frozen=True)
class DispatchDay:
    """The operator schedule for one day beside the realised prices, ready to chart by hour."""

    schedule: pd.DataFrame


def dispatch_day(day_rows: pd.DataFrame, battery: BatteryConfig) -> DispatchDay:
    """Reproduce the forecast driven schedule for one day and lay it beside the realised prices.

    The schedule is the optimiser run on the day's forecast prices, the same deterministic
    solve the backtest dispatched, so the charge and discharge shown are exactly what the
    backtest scored. The profit and capture rate are not recomputed here, they come from the
    backtest table through dispatch_summary.
    """
    ordered = day_rows.sort_values(INTERVAL, kind="stable")
    predicted = ordered[PREDICTED].to_numpy(dtype=float)
    realised = ordered[REALISED].to_numpy(dtype=float)
    schedule = optimise_dispatch(predicted, battery, DA_INTERVAL_HOURS)
    frame = pd.DataFrame(
        {
            HOUR: ordered[HOUR_OF_DAY].to_numpy(),
            PRICE: realised,
            CHARGE_MW: schedule.charge_mw,
            DISCHARGE_MW: schedule.discharge_mw,
            SOE_MWH: schedule.soe_mwh,
        }
    )
    return DispatchDay(frame)


def dispatch_summary(
    backtest: pd.DataFrame, settlement_point: str, delivery_date: pd.Timestamp
) -> dict:
    """The stored forecast driven profit and capture for one day, read straight from the backtest.

    These are the backtest numbers so what the dispatch chart shows and what the results table
    reports cannot drift apart.
    """
    rows = backtest[
        (backtest[SETTLEMENT_POINT] == settlement_point)
        & (backtest[DELIVERY_DATE] == delivery_date)
        & (backtest[SCENARIO] == SCENARIO_FORECAST_DRIVEN)
    ]
    if rows.empty:
        return {PROFIT: None, CAPTURE_RATE: None}
    row = rows.iloc[0]
    return {
        PROFIT: float(row[PROFIT]),
        CAPTURE_RATE: None if pd.isna(row[CAPTURE_RATE]) else float(row[CAPTURE_RATE]),
    }


def annualised_by_duration(sensitivities: pd.DataFrame, settlement_point: str) -> pd.DataFrame:
    """Annualised profit per kW year across the duration sweep for one settlement point.

    The stored sensitivities reduced to one location and kept in the stored unit, so each cell
    is the value the backtest wrote and not a re derived one.
    """
    rows = sensitivities[sensitivities[SETTLEMENT_POINT] == settlement_point]
    kept = rows[[DURATION_H, CYCLING_COST, SCENARIO, USD_PER_KW_YEAR]]
    return kept.sort_values(
        [SCENARIO, DURATION_H, CYCLING_COST], kind="stable"
    ).reset_index(drop=True)


def concentration_curve(
    concentration: pd.DataFrame, settlement_point: str, scenario: str
) -> pd.DataFrame:
    """The cumulative profit share against the ranked day share for one location and scenario.

    Filtered from the stored concentration table and ordered by day share so it plots as a curve
    from the most profitable day to the least.
    """
    rows = concentration[
        (concentration[SETTLEMENT_POINT] == settlement_point)
        & (concentration[SCENARIO] == scenario)
    ].sort_values(DAY_SHARE, kind="stable")
    return pd.DataFrame(
        {
            DAY_SHARE: rows[DAY_SHARE].to_numpy(dtype=float),
            CUMULATIVE_SHARE: rows[CUMULATIVE_SHARE].to_numpy(dtype=float),
        }
    )
