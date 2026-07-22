"""The Streamlit dashboard, a read only front end over the precomputed artifacts.

The app never recomputes a stored result, it reads a table through the data layer, shapes it
with the view functions, and draws it. The one exception is the dispatch schedule, which no
results table stores, so it is reproduced by the optimiser on the day's stored prices, the
same deterministic solve the backtest dispatched. The day's profit and capture rate still come
from the backtest table so nothing shown drifts from the results. The data root comes from the
environment so a test can seed a temp directory and run the app with no network.

The imports are absolute because Streamlit runs this file as a script, so a relative import would
have no parent package to resolve against.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ercot_bess.backtest.aggregate import DAY_SHARE
from ercot_bess.backtest.schema import (
    CAPTURE_RATE,
    PROFIT,
    SCENARIO_CEILING,
    SCENARIO_FORECAST_DRIVEN,
)
from ercot_bess.config import BatteryConfig, Config, load_config
from ercot_bess.dashboard import views
from ercot_bess.dashboard.data import DashboardData, DashboardDataMissing, resolve_root
from ercot_bess.features.schema import HOUR_OF_DAY
from ercot_bess.forecast.schema import MAE, MODEL_LIGHTGBM, REL_MAE, SMAPE
from ercot_bess.validate.schema import PRICE

_SCENARIO_LABELS = ((SCENARIO_CEILING, "ceiling"), (SCENARIO_FORECAST_DRIVEN, "forecast driven"))

# match the read cache in data.py so a pipeline refresh clears the reproduced schedule at the
# same time. The solve is the only recompute the app does.
_DISPATCH_CACHE_TTL_SECONDS = 300


def _fmt(value: float | None, spec: str = "{:.2f}") -> str:
    """Format a number for a metric, or a plain dash when the value is missing."""
    return "n/a" if value is None else spec.format(value)


def _battery_signature(battery: BatteryConfig) -> tuple:
    """A hashable digest of the battery fields the dispatch solve depends on, for the cache key."""
    spec, cost = battery.battery, battery.cost
    return (
        spec.power_mw,
        spec.duration_h,
        spec.round_trip_efficiency,
        spec.initial_soc_frac,
        spec.cycles_per_day_cap,
        cost.cycling_cost_per_mwh,
    )


@st.cache_data(ttl=_DISPATCH_CACHE_TTL_SECONDS, max_entries=256)
def _dispatch_schedule_cached(
    results_root: str,
    settlement_point: str,
    iso_delivery_date: str,
    battery_sig: tuple,
    _day_rows: pd.DataFrame,
    _battery: BatteryConfig,
) -> pd.DataFrame:
    """The reproduced schedule for one day, cached so a rerun does not re solve the MILP.

    The key is the results root, the point, the delivery day, and the battery signature, since
    those determine the solve. The frame and the dataclass lead with an underscore so Streamlit
    passes them through without hashing. The profit and capture shown are read from the backtest,
    not from here, so caching the solve cannot move a displayed number.
    """
    return views.dispatch_day(_day_rows, _battery).schedule


def _index_of(options: list, value, default: int = 0) -> int:
    """The position of value in options for a selectbox default, or a fallback index."""
    return options.index(value) if value in options else default


def _render_prices(data: DashboardData, cfg: Config, point: str, chart_days: int) -> None:
    st.subheader(f"Prices for {point}")
    display_tz = cfg.market.market.timezone_display
    for market, label in (("da", "Day ahead"), ("rt", "Real time")):
        try:
            prices = data.prices(market)
        except DashboardDataMissing:
            st.info(f"{label} prices are not built yet.")
            continue
        st.metric(f"{label} latest price USD per MWh", _fmt(views.latest_price(prices, point)))
        history = views.price_history_for_chart(
            prices, point, display_tz, chart_days, resample_hourly=market == "rt"
        )
        if history.empty:
            st.info(f"no {label} prices for {point}.")
        else:
            st.line_chart(history.set_index(views.LOCAL_TIME))


def _render_forecast(data: DashboardData, cfg: Config, point: str, model: str) -> None:
    st.subheader(f"Forecast against realised for {model}")
    try:
        forecasts = data.forecasts()
        metrics = data.metrics()
    except DashboardDataMissing:
        st.info("forecasts are not built yet.")
        return
    day = views.forecast_day(forecasts, point, model)
    if day.curve.empty:
        st.info("the forecast is not built yet, run the pipeline first.")
        return
    error = views.forecast_error(metrics, point, model)
    st.metric("MAE USD per MWh", _fmt(error[MAE]))
    st.metric("sMAPE percent", _fmt(error[SMAPE]))
    st.metric("MAE relative to naive", _fmt(error[REL_MAE], "{:.3f}"))
    st.caption("relative to naive below one means the model beats last week same hour")
    st.caption(f"delivery day {day.delivery_date.date().isoformat()}")
    st.line_chart(day.curve.set_index(HOUR_OF_DAY))


def _render_dispatch(data: DashboardData, cfg: Config, point: str) -> None:
    st.subheader("Dispatch for a chosen day")
    try:
        forecasts = data.forecasts()
        backtest = data.backtest()
    except DashboardDataMissing:
        st.info("the backtest is not built yet.")
        return
    days = views.available_days(backtest, point)
    if not days:
        st.info("the dispatch backtest is not built yet, run the pipeline first.")
        return
    default_day = views.representative_day(backtest, point)
    day = st.selectbox(
        "Delivery day",
        days,
        index=_index_of(days, default_day),
        format_func=lambda value: pd.Timestamp(value).date().isoformat(),
    )
    st.caption("opens on a typical day, the one with the median forecast driven profit. any day is selectable")
    summary = views.dispatch_summary(backtest, point, day)
    st.metric("Day profit USD", _fmt(summary[PROFIT]))
    capture = summary[CAPTURE_RATE]
    st.metric("Capture rate percent", _fmt(None if capture is None else capture * 100.0))
    day_rows = views.dispatch_rows_for_day(forecasts, point, day)
    if day_rows.empty:
        st.info("no forecast prices to reproduce the schedule for this day.")
        return
    st.caption(
        "the schedule is the deployed lightgbm operator the backtest scored, "
        "not the forecast model chosen in the sidebar"
    )
    schedule = _dispatch_schedule_cached(
        str(resolve_root()),
        point,
        pd.Timestamp(day).date().isoformat(),
        _battery_signature(cfg.battery),
        day_rows,
        cfg.battery,
    )
    dispatch = schedule.set_index(views.HOUR)
    st.line_chart(dispatch[[PRICE]])
    st.line_chart(dispatch[[views.CHARGE_MW, views.DISCHARGE_MW]])
    st.line_chart(dispatch[[views.SOE_MWH]])


def _render_results(data: DashboardData, cfg: Config, point: str) -> None:
    st.subheader("Annualised profit and profit concentration")
    try:
        sensitivities = data.sensitivities()
        concentration = data.concentration()
    except DashboardDataMissing:
        st.info("the backtest summaries are not built yet.")
        return
    table = views.annualised_by_duration(sensitivities, point)
    if table.empty:
        st.info("the results are not built yet, run the pipeline first.")
        return
    st.markdown("Annualised profit per kW year across duration, scenario, and cycling cost")
    st.dataframe(table)
    for scenario, label in _SCENARIO_LABELS:
        curve = views.concentration_curve(concentration, point, scenario)
        if curve.empty:
            continue
        st.markdown(f"Profit concentration, {label}")
        st.line_chart(curve.set_index(DAY_SHARE))
        if scenario == SCENARIO_FORECAST_DRIVEN:
            st.caption(
                "days are ranked most to least profitable, so the height is the cumulative share of "
                "the year's profit. it can pass 100 percent because the worst days lose money, so the "
                "winning days earn more than the net total and the losing days pull the curve back to 100 percent."
            )
        else:
            st.caption(
                "days are ranked most to least profitable, so the height is the cumulative share of the "
                "year's profit. a steep start means a handful of days carry most of it."
            )


def _models(data: DashboardData) -> list[str]:
    """The models to offer in the selector, falling back to the default when none are built."""
    try:
        return views.available_models(data.forecasts()) or [MODEL_LIGHTGBM]
    except DashboardDataMissing:
        return [MODEL_LIGHTGBM]


def main() -> None:
    """Build the dashboard, one sidebar selection driving four views over the artifacts."""
    st.set_page_config(page_title="ERCOT BESS", layout="wide")
    cfg = load_config()
    data = DashboardData(cfg, resolve_root())
    st.title("ERCOT battery arbitrage dashboard")

    try:
        data.prices("da")
    except DashboardDataMissing:
        st.error("no processed prices found, run the pipeline before opening the dashboard.")
        return

    # the analysis is built for the one configured hub, so it drives every view
    point = cfg.market.market.primary_settlement_point

    models = _models(data)
    model = st.sidebar.selectbox("Forecast model", models, index=_index_of(models, MODEL_LIGHTGBM))

    chart_days = st.sidebar.slider(
        "Price chart window in days",
        min_value=7,
        max_value=365,
        value=views.DEFAULT_CHART_DAYS,
    )

    prices_tab, forecast_tab, dispatch_tab, results_tab = st.tabs(
        ["Prices", "Forecast", "Dispatch", "Results"]
    )
    with prices_tab:
        _render_prices(data, cfg, point, chart_days)
    with forecast_tab:
        _render_forecast(data, cfg, point, model)
    with dispatch_tab:
        _render_dispatch(data, cfg, point)
    with results_tab:
        _render_results(data, cfg, point)


main()
