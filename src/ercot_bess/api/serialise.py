"""Shaping the processed and results frames into JSON ready structures.

No disk or network happens here. Each function takes a frame and the query parameters and
returns plain Python containers with timestamps in ISO 8601 UTC and NaN mapped to null. The
disk reads and the FastAPI wiring live in store and app.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from ..backtest.aggregate import N_DAYS, PERIOD, USD_PER_KW_YEAR
from ..backtest.schema import CAPTURE_RATE, EQUIV_FULL_CYCLES, PROFIT, SCENARIO
from ..forecast.schema import DELIVERY_DATE, HOUR_OF_DAY, MODEL, PREDICTED, REALISED
from ..validate.schema import INTERVAL, PRICE, REGIME, SETTLEMENT_POINT

# envelope keys the endpoints wrap their row lists in, kept here so tests share one source
MARKET = "market"
LOCAL_DAY = "local_day"
INTERVALS = "intervals"
HOURS = "hours"
ANNUAL = "annual"
ANNUALISED = "annualised_per_kw_year"


def _clean_float(value: Any) -> float | None:
    """Map a missing value to None so the payload is valid JSON, otherwise a plain float."""
    if pd.isna(value):
        return None
    return float(value)


def _int(value: Any) -> int:
    return int(value)


def _text(value: Any) -> str | None:
    return None if pd.isna(value) else str(value)


def _utc_iso(value: pd.Timestamp) -> str:
    """ISO 8601 in UTC for a timezone aware instant."""
    return value.tz_convert("UTC").isoformat()


def _date_iso(value: pd.Timestamp) -> str:
    """The local calendar date for a tz naive delivery day or period stamp."""
    return value.date().isoformat()


def _records(frame: pd.DataFrame, serialisers: dict[str, Callable[[Any], Any]]) -> list[dict]:
    """Serialise every row of the frame through a column to function mapping."""
    columns = list(serialisers)
    functions = list(serialisers.values())
    rows = zip(*(frame[column] for column in columns), strict=True)
    return [
        {column: fn(value) for column, fn, value in zip(columns, functions, values, strict=True)}
        for values in rows
    ]


_PRICE_RECORD = {INTERVAL: _utc_iso, PRICE: _clean_float, REGIME: _text}
_FORECAST_RECORD = {
    HOUR_OF_DAY: _int,
    INTERVAL: _utc_iso,
    PREDICTED: _clean_float,
    REALISED: _clean_float,
    REGIME: _text,
}
_ANNUAL_RECORD = {
    SETTLEMENT_POINT: _text,
    SCENARIO: _text,
    PERIOD: _date_iso,
    PROFIT: _clean_float,
    EQUIV_FULL_CYCLES: _clean_float,
    CAPTURE_RATE: _clean_float,
    N_DAYS: _int,
}
_ANNUALISED_RECORD = {
    SETTLEMENT_POINT: _text,
    SCENARIO: _text,
    N_DAYS: _int,
    USD_PER_KW_YEAR: _clean_float,
}


def latest_prices(
    prices: pd.DataFrame, settlement_point: str, market: str, display_tz: str
) -> dict:
    """The most recent local day of prices for one settlement point.

    The day is the ERCOT local calendar date so a request returns one clean delivery day
    rather than a ragged UTC slice. An unknown point yields an empty interval list.
    """
    rows = prices[prices[SETTLEMENT_POINT] == settlement_point]
    payload = {SETTLEMENT_POINT: settlement_point, MARKET: market, LOCAL_DAY: None, INTERVALS: []}
    if rows.empty:
        return payload
    local_day = rows[INTERVAL].dt.tz_convert(display_tz).dt.normalize()
    latest = local_day.max()
    kept = rows[local_day == latest].sort_values(INTERVAL, kind="stable")
    payload[LOCAL_DAY] = latest.date().isoformat()
    payload[INTERVALS] = _records(kept, _PRICE_RECORD)
    return payload


def latest_forecast(forecasts: pd.DataFrame, settlement_point: str, model: str) -> dict:
    """The most recent delivery day of predicted against realised prices for one model."""
    rows = forecasts[
        (forecasts[SETTLEMENT_POINT] == settlement_point) & (forecasts[MODEL] == model)
    ]
    payload = {SETTLEMENT_POINT: settlement_point, MODEL: model, DELIVERY_DATE: None, HOURS: []}
    if rows.empty:
        return payload
    latest = rows[DELIVERY_DATE].max()
    kept = rows[rows[DELIVERY_DATE] == latest].sort_values(HOUR_OF_DAY, kind="stable")
    payload[DELIVERY_DATE] = _date_iso(latest)
    payload[HOURS] = _records(kept, _FORECAST_RECORD)
    return payload


def backtest_summary(annual: pd.DataFrame, annualised: pd.DataFrame) -> dict:
    """The headline backtest numbers, annual totals and the annualised per kW year table."""
    order = [SETTLEMENT_POINT, SCENARIO]
    return {
        ANNUAL: _records(annual.sort_values(order, kind="stable"), _ANNUAL_RECORD),
        ANNUALISED: _records(annualised.sort_values(order, kind="stable"), _ANNUALISED_RECORD),
    }
