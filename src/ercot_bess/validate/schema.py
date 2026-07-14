"""Schemas for the processed tables and the check that makes every output match them."""

from __future__ import annotations

import pandas as pd

INTERVAL = "interval_start_utc"
SETTLEMENT_POINT = "settlement_point"
PRICE = "price_usd_per_mwh"
REGIME = "regime"
DA_DEMAND_FORECAST = "da_demand_forecast_mw"
TEMP = "temp_c_ercot"

_UTC = "datetime64[ns, UTC]"

# each schema maps a column to its dtype, in a fixed column order
DA_PRICES_SCHEMA: dict[str, str] = {
    INTERVAL: _UTC,
    SETTLEMENT_POINT: "string",
    PRICE: "float64",
    REGIME: "string",
}

RT_PRICES_SCHEMA: dict[str, str] = dict(DA_PRICES_SCHEMA)

# the day ahead demand forecast is the one load related exogenous the model uses, so the
# table carries the forecast alone and does not keep a realised load column
LOAD_SCHEMA: dict[str, str] = {
    INTERVAL: _UTC,
    DA_DEMAND_FORECAST: "float64",
}

WEATHER_SCHEMA: dict[str, str] = {
    INTERVAL: _UTC,
    TEMP: "float64",
}

NAME_TO_SCHEMA: dict[str, dict[str, str]] = {
    "da_prices": DA_PRICES_SCHEMA,
    "rt_prices": RT_PRICES_SCHEMA,
    "load": LOAD_SCHEMA,
    "weather": WEATHER_SCHEMA,
}


def enforce_schema(frame: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Return the frame reduced to the schema columns, in order, cast to the contract dtypes."""
    missing = [column for column in schema if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing required columns {missing}")
    shaped = frame[list(schema)].copy()
    for column, dtype in schema.items():
        shaped[column] = shaped[column].astype(dtype)
    return shaped
