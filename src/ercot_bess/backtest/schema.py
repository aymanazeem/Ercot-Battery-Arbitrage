"""Schema for the backtest output, one row per day per scenario.

The table carries the daily profit, the equivalent full cycles, and the capture rate for
each of the two scenarios.
"""

from __future__ import annotations

import pandas as pd

DELIVERY_DATE = "delivery_date"
SCENARIO = "scenario"
SETTLEMENT_POINT = "settlement_point"
PROFIT = "profit_usd"
EQUIV_FULL_CYCLES = "equiv_full_cycles"
CAPTURE_RATE = "capture_rate"

# the perfect foresight ceiling and the realistic forecast driven operator
SCENARIO_CEILING = "ceiling"
SCENARIO_FORECAST_DRIVEN = "forecast_driven"

# the delivery day is a tz naive local midnight so it matches the forecasts table join key
_BACKTEST_DTYPES = {
    DELIVERY_DATE: "datetime64[ns]",
    SCENARIO: "string",
    SETTLEMENT_POINT: "string",
    PROFIT: "float64",
    EQUIV_FULL_CYCLES: "float64",
    CAPTURE_RATE: "float64",
}

BACKTEST_COLUMNS = list(_BACKTEST_DTYPES)


def enforce_backtest_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the backtest frame reduced to its columns in order with dtypes pinned."""
    missing = [column for column in _BACKTEST_DTYPES if column not in frame.columns]
    if missing:
        raise ValueError(f"backtest frame is missing required columns {missing}")
    shaped = frame[BACKTEST_COLUMNS].copy()
    for column, dtype in _BACKTEST_DTYPES.items():
        shaped[column] = shaped[column].astype(dtype)
    return shaped
