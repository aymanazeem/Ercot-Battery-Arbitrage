"""Schemas for the forecast outputs, the forecasts table and the forecast_metrics table.

The forecasts table is one row per delivery hour and model with the prediction and the
realised price. The metrics table is one row per model and hour with the aggregate row
carrying an hour of minus one.
"""

from __future__ import annotations

import pandas as pd

from ..features.schema import (
    DELIVERY_DATE,
    HOUR_OF_DAY,
    INTERVAL,
    REGIME,
    SETTLEMENT_POINT,
)

MODEL_NAIVE_WEEK = "naive_week"
MODEL_SEASONAL_DOW = "seasonal_dow"
MODEL_LEAR = "lear"
MODEL_LIGHTGBM = "lightgbm"

# the reference every relative error is measured against, the same hour one week ago
NAIVE_MODEL = MODEL_NAIVE_WEEK

MODEL = "model"
PREDICTED = "predicted_usd_per_mwh"
REALISED = "realised_usd_per_mwh"
FOLD_INDEX = "fold_index"

N_OBS = "n_obs"
MAE = "mae_usd_per_mwh"
SMAPE = "smape_pct"
REL_MAE = "rel_mae_vs_naive"

# the metrics row that pools every hour so a single number per model is available too
ALL_HOURS = -1

_UTC = "datetime64[ns, UTC]"

_FORECASTS_DTYPES = {
    DELIVERY_DATE: "datetime64[ns]",
    HOUR_OF_DAY: "int64",
    INTERVAL: _UTC,
    SETTLEMENT_POINT: "string",
    REGIME: "string",
    MODEL: "string",
    PREDICTED: "float64",
    REALISED: "float64",
    FOLD_INDEX: "int64",
}

_METRICS_DTYPES = {
    MODEL: "string",
    SETTLEMENT_POINT: "string",
    REGIME: "string",
    HOUR_OF_DAY: "int64",
    N_OBS: "int64",
    MAE: "float64",
    SMAPE: "float64",
    REL_MAE: "float64",
}

FORECASTS_COLUMNS = list(_FORECASTS_DTYPES)
METRICS_COLUMNS = list(_METRICS_DTYPES)


def _enforce(frame: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    columns = list(dtypes)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing required columns {missing}")
    shaped = frame[columns].copy()
    for column, dtype in dtypes.items():
        shaped[column] = shaped[column].astype(dtype)
    return shaped


def enforce_forecasts_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the forecasts frame reduced to its columns in order with dtypes pinned."""
    return _enforce(frame, _FORECASTS_DTYPES)


def enforce_metrics_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the metrics frame reduced to its columns in order with dtypes pinned."""
    return _enforce(frame, _METRICS_DTYPES)
