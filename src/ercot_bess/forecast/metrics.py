"""Error metrics for the day ahead forecasts, MAE and sMAPE per hour then pooled.

Errors are measured for each hour of the day first, then pooled. A model that is weak at the
evening peak but accurate off peak then shows that weakness, instead of hiding it in a single
daily average. Every cell also carries its error relative to the naive week baseline, which
shows whether a model beats the baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.schema import HOUR_OF_DAY, REGIME, SETTLEMENT_POINT
from .schema import (
    ALL_HOURS,
    MAE,
    MODEL,
    N_OBS,
    NAIVE_MODEL,
    PREDICTED,
    REALISED,
    REL_MAE,
    SMAPE,
    enforce_metrics_schema,
)

# the floor keeps sMAPE finite when a prediction and a realised price are both zero
_SMAPE_EPSILON = 1e-9

# the columns that identify a metrics cell once the hour is fixed
_CELL = [MODEL, SETTLEMENT_POINT, REGIME]

# temporary column holding the naive baseline error while the relative error is formed
_NAIVE_MAE = "_naive_mae"


def mae(predicted: np.ndarray, realised: np.ndarray) -> float:
    """Mean absolute error in dollars per MWh."""
    return float(np.mean(np.abs(predicted - realised)))


def smape(predicted: np.ndarray, realised: np.ndarray) -> float:
    """Symmetric mean absolute percentage error as a percent bounded by two hundred."""
    denominator = np.maximum(np.abs(predicted) + np.abs(realised), _SMAPE_EPSILON)
    return float(100.0 * np.mean(2.0 * np.abs(predicted - realised) / denominator))


def _aggregate(forecasts: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One metrics row per group formed by the given key columns."""
    records = []
    for values, group in forecasts.groupby(keys, sort=True):
        predicted = group[PREDICTED].to_numpy(dtype=float)
        realised = group[REALISED].to_numpy(dtype=float)
        record = dict(zip(keys, values))
        record[N_OBS] = len(group)
        record[MAE] = mae(predicted, realised)
        record[SMAPE] = smape(predicted, realised)
        records.append(record)
    return pd.DataFrame.from_records(records)


def _add_rel_mae(metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach each cell error divided by the naive week error in the matching cell.

    The naive rows join to themselves and so land on exactly one, the bar every other model
    must beat. A perfect baseline gives a zero divisor, in which case the ratio is left
    undefined rather than infinite.
    """
    naive = metrics[metrics[MODEL] == NAIVE_MODEL]
    reference = naive[[SETTLEMENT_POINT, REGIME, HOUR_OF_DAY, MAE]].rename(
        columns={MAE: _NAIVE_MAE}
    )
    merged = metrics.merge(reference, on=[SETTLEMENT_POINT, REGIME, HOUR_OF_DAY], how="left")
    naive_mae = merged[_NAIVE_MAE].to_numpy(dtype=float)
    model_mae = merged[MAE].to_numpy(dtype=float)
    # divide only where the baseline error is positive, leaving a zero divisor undefined
    merged[REL_MAE] = np.divide(
        model_mae, naive_mae, out=np.full_like(model_mae, np.nan), where=naive_mae > 0.0
    )
    return merged.drop(columns=_NAIVE_MAE)


def build_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Per hour and pooled error metrics for every model in the forecasts frame.

    The pooled rows carry an hour of minus one so a single number per model is available
    alongside the per hour detail.
    """
    per_hour = _aggregate(forecasts, [*_CELL, HOUR_OF_DAY])
    pooled = _aggregate(forecasts, _CELL)
    pooled[HOUR_OF_DAY] = ALL_HOURS
    combined = pd.concat([per_hour, pooled], ignore_index=True)
    combined = _add_rel_mae(combined)
    return enforce_metrics_schema(combined)
