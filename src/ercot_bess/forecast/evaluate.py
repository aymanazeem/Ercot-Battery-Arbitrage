"""Walk forward evaluation of the day ahead forecasters.

A time series model must be tested on its own future, so training always ends before the day
it forecasts. The start point rolls forward by the retrain step, fitting on a trailing multi
year window that grows until it hits the cap. A random split is never used, because it would
let tomorrow leak into today and make the score look better than it is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import CalibrationSpec, Config
from ..features.build import check_no_leakage
from ..features.schema import (
    DELIVERY_DATE,
    HOUR_OF_DAY,
    INTERVAL,
    REGIME,
    SETTLEMENT_POINT,
    TARGET,
)
from .models import build_models
from .schema import FOLD_INDEX, MODEL, PREDICTED, REALISED, enforce_forecasts_schema


class RegimeError(ValueError):
    """Raised when a matrix spans more than one offer cap regime."""


@dataclass(frozen=True)
class Fold:
    """One walk forward origin, the training days and the test days it forecasts."""

    index: int
    train_days: list[pd.Timestamp]
    test_days: list[pd.Timestamp]


def check_single_regime(matrix: pd.DataFrame) -> None:
    """Assert the matrix carries one regime so no fold crosses an offer cap change."""
    regimes = sorted(matrix[REGIME].dropna().unique())
    if len(regimes) != 1:
        raise RegimeError(f"forecasting needs a single regime, found {regimes}")


def walk_forward_folds(days: list[pd.Timestamp], calibration: CalibrationSpec) -> list[Fold]:
    """The train and test day blocks for each walk forward origin.

    The first forecast waits until min train days of history exist, then the start point steps
    forward by the retrain step. Each train block is the trailing window of days before its
    test block, capped at the window length, so it rolls forward on long history and grows on
    short history.
    """
    window = calibration.window_days
    step = calibration.recalibrate_every_days
    start = calibration.min_train_days
    folds = []
    for origin in range(start, len(days), step):
        train_days = days[max(0, origin - window) : origin]
        test_days = days[origin : origin + step]
        if train_days and test_days:
            folds.append(Fold(len(folds), train_days, test_days))
    return folds


def _fold_rows(
    test: pd.DataFrame, model_name: str, predictions: np.ndarray, fold_index: int
) -> pd.DataFrame:
    """Assemble the forecast rows for one model on one test fold."""
    return pd.DataFrame(
        {
            DELIVERY_DATE: test[DELIVERY_DATE].to_numpy(),
            HOUR_OF_DAY: test[HOUR_OF_DAY].to_numpy(),
            INTERVAL: test[INTERVAL].to_numpy(),
            SETTLEMENT_POINT: test[SETTLEMENT_POINT].to_numpy(),
            REGIME: test[REGIME].to_numpy(),
            MODEL: model_name,
            PREDICTED: predictions,
            REALISED: test[TARGET].to_numpy(),
            FOLD_INDEX: fold_index,
        }
    )


def evaluate(matrix: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Run every model across the walk forward folds and return the forecasts frame.

    Leakage and single regime are checked once up front, then each fold fits on its past and
    predicts its future, so no training row is ever dated on or after the hour it forecasts.
    """
    check_no_leakage(matrix, cfg)
    check_single_regime(matrix)
    np.random.seed(cfg.model.seed)

    ordered = matrix.sort_values(INTERVAL, kind="stable")
    by_day = {day: group for day, group in ordered.groupby(DELIVERY_DATE, sort=True)}
    folds = walk_forward_folds(list(by_day), cfg.model.calibration)
    if not folds:
        raise ValueError(
            "not enough history for a single walk forward fold, "
            "min train days exceeds the available delivery days"
        )

    models = build_models(cfg.model)
    frames = []
    for fold in folds:
        train = pd.concat([by_day[day] for day in fold.train_days], ignore_index=True)
        test = pd.concat([by_day[day] for day in fold.test_days], ignore_index=True)
        for model in models:
            predictions = model.fit(train).predict(test)
            frames.append(_fold_rows(test, model.name, predictions, fold.index))

    forecasts = pd.concat(frames, ignore_index=True)
    forecasts = forecasts.sort_values([MODEL, INTERVAL], kind="stable").reset_index(drop=True)
    return enforce_forecasts_schema(forecasts)
