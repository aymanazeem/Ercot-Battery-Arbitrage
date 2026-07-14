"""The four day ahead price forecasters, two baselines and two learned models.

The baselines come first, because a model that cannot beat last week same hour is not useful.
The LEAR model uses the classic design of one Lasso per hour of the day. The LightGBM model is
one pooled tree ensemble that takes the hour as a feature. Every model has the same fit and
predict methods, so the walk forward code treats them the same.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoLarsIC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import ModelConfig
from ..features.schema import (
    DAY_OF_WEEK,
    HOLIDAY_FLAG,
    HOUR_OF_DAY,
    MONTH,
    TARGET,
    feature_names,
    price_lag_name,
)
from .schema import (
    MODEL_LEAR,
    MODEL_LIGHTGBM,
    MODEL_NAIVE_WEEK,
    MODEL_SEASONAL_DOW,
)

# fixed calendar categories so the design matrix is stable even when a fold misses a value
_HOUR_CATEGORIES = list(range(24))
_DOW_CATEGORIES = list(range(7))
_MONTH_CATEGORIES = list(range(1, 13))

# the naive baseline is defined as the price one week earlier, the seven day price lag
_WEEK_LAG_DAYS = 7


class ForecastModel:
    """Common interface, fit on a training frame then predict for a test frame."""

    name: str

    def fit(self, train: pd.DataFrame) -> ForecastModel:
        raise NotImplementedError

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class NaiveWeek(ForecastModel):
    """Tomorrow equals the same hour one week ago, which is exactly the seven day lag."""

    name = MODEL_NAIVE_WEEK

    def __init__(self, model: ModelConfig) -> None:
        self._column = price_lag_name(_WEEK_LAG_DAYS)
        if _WEEK_LAG_DAYS not in model.features.price_lag_days:
            raise ValueError("the naive week baseline needs the seven day price lag feature")

    def fit(self, train: pd.DataFrame) -> NaiveWeek:
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return test[self._column].to_numpy(dtype=float)


class SeasonalDow(ForecastModel):
    """Average realised price for the same weekday and hour over the training window."""

    name = MODEL_SEASONAL_DOW

    def __init__(self, model: ModelConfig) -> None:
        self._profile: pd.Series | None = None
        self._default = 0.0

    def fit(self, train: pd.DataFrame) -> SeasonalDow:
        self._profile = train.groupby([DAY_OF_WEEK, HOUR_OF_DAY])[TARGET].mean()
        self._default = float(train[TARGET].mean())
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        profile = self._profile
        pairs = zip(test[DAY_OF_WEEK].to_numpy(), test[HOUR_OF_DAY].to_numpy())
        values = [profile.get((int(dow), int(hour)), self._default) for dow, hour in pairs]
        return np.asarray(values, dtype=float)


def _lear_numeric_features() -> list[str]:
    return [price_lag_name(1), price_lag_name(2), price_lag_name(3), price_lag_name(7)]


class LearPerHour(ForecastModel):
    """Classic LEAR, a separate Lasso fitted for each of the twenty four hours."""

    name = MODEL_LEAR

    def __init__(self, model: ModelConfig) -> None:
        exogenous = list(model.features.exogenous)
        self._features = [*_lear_numeric_features(), HOLIDAY_FLAG, DAY_OF_WEEK, MONTH, *exogenous]
        self._numeric = [*_lear_numeric_features(), *exogenous]
        self._by_hour: dict[int, Pipeline] = {}
        self._default = 0.0

    def _pipeline(self) -> Pipeline:
        # drop one level from each dummy block so it does not just repeat the intercept
        encoder = OneHotEncoder(
            categories=[_DOW_CATEGORIES, _MONTH_CATEGORIES],
            drop="first",
            sparse_output=False,
        )
        columns = ColumnTransformer(
            [
                ("dummies", encoder, [DAY_OF_WEEK, MONTH]),
                ("flags", "passthrough", [HOLIDAY_FLAG]),
                ("scale", StandardScaler(), self._numeric),
            ]
        )
        return Pipeline([("columns", columns), ("lasso", LassoLarsIC(criterion="aic"))])

    def fit(self, train: pd.DataFrame) -> LearPerHour:
        self._by_hour = {}
        for hour, group in train.groupby(HOUR_OF_DAY):
            pipe = self._pipeline()
            pipe.fit(group[self._features], group[TARGET])
            self._by_hour[int(hour)] = pipe
        self._default = float(train[TARGET].mean())
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        frame = test.reset_index(drop=True)
        hours = frame[HOUR_OF_DAY].to_numpy()
        preds = np.full(len(frame), self._default, dtype=float)
        for hour, pipe in self._by_hour.items():
            mask = hours == hour
            if mask.any():
                preds[mask] = pipe.predict(frame.loc[mask, self._features])
        return preds


class LightGbmPooled(ForecastModel):
    """A single gradient boosted ensemble over all hours with the hour as a feature."""

    name = MODEL_LIGHTGBM

    def __init__(self, model: ModelConfig) -> None:
        self._features = feature_names(model)
        self._categories = {
            HOUR_OF_DAY: _HOUR_CATEGORIES,
            DAY_OF_WEEK: _DOW_CATEGORIES,
            MONTH: _MONTH_CATEGORIES,
        }
        self._seed = model.seed
        self._model = None

    def _design(self, frame: pd.DataFrame) -> pd.DataFrame:
        design = frame[self._features].copy()
        for column, categories in self._categories.items():
            design[column] = design[column].astype(pd.CategoricalDtype(categories=categories))
        return design

    def fit(self, train: pd.DataFrame) -> LightGbmPooled:
        from lightgbm import LGBMRegressor

        self._model = LGBMRegressor(
            random_state=self._seed,
            n_jobs=1,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
        self._model.fit(self._design(train), train[TARGET])
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(self._design(test)), dtype=float)


def build_models(model: ModelConfig) -> list[ForecastModel]:
    """The baselines first, then the two learned models, all sharing one interface."""
    return [NaiveWeek(model), SeasonalDow(model), LearPerHour(model), LightGbmPooled(model)]
