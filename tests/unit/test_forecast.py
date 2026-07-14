"""Tests for the forecasters, the metrics, and the walk forward harness."""

import numpy as np
import pandas as pd
import pytest

from ercot_bess.config import load_config
from ercot_bess.features.schema import (
    DAY_OF_WEEK,
    DELIVERY_DATE,
    HOLIDAY_FLAG,
    HOUR_OF_DAY,
    INTERVAL,
    MONTH,
    REGIME,
    SETTLEMENT_POINT,
    TARGET,
    matrix_columns,
    price_lag_name,
)
from ercot_bess.forecast.evaluate import (
    RegimeError,
    check_single_regime,
    evaluate,
    walk_forward_folds,
)
from ercot_bess.forecast.metrics import build_metrics, mae, smape
from ercot_bess.forecast.models import NaiveWeek, SeasonalDow
from ercot_bess.forecast.schema import (
    ALL_HOURS,
    FORECASTS_COLUMNS,
    MAE,
    METRICS_COLUMNS,
    MODEL,
    MODEL_LEAR,
    MODEL_LIGHTGBM,
    MODEL_NAIVE_WEEK,
    MODEL_SEASONAL_DOW,
    N_OBS,
    REL_MAE,
)

pytestmark = pytest.mark.forecast

_PRIMARY = "HB_HUBAVG"
_REGIME = "swcap5000"
# a Monday so the local day of week is predictable from the day offset
_START_LOCAL = pd.Timestamp("2024-01-01", tz="America/Chicago")


def _matrix(n_days: int = 50, seed: int = 0, noise: float = 1.5) -> pd.DataFrame:
    """A valid day ahead matrix with a weekly and daily price shape and honest lags."""
    rng = np.random.default_rng(seed)
    local = pd.date_range(_START_LOCAL, periods=n_days * 24, freq="h")
    price = (
        30.0
        + 10.0 * np.sin(2 * np.pi * local.hour / 24)
        + 3.0 * local.dayofweek
        + rng.normal(0.0, noise, len(local))
    )
    frame = pd.DataFrame(
        {
            INTERVAL: local.tz_convert("UTC"),
            DELIVERY_DATE: local.normalize().tz_localize(None),
            HOUR_OF_DAY: local.hour.astype("int64"),
            SETTLEMENT_POINT: _PRIMARY,
            REGIME: _REGIME,
            TARGET: price,
        }
    )
    for lag in (1, 2, 3, 7):
        source = frame[[DELIVERY_DATE, HOUR_OF_DAY, TARGET]].copy()
        source[DELIVERY_DATE] = source[DELIVERY_DATE] + pd.Timedelta(lag, "D")
        source = source.rename(columns={TARGET: price_lag_name(lag)})
        frame = frame.merge(source, on=[DELIVERY_DATE, HOUR_OF_DAY], how="left")
    frame[DAY_OF_WEEK] = frame[DELIVERY_DATE].dt.dayofweek.astype("int64")
    frame[MONTH] = frame[DELIVERY_DATE].dt.month.astype("int64")
    frame[HOLIDAY_FLAG] = 0
    frame["da_demand_forecast_mw"] = 40000.0 + local.hour
    frame["temp_c_ercot"] = 10.0
    frame = frame.dropna().reset_index(drop=True)
    frame = frame[matrix_columns(load_config().model)]
    frame[DELIVERY_DATE] = frame[DELIVERY_DATE].astype("datetime64[ns]")
    frame[SETTLEMENT_POINT] = frame[SETTLEMENT_POINT].astype("string")
    frame[REGIME] = frame[REGIME].astype("string")
    return frame


def _small_cfg(min_train: int = 35, window: int = 35, step: int = 1):
    cfg = load_config()
    cfg.model.calibration.min_train_days = min_train
    cfg.model.calibration.window_days = window
    cfg.model.calibration.recalibrate_every_days = step
    return cfg


@pytest.fixture(scope="module")
def evaluated():
    cfg = _small_cfg()
    matrix = _matrix(n_days=50, seed=0)
    forecasts = evaluate(matrix, cfg)
    metrics = build_metrics(forecasts)
    return matrix, cfg, forecasts, metrics


def test_naive_week_prediction_is_exactly_the_seven_day_lag():
    cfg = load_config()
    matrix = _matrix(n_days=30, seed=2)
    preds = NaiveWeek(cfg.model).predict(matrix)
    assert np.array_equal(preds, matrix[price_lag_name(7)].to_numpy(dtype=float))


def test_seasonal_dow_predicts_the_training_weekday_hour_mean():
    cfg = load_config()
    matrix = _matrix(n_days=40, seed=1)
    train = matrix.iloc[: 24 * 30]
    test = matrix.iloc[24 * 30 :]
    preds = SeasonalDow(cfg.model).fit(train).predict(test)
    expected = [
        train[(train[DAY_OF_WEEK] == dow) & (train[HOUR_OF_DAY] == hour)][TARGET].mean()
        for dow, hour in zip(test[DAY_OF_WEEK], test[HOUR_OF_DAY])
    ]
    assert preds == pytest.approx(expected)


def test_mae_is_the_mean_absolute_error():
    predicted = np.array([2.0, 4.0, 6.0])
    realised = np.array([1.0, 2.0, 3.0])
    assert mae(predicted, realised) == pytest.approx(2.0)


def test_smape_is_zero_when_exact_and_two_hundred_when_opposite():
    exact = np.array([5.0, 5.0])
    assert smape(exact, exact) == pytest.approx(0.0)
    assert smape(np.array([1.0]), np.array([-1.0])) == pytest.approx(200.0)


def test_walk_forward_folds_are_causal_and_roll_at_the_window():
    cfg = _small_cfg(min_train=10, window=5, step=1)
    days = list(pd.date_range("2024-01-01", periods=20, freq="D"))
    folds = walk_forward_folds(days, cfg.model.calibration)

    # the origin starts after the required history and steps to the last day
    assert len(folds) == 10
    assert [fold.index for fold in folds] == list(range(10))
    for fold in folds:
        assert len(fold.train_days) == 5
        assert len(fold.test_days) == 1
        # no training day reaches the day being forecast
        assert max(fold.train_days) < min(fold.test_days)


def test_walk_forward_window_expands_on_short_history():
    cfg = _small_cfg(min_train=3, window=100, step=1)
    days = list(pd.date_range("2024-01-01", periods=6, freq="D"))
    folds = walk_forward_folds(days, cfg.model.calibration)
    # with the cap beyond the history the first train block is every prior day
    assert folds[0].train_days == days[:3]
    assert folds[-1].train_days == days[:5]


def test_check_single_regime_rejects_a_mixed_matrix():
    matrix = _matrix(n_days=15, seed=3)
    mixed = matrix.copy()
    mixed.loc[mixed.index[:24], REGIME] = "rtcb"
    with pytest.raises(RegimeError):
        check_single_regime(mixed)


def test_forecasts_carry_the_schema_and_every_model(evaluated):
    _, _, forecasts, _ = evaluated
    assert list(forecasts.columns) == FORECASTS_COLUMNS
    assert set(forecasts[MODEL]) == {
        MODEL_NAIVE_WEEK,
        MODEL_SEASONAL_DOW,
        MODEL_LEAR,
        MODEL_LIGHTGBM,
    }
    assert (forecasts[REGIME] == _REGIME).all()


def test_no_training_timestamp_is_later_than_its_test_fold(evaluated):
    matrix, cfg, _, _ = evaluated
    ordered = matrix.sort_values(INTERVAL, kind="stable")
    by_day = {day: group for day, group in ordered.groupby(DELIVERY_DATE, sort=True)}
    folds = walk_forward_folds(list(by_day), cfg.model.calibration)

    assert folds
    for fold in folds:
        max_train = max(by_day[day][INTERVAL].max() for day in fold.train_days)
        min_test = min(by_day[day][INTERVAL].min() for day in fold.test_days)
        assert max_train < min_test


def test_metrics_are_per_hour_then_pooled_with_the_naive_bar(evaluated):
    _, _, forecasts, metrics = evaluated
    assert list(metrics.columns) == METRICS_COLUMNS

    for model in forecasts[MODEL].unique():
        hours = set(metrics[metrics[MODEL] == model][HOUR_OF_DAY])
        # every delivery hour is scored on its own and then pooled
        assert set(range(24)) <= hours
        assert ALL_HOURS in hours

    naive_rel = metrics[metrics[MODEL] == MODEL_NAIVE_WEEK][REL_MAE].dropna()
    assert np.allclose(naive_rel, 1.0)


def test_pooled_count_is_the_sum_of_the_per_hour_counts(evaluated):
    _, _, _, metrics = evaluated
    lear = metrics[metrics[MODEL] == MODEL_LEAR]
    pooled = lear[lear[HOUR_OF_DAY] == ALL_HOURS][N_OBS].iloc[0]
    per_hour = lear[lear[HOUR_OF_DAY] != ALL_HOURS][N_OBS].sum()
    assert pooled == per_hour


def test_learned_models_have_a_lower_pooled_error_than_naive(evaluated):
    _, _, _, metrics = evaluated
    pooled = metrics[metrics[HOUR_OF_DAY] == ALL_HOURS].set_index(MODEL)[MAE]
    # the whole point of the baseline, a model worth keeping beats last week same hour
    assert pooled[MODEL_LEAR] < pooled[MODEL_NAIVE_WEEK]
    assert pooled[MODEL_LIGHTGBM] < pooled[MODEL_NAIVE_WEEK]


def test_evaluate_is_reproducible_under_the_seed():
    matrix = _matrix(n_days=45, seed=0)
    first = evaluate(matrix.copy(), _small_cfg())
    second = evaluate(matrix.copy(), _small_cfg())
    pd.testing.assert_frame_equal(first, second)
