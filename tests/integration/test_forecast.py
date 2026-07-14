"""Integration tests for the forecast, the model matrix on disk through to the results tables."""

import numpy as np
import pandas as pd
import pytest

from ercot_bess.config import load_config
from ercot_bess.features.build import write_features
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
from ercot_bess.forecast.build import (
    FORECASTS_NAME,
    METRICS_NAME,
    read_forecasts,
    read_metrics,
    run_forecast,
)
from ercot_bess.forecast.schema import (
    ALL_HOURS,
    MODEL,
    MODEL_NAIVE_WEEK,
    REL_MAE,
)

pytestmark = pytest.mark.forecast

_PRIMARY = "HB_HUBAVG"
_REGIME = "swcap5000"
_START_LOCAL = pd.Timestamp("2024-01-01", tz="America/Chicago")


def _matrix(n_days: int, seed: int) -> pd.DataFrame:
    """A valid day ahead matrix with a weekly and daily price shape and honest lags."""
    rng = np.random.default_rng(seed)
    local = pd.date_range(_START_LOCAL, periods=n_days * 24, freq="h")
    price = (
        30.0
        + 10.0 * np.sin(2 * np.pi * local.hour / 24)
        + 3.0 * local.dayofweek
        + rng.normal(0.0, 1.5, len(local))
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


@pytest.fixture
def repo(tmp_path):
    cfg = load_config()
    cfg.model.calibration.min_train_days = 35
    cfg.model.calibration.window_days = 35
    cfg.model.calibration.recalibrate_every_days = 1
    write_features(_matrix(50, 0), tmp_path / cfg.data.paths.features)
    return tmp_path, cfg


def test_run_forecast_writes_both_results_tables(repo):
    root, cfg = repo
    run_forecast(cfg, root)
    results_root = root / cfg.data.paths.results
    assert (results_root / f"{FORECASTS_NAME}.parquet").exists()
    assert (results_root / f"{METRICS_NAME}.parquet").exists()


def test_written_tables_survive_the_parquet_round_trip(repo):
    root, cfg = repo
    forecasts, metrics = run_forecast(cfg, root)
    results_root = root / cfg.data.paths.results
    pd.testing.assert_frame_equal(read_forecasts(results_root), forecasts)
    pd.testing.assert_frame_equal(read_metrics(results_root), metrics)


def test_end_to_end_naive_baseline_sets_the_unit_bar(repo):
    root, cfg = repo
    _, metrics = run_forecast(cfg, root)
    naive_rel = metrics[metrics[MODEL] == MODEL_NAIVE_WEEK][REL_MAE].dropna()
    assert np.allclose(naive_rel, 1.0)


def test_end_to_end_row_counts_match_folds_and_hours(repo):
    root, cfg = repo
    forecasts, metrics = run_forecast(cfg, root)
    folds = forecasts["fold_index"].nunique()
    models = forecasts[MODEL].nunique()
    # one row per fold, hour, and model in the forecasts table
    assert len(forecasts) == folds * 24 * models
    # per model twenty four hourly cells plus the pooled row
    assert len(metrics) == models * 25
    assert ALL_HOURS in set(metrics[HOUR_OF_DAY])
