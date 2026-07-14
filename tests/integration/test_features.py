"""Integration tests for the feature build, processed tables through to the written matrix."""

import json

import pandas as pd
import pytest

from ercot_bess.config import load_config
from ercot_bess.features.build import (
    FEATURE_NAMES_FILE,
    FEATURES_NAME,
    LeakageError,
    check_no_leakage,
    read_feature_names,
    read_features,
    run_features,
)
from ercot_bess.features.schema import DELIVERY_DATE, TARGET, feature_names, matrix_columns
from ercot_bess.validate.build import write_processed

pytestmark = pytest.mark.features

_START_LOCAL = pd.Timestamp("2023-12-25", tz="America/Chicago")
_DAYS = 20
_PRIMARY = "HB_HUBAVG"
_SECONDARY = "HB_WEST"


def _local_index() -> pd.DatetimeIndex:
    return pd.date_range(_START_LOCAL, periods=_DAYS * 24, freq="h")


def _price(day_ord: int, hour: int) -> float:
    return day_ord * 100.0 + hour


def _da_prices() -> pd.DataFrame:
    local = _local_index()
    utc = local.tz_convert("UTC")
    day_ord = (local.normalize() - _START_LOCAL.normalize()).days
    rows = []
    for hub, bump in ((_PRIMARY, 0.0), (_SECONDARY, 1000.0)):
        for interval, day, hour in zip(utc, day_ord, local.hour):
            rows.append(
                {
                    "interval_start_utc": interval,
                    "settlement_point": hub,
                    "price_usd_per_mwh": _price(day, hour) + bump,
                    "regime": "swcap5000",
                }
            )
    frame = pd.DataFrame(rows)
    frame["interval_start_utc"] = pd.to_datetime(frame["interval_start_utc"], utc=True)
    return frame


def _load() -> pd.DataFrame:
    utc = _local_index().tz_convert("UTC")
    return pd.DataFrame(
        {
            "interval_start_utc": utc,
            "da_demand_forecast_mw": [41000.0 for _ in utc],
        }
    )


def _weather() -> pd.DataFrame:
    utc = _local_index().tz_convert("UTC")
    return pd.DataFrame({"interval_start_utc": utc, "temp_c_ercot": [10.0 for _ in utc]})


@pytest.fixture
def repo(tmp_path):
    processed = tmp_path / "data" / "processed"
    write_processed(_da_prices(), processed, "da_prices")
    write_processed(_load(), processed, "load")
    write_processed(_weather(), processed, "weather")
    return tmp_path


def test_run_features_writes_matrix_and_feature_names(repo):
    cfg = load_config()
    run_features(cfg, repo)

    features_root = repo / "data" / "features"
    assert (features_root / f"{FEATURES_NAME}.parquet").exists()
    assert (features_root / FEATURE_NAMES_FILE).exists()

    saved = json.loads((features_root / FEATURE_NAMES_FILE).read_text())
    assert saved == feature_names(cfg.model)
    assert read_feature_names(features_root) == feature_names(cfg.model)


def test_written_matrix_matches_the_schema_and_invariants(repo):
    cfg = load_config()
    run_features(cfg, repo)
    matrix = read_features(repo / "data" / "features", cfg)

    assert list(matrix.columns) == matrix_columns(cfg.model)
    days = matrix[DELIVERY_DATE].nunique()
    assert days == _DAYS - 7
    assert len(matrix) == days * 24
    assert not matrix.isna().any().any()
    # the leakage guard passes on the matrix that was actually persisted
    check_no_leakage(matrix, cfg)


def test_written_lags_recover_the_known_prices(repo):
    cfg = load_config()
    run_features(cfg, repo)
    matrix = read_features(repo / "data" / "features", cfg)

    row = matrix[(matrix[DELIVERY_DATE] == pd.Timestamp("2024-01-05")) & (matrix["hour_of_day"] == 9)]
    row = row.iloc[0]
    # 2024-01-05 is the eleventh local day, day ordinal eleven
    assert row[TARGET] == _price(11, 9)
    assert row["price_lag_1d"] == _price(10, 9)
    assert row["price_lag_7d"] == _price(4, 9)


def test_end_to_end_leakage_guard_rejects_a_leaked_column(repo):
    cfg = load_config()
    matrix = run_features(cfg, repo)
    leaked = matrix.copy()
    leaked["price_same_day"] = leaked[TARGET]
    with pytest.raises(LeakageError):
        check_no_leakage(leaked, cfg)
