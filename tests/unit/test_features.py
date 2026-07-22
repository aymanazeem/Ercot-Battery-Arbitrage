"""Tests for the day ahead model matrix and the leakage guard."""

import pandas as pd
import pytest

from ercot_bess.config import load_config
from ercot_bess.features.build import (
    LeakageError,
    build_model_matrix,
    check_no_leakage,
)
from ercot_bess.features.schema import (
    DA_DEMAND_FORECAST,
    DAY_OF_WEEK,
    DELIVERY_DATE,
    HOLIDAY_FLAG,
    HOUR_OF_DAY,
    MONTH,
    TARGET,
    TEMP,
    feature_names,
    matrix_columns,
)

pytestmark = pytest.mark.features

# start a few days before new year so the warm up clears and a known holiday lands inside
_START_LOCAL = pd.Timestamp("2023-12-25", tz="America/Chicago")
_DAYS = 20
_PRIMARY = "HB_HUBAVG"
_SECONDARY = "HB_WEST"


def _price(day_ord: int, hour: int) -> float:
    return day_ord * 100.0 + hour


def _local_index() -> pd.DatetimeIndex:
    return pd.date_range(_START_LOCAL, periods=_DAYS * 24, freq="h")


def _da_prices() -> pd.DataFrame:
    local = _local_index()
    utc = local.tz_convert("UTC")
    day_ord = (local.normalize() - _START_LOCAL.normalize()).days
    rows = []
    # the secondary hub is pushed far negative so any leak into the matrix is unmistakable
    for hub, bump in ((_PRIMARY, 0.0), (_SECONDARY, -100000.0)):
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
    frame["settlement_point"] = frame["settlement_point"].astype("string")
    frame["regime"] = frame["regime"].astype("string")
    return frame


def _load() -> pd.DataFrame:
    utc = _local_index().tz_convert("UTC")
    return pd.DataFrame(
        {
            "interval_start_utc": utc,
            "da_demand_forecast_mw": [41000.0 + ts.hour for ts in utc],
        }
    )


def _weather() -> pd.DataFrame:
    utc = _local_index().tz_convert("UTC")
    return pd.DataFrame(
        {
            "interval_start_utc": utc,
            "temp_c_ercot": [10.0 + 0.1 * ts.hour for ts in utc],
        }
    )


@pytest.fixture
def matrix():
    cfg = load_config()
    return build_model_matrix(_da_prices(), _load(), _weather(), cfg)


def test_feature_names_follow_the_config_order():
    cfg = load_config()
    assert feature_names(cfg.model) == [
        "price_lag_1d",
        "price_lag_2d",
        "price_lag_3d",
        "price_lag_7d",
        HOUR_OF_DAY,
        DAY_OF_WEEK,
        MONTH,
        HOLIDAY_FLAG,
        DA_DEMAND_FORECAST,
        TEMP,
    ]


def test_matrix_columns_match_the_schema(matrix):
    cfg = load_config()
    assert list(matrix.columns) == matrix_columns(cfg.model)


def test_row_count_is_delivery_days_times_twenty_four(matrix):
    days = matrix[DELIVERY_DATE].nunique()
    assert len(matrix) == days * 24
    # twenty days of history minus the seven day warm up leaves thirteen delivery days
    assert days == _DAYS - 7


def test_no_nulls_after_the_warm_up(matrix):
    assert not matrix.isna().any().any()


def test_only_the_primary_hub_survives(matrix):
    assert (matrix["settlement_point"] == _PRIMARY).all()
    # secondary hub prices were forced negative so their absence proves the hub filter
    assert (matrix[TARGET] < 0).sum() == 0


def test_autoregressive_lags_point_to_the_right_past_day(matrix):
    row = matrix[(matrix[DELIVERY_DATE] == pd.Timestamp("2024-01-02")) & (matrix[HOUR_OF_DAY] == 5)]
    assert len(row) == 1
    row = row.iloc[0]
    # delivery day 2024-01-02 is the eighth local day, day ordinal eight
    assert row[TARGET] == _price(8, 5)
    assert row["price_lag_1d"] == _price(7, 5)
    assert row["price_lag_2d"] == _price(6, 5)
    assert row["price_lag_3d"] == _price(5, 5)
    assert row["price_lag_7d"] == _price(1, 5)


def test_calendar_features_are_local(matrix):
    day = matrix[matrix[DELIVERY_DATE] == pd.Timestamp("2024-01-02")].iloc[0]
    # 2024-01-02 is a Tuesday in month one
    assert day[DAY_OF_WEEK] == 1
    assert day[MONTH] == 1
    assert set(matrix[HOUR_OF_DAY]) == set(range(24))


def test_holiday_flag_marks_new_year(matrix):
    new_year = matrix[matrix[DELIVERY_DATE] == pd.Timestamp("2024-01-01")]
    plain_day = matrix[matrix[DELIVERY_DATE] == pd.Timestamp("2024-01-02")]
    assert (new_year[HOLIDAY_FLAG] == 1).all()
    assert (plain_day[HOLIDAY_FLAG] == 0).all()


def test_exogenous_forecasts_attach_on_the_delivery_hour(matrix):
    row = matrix[(matrix[DELIVERY_DATE] == pd.Timestamp("2024-01-02")) & (matrix[HOUR_OF_DAY] == 5)]
    row = row.iloc[0]
    utc_hour = row["interval_start_utc"].hour
    assert row[DA_DEMAND_FORECAST] == 41000.0 + utc_hour
    assert row[TEMP] == pytest.approx(10.0 + 0.1 * utc_hour)


def test_leakage_guard_passes_on_the_real_matrix(matrix):
    cfg = load_config()
    check_no_leakage(matrix, cfg)


def test_leakage_guard_fails_on_a_realised_column(matrix):
    cfg = load_config()
    leaked = matrix.copy()
    # realised same day demand is only known after close so it must be rejected
    leaked["load_mw"] = 40000.0
    with pytest.raises(LeakageError):
        check_no_leakage(leaked, cfg)


def test_leakage_guard_fails_on_any_unsanctioned_column(matrix):
    cfg = load_config()
    leaked = matrix.copy()
    leaked["surprise_feature"] = 1.0
    with pytest.raises(LeakageError):
        check_no_leakage(leaked, cfg)


def test_leakage_guard_fails_when_a_feature_is_missing(matrix):
    cfg = load_config()
    trimmed = matrix.drop(columns=[TEMP])
    with pytest.raises(LeakageError):
        check_no_leakage(trimmed, cfg)


def _dst_inputs():
    # a window spanning the autumn daylight saving change on 2023-11-05
    start = pd.Timestamp("2023-10-29", tz="America/Chicago")
    local = pd.date_range(start, periods=13 * 24, freq="h")
    utc = local.tz_convert("UTC")
    day_ord = (local.normalize() - start.normalize()).days
    prices = pd.DataFrame(
        {
            "interval_start_utc": utc,
            "settlement_point": pd.array([_PRIMARY] * len(utc), dtype="string"),
            "price_usd_per_mwh": [_price(day, hour) for day, hour in zip(day_ord, local.hour)],
            "regime": pd.array(["swcap5000"] * len(utc), dtype="string"),
        }
    )
    load = pd.DataFrame({"interval_start_utc": utc, "da_demand_forecast_mw": 41000.0})
    weather = pd.DataFrame({"interval_start_utc": utc, "temp_c_ercot": 10.0})
    return prices, load, weather


def test_fall_back_day_does_not_multiply_lag_rows():
    cfg = load_config()
    prices, load, weather = _dst_inputs()
    out = build_model_matrix(prices, load, weather, cfg)

    # the day after the fall back uses the twenty five hour day as its one day lag source,
    # without the one to one dedup it would gain a duplicate hour and be dropped as not full
    day_after = out[out[DELIVERY_DATE] == pd.Timestamp("2023-11-06")]
    assert len(day_after) == 24
    assert out.groupby(DELIVERY_DATE).size().max() == 24


def _spring_inputs():
    # a window spanning the spring daylight saving change on 2024-03-10, the twenty three hour day
    start = pd.Timestamp("2024-03-03", tz="America/Chicago")
    local = pd.date_range(start, periods=15 * 24, freq="h")
    utc = local.tz_convert("UTC")
    day_ord = (local.normalize() - start.normalize()).days
    prices = pd.DataFrame(
        {
            "interval_start_utc": utc,
            "settlement_point": pd.array([_PRIMARY] * len(utc), dtype="string"),
            "price_usd_per_mwh": [_price(day, hour) for day, hour in zip(day_ord, local.hour)],
            "regime": pd.array(["swcap5000"] * len(utc), dtype="string"),
        }
    )
    load = pd.DataFrame({"interval_start_utc": utc, "da_demand_forecast_mw": 41000.0})
    weather = pd.DataFrame({"interval_start_utc": utc, "temp_c_ercot": 10.0})
    return prices, load, weather


def test_spring_forward_gap_does_not_cascade_into_later_days():
    cfg = load_config()
    prices, load, weather = _spring_inputs()
    out = build_model_matrix(prices, load, weather, cfg)
    kept = set(out[DELIVERY_DATE].unique())

    # the twenty three hour transition day itself has no two a.m. delivery price, so it drops
    assert pd.Timestamp("2024-03-10") not in kept
    # but the days that lag one, two, three, and seven days from it must keep their full grid,
    # the gap filled lag source stops the missing hour cascading into them
    for follower in ("2024-03-11", "2024-03-12", "2024-03-13", "2024-03-17"):
        day = out[out[DELIVERY_DATE] == pd.Timestamp(follower)]
        assert len(day) == 24
    assert out.groupby(DELIVERY_DATE).size().max() == 24
