"""Tests for the cleaning helpers and the table builders."""

import pandas as pd
import pytest

from ercot_bess.config import load_config
from ercot_bess.validate.build import (
    build_da_prices,
    build_load,
    build_weather,
    quality_report,
)
from ercot_bess.validate.clean import assign_regime, sort_and_dedup, spacing_gaps, to_utc
from ercot_bess.validate.schema import (
    DA_PRICES_SCHEMA,
    LOAD_SCHEMA,
    WEATHER_SCHEMA,
    enforce_schema,
)

pytestmark = pytest.mark.validate

_ZONES = ["coast", "east", "far_west", "north", "north_central", "south_central", "southern", "west"]


def _raw_da_prices() -> pd.DataFrame:
    # hosted timestamps are UTC, six a.m. UTC is local midnight in Chicago in winter
    start = pd.Timestamp("2024-01-01T06:00", tz="UTC")
    prices = [10.0, -5.0, 6000.0]
    rows = []
    for hour, price in enumerate(prices):
        interval = start + pd.Timedelta(hour, "h")
        rows.append({"interval_start_utc": interval, "location": "HB_HUBAVG", "spp": price})
    return pd.DataFrame(rows)


def _raw_demand() -> pd.DataFrame:
    start = pd.Timestamp("2024-01-01T06:00", tz="UTC")
    rows = []
    for hour in range(2):
        interval = start + pd.Timedelta(hour, "h")
        rows.append({"interval_start_utc": interval, "system_total": 41000.0})
    return pd.DataFrame(rows)


def _raw_weather(zone_temp_f: float, publish: pd.Timestamp | None = None) -> pd.DataFrame:
    interval = pd.Timestamp("2024-01-01T06:00", tz="UTC")
    if publish is None:
        # publish a day before delivery so the day ahead cutoff keeps the row
        publish = interval - pd.Timedelta(1, "D")
    row = {"interval_start_utc": interval, "publish_time_utc": publish}
    row.update({zone: zone_temp_f for zone in _ZONES})
    return pd.DataFrame([row])


def test_to_utc_converts_chicago_to_utc():
    local = pd.Series(pd.to_datetime(["2024-01-01 00:00"]).tz_localize("America/Chicago"))
    out = to_utc(local)
    assert str(out.dt.tz) == "UTC"
    assert out.iloc[0] == pd.Timestamp("2024-01-01 06:00", tz="UTC")


def test_assign_regime_labels_each_window():
    cfg = load_config()
    interval = pd.Series(
        pd.to_datetime(
            ["2021-06-01T12:00Z", "2024-06-01T12:00Z", "2026-01-01T12:00Z"], utc=True
        )
    )
    labels = assign_regime(interval, cfg.market.regimes, cfg.market.market.timezone_display)
    assert list(labels) == ["pre2022", "swcap5000", "rtcb"]


def test_assign_regime_respects_the_local_rtcb_boundary():
    cfg = load_config()
    # 05:00Z is still 4 December locally, 06:00Z has crossed into 5 December locally
    interval = pd.Series(pd.to_datetime(["2025-12-05T05:00Z", "2025-12-05T06:00Z"], utc=True))
    labels = assign_regime(interval, cfg.market.regimes, cfg.market.market.timezone_display)
    assert list(labels) == ["swcap5000", "rtcb"]


def test_sort_and_dedup_collapses_a_repeated_key():
    frame = pd.DataFrame(
        {
            "interval_start_utc": pd.to_datetime(
                ["2024-11-03T07:00Z", "2024-11-03T07:00Z", "2024-11-03T06:00Z"], utc=True
            ),
            "settlement_point": ["HB_HUBAVG", "HB_HUBAVG", "HB_HUBAVG"],
            "price_usd_per_mwh": [10.0, 99.0, 5.0],
        }
    )
    out = sort_and_dedup(frame, ["interval_start_utc", "settlement_point"])
    assert len(out) == 2
    repeated = out.loc[
        out["interval_start_utc"] == pd.Timestamp("2024-11-03T07:00Z"), "price_usd_per_mwh"
    ].iloc[0]
    assert repeated == 10.0


def test_spacing_gaps_counts_missing_intervals():
    interval = pd.Series(
        pd.to_datetime(
            ["2024-01-01T00:00Z", "2024-01-01T01:00Z", "2024-01-01T03:00Z"], utc=True
        )
    )
    assert spacing_gaps(interval, pd.Timedelta(1, "h")) == 1


def test_spacing_gaps_rejects_a_subgrid_interval():
    interval = pd.Series(pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T00:30Z"], utc=True))
    with pytest.raises(ValueError):
        spacing_gaps(interval, pd.Timedelta(1, "h"))


def test_enforce_schema_shapes_orders_and_casts():
    frame = pd.DataFrame(
        {
            "interval_start_utc": pd.to_datetime(["2024-01-01T00:00Z"], utc=True),
            "settlement_point": ["HB_HUBAVG"],
            "price_usd_per_mwh": [10],
            "regime": ["swcap5000"],
            "extra": [1],
        }
    )
    out = enforce_schema(frame, DA_PRICES_SCHEMA)
    assert list(out.columns) == list(DA_PRICES_SCHEMA)
    assert str(out["price_usd_per_mwh"].dtype) == "float64"
    assert str(out["settlement_point"].dtype) == "string"
    assert str(out["interval_start_utc"].dtype) == "datetime64[ns, UTC]"


def test_enforce_schema_raises_on_missing_column():
    frame = pd.DataFrame({"interval_start_utc": pd.to_datetime(["2024-01-01T00:00Z"], utc=True)})
    with pytest.raises(ValueError):
        enforce_schema(frame, DA_PRICES_SCHEMA)


def test_build_da_prices_matches_contract_and_keeps_extreme_prices():
    cfg = load_config()
    out = build_da_prices(_raw_da_prices(), cfg)

    assert list(out.columns) == list(DA_PRICES_SCHEMA)
    assert str(out["interval_start_utc"].dtype) == "datetime64[ns, UTC]"
    assert str(out["settlement_point"].dtype) == "string"
    assert str(out["regime"].dtype) == "string"
    assert (out["regime"] == "swcap5000").all()
    # negative prices are real in ERCOT and must survive
    assert (out["price_usd_per_mwh"] < 0).any()
    # a price above the offer cap is congestion driven and must not be dropped
    assert (out["price_usd_per_mwh"] > 5000.0).any()


def test_build_load_carries_the_day_ahead_demand_forecast():
    cfg = load_config()
    out = build_load(_raw_demand(), cfg)

    assert list(out.columns) == list(LOAD_SCHEMA)
    # the table carries ERCOT's day ahead market system load forecast alone
    assert (out["da_demand_forecast_mw"] == 41000.0).all()


def test_build_weather_averages_zones_and_converts_to_celsius():
    cfg = load_config()
    out = build_weather(_raw_weather(50.0), cfg)

    assert list(out.columns) == list(WEATHER_SCHEMA)
    # fifty fahrenheit is ten celsius
    assert out["temp_c_ercot"].iloc[0] == pytest.approx(10.0)


def test_build_weather_keeps_the_latest_publish_for_an_interval():
    cfg = load_config()
    # both forecasts publish before the delivery day, so the day ahead cutoff keeps both and
    # the later publish must win the interval
    early = _raw_weather(32.0, publish=pd.Timestamp("2023-12-30T06:00", tz="UTC"))
    late = _raw_weather(50.0, publish=pd.Timestamp("2023-12-31T06:00", tz="UTC"))
    out = build_weather(pd.concat([early, late], ignore_index=True), cfg)

    assert len(out) == 1
    # the later publish reports fifty fahrenheit which is ten celsius
    assert out["temp_c_ercot"].iloc[0] == pytest.approx(10.0)


def test_quality_report_counts_negatives_and_above_cap():
    cfg = load_config()
    da = build_da_prices(_raw_da_prices(), cfg)
    load = build_load(_raw_demand(), cfg)
    weather = build_weather(_raw_weather(50.0), cfg)
    tables = {"da_prices": da, "rt_prices": da, "load": load, "weather": weather}

    report = quality_report(tables, cfg)

    assert report["da_prices"]["negative_share"] > 0.0
    assert report["da_prices"]["by_regime"]["swcap5000"]["above_cap"] == 1
    assert report["load"]["rows"] == 2
