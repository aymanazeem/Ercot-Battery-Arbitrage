"""Typed configuration loaded and validated from the YAML files under config."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

# config.py lives at src/ercot_bess, so the repo root is two parents up
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"

_CONFIG_FILES = ("battery.yaml", "market.yaml", "data.yaml", "model.yaml")


class _Strict(BaseModel):
    # forbid unknown keys so a mistyped field raises at load time, not deep in a model
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class BatterySpec(_Strict):
    power_mw: float
    duration_h: float
    round_trip_efficiency: float
    initial_soc_frac: float
    cycles_per_day_cap: float | None


class CostSpec(_Strict):
    cycling_cost_per_mwh: float
    pack_cost_per_kwh: float
    cycle_life_at_80pct_dod: int


class BatteryConfig(_Strict):
    battery: BatterySpec
    cost: CostSpec


class Regime(_Strict):
    name: str
    start: date | None
    end: date | None
    offer_cap_da_usd_per_mwh: float
    offer_cap_rt_usd_per_mwh: float


class MarketSpec(_Strict):
    timezone_display: str
    primary_settlement_point: str
    secondary_settlement_point: str


class MarketConfig(_Strict):
    market: MarketSpec
    regimes: list[Regime]


class Paths(_Strict):
    raw: str
    processed: str
    features: str
    results: str


class Window(_Strict):
    backtest_start: date
    backtest_end: date


class ErcotSource(_Strict):
    provider: str
    api_key_env: str = "GRIDSTATUS_API_KEY"


class Eia930Source(_Strict):
    balancing_authority: str
    api_path: str
    api_key_env: str


class WeatherSource(_Strict):
    source: str


class Sources(_Strict):
    ercot: ErcotSource
    eia930: Eia930Source
    weather: WeatherSource


class DataConfig(_Strict):
    paths: Paths
    window: Window
    sources: Sources


class FeaturesSpec(_Strict):
    price_lag_days: list[int]
    calendar: list[str]
    exogenous: list[str]


class CalibrationSpec(_Strict):
    window_days: int
    recalibrate_every_days: int
    min_train_days: int


class ForecastSpec(_Strict):
    horizon_hours: int


class ModelConfig(_Strict):
    features: FeaturesSpec
    calibration: CalibrationSpec
    forecast: ForecastSpec
    seed: int


class Config(_Strict):
    battery: BatteryConfig
    market: MarketConfig
    data: DataConfig
    model: ModelConfig


def _read_mapping(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing config file {path}")
    with path.open() as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"config file {path.name} did not parse to a mapping")
    return loaded


def load_config(config_dir: Path | str = _CONFIG_DIR) -> Config:
    """Load the four YAML config files into one validated typed object."""
    config_dir = Path(config_dir)
    battery, market, data, model = (
        _read_mapping(config_dir / name) for name in _CONFIG_FILES
    )
    return Config(
        battery=BatteryConfig(**battery),
        market=MarketConfig(**market),
        data=DataConfig(**data),
        model=ModelConfig(**model),
    )
