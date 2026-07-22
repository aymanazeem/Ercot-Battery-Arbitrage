"""Tests for the config loader and the project scaffold."""

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from ercot_bess.config import load_config

pytestmark = pytest.mark.config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_FILES = ("battery.yaml", "market.yaml", "data.yaml", "model.yaml")
_MODULES = (
    "config",
    "ingest",
    "validate",
    "features",
    "forecast",
    "optimise",
    "backtest",
    "api",
    "dashboard",
)


def test_load_config_returns_populated_typed_objects():
    cfg = load_config()

    assert cfg.battery.battery.power_mw == 1.0
    assert cfg.battery.battery.round_trip_efficiency == 0.85
    assert cfg.battery.battery.cycles_per_day_cap is None
    assert isinstance(cfg.battery.cost.cycle_life_at_80pct_dod, int)

    assert cfg.market.market.primary_settlement_point == "HB_HUBAVG"
    assert {r.name for r in cfg.market.regimes} == {"pre2022", "swcap5000", "rtcb"}

    assert cfg.data.window.backtest_start.year == 2025

    assert cfg.model.features.price_lag_days == [1, 2, 3, 7]
    assert isinstance(cfg.model.seed, int)


def test_corrupting_a_key_raises_a_validation_error(tmp_path):
    for name in _CONFIG_FILES:
        (tmp_path / name).write_text((_REPO_ROOT / "config" / name).read_text())

    battery = tmp_path / "battery.yaml"
    battery.write_text(battery.read_text().replace("power_mw", "power_megawatts"))

    with pytest.raises(ValidationError):
        load_config(tmp_path)


def test_a_bad_value_type_raises_a_validation_error(tmp_path):
    for name in _CONFIG_FILES:
        (tmp_path / name).write_text((_REPO_ROOT / "config" / name).read_text())

    battery = tmp_path / "battery.yaml"
    battery.write_text(battery.read_text().replace("power_mw: 1.0", "power_mw: not_a_number"))

    with pytest.raises(ValidationError):
        load_config(tmp_path)


def test_every_declared_make_target_exists():
    makefile = (_REPO_ROOT / "Makefile").read_text()
    targets = {
        line.split(":", 1)[0]
        for line in makefile.splitlines()
        if line and not line[0].isspace() and ":" in line
    }
    required = {"setup", "test", "ingest", "build", "features", "forecast", "backtest", "serve"}
    required |= {f"test-{name}" for name in _MODULES}

    missing = required - targets
    assert not missing, f"missing make targets {sorted(missing)}"


def test_markers_are_registered():
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    declared = pyproject["tool"]["pytest"]["ini_options"]["markers"]
    registered = {entry.split(":", 1)[0] for entry in declared}
    required = set(_MODULES)

    missing = required - registered
    assert not missing, f"unregistered markers {sorted(missing)}"
