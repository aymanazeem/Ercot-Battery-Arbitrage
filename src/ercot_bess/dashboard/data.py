"""Disk reads for the dashboard, the layer that turns the artifact files into frames.

This is the only place the dashboard touches disk. The data root is the repository root that
holds the data directory and it comes from an environment variable so a test can point the app
at a seeded temp directory with no network. Each reader reuses a module reader or reads the
parquet a prior module wrote, so the dashboard never re derives a table. A missing file raises
DashboardDataMissing so the app can show a clean note rather than a stack trace.

Streamlit re executes this whole script on any widget change, so the reads are wrapped in
module level ``@st.cache_data`` functions keyed on the root path as a plain string. A rerun that
does not change the data root serves the frames from memory instead of re reading parquet. The
missing file guard stays outside the cached functions so a missing file is never cached, and the
cache is bounded by a ttl and a max entry count so an out of process pipeline refresh is
eventually picked up.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from ..backtest.build import (
    BACKTEST_NAME,
    CONCENTRATION_NAME,
    SENSITIVITIES_NAME,
    read_backtest,
)
from ..config import Config
from ..forecast.build import FORECASTS_NAME, METRICS_NAME, read_forecasts, read_metrics
from ..validate.build import read_processed

# the app reads its data root from here so a test can point it at a seeded temp directory
DATA_ROOT_ENV = "ERCOT_BESS_DATA_ROOT"

# dashboard/data.py sits at src/ercot_bess/dashboard, so the repo root is three parents up
_REPO_ROOT = Path(__file__).resolve().parents[3]

# the market query value maps to the processed price table name
_MARKET_TABLE = {"da": "da_prices", "rt": "rt_prices"}

# bound the read cache so a pipeline that rewrites the parquet out of process is picked up within
# the ttl, and so the process never pins more than a handful of tables in memory
_CACHE_TTL_SECONDS = 300
_CACHE_MAX_ENTRIES = 16


@st.cache_data(ttl=_CACHE_TTL_SECONDS, max_entries=_CACHE_MAX_ENTRIES)
def _read_processed_cached(processed_root: str, name: str) -> pd.DataFrame:
    """Cache one processed price table, keyed on the root string and the table name."""
    return read_processed(Path(processed_root), name)


@st.cache_data(ttl=_CACHE_TTL_SECONDS, max_entries=_CACHE_MAX_ENTRIES)
def _read_forecasts_cached(results_root: str) -> pd.DataFrame:
    """Cache the walk forward forecasts table, keyed on the results root string."""
    return read_forecasts(Path(results_root))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, max_entries=_CACHE_MAX_ENTRIES)
def _read_metrics_cached(results_root: str) -> pd.DataFrame:
    """Cache the per model error metrics table, keyed on the results root string."""
    return read_metrics(Path(results_root))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, max_entries=_CACHE_MAX_ENTRIES)
def _read_backtest_cached(results_root: str) -> pd.DataFrame:
    """Cache the daily backtest table, keyed on the results root string."""
    return read_backtest(Path(results_root))


@st.cache_data(ttl=_CACHE_TTL_SECONDS, max_entries=_CACHE_MAX_ENTRIES)
def _read_summary_cached(results_root: str, name: str) -> pd.DataFrame:
    """Cache one backtest summary parquet, keyed on the results root string and the table name."""
    return pd.read_parquet(Path(results_root) / f"{name}.parquet")


def resolve_root() -> Path:
    """The repository root that holds the data directory, overridable by the environment."""
    override = os.environ.get(DATA_ROOT_ENV)
    return Path(override) if override else _REPO_ROOT


class DashboardDataMissing(FileNotFoundError):
    """A table the dashboard needs has not been produced by the pipeline yet."""


class DashboardData:
    """Read only access to the processed and results artifacts under one repository root."""

    def __init__(self, cfg: Config, root: Path) -> None:
        self._processed_root = root / cfg.data.paths.processed
        self._results_root = root / cfg.data.paths.results

    def _require(self, path: Path) -> Path:
        if not path.exists():
            raise DashboardDataMissing(f"expected artifact not found at {path}")
        return path

    def prices(self, market: str) -> pd.DataFrame:
        """The day ahead or real time settlement point price table."""
        name = _MARKET_TABLE[market]
        self._require(self._processed_root / f"{name}.parquet")
        return _read_processed_cached(str(self._processed_root), name)

    def forecasts(self) -> pd.DataFrame:
        """The walk forward forecasts table of predicted against realised prices."""
        self._require(self._results_root / f"{FORECASTS_NAME}.parquet")
        return _read_forecasts_cached(str(self._results_root))

    def metrics(self) -> pd.DataFrame:
        """The per model error metrics including the pooled all hours row."""
        self._require(self._results_root / f"{METRICS_NAME}.parquet")
        return _read_metrics_cached(str(self._results_root))

    def backtest(self) -> pd.DataFrame:
        """The daily backtest table of profit, cycles, and capture per scenario."""
        self._require(self._results_root / f"{BACKTEST_NAME}.parquet")
        return _read_backtest_cached(str(self._results_root))

    def sensitivities(self) -> pd.DataFrame:
        """The annualised profit sweep across battery duration and cycling cost."""
        self._require(self._results_root / f"{SENSITIVITIES_NAME}.parquet")
        return _read_summary_cached(str(self._results_root), SENSITIVITIES_NAME)

    def concentration(self) -> pd.DataFrame:
        """The profit concentration curve for both scenarios."""
        self._require(self._results_root / f"{CONCENTRATION_NAME}.parquet")
        return _read_summary_cached(str(self._results_root), CONCENTRATION_NAME)
