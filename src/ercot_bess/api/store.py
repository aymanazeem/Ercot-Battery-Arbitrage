"""Disk reads for the API, the layer that turns the artifact files into frames.

This is the only place the service touches disk. Each reader reuses a module reader so the
API never re-derives a table, and a missing file raises ArtifactMissing so the app can answer
with a clean not found rather than a stack trace. The shaping lives in serialise and the
FastAPI wiring in app.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..backtest.build import ANNUAL_NAME, ANNUALISED_NAME
from ..config import Config
from ..forecast.build import FORECASTS_NAME, read_forecasts
from ..validate.build import read_processed

# the market query value maps to the processed price table name
_MARKET_TABLE = {"da": "da_prices", "rt": "rt_prices"}


class ArtifactMissing(FileNotFoundError):
    """A precomputed table the endpoint needs has not been produced yet."""


class Store:
    """Read only access to the processed and results artifacts under one repo root."""

    def __init__(self, cfg: Config, repo_root: Path) -> None:
        self._processed_root = repo_root / cfg.data.paths.processed
        self._results_root = repo_root / cfg.data.paths.results

    def _require(self, path: Path) -> Path:
        if not path.exists():
            raise ArtifactMissing(f"expected artifact not found at {path}")
        return path

    def prices(self, market: str) -> pd.DataFrame:
        """The day ahead or real time settlement point price table."""
        name = _MARKET_TABLE[market]
        self._require(self._processed_root / f"{name}.parquet")
        return read_processed(self._processed_root, name)

    def forecasts(self) -> pd.DataFrame:
        """The walk forward forecasts table of predicted against realised prices."""
        self._require(self._results_root / f"{FORECASTS_NAME}.parquet")
        return read_forecasts(self._results_root)

    def annual(self) -> pd.DataFrame:
        """The annual backtest totals per settlement point and scenario."""
        path = self._require(self._results_root / f"{ANNUAL_NAME}.parquet")
        return pd.read_parquet(path)

    def annualised(self) -> pd.DataFrame:
        """The annualised per kW year backtest table per settlement point and scenario."""
        path = self._require(self._results_root / f"{ANNUALISED_NAME}.parquet")
        return pd.read_parquet(path)
