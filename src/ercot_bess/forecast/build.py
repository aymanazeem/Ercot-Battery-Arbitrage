"""Run the walk forward forecast and write the results tables.

The main work lives in evaluate and metrics. Here the matrix is read from disk, the models
are run, the errors are summarised, and the two results tables are written. Reading and
writing are the only side effects and they happen here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config
from ..features.build import read_features
from .schema import enforce_forecasts_schema, enforce_metrics_schema

FORECASTS_NAME = "forecasts"
METRICS_NAME = "forecast_metrics"


def read_matrix(cfg: Config, repo_root: Path) -> pd.DataFrame:
    """Read the day ahead model matrix from disk."""
    return read_features(repo_root / cfg.data.paths.features, cfg)


def write_forecasts(forecasts: pd.DataFrame, results_root: Path) -> Path:
    """Write the per delivery hour forecasts as a single parquet file and return its path."""
    path = Path(results_root) / f"{FORECASTS_NAME}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(path, index=False)
    return path


def read_forecasts(results_root: Path) -> pd.DataFrame:
    """Read the forecasts table and re-apply the schema after the parquet round trip."""
    path = Path(results_root) / f"{FORECASTS_NAME}.parquet"
    return enforce_forecasts_schema(pd.read_parquet(path))


def write_metrics(metrics: pd.DataFrame, results_root: Path) -> Path:
    """Write the error metrics as a single parquet file and return its path."""
    path = Path(results_root) / f"{METRICS_NAME}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(path, index=False)
    return path


def read_metrics(results_root: Path) -> pd.DataFrame:
    """Read the metrics table and re-apply the schema after the parquet round trip."""
    path = Path(results_root) / f"{METRICS_NAME}.parquet"
    return enforce_metrics_schema(pd.read_parquet(path))


def run_forecast(
    cfg: Config, repo_root: Path, *, write: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the matrix, run the walk forward forecast, and optionally write the results."""
    from .evaluate import evaluate
    from .metrics import build_metrics

    matrix = read_matrix(cfg, repo_root)
    forecasts = evaluate(matrix, cfg)
    metrics = build_metrics(forecasts)
    if write:
        results_root = repo_root / cfg.data.paths.results
        write_forecasts(forecasts, results_root)
        write_metrics(metrics, results_root)
    return forecasts, metrics
