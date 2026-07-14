"""The FastAPI service, a thin read only layer over the precomputed artifacts.

The app never computes in a request. Each endpoint reads a table through the Store, shapes it
with the serialise functions, and returns JSON. A missing artifact becomes a clean not found
rather than a stack trace. create_app takes the config and repo root so a test can point it at
a seeded directory, and the module level app serves the real data under the repo root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..config import Config, load_config
from ..forecast.schema import MODEL_LIGHTGBM
from . import serialise
from .store import ArtifactMissing, Store

# api/app.py sits at src/ercot_bess/api, so the repo root is three parents up
_REPO_ROOT = Path(__file__).resolve().parents[3]


def create_app(cfg: Config | None = None, repo_root: Path | None = None) -> FastAPI:
    """Build the service bound to one config and repo root so a test can seed its own data."""
    cfg = cfg or load_config()
    repo_root = repo_root or _REPO_ROOT
    store = Store(cfg, repo_root)
    market_cfg = cfg.market.market
    primary = market_cfg.primary_settlement_point
    display_tz = market_cfg.timezone_display

    app = FastAPI(title="ERCOT BESS", version="0.1.0")

    @app.exception_handler(ArtifactMissing)
    async def _artifact_missing(_request, exc: ArtifactMissing) -> JSONResponse:
        # the pipeline has not produced this table yet, a clean not found beats a 500
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict:
        """Liveness only, it touches no disk so it answers even before the first pipeline run."""
        return {"status": "ok"}

    @app.get("/prices/latest")
    def prices_latest(
        settlement_point: str = primary, market: Literal["da", "rt"] = "da"
    ) -> dict:
        """The most recent local day of prices for one settlement point and market."""
        return serialise.latest_prices(store.prices(market), settlement_point, market, display_tz)

    @app.get("/forecast/latest")
    def forecast_latest(settlement_point: str = primary, model: str = MODEL_LIGHTGBM) -> dict:
        """The most recent delivery day of predicted against realised prices for one model."""
        return serialise.latest_forecast(store.forecasts(), settlement_point, model)

    @app.get("/backtest/summary")
    def backtest_summary() -> dict:
        """The headline backtest numbers, annual totals and the annualised per kW year table."""
        return serialise.backtest_summary(store.annual(), store.annualised())

    return app


app = create_app()
