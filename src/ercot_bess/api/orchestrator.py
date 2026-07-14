"""The daily pipeline, the scheduled job that refreshes every artifact end to end.

It ingests each source and then rebuilds the processed tables, the feature matrix, the
forecasts, and the backtest, in that order. Each ingestion source is wrapped on its own so a
single outage is logged and skipped while the rest of the run proceeds. The raw layer is never
overwritten, so it keeps the partitions a skipped source pulled on earlier days, and the
downstream tables still build from history. This is the orchestration layer, the run functions in each module
do the work.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Mapping
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from ..backtest.build import run_backtest_to_disk
from ..config import Config, load_config
from ..features.build import run_features
from ..forecast.build import run_forecast
from ..ingest.ercot import (
    DATASET_DA,
    DATASET_DEMAND,
    DATASET_RT,
    fetch_da_spp,
    fetch_demand_forecast,
    fetch_rt_spp,
)
from ..ingest.weather import DATASET as WEATHER_DATASET
from ..ingest.weather import fetch_weather
from ..validate.build import run_build

_REPO_ROOT = Path(__file__).resolve().parents[3]

Fetcher = Callable[..., pd.DataFrame]

# one entry per raw source, keyed by dataset name so a skip logs the name the partitions use
_FETCHERS: dict[str, Fetcher] = {
    DATASET_DA: fetch_da_spp,
    DATASET_RT: fetch_rt_spp,
    DATASET_DEMAND: fetch_demand_forecast,
    WEATHER_DATASET: fetch_weather,
}

# the errors a live pull actually raises, a network failure or a missing key or bad status,
# caught per source so one failure skips that source rather than the whole run
FETCH_ERRORS = (requests.exceptions.RequestException, RuntimeError)

# the hosted feeds settle by the next day, so the window ends a day or two before today and a
# short lookback re-pulls the last day in case a late correction landed since the prior run
_FIELD_LATENCY_DAYS = 2
_REFRESH_LOOKBACK_DAYS = 1

log = logging.getLogger("ercot_bess.pipeline")


def resolve_window(
    start: date | None, end: date | None, *, today: date | None = None
) -> tuple[date, date]:
    """Fill in a start and end that respect field latency when the caller leaves them open."""
    today = today if today is not None else date.today()
    if end is None:
        end = today - timedelta(days=_FIELD_LATENCY_DAYS)
    if start is None:
        start = end - timedelta(days=_REFRESH_LOOKBACK_DAYS)
    return start, end


def ingest_sources(
    cfg: Config,
    repo_root: Path,
    start: date,
    end: date,
    *,
    fetchers: Mapping[str, Fetcher] = _FETCHERS,
    force: bool = False,
) -> list[str]:
    """Pull each source for the window, skipping any that fail, and return the skipped names."""
    raw_root = repo_root / cfg.data.paths.raw
    skipped: list[str] = []
    for name, fetch in fetchers.items():
        try:
            frame = fetch(cfg, start, end, raw_root, force=force)
        except FETCH_ERRORS as exc:
            log.warning("skipping source %s for %s to %s, %s", name, start, end, exc)
            skipped.append(name)
            continue
        log.info("ingested %s rows for %s from %s to %s", len(frame), name, start, end)
    return skipped


def run_pipeline(
    cfg: Config,
    repo_root: Path,
    start: date,
    end: date,
    *,
    fetchers: Mapping[str, Fetcher] = _FETCHERS,
    force: bool = False,
    sensitivities: bool = True,
) -> list[str]:
    """Ingest the window then rebuild every downstream artifact, returning the skipped sources.

    Only ingestion degrades gracefully. The downstream steps are deterministic transforms over
    whatever raw and processed data exists, so a failure there is a real fault and is left to
    raise rather than be swallowed.
    """
    skipped = ingest_sources(cfg, repo_root, start, end, fetchers=fetchers, force=force)
    run_build(cfg, repo_root)
    run_features(cfg, repo_root)
    run_forecast(cfg, repo_root)
    run_backtest_to_disk(cfg, repo_root, sensitivities=sensitivities)
    return skipped


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ercot_bess.api.orchestrator")
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-sensitivities", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    start, end = resolve_window(args.start, args.end)
    log.info("pipeline start for window %s to %s", start, end)
    skipped = run_pipeline(
        cfg,
        _REPO_ROOT,
        start,
        end,
        force=args.force,
        sensitivities=not args.no_sensitivities,
    )
    if skipped:
        log.warning("pipeline finished with skipped sources %s", skipped)
    else:
        log.info("pipeline finished, all sources ingested")


if __name__ == "__main__":
    main()
