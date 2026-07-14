"""Command line entry point for the walk forward forecast.

Run python -m ercot_bess.forecast to read the day ahead model matrix and write the forecasts
and forecast metrics tables under the results directory.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..config import load_config
from .build import run_forecast

_REPO_ROOT = Path(__file__).resolve().parents[3]

log = logging.getLogger("ercot_bess.forecast")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ercot_bess.forecast")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    forecasts, metrics = run_forecast(cfg, _REPO_ROOT, write=not args.no_write)
    folds = forecasts["fold_index"].nunique()
    log.info("forecast %s rows across %s folds, %s metric rows", len(forecasts), folds, len(metrics))


if __name__ == "__main__":
    main()
