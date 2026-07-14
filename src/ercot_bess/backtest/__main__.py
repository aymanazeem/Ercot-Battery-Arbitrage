"""Command line entry point for the backtest engine.

Run python -m ercot_bess.backtest to read the forecasts and day ahead prices and write the
daily backtest and its summary tables under the results directory. The model that drives the
forecast driven scenario is chosen here since the results backtest table carries no model
column, so the choice is explicit rather than hidden.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..config import load_config
from ..forecast.schema import (
    MODEL_LEAR,
    MODEL_LIGHTGBM,
    MODEL_NAIVE_WEEK,
    MODEL_SEASONAL_DOW,
)
from .build import run_backtest_to_disk
from .schema import DELIVERY_DATE

_REPO_ROOT = Path(__file__).resolve().parents[3]

_MODELS = (MODEL_NAIVE_WEEK, MODEL_SEASONAL_DOW, MODEL_LEAR, MODEL_LIGHTGBM)

log = logging.getLogger("ercot_bess.backtest")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ercot_bess.backtest")
    parser.add_argument(
        "--model",
        choices=_MODELS,
        default=MODEL_LIGHTGBM,
        help="the forecasting model that drives the forecast driven scenario",
    )
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--no-sensitivities", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    backtest, _, _ = run_backtest_to_disk(
        cfg,
        _REPO_ROOT,
        model=args.model,
        write=not args.no_write,
        sensitivities=not args.no_sensitivities,
    )
    days = backtest[DELIVERY_DATE].nunique()
    log.info("backtest %s rows across %s days driven by %s", len(backtest), days, args.model)


if __name__ == "__main__":
    main()
