"""Command line entry point for the feature build.

Run python -m ercot_bess.features to read the processed tables and write the day ahead
model matrix and its feature name list.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..config import load_config
from .build import run_features

_REPO_ROOT = Path(__file__).resolve().parents[3]

log = logging.getLogger("ercot_bess.features")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ercot_bess.features")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    matrix = run_features(cfg, _REPO_ROOT, write=not args.no_write)
    days = matrix["delivery_date"].nunique()
    log.info("built %s rows across %s delivery days", len(matrix), days)


if __name__ == "__main__":
    main()
