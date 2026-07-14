"""Command line entry point that builds the processed tables.

Run python -m ercot_bess.validate to read the raw partitions and write the processed tables.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..config import load_config
from .build import run_build

_REPO_ROOT = Path(__file__).resolve().parents[3]

log = logging.getLogger("ercot_bess.validate")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ercot_bess.validate")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    tables = run_build(cfg, _REPO_ROOT, write=not args.no_write)
    for name, frame in tables.items():
        log.info("built %s rows for %s", len(frame), name)


if __name__ == "__main__":
    main()
