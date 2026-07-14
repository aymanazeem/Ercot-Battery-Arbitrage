"""Command line entry point for ingestion.

Run python -m ercot_bess.ingest --source ercot_spp_da --start 2024-01-01 --end 2024-01-02.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import date
from pathlib import Path

from ..config import Config, load_config
from .eia930 import fetch_eia930
from .ercot import fetch_da_spp, fetch_demand_forecast, fetch_rt_spp
from .weather import fetch_weather

_REPO_ROOT = Path(__file__).resolve().parents[3]

_FETCHERS = {
    "ercot_spp_da": fetch_da_spp,
    "ercot_spp_rt": fetch_rt_spp,
    "ercot_demand": fetch_demand_forecast,
    "weather": fetch_weather,
    "eia930": fetch_eia930,
}

# the sources the pipeline needs, so --source all pulls exactly these. eia930 is left out
# because the day ahead demand forecast now comes from ERCOT, and it stays opt in by name
_PIPELINE_SOURCES = ["ercot_spp_da", "ercot_spp_rt", "ercot_demand", "weather"]

log = logging.getLogger("ercot_bess.ingest")


def _load_dotenv(path: Path) -> None:
    """Populate the environment from a .env file for any key not already set.

    Data collection reads its api key from the environment, and the repo convention is to
    keep it in a .env file, so this loads that file at the start of a run. Values already in
    the environment win, so an explicit export always overrides the file.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def _raw_root(cfg: Config) -> Path:
    return _REPO_ROOT / cfg.data.paths.raw


def _run_source(name: str, cfg: Config, start: date, end: date, raw_root: Path, force: bool) -> int:
    frame = _FETCHERS[name](cfg, start, end, raw_root, force=force)
    log.info("ingested %s rows for %s from %s to %s", len(frame), name, start, end)
    return len(frame)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ercot_bess.ingest")
    parser.add_argument("--source", required=True, choices=[*_FETCHERS, "all"])
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _load_dotenv(_REPO_ROOT / ".env")
    cfg = load_config()
    raw_root = _raw_root(cfg)

    names = _PIPELINE_SOURCES if args.source == "all" else [args.source]
    for name in names:
        _run_source(name, cfg, args.start, args.end, raw_root, args.force)


if __name__ == "__main__":
    main()
