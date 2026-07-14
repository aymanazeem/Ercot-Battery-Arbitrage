"""The raw layer that is never overwritten, dated partitions of parquet with a retrieval stamp.

Nothing here calls the network. A source module supplies a per day fetch function
and this layer handles partitioning, the retrieval stamp, and the skip on rerun.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

RETRIEVED_AT = "retrieved_at_utc"


def date_range(start: date, end: date) -> list[date]:
    """Every calendar day from start to end, both ends included."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    span = (end - start).days
    return [start + timedelta(days=offset) for offset in range(span + 1)]


def partition_path(raw_root: Path, dataset: str, day: date) -> Path:
    """The parquet file for one dataset on one day."""
    return Path(raw_root) / dataset / f"date={day.isoformat()}" / "data.parquet"


def add_retrieved_at(frame: pd.DataFrame, now_utc: pd.Timestamp | None = None) -> pd.DataFrame:
    """Stamp every row with the moment the pull happened so the layer is reproducible."""
    stamp = now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")
    stamped = frame.copy()
    stamped[RETRIEVED_AT] = stamp
    return stamped


def write_partition(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def read_partition(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def cache_days(
    dataset: str,
    start: date,
    end: date,
    fetch_day: Callable[[date], pd.DataFrame],
    raw_root: Path,
    *,
    force: bool = False,
    now_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch each missing day, stamp it, write its partition, and return the whole window.

    A day whose partition already exists is read from disk and never refetched unless
    force is set, so reruns do not duplicate partitions or hit the source again.
    """
    frames: list[pd.DataFrame] = []
    for day in date_range(start, end):
        path = partition_path(raw_root, dataset, day)
        if path.exists() and not force:
            frames.append(read_partition(path))
            continue
        fetched = add_retrieved_at(fetch_day(day), now_utc)
        write_partition(path, fetched)
        frames.append(fetched)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def cache_range(
    dataset: str,
    start: date,
    end: date,
    fetch_range: Callable[[date, date], pd.DataFrame],
    raw_root: Path,
    *,
    interval_column: str = "interval_start_utc",
    force: bool = False,
    now_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch a whole date range in one call, then split it into one partition per day.

    A source with a per month row budget cannot afford a request per day, so the fetch
    covers the span of missing days at once and the result is bucketed by the interval
    start date into the same dated partitions the rest of the pipeline expects. Days
    already on disk are neither refetched nor rewritten unless force is set, so the fetch
    only spans the gap that is actually missing.
    """
    days = date_range(start, end)
    missing = [d for d in days if force or not partition_path(raw_root, dataset, d).exists()]
    if missing:
        fetched = fetch_range(min(missing), max(missing))
        if fetched.empty:
            raise RuntimeError(
                f"{dataset}: the source returned no rows for {min(missing)} to {max(missing)}"
            )
        fetched = add_retrieved_at(fetched, now_utc)
        interval_date = pd.to_datetime(fetched[interval_column], utc=True).dt.date
        for day in missing:
            path = partition_path(raw_root, dataset, day)
            write_partition(path, fetched[interval_date == day].reset_index(drop=True))
    frames = [
        read_partition(partition_path(raw_root, dataset, day))
        for day in days
        if partition_path(raw_root, dataset, day).exists()
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
