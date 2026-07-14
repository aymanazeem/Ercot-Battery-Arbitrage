"""Cleaning helpers shared by every table builder.

These take data and config and return data. No disk or network happens here.
"""

from __future__ import annotations

import pandas as pd

from ..config import Regime


def to_utc(values: pd.Series) -> pd.Series:
    """Parse a timestamp column to timezone aware UTC whatever the source timezone was."""
    return pd.to_datetime(values, utc=True)


def sort_and_dedup(frame: pd.DataFrame, keys: list[str], *, keep: str = "first") -> pd.DataFrame:
    """Sort by the keys and drop rows that repeat a key, collapsing the daylight saving repeat."""
    ordered = frame.sort_values(keys, kind="stable")
    deduped = ordered.drop_duplicates(subset=keys, keep=keep)
    return deduped.reset_index(drop=True)


def assign_regime(interval_utc: pd.Series, regimes: list[Regime], display_tz: str) -> pd.Series:
    """Label each interval with its offer cap regime using the local calendar date.

    The regime dates in market.yaml are ERCOT local effective dates, so the comparison
    happens on the local date rather than the raw UTC date.
    """
    local_dates = interval_utc.dt.tz_convert(display_tz).dt.date
    labels = pd.Series(pd.NA, index=interval_utc.index, dtype="string")
    for regime in regimes:
        mask = pd.Series(True, index=interval_utc.index)
        if regime.start is not None:
            mask &= local_dates >= regime.start
        if regime.end is not None:
            mask &= local_dates <= regime.end
        labels = labels.mask(mask, regime.name)
    if labels.isna().any():
        raise ValueError("some intervals fall outside every configured regime")
    return labels


def spacing_gaps(interval_utc: pd.Series, expected: pd.Timedelta) -> int:
    """Count grid gaps and assert the series sits on the expected interval grid.

    A gap is a step larger than the expected spacing and is allowed since real data has
    missing intervals. A step smaller than the spacing or off the grid is an error.
    """
    if len(interval_utc) < 2:
        return 0
    diffs = interval_utc.sort_values().diff().dropna().dt.total_seconds()
    step = expected.total_seconds()
    if (diffs < step).any():
        raise ValueError("found an interval closer than the expected spacing")
    if (diffs % step != 0).any():
        raise ValueError("found an interval off the expected grid")
    return int((diffs > step).sum())
