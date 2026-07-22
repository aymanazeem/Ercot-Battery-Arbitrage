"""Ingestion. Pull raw source data into the dated raw layer.

One function per source. Each writes dated parquet partitions with a retrieval stamp,
and skips windows already present so reruns are cheap and repeatable.

The public names are re-exported lazily so importing a light submodule, such as the raw
store the dashboard reads through, does not pull in the network clients and their http and
tls dependencies. The names resolve on first attribute access via __getattr__.
"""

from __future__ import annotations

# public name -> submodule that defines it, resolved on first access via __getattr__
_LAZY_EXPORTS = {
    "fetch_da_spp": "ercot",
    "fetch_rt_spp": "ercot",
    "fetch_demand_forecast": "ercot",
    "fetch_weather": "weather",
    "GridStatusClient": "gridstatus_io",
    "cache_days": "raw_store",
    "cache_range": "raw_store",
    "date_range": "raw_store",
    "partition_path": "raw_store",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> object:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
