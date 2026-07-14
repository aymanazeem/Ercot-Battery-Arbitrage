"""Client for the gridstatus.io hosted REST API.

The open source gridstatus scraper only reaches about a month of ERCOT public history,
which is too short to train the forecast, so ingestion runs against the hosted API which
serves years of the same series. This talks to it with requests directly rather than the
gridstatusio package so the pinned environment does not change, and it injects truststore
so a corporate TLS proxy is trusted through the native system trust store.

The free tier caps rows and requests per month, so a query pulls a whole date range in as
few requests as possible, one page of up to fifty thousand rows at a time, advancing by
page number while the response says another page exists. The limit parameter is a hard
total cap that truncates silently, so it is set far above any range and paging does the work.
"""

from __future__ import annotations

import os
import time
from datetime import date

import pandas as pd
import truststore

truststore.inject_into_ssl()

import requests  # noqa: E402  import after truststore so the proxy is trusted

BASE_URL = "https://api.gridstatus.io/v1"
API_KEY_ENV = "GRIDSTATUS_API_KEY"

_MAX_PAGE_SIZE = 50_000
# the limit parameter caps total rows and truncates without warning, so keep it above
# any range we would ever request and let page based paging return everything
_NO_TRUNCATE_LIMIT = 100_000_000
_TIMEOUT_SECONDS = 90
# the free tier allows one request per second, so leave a small margin above that
_MIN_SECONDS_BETWEEN_REQUESTS = 1.2


def require_key(env_name: str = API_KEY_ENV) -> str:
    """The api key from the environment, with a clear message when it is missing."""
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(
            f"gridstatus.io ingestion needs an api key. Set {env_name} in your .env "
            "before ingesting."
        )
    return key


class GridStatusClient:
    """A thin, throttled, paging reader for one gridstatus.io dataset at a time."""

    def __init__(
        self,
        api_key: str,
        *,
        session: requests.Session | None = None,
        base_url: str = BASE_URL,
        min_seconds_between_requests: float = _MIN_SECONDS_BETWEEN_REQUESTS,
    ) -> None:
        self._session = session or requests.Session()
        self._session.headers.update({"x-api-key": api_key})
        self._base_url = base_url.rstrip("/")
        self._min_gap = min_seconds_between_requests
        self._last_request = 0.0

    @classmethod
    def from_config(cls, cfg, **kwargs) -> GridStatusClient:
        """Build a client, reading the api key from the env name in data.yaml."""
        env_name = getattr(cfg.data.sources.ercot, "api_key_env", API_KEY_ENV)
        return cls(require_key(env_name), **kwargs)

    def _throttle(self) -> None:
        gap = time.monotonic() - self._last_request
        if gap < self._min_gap:
            time.sleep(self._min_gap - gap)
        self._last_request = time.monotonic()

    def query(
        self,
        dataset: str,
        start: date,
        end: date,
        *,
        filter_column: str | None = None,
        filter_values: list[str] | None = None,
        page_size: int = _MAX_PAGE_SIZE,
    ) -> pd.DataFrame:
        """Every row of one dataset from start to end inclusive, following all pages.

        The time bounds filter the interval start, both ends included. An optional column
        filter restricts a column to a set of values with the in operator, which is how the
        price datasets are held to the configured hubs instead of every settlement point.
        """
        params: dict[str, object] = {
            "start_time": start.isoformat(),
            "end_time": f"{end.isoformat()}T23:59:59Z",
            "limit": _NO_TRUNCATE_LIMIT,
            "page_size": page_size,
        }
        if filter_column and filter_values:
            params["filter_column"] = filter_column
            params["filter_value"] = ",".join(filter_values)
            params["filter_operator"] = "in"

        rows: list[dict] = []
        page = 1
        while True:
            self._throttle()
            response = self._session.get(
                f"{self._base_url}/datasets/{dataset}/query",
                params={**params, "page": page},
                timeout=_TIMEOUT_SECONDS,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"gridstatus.io query {dataset} page {page} failed with status "
                    f"{response.status_code}: {response.text[:200]}"
                )
            payload = response.json()
            rows.extend(payload.get("data", []))
            if not payload.get("meta", {}).get("hasNextPage"):
                break
            page += 1
        return pd.DataFrame(rows)
