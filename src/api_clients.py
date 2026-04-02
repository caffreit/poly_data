from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class ApiStats:
    requests: int = 0
    retries: int = 0
    failures: int = 0


@dataclass
class PolymarketClient:
    timeout_seconds: float = 25.0
    max_retries: int = 4
    backoff_base_seconds: float = 0.4
    gamma_base: str = "https://gamma-api.polymarket.com"
    clob_base: str = "https://clob.polymarket.com"
    stats: Dict[str, ApiStats] = field(
        default_factory=lambda: {"gamma": ApiStats(), "clob": ApiStats()}
    )

    def __post_init__(self) -> None:
        self.http = httpx.Client(timeout=self.timeout_seconds)

    def close(self) -> None:
        self.http.close()

    def _request_json(
        self,
        service: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        stats = self.stats[service]
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                stats.requests += 1
                response = self.http.get(url, params=params)
                if response.status_code >= 500 or response.status_code == 429:
                    raise httpx.HTTPStatusError(
                        f"Retryable status: {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # broad catch for HTTP + parse errors
                last_err = exc
                if attempt == self.max_retries:
                    stats.failures += 1
                    break
                stats.retries += 1
                sleep_s = self.backoff_base_seconds * (2**attempt)
                time.sleep(sleep_s)

        raise RuntimeError(f"Request failed for {url}: {last_err}") from last_err

    def list_markets(
        self,
        limit: int,
        offset: int,
        closed: Optional[bool] = None,
        end_date_min: Optional[str] = None,
        end_date_max: Optional[str] = None,
        order: Optional[str] = None,
        ascending: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if closed is not None:
            params["closed"] = str(closed).lower()
        if end_date_min is not None:
            params["end_date_min"] = end_date_min
        if end_date_max is not None:
            params["end_date_max"] = end_date_max
        if order is not None:
            params["order"] = order
        if ascending is not None:
            params["ascending"] = str(ascending).lower()
        return self._request_json("gamma", f"{self.gamma_base}/markets", params=params)

    def get_prices_history(
        self,
        token_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        interval: str = "1m",
        fidelity: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"market": token_id, "interval": interval}
        if fidelity is not None:
            params["fidelity"] = fidelity
        if start_ts is not None:
            params["startTs"] = start_ts
        if end_ts is not None:
            params["endTs"] = end_ts
        payload = self._request_json("clob", f"{self.clob_base}/prices-history", params=params)
        return payload.get("history", [])

    def list_simplified_markets(self, next_cursor: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if next_cursor:
            params["next_cursor"] = next_cursor
        return self._request_json("clob", f"{self.clob_base}/simplified-markets", params=params)

