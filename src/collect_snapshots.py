from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

from .api_clients import PolymarketClient
from .collect_markets import MarketRecord
from .config import PipelineConfig


@dataclass
class SnapshotRow:
    market_id: str
    condition_id: str
    question: str
    category: Optional[str]
    token_id_yes: str
    end_time: datetime
    snapshot_time: datetime
    horizon_min: int
    price_last_trade: Optional[float]
    price_midpoint: Optional[float]
    volume_total_market: float
    outcome_yes: int
    is_binary_yes_no: bool = True


def _to_ts(dt: datetime) -> int:
    return int(dt.timestamp())


def _history_series(history: Iterable[Dict[str, float]]) -> Dict[str, List[float]]:
    pairs = []
    for row in history:
        try:
            t = int(row.get("t"))
            p = float(row.get("p"))
        except (TypeError, ValueError):
            continue
        pairs.append((t, p))
    pairs.sort(key=lambda x: x[0])
    return {"t": [x[0] for x in pairs], "p": [x[1] for x in pairs]}


def _nearest_before_price(series: Dict[str, List[float]], target_ts: int) -> Optional[float]:
    ts = series["t"]
    if not ts:
        return None
    idx = bisect_right(ts, target_ts) - 1
    if idx < 0:
        return None
    return float(series["p"][idx])


def build_snapshot_rows(
    client: PolymarketClient,
    markets: List[MarketRecord],
    config: PipelineConfig,
) -> List[SnapshotRow]:
    rows: List[SnapshotRow] = []

    for market in markets:
        horizon_max = max(config.horizons_minutes)
        start_ts = _to_ts(market.end_time - timedelta(minutes=horizon_max + 120))
        end_ts = _to_ts(market.end_time + timedelta(minutes=1))
        history = client.get_prices_history(
            token_id=market.token_id_yes,
            start_ts=start_ts,
            end_ts=end_ts,
            interval=config.clob_history_interval,
            fidelity=config.clob_history_fidelity,
        )
        series = _history_series(history)

        for horizon in config.horizons_minutes:
            snap_dt = market.end_time - timedelta(minutes=horizon)
            snap_ts = _to_ts(snap_dt)
            last_trade = _nearest_before_price(series, snap_ts)
            rows.append(
                SnapshotRow(
                    market_id=market.market_id,
                    condition_id=market.condition_id,
                    question=market.question,
                    category=market.category,
                    token_id_yes=market.token_id_yes,
                    end_time=market.end_time.astimezone(timezone.utc),
                    snapshot_time=snap_dt.astimezone(timezone.utc),
                    horizon_min=horizon,
                    price_last_trade=last_trade,
                    # Historical midpoint is not exposed as a time series in public endpoints.
                    price_midpoint=None,
                    volume_total_market=market.volume_total_market,
                    outcome_yes=market.outcome_yes,
                )
            )

    return rows

