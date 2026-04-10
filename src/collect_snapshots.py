from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

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
    closed_time: Optional[datetime]
    snapshot_time: datetime
    horizon_min: int
    price_last_trade: Optional[float]
    price_midpoint: Optional[float]
    price_source: Optional[str]
    price_timestamp: Optional[datetime]
    price_staleness_min: Optional[float]
    volume_total_market: float
    outcome_yes: int
    outcome_label_positive: str
    outcome_label_negative: Optional[str]
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


def _nearest_before_price(
    series: Dict[str, List[float]], target_ts: int
) -> Optional[Tuple[float, int]]:
    ts = series["t"]
    if not ts:
        return None
    idx = bisect_right(ts, target_ts) - 1
    if idx < 0:
        return None
    return float(series["p"][idx]), int(series["t"][idx])


def build_snapshot_rows(
    client: PolymarketClient,
    markets: List[MarketRecord],
    config: PipelineConfig,
) -> List[SnapshotRow]:
    rows: List[SnapshotRow] = []
    total = len(markets)
    print(f"[snapshots] fetching price history for {total} markets...", flush=True)

    for i, market in enumerate(markets, start=1):
        if i % 100 == 0 or i == total:
            print(f"[snapshots] {i}/{total} markets", flush=True)
        # Pull full history because windowed queries can omit older relevant prints.
        history_yes = client.get_prices_history(
            token_id=market.token_id_yes,
            interval=config.clob_history_interval,
            fidelity=config.clob_history_fidelity,
        )
        yes_series = _history_series(history_yes)

        no_series = {"t": [], "p": []}
        if market.token_id_no:
            history_no = client.get_prices_history(
                token_id=market.token_id_no,
                interval=config.clob_history_interval,
                fidelity=config.clob_history_fidelity,
            )
            no_series = _history_series(history_no)

        for horizon in config.horizons_minutes:
            snap_dt = market.end_time - timedelta(minutes=horizon)
            snap_ts = _to_ts(snap_dt)
            yes_result = _nearest_before_price(yes_series, snap_ts)
            no_result = _nearest_before_price(no_series, snap_ts)

            yes_last_trade: Optional[float] = None
            used_ts: Optional[int] = None
            source: Optional[str] = None
            if yes_result is not None:
                yes_last_trade, used_ts = yes_result
                source = "yes_token_last_trade"
            elif no_result is not None:
                no_last_trade, used_ts = no_result
                yes_last_trade = max(0.0, min(1.0, 1.0 - no_last_trade))
                source = "no_token_complement"

            staleness_min: Optional[float] = None
            used_dt: Optional[datetime] = None
            if used_ts is not None:
                staleness_min = max(0.0, (snap_ts - used_ts) / 60.0)
                used_dt = datetime.fromtimestamp(used_ts, tz=timezone.utc)

            rows.append(
                SnapshotRow(
                    market_id=market.market_id,
                    condition_id=market.condition_id,
                    question=market.question,
                    category=market.category,
                    token_id_yes=market.token_id_yes,
                    end_time=market.end_time.astimezone(timezone.utc),
                    closed_time=market.closed_time.astimezone(timezone.utc) if market.closed_time else None,
                    snapshot_time=snap_dt.astimezone(timezone.utc),
                    horizon_min=horizon,
                    price_last_trade=yes_last_trade,
                    # Historical midpoint is not exposed as a time series in public endpoints.
                    price_midpoint=None,
                    price_source=source,
                    price_timestamp=used_dt,
                    price_staleness_min=staleness_min,
                    volume_total_market=market.volume_total_market,
                    outcome_yes=market.outcome_yes,
                    outcome_label_positive=market.yes_label,
                    outcome_label_negative=market.no_label,
                    is_binary_yes_no=market.is_yes_no,
                )
            )

    print(f"[snapshots] done: {len(rows)} snapshot rows", flush=True)
    return rows

