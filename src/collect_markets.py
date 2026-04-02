from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from dateutil import parser as dt_parser

from .api_clients import PolymarketClient
from .config import PipelineConfig


@dataclass
class MarketRecord:
    market_id: str
    condition_id: str
    question: str
    category: Optional[str]
    end_time: datetime
    closed_time: Optional[datetime]
    volume_total_market: float
    token_id_yes: str
    token_id_no: Optional[str]
    outcome_yes: int
    yes_label: str
    no_label: Optional[str]


def _parse_json_array(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = dt_parser.isoparse(value)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _normalized_label(label: str) -> str:
    return label.strip().lower()


def _is_binary_yes_no(outcomes: Iterable[str]) -> bool:
    vals = [_normalized_label(x) for x in outcomes]
    return len(vals) == 2 and "yes" in vals and "no" in vals


def _extract_yes_no_token_ids(market: Dict[str, Any]) -> Optional[Dict[str, str]]:
    outcomes = _parse_json_array(market.get("outcomes"))
    token_ids = _parse_json_array(market.get("clobTokenIds"))
    if not outcomes or not token_ids or len(outcomes) != len(token_ids):
        return None
    if not _is_binary_yes_no([str(x) for x in outcomes]):
        return None

    mapping = {_normalized_label(str(outcome)): str(token_id) for outcome, token_id in zip(outcomes, token_ids)}
    return {"yes": mapping["yes"], "no": mapping["no"]}


def _extract_outcome_from_prices(market: Dict[str, Any]) -> Optional[int]:
    prices = _parse_json_array(market.get("outcomePrices"))
    outcomes = _parse_json_array(market.get("outcomes"))
    if len(prices) != 2 or len(outcomes) != 2:
        return None

    try:
        parsed = [float(x) for x in prices]
    except (TypeError, ValueError):
        return None

    norm_outcomes = [_normalized_label(str(x)) for x in outcomes]
    if "yes" not in norm_outcomes or "no" not in norm_outcomes:
        return None
    yes_idx = norm_outcomes.index("yes")

    yes_price = parsed[yes_idx]
    no_price = parsed[1 - yes_idx]
    if abs((yes_price + no_price) - 1.0) > 0.05:
        return None
    if yes_price >= 0.97:
        return 1
    if yes_price <= 0.03:
        return 0
    return None


def discover_top_volume_markets(
    client: PolymarketClient,
    config: PipelineConfig,
) -> List[MarketRecord]:
    now_utc = datetime.now(timezone.utc)
    min_end = now_utc - timedelta(days=config.lookback_days)
    min_end_iso = min_end.isoformat()
    now_iso = now_utc.isoformat()
    selected: List[MarketRecord] = []
    selected_ids = set()

    for page_idx in range(config.gamma_max_pages):
        offset = page_idx * config.gamma_page_size
        page = client.list_markets(
            limit=config.gamma_page_size,
            offset=offset,
            closed=True,
            end_date_min=min_end_iso,
            end_date_max=now_iso,
            order="volumeNum",
            ascending=False,
        )
        if not page:
            break
        for raw in page:
            ids = _extract_yes_no_token_ids(raw)
            if not ids:
                continue

            market_id = str(raw.get("id"))
            if market_id in selected_ids:
                continue

            condition_id = str(raw.get("conditionId") or "")
            if not condition_id:
                continue
            end_dt = _parse_dt(raw.get("endDateIso") or raw.get("endDate"))
            if not end_dt or end_dt < min_end or end_dt > now_utc:
                continue

            uma_state = str(raw.get("umaResolutionStatus", "")).lower()
            if uma_state not in {"resolved", "disputed", "finalized"}:
                continue
            outcome_yes = _extract_outcome_from_prices(raw)
            if outcome_yes is None:
                continue

            volume = raw.get("volumeNum", raw.get("volume", 0.0))
            try:
                volume_f = float(volume or 0.0)
            except (TypeError, ValueError):
                volume_f = 0.0

            selected.append(
                MarketRecord(
                    market_id=market_id,
                    condition_id=condition_id,
                    question=str(raw.get("question", "")).strip(),
                    category=raw.get("category"),
                    end_time=end_dt,
                    closed_time=_parse_dt(raw.get("closedTime")),
                    volume_total_market=volume_f,
                    token_id_yes=ids["yes"],
                    token_id_no=ids.get("no"),
                    outcome_yes=outcome_yes,
                    yes_label="Yes",
                    no_label="No",
                )
            )
            selected_ids.add(market_id)

        if len(selected) >= config.target_markets * 2:
            break

    selected.sort(key=lambda m: m.volume_total_market, reverse=True)
    return selected[: config.target_markets]

