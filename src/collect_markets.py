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
    is_yes_no: bool


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


def _preferred_positive_index(outcomes: List[str]) -> int:
    norm = [_normalized_label(x) for x in outcomes]
    preferred_pairs = [
        ("yes", "no"),
        ("true", "false"),
        ("up", "down"),
        ("over", "under"),
    ]
    for pos, neg in preferred_pairs:
        if pos in norm and neg in norm:
            return norm.index(pos)
    # Fallback for generic binary labels (e.g., Team A/Team B): use first label.
    return 0


def _extract_binary_token_ids(market: Dict[str, Any]) -> Optional[Dict[str, str]]:
    outcomes = _parse_json_array(market.get("outcomes"))
    token_ids = _parse_json_array(market.get("clobTokenIds"))
    if not outcomes or not token_ids or len(outcomes) != len(token_ids):
        return None
    if len(outcomes) != 2:
        return None

    labels = [str(x).strip() for x in outcomes]
    idx_pos = _preferred_positive_index(labels)
    idx_neg = 1 - idx_pos
    return {
        "yes": str(token_ids[idx_pos]),
        "no": str(token_ids[idx_neg]),
        "yes_label": labels[idx_pos],
        "no_label": labels[idx_neg],
        "is_yes_no": str(_normalized_label(labels[idx_pos]) == "yes" and _normalized_label(labels[idx_neg]) == "no"),
        "idx_pos": str(idx_pos),
    }


def _extract_outcome_from_prices(market: Dict[str, Any], idx_pos: int) -> Optional[int]:
    prices = _parse_json_array(market.get("outcomePrices"))
    outcomes = _parse_json_array(market.get("outcomes"))
    if len(prices) != 2 or len(outcomes) != 2:
        return None

    try:
        parsed = [float(x) for x in prices]
    except (TypeError, ValueError):
        return None

    pos_price = parsed[idx_pos]
    neg_price = parsed[1 - idx_pos]
    if abs((pos_price + neg_price) - 1.0) > 0.05:
        return None
    if pos_price >= 0.97:
        return 1
    if pos_price <= 0.03:
        return 0
    return None


def _normalized_category(category: Any) -> Optional[str]:
    if category is None:
        return None
    text = str(category).strip()
    if not text:
        return None
    return text


def _extract_primary_event_id(raw_market: Dict[str, Any]) -> Optional[str]:
    events = raw_market.get("events")
    if not isinstance(events, list) or not events:
        return None
    first = events[0]
    if not isinstance(first, dict):
        return None
    event_id = first.get("id")
    if event_id is None:
        return None
    text = str(event_id).strip()
    return text if text else None


def _extract_tag_records(value: Any) -> list[dict[str, Any]]:
    tags = _parse_json_array(value)
    out: list[dict[str, Any]] = []
    for tag in tags:
        if isinstance(tag, dict):
            out.append(tag)
    return out


def _infer_category_from_tags(tags: list[dict[str, Any]]) -> Optional[str]:
    if not tags:
        return None

    canonical_by_exact = {
        "sports": "Sports",
        "politics": "Politics",
        "crypto": "Crypto",
        "economy": "Economy",
        "finance": "Economy",
        "world": "World",
        "technology": "Technology",
        "tech": "Technology",
        "science": "Science",
        "culture": "Culture",
    }
    ignore = {
        "hide from new",
        "recurring",
        "up or down",
        "5m",
        "15m",
        "1h",
        "daily",
        "weekly",
    }
    keyword_map = [
        ("Sports", ["sports", "nba", "nfl", "nhl", "soccer", "tennis", "basketball", "hockey", "mlb", "ncaa", "esports", "premier-league", "epl", "ligue", "bundesliga", "serie-a", "serie-b", "la-liga", "golf", "mma", "boxing", "cricket"]),
        ("Politics", ["politic", "election", "trump", "biden", "congress", "senate", "house", "president", "geopolitic"]),
        ("Crypto", ["crypto", "bitcoin", "ethereum", "solana", "dogecoin", "xrp", "altcoin"]),
        ("Economy", ["econom", "finance", "fed", "rates", "spx", "s&p", "stock", "inflation", "trade-war"]),
        ("World", ["world", "middle-east", "ukraine", "russia", "china", "israel", "syria"]),
        ("Technology", ["tech", "ai", "artificial-intelligence"]),
    ]

    lowered: list[tuple[str, str]] = []
    for tag in tags:
        label = str(tag.get("label", "")).strip().lower()
        slug = str(tag.get("slug", "")).strip().lower()
        lowered.append((label, slug))

    for label, slug in lowered:
        if label in canonical_by_exact:
            return canonical_by_exact[label]
        if slug in canonical_by_exact:
            return canonical_by_exact[slug]

    for label, slug in lowered:
        if label in ignore or slug in ignore:
            continue
        hay = f"{label} {slug}"
        for canonical, needles in keyword_map:
            if any(n in hay for n in needles):
                return canonical

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
    event_category_cache: dict[str, Optional[str]] = {}
    target = config.target_markets
    vol_floor = config.min_volume
    vol_note = f", min volume {vol_floor:,.0f}" if vol_floor > 0 else ""
    print(
        f"[markets] discovering up to {target} markets{vol_note} "
        f"(Gamma pages <= {config.gamma_max_pages}, page size {config.gamma_page_size})...",
        flush=True,
    )

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
            print(
                f"[markets] page {page_idx + 1}: empty response - stopping discovery",
                flush=True,
            )
            break
        for raw in page:
            ids = _extract_binary_token_ids(raw)
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
            outcome_yes = _extract_outcome_from_prices(raw, int(ids["idx_pos"]))
            if outcome_yes is None:
                continue

            volume = raw.get("volumeNum", raw.get("volume", 0.0))
            try:
                volume_f = float(volume or 0.0)
            except (TypeError, ValueError):
                volume_f = 0.0

            if config.min_volume > 0 and volume_f < config.min_volume:
                continue

            category = _normalized_category(raw.get("category"))
            if category is None:
                category = _infer_category_from_tags(_extract_tag_records(raw.get("tags")))
            if category is None:
                event_id = _extract_primary_event_id(raw)
                if event_id:
                    if event_id not in event_category_cache:
                        try:
                            event_payload = client.get_event(event_id)
                            event_category_cache[event_id] = (
                                _normalized_category(event_payload.get("category"))
                                or _infer_category_from_tags(
                                    _extract_tag_records(event_payload.get("tags"))
                                )
                            )
                        except Exception:
                            event_category_cache[event_id] = None
                    category = event_category_cache.get(event_id)

            selected.append(
                MarketRecord(
                    market_id=market_id,
                    condition_id=condition_id,
                    question=str(raw.get("question", "")).strip(),
                    category=category,
                    end_time=end_dt,
                    closed_time=_parse_dt(raw.get("closedTime")),
                    volume_total_market=volume_f,
                    token_id_yes=ids["yes"],
                    token_id_no=ids.get("no"),
                    outcome_yes=outcome_yes,
                    yes_label=ids.get("yes_label", "Yes"),
                    no_label=ids.get("no_label", "No"),
                    is_yes_no=ids.get("is_yes_no", "False").lower() == "true",
                )
            )
            selected_ids.add(market_id)

        print(
            f"[markets] page {page_idx + 1}/{config.gamma_max_pages}: "
            f"{len(selected)} markets pass filters so far (target {target})",
            flush=True,
        )
        if len(selected) >= config.target_markets * 2:
            print(
                f"[markets] enough candidates ({len(selected)} >= {target * 2}); "
                "stopping Gamma pagination",
                flush=True,
            )
            break

    selected.sort(key=lambda m: m.volume_total_market, reverse=True)
    kept = selected[: config.target_markets]
    print(
        f"[markets] done: kept top {len(kept)} by volume "
        f"(had {len(selected)} candidates after filters)",
        flush=True,
    )
    return kept

