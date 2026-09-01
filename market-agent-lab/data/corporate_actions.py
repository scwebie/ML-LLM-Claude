"""Corporate actions ingestion (Phase 4): splits, reverse splits, and
dividends, parsed from the same Yahoo Finance chart response already
fetched for prices (``events=div,splits``).

Adjustment discipline: Yahoo's ``adjclose`` series is already fully
split-and-dividend adjusted -- ``data/real_prices.py`` stores it verbatim
as ``adjusted_close`` and the raw, unadjusted series as ``close``. This
module does NOT recompute or re-apply any adjustment; it only extracts
and persists the underlying corporate-action events for audit purposes.
Re-deriving an adjustment from these events on top of an already-adjusted
series is exactly how double-adjustment bugs happen, so it is deliberately
not done anywhere in this codebase.
"""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb

from core.schemas_v2 import CorporateAction, CorporateActionType
from data.providers.prices.primary import YahooFinancePriceProvider
from database import repository_v2 as repo_v2


def parse_yahoo_events(symbol: str, events: dict, source: str = "yahoo_finance") -> list[CorporateAction]:
    retrieved_at = datetime.now(UTC)
    actions: list[CorporateAction] = []

    for _key, div in (events.get("dividends") or {}).items():
        ex_date = datetime.fromtimestamp(div["date"], tz=UTC).replace(tzinfo=None)
        actions.append(
            CorporateAction(
                symbol=symbol, action_type=CorporateActionType.DIVIDEND, ex_date=ex_date,
                cash_amount=float(div["amount"]), source=source, retrieved_at=retrieved_at,
            )
        )

    for _key, split in (events.get("splits") or {}).items():
        ex_date = datetime.fromtimestamp(split["date"], tz=UTC).replace(tzinfo=None)
        numerator = float(split.get("numerator", 1))
        denominator = float(split.get("denominator", 1)) or 1.0
        ratio = numerator / denominator
        action_type = CorporateActionType.SPLIT if ratio >= 1.0 else CorporateActionType.REVERSE_SPLIT
        actions.append(
            CorporateAction(
                symbol=symbol, action_type=action_type, ex_date=ex_date, ratio=ratio,
                source=source, retrieved_at=retrieved_at,
            )
        )

    return actions


def ingest_corporate_actions(con: duckdb.DuckDBPyConnection, symbols: list[str], start: datetime, end: datetime) -> dict:
    provider = YahooFinancePriceProvider()
    summary: dict[str, int] = {}
    all_actions: list[CorporateAction] = []

    for symbol in symbols:
        try:
            events = provider.get_corporate_action_events(symbol, start, end)
        except Exception:  # noqa: BLE001
            summary[symbol] = 0
            continue
        actions = parse_yahoo_events(symbol, events)
        all_actions.extend(actions)
        summary[symbol] = len(actions)

    if all_actions:
        repo_v2.insert_corporate_actions(con, all_actions)

    return {"per_symbol": summary, "total": len(all_actions)}
