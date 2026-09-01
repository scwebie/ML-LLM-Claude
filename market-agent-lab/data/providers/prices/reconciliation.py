"""Cross-source price reconciliation (Phase 3).

Pure functions, no network I/O -- takes two already-fetched bar sets and
produces a canonical bar series plus a full reconciliation audit trail.
Never silently combines conflicting prices: every date gets an explicit
:class:`~core.schemas_v2.ReconciliationStatus`, and a caller can choose
whether ``MAJOR_DIFFERENCE`` rows are usable for training (default: no).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from core.schemas_v2 import PriceReconciliation, ReconciliationStatus


@dataclass(frozen=True)
class ReconciliationTolerance:
    minor_threshold: float = 0.005  # 0.5%
    major_threshold: float = 0.02  # 2%


def reconcile_bar_sets(
    symbol: str,
    primary_source: str,
    primary_bars: dict[datetime, dict],
    secondary_source: str,
    secondary_bars: dict[datetime, dict],
    tolerance: ReconciliationTolerance | None = None,
    created_at: datetime | None = None,
) -> tuple[list[dict], list[PriceReconciliation]]:
    """Returns (canonical_bars, reconciliation_records).

    ``canonical_bars`` is the bar actually written to ``market_observations``
    for each date: primary when available, secondary as a documented
    fallback when primary is missing, skipped entirely when neither has
    data for that date.
    """
    tolerance = tolerance or ReconciliationTolerance()
    created_at = created_at or datetime.now()

    all_dates = sorted(set(primary_bars) | set(secondary_bars))
    canonical: list[dict] = []
    records: list[PriceReconciliation] = []

    for date in all_dates:
        p = primary_bars.get(date)
        s = secondary_bars.get(date)

        if p is not None and s is not None:
            diff = abs(p["close"] - s["close"]) / p["close"] if p["close"] else float("inf")
            if diff <= tolerance.minor_threshold:
                status = ReconciliationStatus.VALIDATED
            elif diff <= tolerance.major_threshold:
                status = ReconciliationStatus.MINOR_DIFFERENCE
            else:
                status = ReconciliationStatus.MAJOR_DIFFERENCE
            canonical.append({**p, "date": date, "reconciliation_status": status.value})
        elif p is not None and s is None:
            diff = None
            status = ReconciliationStatus.SECONDARY_MISSING
            canonical.append({**p, "date": date, "reconciliation_status": status.value})
        elif p is None and s is not None:
            diff = None
            status = ReconciliationStatus.PRIMARY_MISSING
            # Documented fallback: use the secondary bar since primary has nothing.
            canonical.append({**s, "date": date, "reconciliation_status": status.value})
        else:
            continue  # unreachable given all_dates construction, kept for clarity

        records.append(
            PriceReconciliation(
                symbol=symbol, date=date,
                primary_source=primary_source if p is not None else None,
                primary_close=p["close"] if p is not None else None,
                secondary_source=secondary_source if s is not None else None,
                secondary_close=s["close"] if s is not None else None,
                abs_pct_diff=diff, status=status, created_at=created_at,
            )
        )

    return canonical, records


def filter_trainable_bars(canonical_bars: list[dict], allow_major_difference: bool = False) -> list[dict]:
    """Drop MAJOR_DIFFERENCE rows by default -- 'do not train on records
    marked invalid unless explicitly permitted.'"""
    if allow_major_difference:
        return canonical_bars
    return [b for b in canonical_bars if b["reconciliation_status"] != ReconciliationStatus.MAJOR_DIFFERENCE.value]
