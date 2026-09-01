"""Tests for cross-source price reconciliation (pure logic, no network)."""

from __future__ import annotations

from datetime import datetime

import pytest

from core.schemas_v2 import ReconciliationStatus
from data.providers.prices.reconciliation import (
    ReconciliationTolerance,
    filter_trainable_bars,
    reconcile_bar_sets,
)


def _bar(close: float) -> dict:
    return {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "adjusted_close": close, "volume": 1_000_000}


def test_matching_prices_are_validated():
    date = datetime(2024, 1, 2)
    canonical, records = reconcile_bar_sets("AAPL", "yahoo", {date: _bar(100.0)}, "stockanalysis", {date: _bar(100.05)})
    assert records[0].status == ReconciliationStatus.VALIDATED
    assert canonical[0]["reconciliation_status"] == ReconciliationStatus.VALIDATED.value


def test_minor_difference_detected():
    date = datetime(2024, 1, 2)
    # 1% difference: above 0.5% minor threshold, below 2% major threshold.
    canonical, records = reconcile_bar_sets("AAPL", "yahoo", {date: _bar(100.0)}, "stockanalysis", {date: _bar(101.0)})
    assert records[0].status == ReconciliationStatus.MINOR_DIFFERENCE


def test_major_difference_detected():
    date = datetime(2024, 1, 2)
    canonical, records = reconcile_bar_sets("AAPL", "yahoo", {date: _bar(100.0)}, "stockanalysis", {date: _bar(110.0)})
    assert records[0].status == ReconciliationStatus.MAJOR_DIFFERENCE
    assert records[0].abs_pct_diff == pytest.approx(0.10, rel=1e-6)


def test_secondary_missing_falls_back_to_primary():
    date = datetime(2024, 1, 2)
    canonical, records = reconcile_bar_sets("AAPL", "yahoo", {date: _bar(100.0)}, "stockanalysis", {})
    assert records[0].status == ReconciliationStatus.SECONDARY_MISSING
    assert canonical[0]["close"] == 100.0


def test_primary_missing_falls_back_to_secondary_with_documented_status():
    date = datetime(2024, 1, 2)
    canonical, records = reconcile_bar_sets("AAPL", "yahoo", {}, "stockanalysis", {date: _bar(99.5)})
    assert records[0].status == ReconciliationStatus.PRIMARY_MISSING
    # Fallback used the secondary value -- not silently substituted, explicitly tagged.
    assert canonical[0]["close"] == 99.5
    assert canonical[0]["reconciliation_status"] == ReconciliationStatus.PRIMARY_MISSING.value


def test_configurable_tolerances_change_classification():
    date = datetime(2024, 1, 2)
    tight = ReconciliationTolerance(minor_threshold=0.001, major_threshold=0.005)
    _, records = reconcile_bar_sets("AAPL", "yahoo", {date: _bar(100.0)}, "stockanalysis", {date: _bar(100.3)}, tolerance=tight)
    # 0.3% diff exceeds the tight minor threshold (0.1%) but is under major (0.5%).
    assert records[0].status == ReconciliationStatus.MINOR_DIFFERENCE


def test_filter_trainable_bars_excludes_major_difference_by_default():
    date1, date2 = datetime(2024, 1, 2), datetime(2024, 1, 3)
    canonical, _ = reconcile_bar_sets(
        "AAPL", "yahoo", {date1: _bar(100.0), date2: _bar(100.0)},
        "stockanalysis", {date1: _bar(100.05), date2: _bar(150.0)},
    )
    trainable = filter_trainable_bars(canonical)
    assert len(trainable) == 1
    assert trainable[0]["date"] == date1


def test_filter_trainable_bars_can_be_explicitly_overridden():
    date = datetime(2024, 1, 2)
    canonical, _ = reconcile_bar_sets("AAPL", "yahoo", {date: _bar(100.0)}, "stockanalysis", {date: _bar(200.0)})
    assert len(filter_trainable_bars(canonical)) == 0
    assert len(filter_trainable_bars(canonical, allow_major_difference=True)) == 1


def test_never_fabricates_a_bar_when_both_sources_missing():
    date_present = datetime(2024, 1, 2)
    date_absent = datetime(2024, 1, 3)
    canonical, records = reconcile_bar_sets(
        "AAPL", "yahoo", {date_present: _bar(100.0)}, "stockanalysis", {date_present: _bar(100.0)}
    )
    dates_seen = {r["date"] for r in canonical}
    assert date_absent not in dates_seen  # never invented for a date neither source had
