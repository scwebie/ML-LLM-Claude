"""Tests for data/event_coverage_diagnostics.py (V0.3 Stage 9)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from data.event_coverage_diagnostics import (
    event_probability_coverage_report,
    eventprob_feature_missingness_report,
)
from database.db import fresh_connection
from database.schema import init_schema


def _insert_observation(con, event_id, category, observed_timestamp, prob=0.5):
    con.execute(
        "INSERT INTO event_probability_observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            f"{event_id}-{observed_timestamp.isoformat()}", event_id, f"question about {event_id}", category,
            observed_timestamp, None, prob, None, None, "polymarket_readonly", observed_timestamp,
        ],
    )


def test_event_probability_coverage_report_empty_database():
    with fresh_connection(":memory:") as con:
        init_schema(con)
        report = event_probability_coverage_report(con)
    assert report["n_observations"] == 0
    assert report["is_single_snapshot_only"] is None


def test_event_probability_coverage_report_detects_single_snapshot():
    """Reproduces the actual production finding: every observation
    fetched in one ingestion run shares the same observed_timestamp."""
    now = datetime(2026, 9, 1, 7, 16, 38)
    with fresh_connection(":memory:") as con:
        init_schema(con)
        for i in range(5):
            _insert_observation(con, f"event{i}", "monetary_policy", now)
        report = event_probability_coverage_report(con)

    assert report["n_observations"] == 5
    assert report["n_distinct_observation_days"] == 1
    assert report["is_single_snapshot_only"] is True
    assert "single current snapshot" in report["finding"]
    assert report["coverage_by_category"] == {"monetary_policy": 5}


def test_event_probability_coverage_report_detects_accumulating_history():
    """Once multiple ingestion runs have happened on different days
    (exactly what running real-demo repeatedly over time produces), the
    report must reflect a genuine multi-day history, not claim
    single-snapshot-only."""
    day1 = datetime(2026, 9, 1, 7, 0, 0)
    day2 = day1 + timedelta(days=1)
    day3 = day1 + timedelta(days=2)
    with fresh_connection(":memory:") as con:
        init_schema(con)
        for day in (day1, day2, day3):
            _insert_observation(con, "event0", "geopolitical", day)
        report = event_probability_coverage_report(con)

    assert report["n_distinct_observation_days"] == 3
    assert report["is_single_snapshot_only"] is False
    assert "prospective" in report["finding"]


def test_eventprob_feature_missingness_report():
    df = pd.DataFrame(
        {
            "eventprob_monetary_policy_probability": [0.5, float("nan"), float("nan")],
            "eventprob_geopolitical_probability": [float("nan"), float("nan"), float("nan")],
            "eventprob_missing": [0.0, 1.0, 1.0],
        }
    )
    report = eventprob_feature_missingness_report(df)
    assert report["n_rows"] == 3
    assert report["columns"]["eventprob_monetary_policy_probability"] == pytest.approx(1 / 3)
    assert report["columns"]["eventprob_geopolitical_probability"] == 0.0
    assert report["overall_any_present_fraction"] == pytest.approx(1 / 3)


def test_eventprob_feature_missingness_report_empty_frame():
    report = eventprob_feature_missingness_report(pd.DataFrame())
    assert report["n_rows"] == 0
    assert report["columns"] == {}
