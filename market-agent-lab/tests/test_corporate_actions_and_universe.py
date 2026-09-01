"""Tests for corporate-action parsing and point-in-time universe membership."""

from __future__ import annotations

from datetime import datetime

import pytest

from core.schemas_v2 import CorporateActionType
from data.corporate_actions import parse_yahoo_events
from data.universe import get_point_in_time_universe, seed_universe_membership
from database.db import fresh_connection


def test_parse_yahoo_split_event_produces_correct_ratio():
    events = {"splits": {"1598880600": {"date": 1598880600, "numerator": 4, "denominator": 1, "splitRatio": "4:1"}}}
    actions = parse_yahoo_events("AAPL", events)
    assert len(actions) == 1
    assert actions[0].action_type == CorporateActionType.SPLIT
    assert actions[0].ratio == pytest.approx(4.0)


def test_parse_yahoo_reverse_split_event():
    events = {"splits": {"1": {"date": 1000000, "numerator": 1, "denominator": 10, "splitRatio": "1:10"}}}
    actions = parse_yahoo_events("XYZ", events)
    assert actions[0].action_type == CorporateActionType.REVERSE_SPLIT
    assert actions[0].ratio == pytest.approx(0.1)


def test_parse_yahoo_dividend_event():
    events = {"dividends": {"1": {"date": 1000000, "amount": 0.24}}}
    actions = parse_yahoo_events("AAPL", events)
    assert len(actions) == 1
    assert actions[0].action_type == CorporateActionType.DIVIDEND
    assert actions[0].cash_amount == pytest.approx(0.24)


def test_parse_yahoo_events_handles_empty_payload():
    assert parse_yahoo_events("AAPL", {}) == []


def test_parse_yahoo_events_handles_both_dividends_and_splits():
    events = {
        "dividends": {"1": {"date": 1000000, "amount": 0.5}},
        "splits": {"2": {"date": 2000000, "numerator": 2, "denominator": 1}},
    }
    actions = parse_yahoo_events("AAPL", events)
    assert len(actions) == 2
    types = {a.action_type for a in actions}
    assert types == {CorporateActionType.DIVIDEND, CorporateActionType.SPLIT}


# --- point-in-time universe --------------------------------------------------------


@pytest.fixture
def con():
    with fresh_connection(":memory:") as c:
        yield c


def test_universe_membership_respects_start_date(con):
    seed_universe_membership(con, "test_universe", ["AAPL", "MSFT"], start_date=datetime(2020, 1, 1))
    before = get_point_in_time_universe(con, "test_universe", datetime(2019, 6, 1))
    after = get_point_in_time_universe(con, "test_universe", datetime(2021, 1, 1))
    assert before == []
    assert set(after) == {"AAPL", "MSFT"}


def test_universe_membership_excludes_after_end_date(con):
    from core.schemas_v2 import UniverseMembership
    from database import repository_v2 as repo_v2

    repo_v2.insert_universe_membership(
        con,
        [
            UniverseMembership(
                universe_name="test_universe", symbol="DELISTED_CO",
                start_date=datetime(2015, 1, 1), end_date=datetime(2018, 1, 1), source="test",
            )
        ],
    )
    before_delisting = get_point_in_time_universe(con, "test_universe", datetime(2017, 1, 1))
    after_delisting = get_point_in_time_universe(con, "test_universe", datetime(2019, 1, 1))
    assert "DELISTED_CO" in before_delisting
    assert "DELISTED_CO" not in after_delisting


def test_universe_membership_never_uses_future_constituents_for_past_dates(con):
    """A symbol added to the universe starting in 2022 must never appear
    in a point-in-time query for a 2015 date -- this is the core
    survivorship-bias guard the mechanism provides."""
    seed_universe_membership(con, "test_universe", ["OLDCO"], start_date=datetime(2010, 1, 1))
    seed_universe_membership(con, "test_universe", ["NEWCO_IPO_2022"], start_date=datetime(2022, 6, 1))

    universe_2015 = get_point_in_time_universe(con, "test_universe", datetime(2015, 1, 1))
    universe_2023 = get_point_in_time_universe(con, "test_universe", datetime(2023, 1, 1))

    assert "NEWCO_IPO_2022" not in universe_2015
    assert "NEWCO_IPO_2022" in universe_2023
    assert "OLDCO" in universe_2015
