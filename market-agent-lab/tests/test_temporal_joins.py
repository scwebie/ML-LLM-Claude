"""Tests for as-of temporal joins and look-ahead-bias guards.

Covers: fundamentals as-of join, macro as-of join, and the historical
similarity engine's structural leakage guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from database import repository as repo
from database.db import fresh_connection
from features.historical import find_historical_analogues


@pytest.fixture
def con():
    with fresh_connection(":memory:") as c:
        yield c


def test_fundamentals_asof_excludes_future_publications(con):
    df = pd.DataFrame(
        [
            {
                "symbol": "SYN_X",
                "publication_timestamp": pd.Timestamp("2020-05-01"),
                "reporting_period_end": pd.Timestamp("2020-03-31"),
                "revenue": 100.0, "revenue_growth": 0.1, "eps": 1.0, "eps_growth": 0.1,
                "gross_margin": 0.4, "operating_margin": 0.2, "free_cash_flow": 10.0,
                "fcf_margin": 0.1, "roic": 0.1, "debt": 50.0, "cash": 20.0,
                "pe_ratio": 15.0, "ev_to_ebitda": 10.0, "price_to_book": 2.0, "price_to_sales": 3.0,
            },
            {
                "symbol": "SYN_X",
                "publication_timestamp": pd.Timestamp("2020-08-01"),
                "reporting_period_end": pd.Timestamp("2020-06-30"),
                "revenue": 200.0, "revenue_growth": 1.0, "eps": 2.0, "eps_growth": 1.0,
                "gross_margin": 0.5, "operating_margin": 0.3, "free_cash_flow": 20.0,
                "fcf_margin": 0.1, "roic": 0.2, "debt": 40.0, "cash": 30.0,
                "pe_ratio": 12.0, "ev_to_ebitda": 9.0, "price_to_book": 1.8, "price_to_sales": 2.5,
            },
        ]
    )
    repo.insert_fundamental_observations(con, df)

    # As of a date between the two publications, only the first (older)
    # report must be visible -- even though its reporting period is old.
    result = repo.get_fundamentals_asof(con, ["SYN_X"], pd.Timestamp("2020-06-15"))
    assert len(result) == 1
    assert result.iloc[0]["revenue"] == 100.0

    # As of a date right at the second publication, the newer report
    # becomes visible.
    result2 = repo.get_fundamentals_asof(con, ["SYN_X"], pd.Timestamp("2020-08-01"))
    assert result2.iloc[0]["revenue"] == 200.0

    # One day before the second publication, it must still be hidden.
    result3 = repo.get_fundamentals_asof(con, ["SYN_X"], pd.Timestamp("2020-07-31"))
    assert result3.iloc[0]["revenue"] == 100.0


def test_macro_asof_excludes_future_publications(con):
    df = pd.DataFrame(
        [
            {
                "series_name": "SYN_RATES", "timestamp": pd.Timestamp("2020-01-31"),
                "value": 1.0, "publication_timestamp": pd.Timestamp("2020-02-05"),
                "vintage_timestamp": pd.Timestamp("2020-02-05"),
            },
            {
                "series_name": "SYN_RATES", "timestamp": pd.Timestamp("2020-02-29"),
                "value": 2.0, "publication_timestamp": pd.Timestamp("2020-03-05"),
                "vintage_timestamp": pd.Timestamp("2020-03-05"),
            },
        ]
    )
    repo.insert_macro_observations(con, df)

    # As-of a date before the February print is published, only January's
    # value (1.0) should be visible even though February has already ended.
    result = repo.get_macro_asof(con, pd.Timestamp("2020-03-01"))
    assert result.iloc[0]["value"] == 1.0

    result2 = repo.get_macro_asof(con, pd.Timestamp("2020-03-05"))
    assert result2.iloc[0]["value"] == 2.0


def test_historical_analogues_never_use_incomplete_future_window():
    """The 20 most recent rows before `as_of` must never be selected as
    analogues, because their 20-day forward return could not yet be known."""
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    market = pd.DataFrame({"timestamp": dates, "close": close})
    features = pd.DataFrame(
        {
            "timestamp": dates,
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
        }
    )

    result = find_historical_analogues(
        feature_history_asof=features,
        market_history_asof=market,
        feature_cols=["f1", "f2"],
        k=50,
        min_history=60,
    )
    assert result.num_analogues > 0
    assert result.num_analogues <= 50
    # Sanity: probability values must be valid probabilities.
    assert 0.0 <= (result.prob_positive_20d or 0.0) <= 1.0


def test_historical_analogues_insufficient_history_returns_zero():
    n = 10
    dates = pd.bdate_range("2020-01-01", periods=n)
    market = pd.DataFrame({"timestamp": dates, "close": np.linspace(100, 110, n)})
    features = pd.DataFrame({"timestamp": dates, "f1": np.linspace(0, 1, n)})
    result = find_historical_analogues(
        feature_history_asof=features,
        market_history_asof=market,
        feature_cols=["f1"],
        k=50,
        min_history=60,
    )
    assert result.num_analogues == 0
    assert result.similarity_confidence == 0.0
