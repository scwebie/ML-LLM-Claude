"""Tests for fundamental and macro feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from features.fundamental import compute_fundamental_features
from features.macro import compute_macro_features


def test_fundamental_features_cheaper_valuation_scores_higher():
    universe = pd.DataFrame(
        [
            {
                "symbol": "CHEAP", "revenue_growth": 0.1, "eps_growth": 0.1,
                "gross_margin": 0.4, "operating_margin": 0.2, "fcf_margin": 0.15, "roic": 0.1,
                "debt": 10.0, "cash": 20.0,
                "pe_ratio": 8.0, "ev_to_ebitda": 5.0, "price_to_book": 1.0, "price_to_sales": 1.0,
            },
            {
                "symbol": "EXPENSIVE", "revenue_growth": 0.1, "eps_growth": 0.1,
                "gross_margin": 0.4, "operating_margin": 0.2, "fcf_margin": 0.15, "roic": 0.1,
                "debt": 10.0, "cash": 20.0,
                "pe_ratio": 40.0, "ev_to_ebitda": 30.0, "price_to_book": 10.0, "price_to_sales": 12.0,
            },
        ]
    )
    result = compute_fundamental_features(universe)
    cheap_z = result.loc[result["symbol"] == "CHEAP", "valuation_zscore"].iloc[0]
    expensive_z = result.loc[result["symbol"] == "EXPENSIVE", "valuation_zscore"].iloc[0]
    assert cheap_z > expensive_z


def test_fundamental_features_higher_profitability_scores_higher():
    universe = pd.DataFrame(
        [
            {
                "symbol": "PROFITABLE", "revenue_growth": 0.1, "eps_growth": 0.1,
                "gross_margin": 0.6, "operating_margin": 0.4, "fcf_margin": 0.3, "roic": 0.25,
                "debt": 10.0, "cash": 20.0, "pe_ratio": 15.0, "ev_to_ebitda": 10.0,
                "price_to_book": 2.0, "price_to_sales": 3.0,
            },
            {
                "symbol": "UNPROFITABLE", "revenue_growth": 0.1, "eps_growth": 0.1,
                "gross_margin": 0.1, "operating_margin": -0.1, "fcf_margin": -0.2, "roic": -0.1,
                "debt": 10.0, "cash": 20.0, "pe_ratio": 15.0, "ev_to_ebitda": 10.0,
                "price_to_book": 2.0, "price_to_sales": 3.0,
            },
        ]
    )
    result = compute_fundamental_features(universe)
    profitable_z = result.loc[result["symbol"] == "PROFITABLE", "profitability_zscore"].iloc[0]
    unprofitable_z = result.loc[result["symbol"] == "UNPROFITABLE", "profitability_zscore"].iloc[0]
    assert profitable_z > unprofitable_z


def test_fundamental_features_debt_to_cash_computed():
    universe = pd.DataFrame(
        [
            {
                "symbol": "X", "revenue_growth": 0.1, "eps_growth": 0.1,
                "gross_margin": 0.4, "operating_margin": 0.2, "fcf_margin": 0.15, "roic": 0.1,
                "debt": 50.0, "cash": 25.0, "pe_ratio": 15.0, "ev_to_ebitda": 10.0,
                "price_to_book": 2.0, "price_to_sales": 3.0,
            }
        ]
    )
    result = compute_fundamental_features(universe)
    assert result["debt_to_cash"].iloc[0] == pytest.approx(2.0)


def test_fundamental_features_empty_universe_returns_empty_frame():
    result = compute_fundamental_features(pd.DataFrame())
    assert result.empty


def test_macro_features_zscore_reflects_deviation_from_trailing_baseline():
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    values = [2.0] * 23 + [10.0]  # last observation is a big outlier
    df = pd.DataFrame(
        {
            "series_name": "SYN_RATES",
            "timestamp": dates,
            "value": values,
            "publication_timestamp": dates + pd.Timedelta(days=5),
        }
    )
    features = compute_macro_features(df, as_of=dates[-1] + pd.Timedelta(days=5))
    assert features["SYN_RATES_zscore"] > 2.0  # far above trailing mean
    assert features["SYN_RATES_level"] == pytest.approx(10.0)


def test_macro_features_empty_history_returns_empty_dict():
    assert compute_macro_features(pd.DataFrame(), as_of=pd.Timestamp("2020-01-01")) == {}


def test_macro_features_flat_series_has_zero_zscore():
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    df = pd.DataFrame(
        {
            "series_name": "SYN_INFLATION",
            "timestamp": dates,
            "value": [2.0] * 12,
            "publication_timestamp": dates + pd.Timedelta(days=5),
        }
    )
    features = compute_macro_features(df, as_of=dates[-1] + pd.Timedelta(days=5))
    assert features["SYN_INFLATION_zscore"] == pytest.approx(0.0)
