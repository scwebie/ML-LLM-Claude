"""Tests for cross-sectional percentile ranks and market breadth."""

from __future__ import annotations

import pandas as pd
import pytest

from features.cross_sectional import compute_percentile_ranks, compute_percentile_ranks_multi_date
from features.market_breadth import compute_market_breadth, compute_market_breadth_series


def test_momentum_percentile_ranks_highest_return_at_top():
    cross_section = pd.DataFrame({"symbol": ["A", "B", "C"], "return_60d": [0.01, 0.10, 0.05]})
    ranks = compute_percentile_ranks(cross_section, {"momentum_percentile": ("return_60d", True)})
    row_b = ranks[ranks.symbol == "B"].iloc[0]
    row_a = ranks[ranks.symbol == "A"].iloc[0]
    assert row_b["momentum_percentile"] == 1.0  # highest return -> top percentile
    assert row_a["momentum_percentile"] == pytest.approx(1 / 3)  # lowest return -> bottom percentile


def test_volatility_percentile_inverts_so_lower_vol_ranks_higher():
    cross_section = pd.DataFrame({"symbol": ["A", "B", "C"], "realised_vol_20d": [0.10, 0.50, 0.30]})
    ranks = compute_percentile_ranks(cross_section, {"volatility_percentile": ("realised_vol_20d", False)})
    row_a = ranks[ranks.symbol == "A"].iloc[0]  # lowest vol
    row_b = ranks[ranks.symbol == "B"].iloc[0]  # highest vol
    assert row_a["volatility_percentile"] == 1.0
    assert row_b["volatility_percentile"] == pytest.approx(1 / 3)


def test_missing_metric_column_is_skipped_not_fabricated():
    cross_section = pd.DataFrame({"symbol": ["A", "B"], "return_60d": [0.1, 0.2]})
    ranks = compute_percentile_ranks(cross_section, {"value_percentile": ("valuation_zscore", True)})
    assert "value_percentile" not in ranks.columns


def test_percentile_ranks_computed_independently_per_date():
    """A date with only 2 symbols must not be influenced by a different
    date's 5-symbol universe -- confirms no cross-date leakage. Also
    covers the SECOND date's group, whose rows carry original panel
    indices [2, 3, 4] -- a regression guard for a real index-alignment
    bug where only a group whose original index happened to start at 0
    (like day 1 here) produced non-NaN ranks; every later group produced
    NaN, silently, because .rank()'s output Series carried the group's
    original (non-reset) index while the output frame did not."""
    panel = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2023-01-01")] * 2 + [pd.Timestamp("2023-01-02")] * 3,
            "symbol": ["A", "B", "C", "D", "E"],
            "return_60d": [0.1, 0.2, 0.05, 0.15, 0.25],
        }
    )
    ranks = compute_percentile_ranks_multi_date(panel, {"momentum_percentile": ("return_60d", True)})
    day1 = ranks[ranks.timestamp == pd.Timestamp("2023-01-01")]
    assert len(day1) == 2
    assert set(day1["momentum_percentile"]) == {0.5, 1.0}  # only ranked among that day's 2 symbols

    day2 = ranks[ranks.timestamp == pd.Timestamp("2023-01-02")]
    assert len(day2) == 3
    assert day2["momentum_percentile"].notna().all()
    row_e = day2[day2.symbol == "E"].iloc[0]  # highest return_60d (0.25) among C/D/E
    assert row_e["momentum_percentile"] == 1.0


def test_compute_percentile_ranks_is_positional_not_index_aligned():
    """Direct regression test for the index-alignment bug: a cross_section
    slice carrying a non-zero-starting, non-contiguous original index
    (exactly what a groupby() split of a larger frame produces) must
    still produce real percentile ranks, not NaN."""
    cross_section = pd.DataFrame(
        {"symbol": ["A", "B", "C"], "dollar_volume": [100.0, 300.0, 200.0]}, index=[57, 58, 59]
    )
    ranks = compute_percentile_ranks(cross_section, {"liquidity_percentile": ("dollar_volume", True)})
    assert ranks["liquidity_percentile"].notna().all()
    row_b = ranks[ranks.symbol == "B"].iloc[0]
    assert row_b["liquidity_percentile"] == 1.0  # highest dollar_volume


def test_market_breadth_pct_above_sma():
    cross_section = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "dist_sma_20": [0.05, -0.02, 0.01, -0.01]})
    breadth = compute_market_breadth(cross_section)
    assert breadth["pct_above_sma20"] == pytest.approx(0.5)


def test_market_breadth_advance_decline_and_dispersion():
    cross_section = pd.DataFrame({"symbol": ["A", "B", "C"], "return_1d": [0.02, -0.01, 0.0]})
    breadth = compute_market_breadth(cross_section)
    assert breadth["advance_decline_proxy"] == pytest.approx(1 / 3 - 1 / 3)
    assert breadth["median_stock_return"] == pytest.approx(0.0)
    assert breadth["equal_weight_return"] == pytest.approx((0.02 - 0.01 + 0.0) / 3)


def test_market_breadth_volume_weighted_spread():
    cross_section = pd.DataFrame(
        {"symbol": ["A", "B"], "return_1d": [0.10, -0.10], "dollar_volume": [900.0, 100.0]}
    )
    breadth = compute_market_breadth(cross_section)
    expected_weighted = 0.10 * 0.9 + (-0.10) * 0.1
    assert breadth["dollar_volume_weighted_return"] == pytest.approx(expected_weighted)
    assert breadth["equal_weight_return"] == pytest.approx(0.0)


def test_market_breadth_empty_frame_returns_empty_dict():
    assert compute_market_breadth(pd.DataFrame(columns=["symbol", "return_1d"])) == {}


def test_market_breadth_series_grouped_by_date():
    panel = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2023-01-01")] * 2 + [pd.Timestamp("2023-01-02")] * 2,
            "symbol": ["A", "B", "A", "B"],
            "return_1d": [0.01, -0.01, 0.02, 0.02],
        }
    )
    series = compute_market_breadth_series(panel)
    assert len(series) == 2
    day2 = series[series.timestamp == pd.Timestamp("2023-01-02")].iloc[0]
    assert day2["advance_decline_proxy"] == pytest.approx(1.0)  # both up on day 2
