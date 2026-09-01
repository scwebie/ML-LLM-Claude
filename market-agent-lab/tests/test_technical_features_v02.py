"""Tests for the Version 0.2 technical indicator additions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import technical as tech


def _ohlcv(closes: list[float], opens: list[float] | None = None, symbol: str = "SYN_TEST") -> pd.DataFrame:
    n = len(closes)
    dates = pd.bdate_range("2020-01-01", periods=n)
    closes_s = pd.Series(closes, dtype=float)
    opens_s = pd.Series(opens, dtype=float) if opens is not None else closes_s
    return pd.DataFrame(
        {
            "symbol": symbol, "timestamp": dates, "open": opens_s, "high": closes_s * 1.001,
            "low": closes_s * 0.999, "close": closes_s, "adjusted_close": closes_s,
            "volume": np.full(n, 1_000_000),
        }
    )


def test_ema_converges_to_constant_price():
    df = _ohlcv([50.0] * 60)
    result = tech.ema(df["close"], span=12)
    assert result.iloc[-1] == pytest.approx(50.0)
    assert result.iloc[:11].isna().all()


def test_macd_histogram_is_macd_minus_signal():
    rng = np.random.default_rng(1)
    closes = 100 + np.cumsum(rng.normal(0, 1, 80))
    df = _ohlcv(list(closes))
    macd_line, signal_line = tech.macd(df["close"])
    hist = tech.macd_histogram(macd_line, signal_line)
    pd.testing.assert_series_equal(hist, macd_line - signal_line, check_names=False)


def test_bollinger_bandwidth_zero_for_constant_price():
    df = _ohlcv([100.0] * 30)
    bw = tech.bollinger_bandwidth(df["close"], window=20)
    assert bw.iloc[-1] == pytest.approx(0.0)


def test_bollinger_bandwidth_positive_for_volatile_price():
    rng = np.random.default_rng(2)
    closes = 100 + np.cumsum(rng.normal(0, 2, 40))
    df = _ohlcv(list(closes))
    bw = tech.bollinger_bandwidth(df["close"], window=20)
    assert bw.iloc[-1] > 0


def test_downside_volatility_zero_when_all_returns_positive():
    closes = list(np.linspace(100, 150, 30))  # strictly increasing -> no negative daily returns
    df = _ohlcv(closes)
    dvol = tech.downside_volatility(df["close"], window=10)
    assert dvol.iloc[-1] == pytest.approx(0.0)


def test_downside_volatility_positive_when_price_falls():
    closes = list(np.linspace(150, 100, 30))  # strictly decreasing -> all negative returns
    df = _ohlcv(closes)
    dvol = tech.downside_volatility(df["close"], window=10)
    assert dvol.iloc[-1] > 0


def test_gap_return_matches_manual_calc():
    df = _ohlcv(closes=[100.0, 102.0, 101.0], opens=[100.0, 103.0, 99.0])
    gaps = tech.gap_return(df["open"], df["close"])
    assert gaps.iloc[1] == pytest.approx(103.0 / 100.0 - 1.0)
    assert gaps.iloc[2] == pytest.approx(99.0 / 102.0 - 1.0)


def test_overnight_return_equals_gap_return():
    df = _ohlcv(closes=[100.0, 102.0, 101.0], opens=[100.0, 103.0, 99.0])
    gaps = tech.gap_return(df["open"], df["close"])
    overnight = tech.overnight_return(df["open"], df["close"])
    pd.testing.assert_series_equal(gaps, overnight)


def test_drawdown_from_high_is_zero_at_new_high():
    closes = [100.0, 105.0, 110.0]  # strictly increasing -> always at a new high
    df = _ohlcv(closes)
    dd = tech.drawdown_from_rolling_high(df["close"], window=252)
    assert (dd == 0.0).all()


def test_drawdown_from_high_negative_after_decline():
    closes = [100.0, 120.0, 90.0]
    df = _ohlcv(closes)
    dd = tech.drawdown_from_rolling_high(df["close"], window=252)
    assert dd.iloc[-1] == pytest.approx(90.0 / 120.0 - 1.0)


def test_additional_return_horizons_match_manual_calc():
    closes = [100 * (1.01**i) for i in range(260)]
    df = _ohlcv(closes)
    r2 = tech.returns(df["close"], 2)
    r120 = tech.returns(df["close"], 120)
    r252 = tech.returns(df["close"], 252)
    assert r2.iloc[-1] == pytest.approx(1.01**2 - 1, rel=1e-6)
    assert r120.iloc[-1] == pytest.approx(1.01**120 - 1, rel=1e-6)
    assert r252.iloc[-1] == pytest.approx(1.01**252 - 1, rel=1e-6)


def test_rolling_beta_of_symbol_to_itself_is_one():
    rng = np.random.default_rng(3)
    returns_series = pd.Series(rng.normal(0, 0.01, 100))
    beta, corr = tech.compute_rolling_beta_correlation(returns_series, returns_series, window=30)
    assert beta.iloc[-1] == pytest.approx(1.0, abs=1e-6)
    assert corr.iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_rolling_beta_of_double_leveraged_series_is_two():
    rng = np.random.default_rng(4)
    benchmark_returns = pd.Series(rng.normal(0, 0.01, 100))
    leveraged_returns = benchmark_returns * 2.0
    beta, _ = tech.compute_rolling_beta_correlation(leveraged_returns, benchmark_returns, window=30)
    assert beta.iloc[-1] == pytest.approx(2.0, abs=1e-6)


def test_relative_momentum_matches_manual_subtraction():
    symbol_cum = pd.Series([0.10, 0.15, 0.20])
    reference_cum = pd.Series([0.05, 0.05, 0.05])
    rel = tech.relative_momentum(symbol_cum, reference_cum)
    assert list(rel) == pytest.approx([0.05, 0.10, 0.15])


def test_compute_technical_features_includes_all_v02_columns_and_no_lookahead():
    rng = np.random.default_rng(5)
    closes = 100 + np.cumsum(rng.normal(0.05, 1, 300))
    df = _ohlcv(list(closes))
    features = tech.compute_technical_features(df)
    for col in ("return_2d", "return_120d", "return_252d", "ema_12", "ema_26", "macd_histogram",
                "bollinger_bandwidth", "realised_vol_120d", "downside_vol_20d",
                "gap_return", "overnight_return", "drawdown_from_high"):
        assert col in features.columns

    cutoff = 250
    truncated = tech.compute_technical_features(df.iloc[: cutoff + 1].reset_index(drop=True))
    pd.testing.assert_frame_equal(
        features.iloc[: cutoff + 1].reset_index(drop=True), truncated.reset_index(drop=True)
    )
