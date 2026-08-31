"""Unit tests for features/technical.py against hand-computed known examples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import technical as tech


def _ohlcv(closes: list[float], symbol: str = "SYN_TEST") -> pd.DataFrame:
    n = len(closes)
    dates = pd.bdate_range("2020-01-01", periods=n)
    closes = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": dates,
            "open": closes,
            "high": closes * 1.001,
            "low": closes * 0.999,
            "close": closes,
            "adjusted_close": closes,
            "volume": np.full(n, 1_000_000),
        }
    )


def test_sma_matches_manual_average():
    df = _ohlcv([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    result = tech.sma(df["close"], window=5)
    # last value: mean of the last 5 closes (16..20)
    assert result.iloc[-1] == pytest.approx(np.mean([16, 17, 18, 19, 20]))
    # first 4 values (insufficient window) must be NaN, not back-filled
    assert result.iloc[:4].isna().all()


def test_returns_known_values():
    df = _ohlcv([100, 110, 121, 133.1])
    r1 = tech.returns(df["close"], 1)
    assert r1.iloc[1] == pytest.approx(0.10)
    assert r1.iloc[2] == pytest.approx(0.10)
    r3 = tech.returns(df["close"], 3)
    assert r3.iloc[3] == pytest.approx(0.331)


def test_rsi_all_gains_is_100():
    # Strictly increasing prices -> no losses -> RSI should saturate at 100.
    closes = list(range(1, 30))
    df = _ohlcv([float(c) for c in closes])
    result = tech.rsi(df["close"], window=14)
    assert result.iloc[-1] == pytest.approx(100.0)


def test_rsi_all_losses_is_0():
    closes = list(range(30, 1, -1))
    df = _ohlcv([float(c) for c in closes])
    result = tech.rsi(df["close"], window=14)
    assert result.iloc[-1] == pytest.approx(0.0)


def test_rsi_flat_price_is_neutral_100_no_losses():
    # A perfectly flat series has zero gains AND zero losses; by convention
    # (no losses in the window) RSI reports 100.
    df = _ohlcv([50.0] * 30)
    result = tech.rsi(df["close"], window=14)
    assert result.iloc[-1] == pytest.approx(100.0)


def test_macd_zero_for_constant_price():
    df = _ohlcv([50.0] * 60)
    macd_line, signal_line = tech.macd(df["close"])
    assert macd_line.iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert signal_line.iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_atr_constant_high_low_spread():
    # high/low always exactly +/-1 around a constant close -> true range is
    # constant at 2 once past the first bar, so ATR converges to 2.
    n = 40
    closes = [100.0] * n
    df = pd.DataFrame(
        {
            "symbol": "SYN_TEST",
            "timestamp": pd.bdate_range("2020-01-01", periods=n),
            "open": closes,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": closes,
            "adjusted_close": closes,
            "volume": np.full(n, 1_000_000),
        }
    )
    result = tech.atr(df["high"], df["low"], df["close"], window=14)
    assert result.iloc[-1] == pytest.approx(2.0, rel=1e-6)


def test_bollinger_position_bounds():
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 1, 100))
    df = _ohlcv(list(closes))
    pos = tech.bollinger_band_position(df["close"], window=20)
    valid = pos.dropna()
    assert len(valid) > 0
    # By construction position should mostly sit within [0, 1], with rare
    # excursions outside on strong moves -- but never NaN/inf once valid.
    assert np.isfinite(valid).all()


def test_realised_volatility_zero_for_constant_price():
    df = _ohlcv([100.0] * 30)
    vol = tech.realised_volatility(df["close"], window=10)
    assert vol.iloc[-1] == pytest.approx(0.0)


def test_volume_zscore_and_relative_volume():
    n = 30
    volumes = [1_000_000] * (n - 1) + [3_000_000]
    df = _ohlcv([100.0] * n)
    df["volume"] = volumes
    z = tech.volume_zscore(df["volume"], window=20)
    rel = tech.relative_volume(df["volume"], window=20)
    assert z.iloc[-1] > 0
    assert rel.iloc[-1] > 1.0


def test_52_week_high_low_distance():
    closes = [100.0] * 250 + [150.0, 50.0]
    df = _ohlcv(closes)
    dist_high = tech.distance_from_52w_high(df["close"])
    dist_low = tech.distance_from_52w_low(df["close"])
    # At the 150 print, price == new 52w high -> distance 0
    assert dist_high.iloc[-2] == pytest.approx(0.0)
    # At the 50 print (after the 150 print), 52w low is still 50 itself -> 0
    assert dist_low.iloc[-1] == pytest.approx(0.0)


def test_compute_technical_features_shape_and_no_lookahead():
    rng = np.random.default_rng(1)
    closes = 100 + np.cumsum(rng.normal(0.05, 1, 300))
    df = _ohlcv(list(closes))
    features = tech.compute_technical_features(df)
    assert list(features.columns) == ["timestamp", "symbol"] + tech.TECHNICAL_FEATURE_COLUMNS
    assert len(features) == len(df)
    # Truncating the input to day t must not change any feature value at
    # or before day t -- i.e. no feature peeks into the future.
    cutoff = 250
    truncated = tech.compute_technical_features(df.iloc[: cutoff + 1].reset_index(drop=True))
    pd.testing.assert_frame_equal(
        features.iloc[: cutoff + 1].reset_index(drop=True),
        truncated.reset_index(drop=True),
    )


def test_compute_technical_features_multi_symbol_independent():
    rng = np.random.default_rng(2)
    closes_a = 100 + np.cumsum(rng.normal(0, 1, 60))
    closes_b = 200 + np.cumsum(rng.normal(0, 1, 60))
    df_a = _ohlcv(list(closes_a), symbol="SYN_A")
    df_b = _ohlcv(list(closes_b), symbol="SYN_B")
    combined = pd.concat([df_a, df_b], ignore_index=True)
    result = tech.compute_technical_features_multi(combined)
    assert set(result["symbol"].unique()) == {"SYN_A", "SYN_B"}
    solo_a = tech.compute_technical_features(df_a)
    joined = result[result["symbol"] == "SYN_A"].reset_index(drop=True)
    pd.testing.assert_frame_equal(joined, solo_a.reset_index(drop=True))
