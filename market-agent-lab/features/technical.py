"""Deterministic technical feature engine.

Every function here is pure arithmetic over a single symbol's OHLCV
history -- there is no LLM involvement anywhere in this module. Research
agents (Phase 3) only ever *consume* these already-computed numbers; they
never recompute or approximate them.

All functions expect a DataFrame sorted ascending by ``timestamp`` for one
symbol, with columns: ``open, high, low, close, adjusted_close, volume``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _require_sorted(df: pd.DataFrame) -> None:
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("OHLCV frame must be sorted ascending by timestamp")


def returns(close: pd.Series, window: int) -> pd.Series:
    """Simple percentage return over ``window`` trading days."""
    return close.pct_change(window)


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def distance_from_sma(close: pd.Series, sma_series: pd.Series) -> pd.Series:
    """(price / SMA) - 1, i.e. percentage distance above/below the moving average."""
    return close / sma_series - 1.0


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Classic Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    # RSI is defined as 100 when there are no losses at all in the window.
    result = result.where(avg_loss != 0.0, 100.0)
    result[avg_gain.isna() | avg_loss.isna()] = np.nan
    return result


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    """Returns (macd_line, signal_line) using standard EMA parameters."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


def bollinger_band_position(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """Where price sits within the Bollinger bands: 0 = lower band, 1 = upper band.

    Values outside [0, 1] indicate a breakout beyond the bands.
    """
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    band_width = (upper - lower).replace(0.0, np.nan)
    return (close - lower) / band_width


def realised_volatility(close: pd.Series, window: int) -> pd.Series:
    """Annualised realised volatility of daily simple returns over ``window`` days."""
    daily_returns = close.pct_change()
    return daily_returns.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    mean = volume.rolling(window, min_periods=window).mean()
    std = volume.rolling(window, min_periods=window).std()
    return (volume - mean) / std.replace(0.0, np.nan)


def relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    mean = volume.rolling(window, min_periods=window).mean()
    return volume / mean.replace(0.0, np.nan)


def distance_from_52w_high(close: pd.Series) -> pd.Series:
    rolling_high = close.rolling(TRADING_DAYS_PER_YEAR, min_periods=20).max()
    return close / rolling_high - 1.0


def distance_from_52w_low(close: pd.Series) -> pd.Series:
    rolling_low = close.rolling(TRADING_DAYS_PER_YEAR, min_periods=20).min()
    return close / rolling_low - 1.0


# --------------------------------------------------------------------------
# Version 0.2 additions (Phase 10 of the brief): more return horizons, EMAs,
# MACD histogram, Bollinger bandwidth, additional volatility windows,
# downside volatility, gap/overnight returns, and drawdown-from-high.
# All still pure single-symbol arithmetic -- no benchmark/cross-sectional
# data needed here (that's features/cross_sectional.py and
# features/market_breadth.py, plus compute_rolling_beta_correlation below
# which explicitly takes a second, benchmark return series).
# --------------------------------------------------------------------------


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def macd_histogram(macd_line: pd.Series, signal_line: pd.Series) -> pd.Series:
    return macd_line - signal_line


def bollinger_bandwidth(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """(upper - lower) / mid -- band width relative to price, distinct from
    ``bollinger_band_position`` (where price sits within the bands)."""
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return (upper - lower) / mid


def downside_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualised standard deviation of NEGATIVE daily returns only
    (semi-deviation) -- distinguishes downside risk from symmetric
    realised volatility."""
    daily_returns = close.pct_change()
    downside = daily_returns.where(daily_returns < 0, 0.0)
    return downside.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def gap_return(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Today's open vs. yesterday's close -- the overnight/weekend gap."""
    prev_close = close.shift(1)
    return open_ / prev_close - 1.0


def overnight_return(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Alias of gap_return kept as a distinct, explicitly-named column per
    the brief's feature list; identical definition."""
    return gap_return(open_, close)


def drawdown_from_rolling_high(close: pd.Series, window: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    rolling_high = close.rolling(window, min_periods=1).max()
    return close / rolling_high - 1.0


def compute_rolling_beta_correlation(
    symbol_returns: pd.Series, benchmark_returns: pd.Series, window: int = 60
) -> tuple[pd.Series, pd.Series]:
    """Rolling beta and correlation of ``symbol_returns`` against
    ``benchmark_returns`` (both daily simple returns, same index). Returns
    ``(beta, correlation)``. Requires benchmark data, so this is called
    separately from ``compute_technical_features`` (which only ever sees
    one symbol's own OHLCV) -- see ``data/real_technical_features.py``."""
    aligned = pd.concat([symbol_returns, benchmark_returns], axis=1, join="inner")
    aligned.columns = ["symbol", "benchmark"]
    rolling_cov = aligned["symbol"].rolling(window, min_periods=window).cov(aligned["benchmark"])
    rolling_var = aligned["benchmark"].rolling(window, min_periods=window).var()
    beta = rolling_cov / rolling_var
    correlation = aligned["symbol"].rolling(window, min_periods=window).corr(aligned["benchmark"])
    return beta, correlation


def relative_momentum(symbol_returns_cum: pd.Series, reference_returns_cum: pd.Series) -> pd.Series:
    """Symbol's cumulative return minus a reference's (benchmark or
    sector-average) cumulative return over the same window -- used for both
    'sector relative momentum' and 'market relative momentum'."""
    aligned = pd.concat([symbol_returns_cum, reference_returns_cum], axis=1, join="inner")
    aligned.columns = ["symbol", "reference"]
    return aligned["symbol"] - aligned["reference"]


TECHNICAL_FEATURE_COLUMNS: list[str] = [
    "return_1d", "return_2d", "return_5d", "return_10d", "return_20d", "return_60d", "return_120d", "return_252d",
    "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
    "ema_12", "ema_26",
    "dist_sma_20", "dist_sma_50", "dist_sma_200",
    "rsi_14", "macd", "macd_signal", "macd_histogram", "atr_14",
    "bollinger_position", "bollinger_bandwidth",
    "realised_vol_10d", "realised_vol_20d", "realised_vol_60d", "realised_vol_120d", "downside_vol_20d",
    "volume_zscore_20d", "relative_volume_20d",
    "dist_52w_high", "dist_52w_low",
    "gap_return", "overnight_return", "drawdown_from_high",
]


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the full technical feature set for one symbol's OHLCV history.

    Returns a DataFrame indexed the same as ``df`` with ``timestamp``,
    ``symbol`` and every column in :data:`TECHNICAL_FEATURE_COLUMNS`. Rows
    without enough history for a given window are left as ``NaN`` for that
    column (never back-filled -- that would itself be a form of leakage).
    """
    _require_sorted(df)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    open_ = df["open"]
    sma_10 = sma(close, 10)
    sma_20 = sma(close, 20)
    sma_50 = sma(close, 50)
    sma_100 = sma(close, 100)
    sma_200 = sma(close, 200)
    macd_line, macd_signal_line = macd(close)

    out = pd.DataFrame(
        {
            "timestamp": df["timestamp"].to_numpy(),
            "symbol": df["symbol"].to_numpy(),
            "return_1d": returns(close, 1),
            "return_2d": returns(close, 2),
            "return_5d": returns(close, 5),
            "return_10d": returns(close, 10),
            "return_20d": returns(close, 20),
            "return_60d": returns(close, 60),
            "return_120d": returns(close, 120),
            "return_252d": returns(close, 252),
            "sma_10": sma_10,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_100": sma_100,
            "sma_200": sma_200,
            "ema_12": ema(close, 12),
            "ema_26": ema(close, 26),
            "dist_sma_20": distance_from_sma(close, sma_20),
            "dist_sma_50": distance_from_sma(close, sma_50),
            "dist_sma_200": distance_from_sma(close, sma_200),
            "rsi_14": rsi(close, 14),
            "macd": macd_line,
            "macd_signal": macd_signal_line,
            "macd_histogram": macd_histogram(macd_line, macd_signal_line),
            "atr_14": atr(high, low, close, 14),
            "bollinger_position": bollinger_band_position(close, 20),
            "bollinger_bandwidth": bollinger_bandwidth(close, 20),
            "realised_vol_10d": realised_volatility(close, 10),
            "realised_vol_20d": realised_volatility(close, 20),
            "realised_vol_60d": realised_volatility(close, 60),
            "realised_vol_120d": realised_volatility(close, 120),
            "downside_vol_20d": downside_volatility(close, 20),
            "volume_zscore_20d": volume_zscore(volume, 20),
            "relative_volume_20d": relative_volume(volume, 20),
            "dist_52w_high": distance_from_52w_high(close),
            "dist_52w_low": distance_from_52w_low(close),
            "gap_return": gap_return(open_, close),
            "overnight_return": overnight_return(open_, close),
            "drawdown_from_high": drawdown_from_rolling_high(close),
        }
    )
    return out


def compute_technical_features_multi(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`compute_technical_features` per symbol on a multi-symbol frame."""
    parts = []
    for _symbol, group in df.sort_values(["symbol", "timestamp"]).groupby("symbol", sort=False):
        parts.append(compute_technical_features(group.reset_index(drop=True)))
    return pd.concat(parts, ignore_index=True)
