"""Deterministic synthetic financial dataset generator.

market-agent-lab v0.1 must work fully offline, without any paid market-data
API. This module generates a self-contained synthetic universe:

* multiple fictional symbols across a few fictional "sectors"
* several years of daily OHLCV data driven by a regime-switching market
  factor (BULL / BEAR / SIDEWAYS / HIGH_VOL segments)
* fictional quarterly fundamentals, published with a realistic reporting lag
* a handful of fictional macro series, published with a realistic lag
* daily news/event sentiment per symbol

Everything is generated from a single fixed seed (``SYNTHETIC_SEED``), so
runs are fully reproducible. The data is intentionally NOT meant to
resemble any real company, index, or macro series -- every symbol and
series name is clearly fictional (``SYN_*`` prefix) to avoid any
implication that this could be mistaken for real market data.

Weak, learnable relationships are deliberately injected into forward
returns (via lagged sentiment and macro "risk appetite") so that the
downstream ML pipeline (Phase 4-6) has *something* real, if faint, to
learn -- this is what "intentionally contains weak learnable
relationships" means in the project brief. Everything else is noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Universe definition
# --------------------------------------------------------------------------

SECTORS = ["TECH", "ENERGY", "FINANCE", "HEALTH", "CONSUMER"]

SYMBOLS: list[tuple[str, str]] = [
    ("SYN_ALPS", "TECH"),
    ("SYN_BRIX", "TECH"),
    ("SYN_CIRC", "ENERGY"),
    ("SYN_DYNA", "ENERGY"),
    ("SYN_ECHO", "FINANCE"),
    ("SYN_FLUX", "FINANCE"),
    ("SYN_GLEN", "HEALTH"),
    ("SYN_HALO", "HEALTH"),
    ("SYN_IONX", "CONSUMER"),
    ("SYN_JOLT", "CONSUMER"),
]

BENCHMARK_SYMBOL = "SYN_BENCH"

MACRO_SERIES = ["SYN_RATES", "SYN_INFLATION", "SYN_GROWTH_INDEX", "SYN_VOL_INDEX"]

REGIME_TYPES = ["BULL", "BEAR", "SIDEWAYS", "HIGH_VOL"]
# (annualised drift, annualised vol) for the shared market factor, per regime
REGIME_PARAMS = {
    "BULL": (0.18, 0.13),
    "BEAR": (-0.22, 0.22),
    "SIDEWAYS": (0.01, 0.10),
    "HIGH_VOL": (0.00, 0.32),
}

TRADING_DAYS_PER_YEAR = 252


@dataclass
class SyntheticDataset:
    market: pd.DataFrame  # OHLCV, all symbols + benchmark
    fundamentals: pd.DataFrame
    macro: pd.DataFrame
    news: pd.DataFrame
    regimes: pd.DataFrame  # date -> regime label, for reference/plots
    symbols: list[str] = field(default_factory=lambda: [s for s, _ in SYMBOLS])
    sector_map: dict[str, str] = field(
        default_factory=lambda: {s: sector for s, sector in SYMBOLS}
    )


def _trading_calendar(start_date: str, end_date: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start_date, end=end_date, freq="B")


def _regime_schedule(rng: np.random.Generator, n_days: int) -> np.ndarray:
    """Build a piecewise regime schedule covering ``n_days`` sessions."""
    schedule = np.empty(n_days, dtype=object)
    idx = 0
    # Cycle through regimes with randomised segment lengths so the dataset
    # deterministically contains every regime type multiple times.
    order = REGIME_TYPES * (n_days // (60 * len(REGIME_TYPES)) + 4)
    for regime in order:
        if idx >= n_days:
            break
        length = int(rng.integers(60, 260))
        end = min(idx + length, n_days)
        schedule[idx:end] = regime
        idx = end
    return schedule


def _simulate_market_factor(rng: np.random.Generator, regimes: np.ndarray) -> np.ndarray:
    """Simulate one shared log-return factor path driven by the regime schedule."""
    n = len(regimes)
    daily_returns = np.empty(n)
    for t in range(n):
        mu_annual, sigma_annual = REGIME_PARAMS[regimes[t]]
        mu_daily = mu_annual / TRADING_DAYS_PER_YEAR
        sigma_daily = sigma_annual / np.sqrt(TRADING_DAYS_PER_YEAR)
        daily_returns[t] = rng.normal(mu_daily, sigma_daily)
    return daily_returns


def _ar1_noise(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """Zero-mean AR(1) process, used for idiosyncratic sentiment / spread noise."""
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, sigma)
    return x


def _build_ohlcv_from_close(
    dates: pd.DatetimeIndex, close: np.ndarray, rng: np.random.Generator, base_volume: float
) -> pd.DataFrame:
    n = len(dates)
    open_ = np.empty(n)
    high = np.empty(n)
    low = np.empty(n)
    prev_close = close[0] / (1 + rng.normal(0, 0.002))
    for t in range(n):
        gap = rng.normal(0, 0.002)
        open_[t] = (prev_close if t == 0 else close[t - 1]) * (1 + gap)
        intraday_range = abs(rng.normal(0, 0.006)) + 1e-4
        high[t] = max(open_[t], close[t]) * (1 + intraday_range)
        low[t] = min(open_[t], close[t]) * (1 - intraday_range)
        low[t] = max(low[t], 0.01)
    volume = base_volume * (1 + 0.4 * rng.standard_normal(n))
    volume = np.clip(volume, base_volume * 0.1, None).astype(np.int64)
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adjusted_close": close,
            "volume": volume,
        }
    )


def generate_synthetic_dataset(
    seed: int = 42,
    start_date: str = "2015-01-01",
    end_date: str = "2023-12-31",
) -> SyntheticDataset:
    """Generate the full synthetic universe deterministically for ``seed``."""

    master_rng = np.random.default_rng(seed)
    dates = _trading_calendar(start_date, end_date)
    n = len(dates)

    regime_schedule = _regime_schedule(master_rng, n)
    market_factor_returns = _simulate_market_factor(master_rng, regime_schedule)
    market_price = 100.0 * np.exp(np.cumsum(market_factor_returns))

    regimes_df = pd.DataFrame({"timestamp": dates, "regime": regime_schedule})

    # Macro "risk appetite" proxy: smoothed market factor z-score, used to
    # inject a genuine (weak) macro -> forward-return relationship.
    roll = pd.Series(market_factor_returns).rolling(20, min_periods=1).mean().to_numpy()
    risk_appetite = (roll - roll.mean()) / (roll.std() + 1e-9)

    market_rows = []
    news_rows = []
    fundamental_rows = []

    for symbol, _sector in SYMBOLS:
        sym_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
        beta = float(sym_rng.uniform(0.5, 1.6))
        idio_vol_annual = float(sym_rng.uniform(0.15, 0.45))
        idio_daily_vol = idio_vol_annual / np.sqrt(TRADING_DAYS_PER_YEAR)

        # Daily sentiment: AR(1) noise plus a slice of forward-looking signal
        # is added further down (sentiment leads price by design).
        sentiment_raw = _ar1_noise(sym_rng, n, phi=0.85, sigma=0.35)
        sentiment = np.tanh(sentiment_raw)  # bound to [-1, 1]

        idio_shock = sym_rng.normal(0, idio_daily_vol, n)

        # Weak learnable signal: today's forward return nudged by
        # yesterday's sentiment and yesterday's macro risk appetite. Both
        # coefficients are deliberately small relative to noise so the ML
        # model has to actually learn something instead of trivially
        # reconstructing the target -- this mirrors the intent of the brief.
        signal = np.zeros(n)
        signal[1:] += 0.0009 * sentiment[:-1]
        signal[1:] += 0.0006 * risk_appetite[:-1]

        symbol_log_returns = beta * market_factor_returns + idio_shock + signal
        close = 50.0 * float(sym_rng.uniform(0.6, 2.0)) * np.exp(np.cumsum(symbol_log_returns))

        base_volume = float(sym_rng.uniform(3e5, 4e6))
        ohlcv = _build_ohlcv_from_close(dates, close, sym_rng, base_volume)
        ohlcv.insert(0, "symbol", symbol)
        market_rows.append(ohlcv)

        news_rows.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "timestamp": dates,
                    "news_sentiment": sentiment,
                    "event_uncertainty": np.clip(np.abs(_ar1_noise(sym_rng, n, 0.6, 0.25)), 0, 1),
                    "is_earnings_event": False,
                }
            )
        )

        # --- Fundamentals: quarterly, published ~45 days after quarter end
        base_revenue = float(sym_rng.uniform(200, 5000))  # millions, fictional
        quarter_ends = pd.date_range(start=dates[0], end=dates[-1] + pd.Timedelta(days=100), freq="QE")
        revenue = base_revenue
        eps = float(sym_rng.uniform(0.5, 6.0))
        prev_revenue = revenue
        prev_eps = eps
        for q_end in quarter_ends:
            growth_drift = float(sym_rng.normal(0.015, 0.03))
            revenue = max(prev_revenue * (1 + growth_drift), 1.0)
            gross_margin = float(np.clip(sym_rng.normal(0.45, 0.08), 0.05, 0.85))
            operating_margin = float(np.clip(gross_margin - sym_rng.uniform(0.05, 0.25), -0.2, 0.6))
            eps = max(prev_eps * (1 + growth_drift + sym_rng.normal(0, 0.05)), 0.01)
            fcf_margin = float(np.clip(operating_margin - sym_rng.uniform(0.0, 0.08), -0.3, 0.5))
            free_cash_flow = revenue * fcf_margin
            debt = float(sym_rng.uniform(0.2, 2.0) * revenue)
            cash = float(sym_rng.uniform(0.1, 1.0) * revenue)
            roic = float(np.clip(sym_rng.normal(0.10, 0.06), -0.2, 0.4))
            publication_ts = q_end + pd.Timedelta(days=int(sym_rng.integers(35, 55)))
            fundamental_rows.append(
                {
                    "symbol": symbol,
                    "publication_timestamp": publication_ts,
                    "reporting_period_end": q_end,
                    "revenue": revenue,
                    "revenue_growth": (revenue - prev_revenue) / prev_revenue,
                    "eps": eps,
                    "eps_growth": (eps - prev_eps) / prev_eps,
                    "gross_margin": gross_margin,
                    "operating_margin": operating_margin,
                    "free_cash_flow": free_cash_flow,
                    "fcf_margin": fcf_margin,
                    "roic": roic,
                    "debt": debt,
                    "cash": cash,
                    "pe_ratio": float(np.clip(sym_rng.normal(20, 8), 3, 80)),
                    "ev_to_ebitda": float(np.clip(sym_rng.normal(12, 5), 2, 50)),
                    "price_to_book": float(np.clip(sym_rng.normal(3, 1.5), 0.3, 15)),
                    "price_to_sales": float(np.clip(sym_rng.normal(4, 2), 0.2, 25)),
                }
            )
            prev_revenue, prev_eps = revenue, eps

    market_df = pd.concat(market_rows, ignore_index=True)
    news_df = pd.concat(news_rows, ignore_index=True)
    fundamentals_df = pd.DataFrame(fundamental_rows)

    # Mark a subset of earnings-adjacent dates as earnings events (the date
    # nearest each fundamental publication for that symbol).
    news_df = news_df.set_index(["symbol", "timestamp"])
    for row in fundamentals_df.itertuples():
        near = market_df[market_df["symbol"] == row.symbol]["timestamp"]
        nearest = near.iloc[(near - row.publication_timestamp).abs().argsort()[:1]]
        if not nearest.empty:
            key = (row.symbol, nearest.iloc[0])
            if key in news_df.index:
                news_df.loc[key, "is_earnings_event"] = True
    news_df = news_df.reset_index()

    benchmark_ohlcv = _build_ohlcv_from_close(dates, market_price, master_rng, base_volume=1e7)
    benchmark_ohlcv.insert(0, "symbol", BENCHMARK_SYMBOL)
    market_df = pd.concat([market_df, benchmark_ohlcv], ignore_index=True)

    # --- Macro series, monthly, published with a short lag
    macro_rows = []
    month_ends = pd.date_range(start=dates[0], end=dates[-1], freq="ME")
    macro_state = {"SYN_RATES": 2.0, "SYN_INFLATION": 2.0, "SYN_GROWTH_INDEX": 100.0, "SYN_VOL_INDEX": 16.0}
    macro_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
    for m_end in month_ends:
        idx_pos = int(dates.searchsorted(m_end))
        idx_pos = min(idx_pos, n - 1)
        regime = regime_schedule[idx_pos]
        for series in MACRO_SERIES:
            drift = {
                "SYN_RATES": {"BULL": 0.02, "BEAR": -0.05, "SIDEWAYS": 0.0, "HIGH_VOL": -0.02}[regime],
                "SYN_INFLATION": {"BULL": 0.03, "BEAR": -0.02, "SIDEWAYS": 0.0, "HIGH_VOL": 0.04}[regime],
                "SYN_GROWTH_INDEX": {"BULL": 0.6, "BEAR": -0.8, "SIDEWAYS": 0.05, "HIGH_VOL": -0.3}[regime],
                "SYN_VOL_INDEX": {"BULL": -0.3, "BEAR": 1.2, "SIDEWAYS": -0.05, "HIGH_VOL": 2.0}[regime],
            }[series]
            noise = macro_rng.normal(0, {"SYN_RATES": 0.08, "SYN_INFLATION": 0.1, "SYN_GROWTH_INDEX": 0.9, "SYN_VOL_INDEX": 1.1}[series])
            macro_state[series] = max(macro_state[series] + drift + noise, 0.01)
            publication_lag_days = int(macro_rng.integers(3, 12))
            macro_rows.append(
                {
                    "series_name": series,
                    "timestamp": m_end,
                    "value": macro_state[series],
                    "publication_timestamp": m_end + pd.Timedelta(days=publication_lag_days),
                    "vintage_timestamp": m_end + pd.Timedelta(days=publication_lag_days),
                }
            )
    macro_df = pd.DataFrame(macro_rows)

    return SyntheticDataset(
        market=market_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True),
        fundamentals=fundamentals_df.sort_values(["symbol", "publication_timestamp"]).reset_index(drop=True),
        macro=macro_df.sort_values(["series_name", "timestamp"]).reset_index(drop=True),
        news=news_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True),
        regimes=regimes_df,
    )


def write_synthetic_dataset(dataset: SyntheticDataset, out_dir) -> None:
    """Persist the synthetic dataset to Parquet files under ``out_dir``."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.market.to_parquet(out_dir / "market.parquet", index=False)
    dataset.fundamentals.to_parquet(out_dir / "fundamentals.parquet", index=False)
    dataset.macro.to_parquet(out_dir / "macro.parquet", index=False)
    dataset.news.to_parquet(out_dir / "news.parquet", index=False)
    dataset.regimes.to_parquet(out_dir / "regimes.parquet", index=False)
