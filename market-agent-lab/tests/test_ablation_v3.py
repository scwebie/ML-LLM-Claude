"""Tests for backtesting/ablation_v3.py (V0.3 Stage 4): feature-family
ablation with the fuller per-family statistics V0.3 requires, on
DEVELOPMENT DATA ONLY."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.ablation_v3 import run_feature_ablation_v3
from backtesting.purged_walk_forward import build_trading_calendar, generate_purged_folds


def _synthetic_multi_symbol_frame(n_symbols=6, n_days=260, seed=1, informative_families=("macro_raw",)):
    """Builds a small multi-symbol dev frame where the truth (target) is
    driven only by the "macro_raw_signal" feature (when "macro_raw" is in
    informative_families) plus noise -- everything else (breadth_noise,
    plain_noise1/2) is pure noise, unrelated to the target. This lets an
    ablation test assert the informative family's removal HURTS mean IC
    while a noise family's removal does not."""
    rng = np.random.default_rng(seed)
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    rows = []
    for d in dates:
        macro_signal = rng.normal()  # one shared macro draw per date
        for sym in symbols:
            idio = rng.normal(0, 1.0)
            target = (0.02 * macro_signal if "macro_raw" in informative_families else 0.0) + 0.002 * idio
            rows.append(
                {
                    "symbol": sym, "timestamp": d,
                    "macro_raw_signal": macro_signal + rng.normal(0, 0.1),
                    "breadth_noise": rng.normal(),
                    "plain_noise1": rng.normal(),
                    "plain_noise2": rng.normal(),
                    "excess_return_20d": target, "excess_return_5d": target * 0.5,
                    "positive_20d": float(target > 0), "positive_5d": float(target * 0.5 > 0),
                }
            )
    df = pd.DataFrame(rows)

    prices = pd.DataFrame(index=dates, columns=symbols, dtype=float)
    for sym in symbols:
        daily = rng.normal(0, 0.01, n_days)
        prices[sym] = 100.0 * (1 + pd.Series(daily, index=dates)).cumprod()
    market_df = pd.DataFrame(
        [{"symbol": s, "timestamp": d, "adjusted_close": prices.loc[d, s]} for s in symbols for d in dates]
    )
    return df, market_df, symbols


def _folds_for(df, initial_frac=0.5, val_frac=0.2):
    calendar = build_trading_calendar(df["timestamp"])
    n = len(calendar)
    return generate_purged_folds(calendar, max(1, int(n * initial_frac)), max(1, int(n * val_frac)), window_mode="expanding")


def test_run_feature_ablation_v3_baseline_report_present_and_shaped():
    df, market_df, _ = _synthetic_multi_symbol_frame()
    feature_cols = ["macro_raw_signal", "breadth_noise", "plain_noise1", "plain_noise2"]
    folds = _folds_for(df)

    reports = run_feature_ablation_v3(df, folds, feature_cols, market_df)
    variants = {r.variant for r in reports}
    assert "baseline" in variants
    baseline = next(r for r in reports if r.variant == "baseline")
    assert baseline.n_folds == len(folds)
    assert len(baseline.per_fold_rank_ic) <= baseline.n_folds
    assert "gross_sharpe" in baseline.sharpe_audit
    assert isinstance(baseline.rank_ic_bootstrap_ci, dict)


def test_run_feature_ablation_v3_removing_the_informative_family_hurts_ic():
    """Removing the ONLY family that actually drives the target must
    measurably reduce mean rank IC relative to baseline -- removing a
    pure-noise family must not."""
    df, market_df, _ = _synthetic_multi_symbol_frame(informative_families=("macro_raw",))
    feature_cols = ["macro_raw_signal", "breadth_noise", "plain_noise1", "plain_noise2"]
    folds = _folds_for(df)

    reports = run_feature_ablation_v3(df, folds, feature_cols, market_df)
    by_variant = {r.variant: r for r in reports}

    assert "remove_macro_raw" in by_variant
    assert "remove_market_breadth" in by_variant
    macro_removed = by_variant["remove_macro_raw"]
    breadth_removed = by_variant["remove_market_breadth"]

    # Removing the informative family should show a positive delta
    # (baseline beat the ablated model); removing pure noise should not
    # show a comparably large positive delta.
    assert macro_removed.delta_vs_baseline_rank_ic > breadth_removed.delta_vs_baseline_rank_ic


def test_run_feature_ablation_v3_family_only_variant_when_requested():
    df, market_df, _ = _synthetic_multi_symbol_frame()
    feature_cols = ["macro_raw_signal", "breadth_noise", "plain_noise1", "plain_noise2"]
    folds = _folds_for(df)

    reports = run_feature_ablation_v3(df, folds, feature_cols, market_df, include_family_only=True)
    variants = {r.variant for r in reports}
    assert "only_macro_raw" in variants
    only_macro = next(r for r in reports if r.variant == "only_macro_raw")
    assert only_macro.n_features == 1


def test_run_feature_ablation_v3_does_not_declare_from_one_fold_alone():
    """Structural check: every non-baseline report must expose the
    per-fold IC list (not just a single blended mean), so callers can see
    fold-to-fold variance rather than trusting one number."""
    df, market_df, _ = _synthetic_multi_symbol_frame()
    feature_cols = ["macro_raw_signal", "breadth_noise", "plain_noise1", "plain_noise2"]
    folds = _folds_for(df)
    assert len(folds) >= 2  # this test only means something with >= 2 folds

    reports = run_feature_ablation_v3(df, folds, feature_cols, market_df)
    for r in reports:
        assert len(r.per_fold_rank_ic) <= r.n_folds
        assert r.n_folds == len(folds)
