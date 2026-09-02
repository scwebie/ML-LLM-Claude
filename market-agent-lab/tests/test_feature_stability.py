"""Tests for backtesting/feature_stability.py (V0.3 Stage 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.feature_stability import compute_feature_stability, permutation_importance_for_fold
from backtesting.purged_walk_forward import (
    build_trading_calendar,
    generate_purged_folds,
    run_purged_walk_forward,
)


def _synthetic_frame(n_symbols=6, n_days=300, seed=1):
    """A dev frame where "stable_signal" always drives the target across
    the whole period, "noise1"/"noise2" never do, and "early_only_signal"
    drives the target only in the FIRST half of the period (a feature
    that should show up as important in an early fold and not a later
    one -- i.e. exactly the "one period only" case)."""
    rng = np.random.default_rng(seed)
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    midpoint = n_days // 2
    rows = []
    for day_idx, d in enumerate(dates):
        stable = rng.normal()
        early = rng.normal()
        for sym in symbols:
            target = 0.02 * stable + (0.02 * early if day_idx < midpoint else 0.0) + 0.002 * rng.normal()
            rows.append(
                {
                    "symbol": sym, "timestamp": d,
                    "stable_signal": stable + rng.normal(0, 0.1),
                    "early_only_signal": early + rng.normal(0, 0.1),
                    "noise1": rng.normal(), "noise2": rng.normal(),
                    "excess_return_20d": target, "excess_return_5d": target * 0.5,
                    "positive_20d": float(target > 0), "positive_5d": float(target * 0.5 > 0),
                }
            )
    return pd.DataFrame(rows)


def _folds_for(df, initial_frac=0.4, val_frac=0.2):
    calendar = build_trading_calendar(df["timestamp"])
    n = len(calendar)
    return generate_purged_folds(calendar, max(1, int(n * initial_frac)), max(1, int(n * val_frac)), window_mode="expanding")


def test_permutation_importance_ranks_the_real_signal_above_noise():
    df = _synthetic_frame()
    feature_cols = ["stable_signal", "early_only_signal", "noise1", "noise2"]
    folds = _folds_for(df)
    fold_results = run_purged_walk_forward(df, folds, feature_cols)
    last_fold = fold_results[-1]

    val_mask = (df["timestamp"] >= last_fold.fold.validation_start) & (df["timestamp"] <= last_fold.fold.validation_end)
    val_df = df.loc[val_mask]
    perm = permutation_importance_for_fold(last_fold.trained, val_df, feature_cols, "excess_return_20d")
    assert not perm.empty
    top_feature = perm.iloc[0]["feature"]
    assert top_feature == "stable_signal"


def test_compute_feature_stability_end_to_end_shape():
    df = _synthetic_frame()
    feature_cols = ["stable_signal", "early_only_signal", "noise1", "noise2"]
    folds = _folds_for(df)
    assert len(folds) >= 2
    fold_results = run_purged_walk_forward(df, folds, feature_cols)

    report = compute_feature_stability(fold_results, df, feature_cols, top_k=2, n_permutation_repeats=3)
    assert report.n_folds == len(fold_results)
    assert len(report.top_features_native_per_fold) == len(fold_results)
    assert isinstance(report.native_rank_correlation, float)  # may be NaN, but never crashes/missing
    assert isinstance(report.family_stability, dict)
    assert "mean_importance_by_family" in report.family_stability


def test_compute_feature_stability_flags_the_stable_signal_as_consistently_important():
    """stable_signal drives the target in EVERY fold -- it should appear
    in the native top-K list in every fold, i.e. never flagged as
    one-period-only."""
    df = _synthetic_frame()
    feature_cols = ["stable_signal", "early_only_signal", "noise1", "noise2"]
    folds = _folds_for(df)
    fold_results = run_purged_walk_forward(df, folds, feature_cols)

    report = compute_feature_stability(fold_results, df, feature_cols, top_k=2)
    assert "stable_signal" not in report.one_period_only_features


def test_compute_feature_stability_too_few_folds_returns_empty_report():
    report = compute_feature_stability([], pd.DataFrame(), ["f1"])
    assert report.n_folds == 0
    assert report.native_rank_correlation != report.native_rank_correlation  # NaN
