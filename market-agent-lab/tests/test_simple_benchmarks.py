"""Tests for backtesting/simple_benchmarks.py (V0.3 Stage 11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.purged_walk_forward import (
    build_trading_calendar,
    generate_purged_folds,
    run_purged_walk_forward,
)
from backtesting.simple_benchmarks import (
    equal_weight_composite_ic,
    factor_signal_ic,
    logistic_baseline_ic,
    ridge_baseline_ic,
    run_simple_benchmarks,
)


def _linear_frame(n_days=300, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    symbols = [f"SYM{i}" for i in range(6)]
    rows = []
    for d in dates:
        for sym in symbols:
            momentum = rng.normal()
            reversion = rng.normal()  # HIGH reversion factor -> LOWER forward return
            target = 0.02 * momentum - 0.015 * reversion + 0.002 * rng.normal()
            rows.append(
                {
                    "symbol": sym, "timestamp": d, "raw_return_60d": momentum, "raw_rsi_14": reversion,
                    "f1": momentum, "f2": reversion,
                    "excess_return_20d": target, "excess_return_5d": target * 0.5,
                    "positive_20d": float(target > 0), "positive_5d": float(target * 0.5 > 0),
                }
            )
    return pd.DataFrame(rows)


def _folds_for(df, initial_frac=0.4, val_frac=0.2):
    calendar = build_trading_calendar(df["timestamp"])
    n = len(calendar)
    return generate_purged_folds(calendar, max(1, int(n * initial_frac)), max(1, int(n * val_frac)), window_mode="expanding")


def test_ridge_baseline_ic_detects_linear_relationship():
    df = _linear_frame()
    train, val = df.iloc[:1000], df.iloc[1000:1400]
    ic, n = ridge_baseline_ic(train, val, ["f1", "f2"], "excess_return_20d")
    assert n > 0
    assert ic > 0.3


def test_logistic_baseline_ic_detects_relationship():
    df = _linear_frame()
    train, val = df.iloc[:1000], df.iloc[1000:1400]
    ic, n = logistic_baseline_ic(train, val, ["f1", "f2"], "positive_20d")
    assert n > 0
    assert ic > 0.1


def test_factor_signal_ic_momentum_positive_relationship_no_invert_needed():
    df = _linear_frame()
    val = df.iloc[1000:1400]
    ic, n = factor_signal_ic(val, "raw_return_60d", "excess_return_20d", invert=False)
    assert n > 0
    assert ic > 0.1


def test_factor_signal_ic_mean_reversion_requires_invert_to_show_positive_ic():
    df = _linear_frame()
    val = df.iloc[1000:1400]
    inverted_ic, _ = factor_signal_ic(val, "raw_rsi_14", "excess_return_20d", invert=True)
    non_inverted_ic, _ = factor_signal_ic(val, "raw_rsi_14", "excess_return_20d", invert=False)
    assert inverted_ic > 0.1
    assert non_inverted_ic < 0  # the raw (non-inverted) relationship is negative by construction
    assert inverted_ic == pytest.approx(-non_inverted_ic, abs=1e-9)


def test_equal_weight_composite_combines_both_factors():
    df = _linear_frame()
    val = df.iloc[1000:1400]
    combo_ic, n = equal_weight_composite_ic(val, ["raw_return_60d", "raw_rsi_14"], "excess_return_20d", invert=[False, True])
    assert n > 0
    assert combo_ic > 0.1


def test_run_simple_benchmarks_end_to_end_with_lightgbm_comparison():
    df = _linear_frame(n_days=260)
    feature_cols = ["f1", "f2"]
    folds = _folds_for(df)
    assert len(folds) >= 2

    lgbm_fold_results = run_purged_walk_forward(df, folds, feature_cols)
    lgbm_ics = [r.metrics.get("excess_return_20d", {}).get("information_coefficient", float("nan")) for r in lgbm_fold_results]

    report = run_simple_benchmarks(df, folds, feature_cols, lightgbm_per_fold_ic=lgbm_ics)
    assert report.target_col == "excess_return_20d"
    assert "ridge" in report.mean_ic_by_model
    assert "logistic" in report.mean_ic_by_model
    assert "momentum" in report.mean_ic_by_model
    assert "mean_reversion" in report.mean_ic_by_model
    assert "equal_weight_composite" in report.mean_ic_by_model
    assert "lightgbm" in report.mean_ic_by_model
    assert report.best_baseline != ""
    assert report.incremental_ic_over_best_baseline == report.mean_ic_by_model["lightgbm"] - report.mean_ic_by_model[report.best_baseline]
    assert len(report.per_fold) > 0


def test_run_simple_benchmarks_without_lightgbm_still_compares_baselines():
    df = _linear_frame(n_days=260)
    feature_cols = ["f1", "f2"]
    folds = _folds_for(df)
    report = run_simple_benchmarks(df, folds, feature_cols)
    assert "lightgbm" not in report.mean_ic_by_model
    assert report.incremental_ic_over_best_baseline != report.incremental_ic_over_best_baseline  # NaN
    assert report.best_baseline != ""
