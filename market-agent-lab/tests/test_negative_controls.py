"""Tests for backtesting/negative_controls.py (V0.3 Stage 6): the five
falsification tests, and that they actually detect what they're built to
detect on controlled synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.negative_controls import (
    NegativeControlResult,
    assert_negative_controls_pass,
    inject_future_leak_feature,
    run_negative_controls,
    shuffle_target_control,
    symbol_label_permutation_control,
    time_shift_target_control,
)
from backtesting.purged_walk_forward import build_trading_calendar, generate_purged_folds


def _informative_frame(n_symbols=10, n_days=420, seed=1):
    """A dev frame with a REAL, persistent, per-(symbol, date) signal:
    excess_return_20d is driven by "signal", independently drawn for
    every symbol on every date (a genuinely cross-sectional/idiosyncratic
    relationship, not a shared date-level common factor) plus noise.
    Every control's expectation is evaluated against this -- a real
    signal exists, so the near-zero controls must destroy it (including
    symbol_label_permutation, which specifically needs the true
    signal<->target correspondence to be AT THE SYMBOL LEVEL within a
    date to have anything to break), while the future-data trap must not
    need to destroy anything (it adds a NEW, perfect feature)."""
    rng = np.random.default_rng(seed)
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    rows = []
    for d in dates:
        for sym in symbols:
            signal = rng.normal()
            target = 0.02 * signal + 0.002 * rng.normal()
            rows.append(
                {
                    "symbol": sym, "timestamp": d, "signal": signal + rng.normal(0, 0.1),
                    "noise_feat": rng.normal(),
                    "excess_return_20d": target, "excess_return_5d": target * 0.5,
                    "positive_20d": float(target > 0), "positive_5d": float(target * 0.5 > 0),
                }
            )
    return pd.DataFrame(rows)


def _folds_for(df, initial_frac=0.4, val_frac=0.2):
    calendar = build_trading_calendar(df["timestamp"])
    n = len(calendar)
    return generate_purged_folds(calendar, max(1, int(n * initial_frac)), max(1, int(n * val_frac)), window_mode="expanding")


# --- individual control constructions --------------------------------------------------------------


def test_shuffle_target_control_destroys_feature_target_alignment():
    df = _informative_frame(n_symbols=4, n_days=40)
    shuffled = shuffle_target_control(df)
    # Row count and column set preserved.
    assert len(shuffled) == len(df)
    assert set(shuffled.columns) == set(df.columns)
    # But the specific (symbol, timestamp) -> target pairing has changed
    # for at least some rows (astronomically likely with real randomness).
    original = df.sort_values(["symbol", "timestamp"])["excess_return_20d"].to_numpy()
    new = shuffled.sort_values(["symbol", "timestamp"])["excess_return_20d"].to_numpy()
    assert not np.array_equal(original, new)


def test_time_shift_target_control_shifts_within_symbol():
    df = _informative_frame(n_symbols=3, n_days=100)
    shifted = time_shift_target_control(df, shift_days=10)
    # Every symbol's target series should now be the ORIGINAL series
    # shifted -- confirm for one symbol directly.
    sym = "SYM0"
    orig = df[df["symbol"] == sym].sort_values("timestamp")["excess_return_20d"].to_numpy()
    new = shifted[shifted["symbol"] == sym].sort_values("timestamp")["excess_return_20d"].to_numpy()
    # pandas .shift(10) semantics: new[i] = orig[i - 10] for i >= 10, and
    # the first 10 entries are NaN (no data to shift forward from).
    assert np.isnan(new[:10]).all()
    assert np.allclose(new[10:], orig[:-10], equal_nan=True)


def test_inject_future_leak_feature_adds_exact_copy_of_target():
    df = _informative_frame(n_symbols=3, n_days=30)
    leaked, leak_col = inject_future_leak_feature(df, "excess_return_20d")
    assert (leaked[leak_col] == leaked["excess_return_20d"]).all()


def test_symbol_label_permutation_preserves_date_level_target_distribution():
    df = _informative_frame(n_symbols=6, n_days=20)
    permuted = symbol_label_permutation_control(df)
    for ts, group in df.groupby("timestamp"):
        permuted_group = permuted[permuted["timestamp"] == ts]
        # Same MULTISET of target values that date, just reassigned across symbols.
        assert sorted(group["excess_return_20d"].round(10)) == sorted(permuted_group["excess_return_20d"].round(10))


# --- run_negative_controls: full pipeline ------------------------------------------------------------


def test_run_negative_controls_all_five_present_and_behave_as_expected():
    df = _informative_frame()
    feature_cols = ["signal", "noise_feat"]
    folds = _folds_for(df)

    results = run_negative_controls(df, folds, feature_cols)
    names = {r.name for r in results}
    assert names == {
        "shuffled_target", "time_shifted_target", "random_feature", "future_data_trap", "symbol_label_permutation",
    }

    by_name = {r.name: r for r in results}
    # The four "near_zero" controls must pass (no leakage in this clean setup).
    for name in ("shuffled_target", "time_shifted_target", "random_feature", "symbol_label_permutation"):
        assert by_name[name].passed, f"{name} unexpectedly failed: {by_name[name].detail}"
    # The future-data trap must show the harness CAN detect an obvious leak.
    assert by_name["future_data_trap"].passed, by_name["future_data_trap"].detail
    assert by_name["future_data_trap"].statistic > 0.9


def test_assert_negative_controls_pass_raises_on_a_failed_near_zero_control():
    results = [
        NegativeControlResult("shuffled_target", 0.5, "near_zero", passed=False, detail="IC=0.5000 (threshold 0.05)"),
        NegativeControlResult("future_data_trap", 0.99, "strong_signal_expected", passed=True, detail="IC=0.9900"),
    ]
    with pytest.raises(AssertionError, match="shuffled_target"):
        assert_negative_controls_pass(results)


def test_assert_negative_controls_pass_is_silent_when_everything_passed():
    results = [
        NegativeControlResult("shuffled_target", 0.01, "near_zero", passed=True, detail="IC=0.0100"),
        NegativeControlResult("future_data_trap", 0.99, "strong_signal_expected", passed=True, detail="IC=0.9900"),
    ]
    assert_negative_controls_pass(results)  # must not raise
