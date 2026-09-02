"""V0.3 Stage 6: negative-control falsification tests, development data only.

Five automated controls. Four ("near_zero" expectation) construct data
where any genuine predictive relationship has been deliberately destroyed
-- if the pipeline still reports a real IC on one of these, that is
evidence of leakage or an evaluation-methodology bug, not of skill. The
fifth (the future-data trap) does the OPPOSITE: it deliberately injects a
feature that IS the answer, and the harness is expected to catch it with
a very high IC -- proving the evaluation methodology has the power to
detect a real leak when one exists, rather than being blind to it.

Reuses V0.2's ``backtesting/robustness.py::add_negative_control_feature``/
``negative_control_report`` verbatim for the random-feature control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.purged_walk_forward import (
    DEFAULT_EMBARGO_DAYS,
    MAX_TARGET_HORIZON_DAYS,
    TARGET_TO_PRED_COL,
    PurgedFold,
    run_purged_walk_forward,
)
from backtesting.robustness import (
    NEGATIVE_CONTROL_FEATURE_NAME,
    add_negative_control_feature,
    build_evaluation_frame,
    negative_control_report,
)
from models.evaluate import feature_importance, information_coefficient

_TARGET_COLS = ("excess_return_5d", "excess_return_20d")


def _resync_positive_targets(df: pd.DataFrame) -> pd.DataFrame:
    for tc in _TARGET_COLS:
        pos_col = tc.replace("excess_return_", "positive_")
        if tc in df.columns and pos_col in df.columns:
            df[pos_col] = (df[tc] > 0).astype(float)
    return df


def shuffle_target_control(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Randomly permutes every target column's VALUES across rows,
    completely severing any feature<->target correspondence. Any surviving
    IC is spurious by construction."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    perm = rng.permutation(len(out))
    for tc in _TARGET_COLS:
        if tc in out.columns:
            out[tc] = out[tc].to_numpy()[perm]
    return _resync_positive_targets(out)


def time_shift_target_control(df: pd.DataFrame, shift_days: int = 50) -> pd.DataFrame:
    """Shifts each symbol's target series by ``shift_days`` ROWS along its
    own timeline, so a given date's row now carries a different date's
    realised outcome. A genuinely point-in-time-safe model, whose skill
    comes from features observed as of that date, must not predict a
    DIFFERENT date's outcome; any surviving IC here suggests the model (or
    the harness) is not actually anchored to the claimed timestamp."""
    out = df.sort_values(["symbol", "timestamp"]).copy()
    for tc in _TARGET_COLS:
        if tc in out.columns:
            out[tc] = out.groupby("symbol")[tc].shift(shift_days)
    return _resync_positive_targets(out)


def inject_future_leak_feature(
    df: pd.DataFrame, target_col: str = "excess_return_20d", leak_col_name: str = "FUTURE_LEAK_actual_target"
) -> tuple[pd.DataFrame, str]:
    """Adds a feature that IS the target itself -- the ultimate leak. A
    correctly-functioning evaluation harness must show a very high IC
    when this feature is included (it is being handed the answer); if it
    does NOT, the harness has a blind spot elsewhere and this control has
    done its job by exposing that."""
    out = df.copy()
    out[leak_col_name] = out[target_col]
    return out, leak_col_name


def symbol_label_permutation_control(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Within each DATE, randomly permutes which row's target values are
    attached to which row's features (symbol and features stay put; only
    the target pairing is scrambled), preserving both the cross-sectional
    feature distribution and the cross-sectional target distribution for
    that date, but destroying genuine per-symbol feature<->outcome
    correspondence. If IC survives this, the apparent skill was never
    really about which specific symbol had which feature values -- a red
    flag for a date-level (not symbol-level) artifact."""
    rng = np.random.default_rng(seed)
    out = df.reset_index(drop=True).copy()
    for _, idx in out.groupby("timestamp").groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        perm = rng.permutation(len(idx))
        for tc in _TARGET_COLS:
            if tc in out.columns:
                out.loc[idx, tc] = out.loc[idx, tc].to_numpy()[perm]
    return _resync_positive_targets(out)


@dataclass
class NegativeControlResult:
    name: str
    statistic: float
    expectation: str  # "near_zero" or "strong_signal_expected"
    passed: bool
    detail: str


def run_negative_controls(
    development_df: pd.DataFrame,
    folds: list[PurgedFold],
    feature_cols: list[str],
    target_col: str = "excess_return_20d",
    horizon_days: int = MAX_TARGET_HORIZON_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    hyperparameters: dict | None = None,
    ic_threshold: float = 0.10,
    leak_ic_threshold: float = 0.9,
) -> list[NegativeControlResult]:
    """Runs all five controls against ``development_df`` (pre-holdout
    only) and ``folds`` (the caller's own purged walk-forward folds).
    Never touches the holdout or post-holdout regions."""
    pred_col = TARGET_TO_PRED_COL[target_col]

    def _mean_ic(variant_df: pd.DataFrame, cols: list[str], prefix: str) -> float:
        fold_results = run_purged_walk_forward(
            variant_df, folds, cols, horizon_days, embargo_days, hyperparameters, model_version_prefix=prefix
        )
        eval_frame = build_evaluation_frame(fold_results, variant_df, target_col)
        if eval_frame.empty:
            return float("nan")
        return information_coefficient(eval_frame[target_col], eval_frame[pred_col])

    results: list[NegativeControlResult] = []

    shuffled_ic = _mean_ic(shuffle_target_control(development_df), feature_cols, "negctrl_shuffled_target")
    results.append(
        NegativeControlResult(
            "shuffled_target", shuffled_ic, "near_zero", (shuffled_ic != shuffled_ic) or abs(shuffled_ic) < ic_threshold,
            f"IC={shuffled_ic:.4f} (threshold {ic_threshold})",
        )
    )

    shifted_ic = _mean_ic(time_shift_target_control(development_df), feature_cols, "negctrl_time_shift")
    results.append(
        NegativeControlResult(
            "time_shifted_target", shifted_ic, "near_zero", (shifted_ic != shifted_ic) or abs(shifted_ic) < ic_threshold,
            f"IC={shifted_ic:.4f} (threshold {ic_threshold})",
        )
    )

    with_noise = add_negative_control_feature(development_df)
    noise_fold_results = run_purged_walk_forward(
        with_noise, folds, [*feature_cols, NEGATIVE_CONTROL_FEATURE_NAME], horizon_days, embargo_days,
        hyperparameters, model_version_prefix="negctrl_random_feature",
    )
    if noise_fold_results:
        booster = noise_fold_results[-1].trained.boosters.get(target_col)
        importances = feature_importance(booster, noise_fold_results[-1].trained.feature_names) if booster else pd.DataFrame()
        if not importances.empty:
            ctrl = negative_control_report(importances)
            results.append(
                NegativeControlResult(
                    "random_feature", ctrl.control_importance, "near_zero", ctrl.passed,
                    f"rank {ctrl.control_rank}/{ctrl.n_features} by native importance",
                )
            )

    leaked_df, leak_col = inject_future_leak_feature(development_df, target_col)
    leak_ic = _mean_ic(leaked_df, [*feature_cols, leak_col], "negctrl_future_leak")
    results.append(
        NegativeControlResult(
            "future_data_trap", leak_ic, "strong_signal_expected", (leak_ic == leak_ic) and leak_ic >= leak_ic_threshold,
            f"IC={leak_ic:.4f} (expected >= {leak_ic_threshold} -- the harness must detect an obvious leak)",
        )
    )

    permuted_ic = _mean_ic(symbol_label_permutation_control(development_df), feature_cols, "negctrl_symbol_permutation")
    results.append(
        NegativeControlResult(
            "symbol_label_permutation", permuted_ic, "near_zero", (permuted_ic != permuted_ic) or abs(permuted_ic) < ic_threshold,
            f"IC={permuted_ic:.4f} (threshold {ic_threshold})",
        )
    )

    return results


def assert_negative_controls_pass(results: list[NegativeControlResult]) -> None:
    """Fails loudly (raises) if any control did not behave as expected --
    a "near_zero" control showing material signal, or the future-data
    trap FAILING to show strong signal (meaning the harness has a blind
    spot). Call this after :func:`run_negative_controls` wherever the
    result should hard-fail a CI/report run, not just be logged."""
    failures = [r for r in results if not r.passed]
    if failures:
        lines = "\n".join(f"  - {r.name} ({r.expectation}): {r.detail}" for r in failures)
        raise AssertionError(f"negative control(s) failed:\n{lines}")
