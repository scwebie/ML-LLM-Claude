"""Initial-champion qualification bar (Stage 14, V0.2).

V0.1's champion/challenger gate (``learning/champion_challenger.py``, left
completely untouched) auto-promotes the very first challenger whenever
there is no existing champion, gated only on
``information_coefficient > 0`` -- appropriate for V0.1's small synthetic
demo, whose point is to exercise the full pipeline rather than to certify
a model. For V0.2's real-data pipeline, a merely-positive IC is not
evidence of real skill: with a modest number of out-of-sample
observations, a positive IC is easily noise. This module adds a strictly
higher bar for becoming the FIRST champion in the real-data pipeline --
every criterion below must pass, so a weak first model is rejected rather
than auto-promoted for lack of anything to compare it against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtesting.robustness import permutation_test_ic


@dataclass(frozen=True)
class InitialQualificationBar:
    """Every criterion must pass for a first model (no existing champion)
    to be promoted. Failing ANY of them means "reject the model, collect
    more data or improve it" -- there is no partial credit."""

    min_out_of_sample_observations: int = 500
    min_information_coefficient: float = 0.02
    min_sharpe_ratio: float = 0.30
    max_permutation_p_value: float = 0.10  # observed IC must beat >=90% of the noise-null distribution
    n_permutations: int = 500


@dataclass(frozen=True)
class QualificationResult:
    qualified: bool
    reasons: list[str] = field(default_factory=list)  # every failing criterion, not just the first


def evaluate_initial_qualification(
    challenger_metrics: dict,
    predictions_df: pd.DataFrame,
    target_col: str = "excess_return_20d",
    pred_col: str = "predicted_excess_return_20d",
    bar: InitialQualificationBar | None = None,
) -> QualificationResult:
    """Whether a challenger with NO existing champion is strong enough to
    become the initial champion at all. Checks every criterion (rather
    than short-circuiting on the first failure) so a rejection always
    shows the full picture.

    ``challenger_metrics[target_col]`` is expected to carry
    ``information_coefficient`` (from ``models.evaluate.evaluate_regression``)
    and a caller-supplied ``sharpe_ratio`` -- the latter is deliberately
    NOT defaulted from IC (unlike V0.1's looser gate): a missing Sharpe
    fails this bar rather than silently reusing IC as a stand-in, since
    the two are on different scales and conflating them would understate
    what "no real backtest-level risk-adjusted return was computed" means.
    """
    bar = bar or InitialQualificationBar()
    reasons: list[str] = []

    clean = predictions_df.dropna(subset=[target_col, pred_col]) if not predictions_df.empty else predictions_df
    n_obs = len(clean)
    if n_obs < bar.min_out_of_sample_observations:
        reasons.append(f"only {n_obs} out-of-sample observations, need >= {bar.min_out_of_sample_observations}")

    reg_metrics = challenger_metrics.get(target_col, {})
    ic = reg_metrics.get("information_coefficient", float("nan"))
    if ic != ic:
        reasons.append("information_coefficient is missing/NaN")
    elif ic < bar.min_information_coefficient:
        reasons.append(f"information_coefficient={ic:.4f} below minimum {bar.min_information_coefficient}")

    sharpe = reg_metrics.get("sharpe_ratio", float("nan"))
    if sharpe != sharpe:
        reasons.append("sharpe_ratio is missing/NaN -- a real backtest-level Sharpe must be supplied")
    elif sharpe < bar.min_sharpe_ratio:
        reasons.append(f"sharpe_ratio={sharpe:.4f} below minimum {bar.min_sharpe_ratio}")

    if n_obs >= 3:
        perm = permutation_test_ic(clean[target_col], clean[pred_col], n_permutations=bar.n_permutations)
        p_value = perm["p_value"]
        if p_value != p_value:
            reasons.append("permutation test could not be computed (insufficient valid rows)")
        elif p_value > bar.max_permutation_p_value:
            reasons.append(
                f"permutation test p-value={p_value:.4f} exceeds {bar.max_permutation_p_value} "
                "-- observed IC is not distinguishable from noise"
            )
    else:
        reasons.append("too few observations to run a permutation significance test")

    return QualificationResult(qualified=not reasons, reasons=reasons)
