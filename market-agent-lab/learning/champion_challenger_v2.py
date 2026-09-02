"""Champion/challenger promotion for the real-data pipeline (Stage 14, V0.2).

When there is no existing champion, the challenger must pass the
initial-champion qualification bar (``learning/initial_qualification.py``)
before being promoted -- see ``decide_promotion_v2`` below.

When a champion already exists, this module does its OWN comparison on
``target_col``/``pred_col`` (default ``excess_return_20d`` -- V0.2's
primary target) rather than delegating to V0.1's
``learning.champion_challenger.decide_promotion``. V0.1's gate is left
completely untouched (its own synthetic demo pipeline and tests still use
it directly), but it internally hardcodes its metric lookups to
``excess_return_5d``/``positive_5d`` regardless of any ``target_col``
passed in -- silently scoring V0.2's real-data promotions on the wrong
horizon. ``_ic_of_v2``/``_sharpe_of_v2``/``_brier_of_v2`` below mirror
V0.1's logic and thresholds exactly (same ``PromotionCriteria`` defaults,
nothing loosened), just keyed on the actual target being evaluated.

This module also asserts that a challenger is never compared against
itself: ``run_promotion_cycle_v2`` loads the incumbent champion's
``model_version`` from the registry and asserts it differs from the
challenger's before treating the comparison as valid. Two independently
computed metrics dicts being deep-equal (e.g. because ``evaluate-real``
was re-run against unchanged development data, and LightGBM training is
deterministic) is a different, non-fatal condition -- it produces a
correct but vacuous "passed all criteria" comparison and is not treated as
an assertion failure here, but the version-identity check guards against a
real invariant violation: a colliding ``model_version`` (e.g. from the
registry's ``COUNT(*)``-based numbering scheme being reused, or two
databases being conflated) would make ``get_champion`` return the very row
being registered as the challenger, which must never be silently accepted
as a legitimate comparison.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import duckdb
import pandas as pd

from database import repository as repo
from learning.champion_challenger import PromotionCriteria
from learning.initial_qualification import InitialQualificationBar, evaluate_initial_qualification
from models.registry import promote_to_champion

# excess_return_{H}d's matching binary-classification target, for the
# calibration (Brier score) leg of the comparison -- same convention used
# throughout the real pipeline (models/train.py, backtesting/purged_walk_forward.py).
_CLASSIFICATION_TARGET_FOR = {
    "excess_return_5d": "positive_5d",
    "excess_return_20d": "positive_20d",
}


def _ic_of_v2(metrics: dict, target_col: str) -> float:
    reg = metrics.get(target_col, {})
    return reg.get("information_coefficient", 0.0) or 0.0


def _sharpe_of_v2(metrics: dict, target_col: str) -> float:
    reg = metrics.get(target_col, {})
    return reg.get("sharpe_ratio", reg.get("information_coefficient", 0.0)) or 0.0


def _brier_of_v2(metrics: dict, target_col: str) -> float:
    pos_col = _CLASSIFICATION_TARGET_FOR.get(target_col)
    cls = metrics.get(pos_col, {}) if pos_col else {}
    return cls.get("brier_score", 0.25) or 0.25


def _drawdown_of_v2(metrics: dict) -> float:
    return metrics.get("backtest", {}).get("max_drawdown", 0.0) or 0.0


def decide_promotion_v2(
    challenger_metrics: dict,
    champion_metrics: dict | None,
    predictions_df: pd.DataFrame,
    target_col: str = "excess_return_20d",
    pred_col: str = "predicted_excess_return_20d",
    criteria: PromotionCriteria | None = None,
    qualification_bar: InitialQualificationBar | None = None,
) -> tuple[bool, str]:
    if champion_metrics is None:
        qualification = evaluate_initial_qualification(
            challenger_metrics, predictions_df, target_col, pred_col, qualification_bar
        )
        if not qualification.qualified:
            return False, "no existing champion, and the challenger failed the initial-qualification bar: " + "; ".join(
                qualification.reasons
            )
        return True, "no existing champion; challenger passed every initial-qualification criterion"

    criteria = criteria or PromotionCriteria()

    challenger_ic = _ic_of_v2(challenger_metrics, target_col)
    if challenger_ic < criteria.min_information_coefficient or challenger_ic != challenger_ic:
        return False, f"challenger information_coefficient={challenger_ic:.4f} below minimum {criteria.min_information_coefficient}"

    champ_ic = _ic_of_v2(champion_metrics, target_col)
    champ_brier = _brier_of_v2(champion_metrics, target_col)
    chal_brier = _brier_of_v2(challenger_metrics, target_col)
    champ_dd = _drawdown_of_v2(champion_metrics)
    chal_dd = _drawdown_of_v2(challenger_metrics)

    if chal_brier > champ_brier + criteria.max_brier_regression_tolerance:
        return False, f"challenger calibration worse: brier {chal_brier:.4f} vs champion {champ_brier:.4f}"

    if abs(chal_dd) > abs(champ_dd) + criteria.max_drawdown_regression_tolerance:
        return False, f"challenger drawdown worse: {chal_dd:.2%} vs champion {champ_dd:.2%}"

    champ_sharpe = _sharpe_of_v2(champion_metrics, target_col)
    chal_sharpe = _sharpe_of_v2(challenger_metrics, target_col)
    if chal_sharpe < champ_sharpe + criteria.min_sharpe_improvement:
        return False, f"challenger sharpe {chal_sharpe:.3f} not sufficiently close to/better than champion {champ_sharpe:.3f}"

    if challenger_ic < champ_ic - 0.01:
        return False, f"challenger information_coefficient {challenger_ic:.4f} worse than champion {champ_ic:.4f}"

    return True, (
        f"challenger passed all criteria on {target_col}: IC {challenger_ic:.4f} (champion {champ_ic:.4f}), "
        f"sharpe {chal_sharpe:.3f} (champion {champ_sharpe:.3f}), brier {chal_brier:.4f} (champion {champ_brier:.4f}), "
        f"drawdown {chal_dd:.2%} (champion {champ_dd:.2%})"
    )


def run_promotion_cycle_v2(
    con: duckdb.DuckDBPyConnection,
    challenger_version: str,
    challenger_metrics: dict,
    predictions_df: pd.DataFrame,
    target_col: str = "excess_return_20d",
    pred_col: str = "predicted_excess_return_20d",
    criteria: PromotionCriteria | None = None,
    qualification_bar: InitialQualificationBar | None = None,
) -> tuple[bool, str]:
    """Same audit-trail behaviour as V0.1's
    ``champion_challenger.run_promotion_cycle`` (every decision, promoted
    or rejected, is written to ``promotion_log``), routed through
    :func:`decide_promotion_v2` instead of the plain V0.1 gate."""
    champion_record = repo.get_champion(con)
    champion_metrics = None
    champion_version = None
    if champion_record is not None:
        champion_metrics = json.loads(champion_record["metrics_json"])
        champion_version = champion_record["model_version"]
        # A challenger must never be compared against itself: if the
        # registry ever produced a colliding model_version (e.g. the
        # COUNT(*)-based numbering in models.registry._next_model_version
        # reused an id, or two databases' registries were conflated),
        # get_champion() could return the very row being registered as
        # this run's challenger. Silently "comparing" a model to itself
        # would trivially pass every criterion (every delta is exactly
        # zero) and is never a legitimate promotion decision.
        assert challenger_version != champion_version, (
            "model-selection invariant violated: challenger_version and the existing "
            f"champion_version are identical ({challenger_version!r}) -- a model can "
            "never be legitimately compared against itself"
        )

    promoted, rationale = decide_promotion_v2(
        challenger_metrics, champion_metrics, predictions_df, target_col, pred_col, criteria, qualification_bar
    )

    repo.insert_promotion_log(
        con,
        {
            "id": uuid.uuid4().hex,
            "timestamp": datetime.now(UTC),
            "challenger_version": challenger_version,
            "champion_version": champion_version,
            "decision": "PROMOTED" if promoted else "REJECTED",
            "rationale": rationale,
            "metrics": {"challenger": challenger_metrics, "champion": champion_metrics},
        },
    )

    if promoted:
        promote_to_champion(con, challenger_version)

    return promoted, rationale
