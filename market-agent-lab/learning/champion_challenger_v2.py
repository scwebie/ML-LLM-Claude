"""Champion/challenger promotion for the real-data pipeline (Stage 14, V0.2;
self-promotion hardening in V0.3 Stage 1).

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

Two structural guards close the "challenger metrics exactly equal champion
metrics" failure mode audited in V0.3 Stage 1:

1. ``run_promotion_cycle_v2`` takes the incumbent ``champion_record`` as a
   REQUIRED parameter (no default) -- there is no code path through which
   this function can run a comparison without the caller having already
   looked up the incumbent. Combined with real_pipeline.py's
   ``evaluate_real_step`` calling ``get_champion()`` before
   ``register_model()``, this guarantees the incumbent is read before the
   challenger is written to the registry, so a challenger registration can
   never race or overwrite the record this comparison reads. If the
   incumbent's own ``model_version`` is ever identical to the challenger's
   (e.g. a model-registry versioning collision, or two databases'
   registries conflated), this is asserted to never happen.
2. A *different*, non-identity failure mode: re-running ``evaluate-real``
   against development data that hasn't changed retrains a bit-identical
   model (LightGBM's ``seed=42`` makes training deterministic) under a
   genuinely new ``model_version``. Comparing it to the incumbent would
   trivially "pass every criterion" (every delta is exactly zero) and log
   a fresh, uninformative PROMOTED decision. ``run_promotion_cycle_v2``
   now requires the challenger's ``validation_period_end`` to extend
   strictly past the incumbent's before running any metric comparison at
   all; otherwise it rejects immediately with an explicit "no new
   information since the incumbent" rationale, never silently re-promoting
   a re-run of the same evaluation.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import duckdb
import pandas as pd

from core.logging import get_logger
from database import repository as repo
from learning.champion_challenger import PromotionCriteria
from learning.initial_qualification import InitialQualificationBar, evaluate_initial_qualification
from models.registry import promote_to_champion

logger = get_logger(__name__)

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
    champion_record: dict | None,
    challenger_validation_end: pd.Timestamp | datetime | None = None,
    target_col: str = "excess_return_20d",
    pred_col: str = "predicted_excess_return_20d",
    criteria: PromotionCriteria | None = None,
    qualification_bar: InitialQualificationBar | None = None,
) -> tuple[bool, str]:
    """Same audit-trail behaviour as V0.1's
    ``champion_challenger.run_promotion_cycle`` (every decision, promoted
    or rejected, is written to ``promotion_log``), routed through
    :func:`decide_promotion_v2` instead of the plain V0.1 gate.

    ``champion_record`` is REQUIRED (pass ``None`` explicitly when there is
    no incumbent) -- the caller must look it up (``models.registry.get_champion``
    or ``database.repository.get_champion``) BEFORE calling this function
    and, critically, before registering the challenger in ``model_registry``.
    This function never queries the registry for the incumbent itself, so
    it structurally cannot race against, or be confused by, the
    challenger's own registration.

    ``challenger_validation_end``, when supplied alongside an existing
    incumbent, gates the entire comparison on genuinely new information: a
    challenger whose validation window does not extend past the
    incumbent's is rejected immediately, before any metric is even read,
    rather than running a comparison that (thanks to deterministic
    training on unchanged data) would otherwise trivially "pass"."""
    champion_metrics = None
    champion_version = None
    champion_validation_end = None
    if champion_record is not None:
        champion_metrics = json.loads(champion_record["metrics_json"])
        champion_version = champion_record["model_version"]
        champion_validation_end = champion_record.get("validation_period_end")
        # A challenger must never be compared against itself: if the
        # registry ever produced a colliding model_version (e.g. the
        # COUNT(*)-based numbering in models.registry._next_model_version
        # reused an id, or two databases' registries were conflated), the
        # caller could hand us a champion_record that IS the row being
        # registered as this run's challenger. Silently "comparing" a
        # model to itself would trivially pass every criterion (every
        # delta is exactly zero) and is never a legitimate decision.
        assert challenger_version != champion_version, (
            "model-selection invariant violated: challenger_version and the existing "
            f"champion_version are identical ({challenger_version!r}) -- a model can "
            "never be legitimately compared against itself"
        )

    incumbent_metric_source = (
        f"model_registry.metrics_json for model_version={champion_version!r}, loaded before the "
        "challenger was registered"
        if champion_version is not None
        else "none -- no incumbent champion exists"
    )
    challenger_metric_source = "freshly computed this run from the current purged walk-forward fold"
    logger.info(
        "promotion_audit",
        incumbent_champion_version=champion_version,
        challenger_version=challenger_version,
        incumbent_metric_source=incumbent_metric_source,
        challenger_metric_source=challenger_metric_source,
    )

    def _log_and_return(promoted: bool, rationale: str) -> tuple[bool, str]:
        repo.insert_promotion_log(
            con,
            {
                "id": uuid.uuid4().hex,
                "timestamp": datetime.now(UTC),
                "challenger_version": challenger_version,
                "champion_version": champion_version,
                "decision": "PROMOTED" if promoted else "REJECTED",
                "rationale": rationale,
                "metrics": {
                    "challenger": challenger_metrics,
                    "champion": champion_metrics,
                    "challenger_metric_source": challenger_metric_source,
                    "incumbent_metric_source": incumbent_metric_source,
                },
            },
        )
        if promoted:
            promote_to_champion(con, challenger_version)
        return promoted, rationale

    if champion_version is not None and challenger_validation_end is not None and champion_validation_end is not None:
        if pd.Timestamp(challenger_validation_end) <= pd.Timestamp(champion_validation_end):
            rationale = (
                f"challenger's validation window (ending {challenger_validation_end}) does not extend past the "
                f"incumbent champion's (ending {champion_validation_end}) -- no new development data has been "
                "added since the incumbent was selected, so this would be re-evaluating an unchanged (or bit-"
                "identical, given deterministic training) result against itself; refusing to promote or reject "
                "on a vacuous comparison"
            )
            return _log_and_return(False, rationale)

    promoted, rationale = decide_promotion_v2(
        challenger_metrics, champion_metrics, predictions_df, target_col, pred_col, criteria, qualification_bar
    )
    return _log_and_return(promoted, rationale)
