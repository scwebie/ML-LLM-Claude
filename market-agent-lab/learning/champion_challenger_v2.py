"""Champion/challenger promotion for the real-data pipeline (Stage 14, V0.2).

Wraps ``learning.champion_challenger.decide_promotion`` (V0.1, left
completely untouched) with one additional gate: when there is no existing
champion, the challenger must ALSO pass the initial-champion qualification
bar (``learning/initial_qualification.py``) before being promoted. When a
champion already exists, behaviour is identical to V0.1's
``decide_promotion`` -- a positive track record has already been
established, so the ongoing sharpe/IC/drawdown/calibration comparison is
the right test, not the one-off qualification bar.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import duckdb
import pandas as pd

from database import repository as repo
from learning.champion_challenger import PromotionCriteria, decide_promotion
from learning.initial_qualification import InitialQualificationBar, evaluate_initial_qualification
from models.registry import promote_to_champion


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
    return decide_promotion(challenger_metrics, champion_metrics, criteria)


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
        import json

        champion_metrics = json.loads(champion_record["metrics_json"])
        champion_version = champion_record["model_version"]

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
