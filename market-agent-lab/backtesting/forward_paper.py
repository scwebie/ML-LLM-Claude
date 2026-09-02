"""V0.3 Stage 10: forward-paper evaluation on the post-holdout period.

The post-holdout region (everything after the fixed historical holdout's
end date -- ``backtesting/holdout.py::split_temporal_partitions``'s
``post_holdout_df``) was preserved but never used in V0.2/V0.3 model
selection. This module provides the ONE legitimate way to use it: a
single, logged, no-training evaluation of an ALREADY-FROZEN model (the
current champion in ``model_registry`` -- selected entirely on
development data, per ``learning/champion_challenger_v2.py``) against
that period.

Mirrors ``backtesting/holdout.py::evaluate_on_holdout``'s structure and
audit-trail discipline exactly, logging to ``forward_paper_access_log``
instead of ``holdout_access_log``. This function trains nothing and
performs no model selection, comparison, or promotion decision -- it only
scores a model that was already chosen before this function is ever
called.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb
import pandas as pd

from backtesting.purged_walk_forward import TARGET_TO_PRED_COL
from core.schemas_v2 import ForwardPaperAccessLog
from database import repository_v2 as repo_v2
from models.evaluate import evaluate_classification, evaluate_regression
from models.predict import predict_batch
from models.registry import get_champion, load_model
from models.train import TARGET_KIND


@dataclass
class ForwardPaperEvaluationResult:
    metrics: dict[str, dict[str, float]]
    predictions: pd.DataFrame
    n_rows: int
    model_version: str
    log_entry: ForwardPaperAccessLog


def evaluate_on_forward_paper(
    con: duckdb.DuckDBPyConnection,
    post_holdout_df: pd.DataFrame,
    feature_cols: list[str],
    model_version: str,
    purpose: str,
    feature_version: str = "fv1",
) -> ForwardPaperEvaluationResult:
    """Loads the ALREADY-TRAINED model artifact for ``model_version`` from
    the registry (never retrains) and scores it once against
    ``post_holdout_df``. Every call is logged, unconditionally, to
    ``forward_paper_access_log``."""
    if post_holdout_df.empty:
        raise ValueError("evaluate_on_forward_paper called with an empty post_holdout_df -- nothing to evaluate")

    boosters, _record = load_model(con, model_version)
    preds = predict_batch(boosters, post_holdout_df, feature_cols, model_version, feature_version)
    pred_df = pd.DataFrame([p.model_dump() for p in preds])

    metrics: dict[str, dict[str, float]] = {}
    if not pred_df.empty:
        merged = pred_df.merge(post_holdout_df[["symbol", "timestamp", *TARGET_KIND.keys()]], on=["symbol", "timestamp"])
        for target_col, kind in TARGET_KIND.items():
            sub = merged.dropna(subset=[target_col])
            if sub.empty:
                continue
            pred_col = TARGET_TO_PRED_COL[target_col]
            if kind == "regression":
                metrics[target_col] = evaluate_regression(sub[target_col], sub[pred_col])
            else:
                metrics[target_col] = evaluate_classification(sub[target_col], sub[pred_col])

    log_entry = ForwardPaperAccessLog(
        accessed_at=datetime.now(UTC),
        purpose=purpose,
        model_version=model_version,
        forward_paper_start=post_holdout_df["timestamp"].min(),
        forward_paper_end=post_holdout_df["timestamp"].max(),
        n_rows=len(post_holdout_df),
        symbols=sorted(post_holdout_df["symbol"].unique().tolist()),
    )
    repo_v2.insert_forward_paper_access_log(con, log_entry)

    return ForwardPaperEvaluationResult(
        metrics=metrics, predictions=pred_df, n_rows=len(post_holdout_df), model_version=model_version,
        log_entry=log_entry,
    )


def load_frozen_champion_for_forward_paper(con: duckdb.DuckDBPyConnection) -> str:
    """The only "model selection" this workflow is allowed to perform:
    reading which model_version is ALREADY the champion (a decision made
    entirely on development data, before this function is ever called).
    Raises if no champion exists yet -- forward-paper evaluation cannot
    proceed without a model that has already cleared development-side
    selection."""
    champion = get_champion(con)
    if champion is None:
        raise ValueError(
            "no champion model exists in model_registry -- forward-paper evaluation requires a model already "
            "selected via development-only evaluation (run evaluate-real/evaluate-development first)"
        )
    return champion["model_version"]
