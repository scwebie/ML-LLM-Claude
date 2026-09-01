"""Purged + embargoed + nested walk-forward evaluation (Stage 11, V0.2).

Extends the plain expanding-window walk-forward in
``backtesting/walk_forward.py`` (V0.1, left completely untouched) for the
real-data feature matrix, whose targets look up to 20 trading days into
the future (``excess_return_20d`` / ``positive_20d`` -- see
``models/train.py``).

A plain "train rows before validation_start, validate after" split is not
safe once targets have a multi-day forward-looking realization window: a
training row timestamped strictly before the validation start can still
have its *target* window extend into the validation period, silently
leaking validation-period price action into the training set. This
module adds the standard purged + embargoed cross-validation machinery
(Lopez de Prado, "Advances in Financial Machine Learning"):

* PURGE -- drop any candidate training row whose target-realization
  window ``[timestamp, timestamp + horizon_days]`` (in trading days, not
  calendar days) overlaps the evaluation window.
* EMBARGO -- additionally drop a configurable buffer of trading days
  immediately following the evaluation window from ever being used as
  training data, guarding against residual serial-correlation leakage.
* NESTED INNER CV -- hyperparameters are selected using an inner
  purged+embargoed split carved *only* out of the outer training window;
  the outer validation window is never seen until the final, single
  evaluation of the selected model.

Everything here reasons in trading-day units on the real trading
calendar derived from the data itself (not fixed calendar-day offsets),
since ``horizon_days`` is defined in trading days
(``close.shift(-5)`` / ``close.shift(-20)`` in ``models/train.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from models.evaluate import evaluate_classification, evaluate_regression
from models.predict import predict_batch
from models.train import (
    DEFAULT_HYPERPARAMETERS,
    TARGET_KIND,
    TrainedModels,
    get_feature_columns,
    train_all_targets,
)

# The longest-horizon target is excess_return_20d / positive_20d (see
# models/train.py::TARGET_KIND) -- purge must cover its full realization
# window or a training row's target could straddle the validation start.
MAX_TARGET_HORIZON_DAYS = 20

# Buffer of trading days excluded after every evaluation window, on top of
# the horizon-based purge. Matches the shorter 5-day target horizon: even
# after purging, rows just past an evaluation window can carry residual
# serial correlation into a subsequent fold's training set.
DEFAULT_EMBARGO_DAYS = 5

TARGET_TO_PRED_COL = {
    "excess_return_5d": "predicted_excess_return_5d",
    "excess_return_20d": "predicted_excess_return_20d",
    "positive_5d": "probability_positive_5d",
    "positive_20d": "probability_positive_20d",
}


def build_trading_calendar(timestamps: pd.Series) -> pd.DatetimeIndex:
    """The sorted, deduplicated set of trading days present in the data --
    the basis for all trading-day (not calendar-day) offset arithmetic
    below."""
    return pd.DatetimeIndex(sorted(pd.Series(timestamps).unique()))


def _trading_day_offset(ts: pd.Timestamp, calendar: pd.DatetimeIndex, steps: int) -> pd.Timestamp:
    """The trading-calendar date ``steps`` trading days after ``ts``,
    clamped to the last known calendar date."""
    idx = calendar.searchsorted(ts)
    target_idx = min(idx + steps, len(calendar) - 1)
    return calendar[target_idx]


def compute_purge_embargo_mask(
    timestamps: pd.Series,
    calendar: pd.DatetimeIndex,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    horizon_days: int = MAX_TARGET_HORIZON_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> pd.Series:
    """Boolean mask, aligned positionally to ``timestamps``, True for rows
    that remain ELIGIBLE to be used as training data for an evaluation
    window ``[eval_start, eval_end]``.

    A row is excluded (mask False) if either:
      1. its own timestamp falls inside the evaluation window, or its
         target-realization window ``[ts, ts + horizon_days]`` (trading
         days) overlaps the evaluation window at all -- PURGE; or
      2. it falls within ``embargo_days`` trading days strictly after
         ``eval_end`` -- EMBARGO.

    Rows with timestamps before the evaluation window whose target window
    does not reach ``eval_start`` are unaffected -- this is what lets an
    expanding-window fold keep using most of its historical training data.
    """
    ts = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True))
    embargo_end = _trading_day_offset(pd.Timestamp(eval_end), calendar, embargo_days)

    target_end = ts.map(lambda t: _trading_day_offset(t, calendar, horizon_days))

    overlaps_eval = (target_end >= eval_start) & (ts <= eval_end)
    inside_embargo = (ts > eval_end) & (ts <= embargo_end)

    eligible = ~(overlaps_eval | inside_embargo)
    eligible.index = pd.Series(timestamps).index
    return eligible


@dataclass(frozen=True)
class PurgedFold:
    fold_id: int
    train_start: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    window_mode: str  # "expanding" | "rolling"


def generate_purged_folds(
    calendar: pd.DatetimeIndex,
    initial_train_days: int,
    validation_days: int,
    step_days: int | None = None,
    window_mode: str = "expanding",
    rolling_train_days: int | None = None,
) -> list[PurgedFold]:
    """Generate folds by trading-day counts on ``calendar`` (not calendar
    dates), so window sizes are exact regardless of holidays/weekends.

    ``window_mode="expanding"`` (V0.1's default behaviour): every fold's
    training window starts at the beginning of the calendar.
    ``window_mode="rolling"``: every fold's training window is the most
    recent ``rolling_train_days`` trading days -- required by section
    24-27 of the V0.2 spec alongside expanding-window evaluation, since a
    rolling window tests whether the model needs old data at all and is
    more robust to regime change than an ever-growing training set.
    """
    if window_mode not in ("expanding", "rolling"):
        raise ValueError(f"window_mode must be 'expanding' or 'rolling', got {window_mode!r}")
    if window_mode == "rolling" and not rolling_train_days:
        raise ValueError("rolling_train_days is required when window_mode='rolling'")
    step_days = step_days or validation_days

    folds: list[PurgedFold] = []
    fold_id = 0
    train_end_idx = initial_train_days - 1
    while True:
        val_start_idx = train_end_idx + 1
        val_end_idx = val_start_idx + validation_days - 1
        if val_end_idx >= len(calendar):
            break
        train_start_idx = 0 if window_mode == "expanding" else max(0, train_end_idx - rolling_train_days + 1)
        folds.append(
            PurgedFold(
                fold_id=fold_id,
                train_start=calendar[train_start_idx],
                validation_start=calendar[val_start_idx],
                validation_end=calendar[val_end_idx],
                window_mode=window_mode,
            )
        )
        fold_id += 1
        train_end_idx += step_days
    return folds


def _predict_and_evaluate(
    trained: TrainedModels, val_df: pd.DataFrame, feature_cols: list[str], model_version: str, feature_version: str
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    preds = predict_batch(trained.boosters, val_df, feature_cols, model_version, feature_version)
    pred_df = pd.DataFrame([p.model_dump() for p in preds])

    metrics: dict[str, dict[str, float]] = {}
    if not pred_df.empty:
        merged = pred_df.merge(val_df[["symbol", "timestamp", *TARGET_KIND.keys()]], on=["symbol", "timestamp"])
        for target_col, kind in TARGET_KIND.items():
            sub = merged.dropna(subset=[target_col])
            if sub.empty:
                continue
            pred_col = TARGET_TO_PRED_COL[target_col]
            if kind == "regression":
                metrics[target_col] = evaluate_regression(sub[target_col], sub[pred_col])
            else:
                metrics[target_col] = evaluate_classification(sub[target_col], sub[pred_col])
    return pred_df, metrics


def _score(metrics: dict[str, dict[str, float]], primary_target: str, primary_metric: str) -> float:
    value = metrics.get(primary_target, {}).get(primary_metric, float("nan"))
    return float(value) if value == value else float("-inf")  # NaN-safe: NaN never wins a comparison


@dataclass
class PurgedFoldResult:
    fold: PurgedFold
    trained: TrainedModels
    metrics: dict[str, dict[str, float]]
    predictions: pd.DataFrame
    n_train_rows: int
    n_purged_or_embargoed: int


def build_outer_train_eligibility(
    df: pd.DataFrame,
    fold: PurgedFold,
    calendar: pd.DatetimeIndex,
    horizon_days: int = MAX_TARGET_HORIZON_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
) -> tuple[pd.Series, pd.Series]:
    """Returns ``(candidate_mask, eligible_mask)`` for ``fold``'s training
    window: ``candidate_mask`` is the naive "before validation_start, on
    or after train_start" window; ``eligible_mask`` additionally applies
    purge+embargo. ``train_mask = candidate_mask & eligible_mask``."""
    candidate_mask = (df["timestamp"] >= fold.train_start) & (df["timestamp"] < fold.validation_start)
    eligible_mask = compute_purge_embargo_mask(
        df["timestamp"], calendar, fold.validation_start, fold.validation_end, horizon_days, embargo_days
    )
    return candidate_mask, eligible_mask


def run_purged_walk_forward(
    df: pd.DataFrame,
    folds: list[PurgedFold],
    feature_cols: list[str] | None = None,
    horizon_days: int = MAX_TARGET_HORIZON_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    hyperparameters: dict[str, dict] | None = None,
    model_version_prefix: str = "pwf",
    feature_version: str = "fv1",
) -> list[PurgedFoldResult]:
    """Train + evaluate one model per fold, with purge+embargo applied to
    every fold's training window. Mirrors
    ``backtesting/walk_forward.py::run_walk_forward``'s structure and
    return shape, but is the leakage-safe variant required for the real
    (multi-day-horizon-target) feature matrix."""
    feature_cols = feature_cols or get_feature_columns(df)
    calendar = build_trading_calendar(df["timestamp"])
    results: list[PurgedFoldResult] = []

    for fold in folds:
        candidate_mask, eligible_mask = build_outer_train_eligibility(df, fold, calendar, horizon_days, embargo_days)
        train_mask = candidate_mask & eligible_mask
        val_mask = (df["timestamp"] >= fold.validation_start) & (df["timestamp"] <= fold.validation_end)

        train_df = df.loc[train_mask]
        val_df = df.loc[val_mask]

        # Hard runtime leakage guard, independent of the mask construction
        # above: assert no training row's target-realization window
        # reaches the earliest validation timestamp actually used.
        if len(train_df) and len(val_df):
            target_ends = train_df["timestamp"].map(lambda t: _trading_day_offset(t, calendar, horizon_days))
            assert (target_ends < val_df["timestamp"].min()).all(), (
                f"fold {fold.fold_id}: leakage detected -- a training row's target realization "
                "window extends into the validation period despite purging"
            )

        if train_df.empty or val_df.empty:
            continue

        trained = train_all_targets(train_df, val_df, feature_cols, hyperparameters)
        model_version = f"{model_version_prefix}_fold{fold.fold_id}"
        pred_df, metrics = _predict_and_evaluate(trained, val_df, feature_cols, model_version, feature_version)

        results.append(
            PurgedFoldResult(
                fold=fold, trained=trained, metrics=metrics, predictions=pred_df,
                n_train_rows=len(train_df),
                n_purged_or_embargoed=int((candidate_mask & ~eligible_mask).sum()),
            )
        )
    return results


def default_hyperparameter_grid(
    num_leaves_options: tuple[int, ...] = (7, 15, 31),
    learning_rate_options: tuple[float, ...] = (0.03, 0.05,),
) -> list[dict[str, dict]]:
    """A small grid of candidate hyperparameter sets for nested inner-CV
    model selection, built by varying ``num_leaves``/``learning_rate``
    around ``models.train.DEFAULT_HYPERPARAMETERS``."""
    candidates = []
    for num_leaves in num_leaves_options:
        for learning_rate in learning_rate_options:
            candidates.append(
                {
                    kind: {**params, "num_leaves": num_leaves, "learning_rate": learning_rate}
                    for kind, params in DEFAULT_HYPERPARAMETERS.items()
                }
            )
    return candidates


@dataclass
class NestedFoldResult(PurgedFoldResult):
    selected_hyperparameters: dict[str, dict] = field(default_factory=dict)
    inner_cv_scores: list[float] = field(default_factory=list)
    n_inner_folds_used: int = 0


def run_nested_purged_walk_forward(
    df: pd.DataFrame,
    outer_folds: list[PurgedFold],
    hyperparameter_candidates: list[dict[str, dict]] | None = None,
    feature_cols: list[str] | None = None,
    horizon_days: int = MAX_TARGET_HORIZON_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    n_inner_folds: int = 2,
    inner_validation_days: int = 42,
    primary_target: str = "excess_return_20d",
    primary_metric: str = "information_coefficient",
    model_version_prefix: str = "npwf",
    feature_version: str = "fv1",
) -> list[NestedFoldResult]:
    """Nested purged+embargoed walk-forward: for every outer fold, select
    hyperparameters using ONLY an inner purged+embargoed split carved out
    of that fold's outer-training window, then retrain on the full outer
    training window with the selected hyperparameters and evaluate -- for
    the first and only time -- on the outer validation window.

    This is what makes model selection itself leakage-safe: without it, a
    single walk-forward run picked to maximize the validation-fold score
    is itself a (subtle) form of look-ahead bias, since the "test" fold
    influenced the choice of model.
    """
    feature_cols = feature_cols or get_feature_columns(df)
    hyperparameter_candidates = hyperparameter_candidates or default_hyperparameter_grid()
    calendar = build_trading_calendar(df["timestamp"])
    results: list[NestedFoldResult] = []

    for fold in outer_folds:
        candidate_mask, eligible_mask = build_outer_train_eligibility(df, fold, calendar, horizon_days, embargo_days)
        outer_train_mask = candidate_mask & eligible_mask
        outer_train_df = df.loc[outer_train_mask]
        val_mask = (df["timestamp"] >= fold.validation_start) & (df["timestamp"] <= fold.validation_end)
        val_df = df.loc[val_mask]
        if outer_train_df.empty or val_df.empty:
            continue

        inner_calendar = build_trading_calendar(outer_train_df["timestamp"])
        min_inner_train_days = max(1, len(inner_calendar) - n_inner_folds * inner_validation_days)
        inner_folds = generate_purged_folds(
            inner_calendar,
            initial_train_days=min_inner_train_days,
            validation_days=inner_validation_days,
            window_mode="expanding",
        )[:n_inner_folds]

        best_candidate = hyperparameter_candidates[0]
        best_mean_score = float("-inf")
        candidate_scores: list[float] = []
        if inner_folds:
            for candidate in hyperparameter_candidates:
                inner_results = run_purged_walk_forward(
                    outer_train_df, inner_folds, feature_cols, horizon_days, embargo_days,
                    hyperparameters=candidate, model_version_prefix="inner", feature_version=feature_version,
                )
                scores = [_score(r.metrics, primary_target, primary_metric) for r in inner_results]
                mean_score = float(np.mean(scores)) if scores else float("-inf")
                candidate_scores.append(mean_score)
                if mean_score > best_mean_score:
                    best_mean_score = mean_score
                    best_candidate = candidate
        else:
            candidate_scores = [float("nan")] * len(hyperparameter_candidates)

        # Final, single fit on the FULL outer training window -- the outer
        # validation fold has not influenced hyperparameter choice at all.
        trained = train_all_targets(outer_train_df, val_df, feature_cols, best_candidate)
        model_version = f"{model_version_prefix}_fold{fold.fold_id}"
        pred_df, metrics = _predict_and_evaluate(trained, val_df, feature_cols, model_version, feature_version)

        results.append(
            NestedFoldResult(
                fold=fold, trained=trained, metrics=metrics, predictions=pred_df,
                n_train_rows=len(outer_train_df),
                n_purged_or_embargoed=int((candidate_mask & ~eligible_mask).sum()),
                selected_hyperparameters=best_candidate,
                inner_cv_scores=candidate_scores,
                n_inner_folds_used=len(inner_folds),
            )
        )
    return results
