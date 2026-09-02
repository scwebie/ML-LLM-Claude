"""V0.2 real-data pipeline orchestration (Stage 15).

Plain, Typer-independent functions called by the ``ingest-*``,
``build-real-features``, ``evaluate-real``, and ``real-demo`` commands in
``main.py`` -- kept in their own module so they're directly testable and
importable without going through the CLI layer. ``main.py demo`` (V0.1's
synthetic pipeline) is completely untouched; this module is purely
additive and only ever writes to the V0.2 tables (``core/schemas_v2.py``)
plus the shared, provider-agnostic ``market_observations``/
``feature_snapshots`` tables V0.1 already uses.

Pipeline stages, in the order ``real-demo`` runs them:

1. Ingest prices, fundamentals, macro, news, and (read-only) event
   probabilities for a configured universe -- ingestion keeps running
   through "today" regardless of where the holdout period falls.
2. Build the point-in-time real feature matrix and store it under a
   dedicated ``feature_version`` (``REAL_FEATURE_VERSION``), reusing
   V0.1's ``feature_snapshots`` table verbatim.
3. Join to excess-return targets against the real benchmark (SPY) and
   split into the three temporal regions (``backtesting/holdout.py::
   split_temporal_partitions``): PRE-HOLDOUT development, the FINAL
   holdout, and POST-HOLDOUT data (real ingestion routinely runs past the
   holdout end -- those rows are preserved, never merged into
   development, and never used for any model-selection decision here).
4. Run the purged+embargoed walk-forward evaluator
   (``backtesting/purged_walk_forward.py``) on PRE-HOLDOUT development
   only.
5. Register the latest fold's model and route it through the V0.2
   champion/challenger gate (``learning/champion_challenger_v2.py``),
   which rejects a weak first model rather than auto-promoting it. This
   is the point at which the model is effectively FROZEN for the rest of
   this run -- nothing after this step feeds back into selection.
6. Run the SAME, unmodified Portfolio/Risk/Execution engine V0.1 uses
   (``backtesting/engine.py::run_ml_strategy_backtest``) over the most
   recent development-set fold's validation window -- a DIAGNOSTIC
   backtest, not the final out-of-sample result (it's still evaluated on
   pre-holdout data the walk-forward process already saw the surrounding
   context of).
7. Evaluate the frozen, selected model on the FINAL holdout, exactly
   once, via ``backtesting/holdout.py::evaluate_on_holdout`` -- this,
   not step 6, is the genuine final out-of-sample result, and it never
   feeds back into anything above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import duckdb
import pandas as pd

from backtesting.daily_portfolio import (
    build_daily_rebalanced_portfolio_returns,
    sharpe_audit_report,
)
from backtesting.engine import run_ml_strategy_backtest
from backtesting.holdout import (
    HoldoutConfig,
    HoldoutEvaluationResult,
    assert_no_fold_touches_holdout,
    default_holdout_config,
    evaluate_on_holdout,
    split_temporal_partitions,
)
from backtesting.purged_walk_forward import (
    DEFAULT_EMBARGO_DAYS,
    MAX_TARGET_HORIZON_DAYS,
    TARGET_TO_PRED_COL,
    PurgedFoldResult,
    build_trading_calendar,
    generate_purged_folds,
    run_purged_walk_forward,
)
from backtesting.robustness import (
    permutation_test_ic,
    rank_ic_report,
)
from core.logging import get_logger
from data.real_events import ingest_event_probabilities
from data.real_features import build_real_feature_matrix
from data.real_fundamentals import ingest_fundamentals
from data.real_macro import ingest_macro
from data.real_news import ingest_news
from data.real_prices import REAL_BENCHMARK_SYMBOL, ingest_prices
from data.universe import (
    DEFAULT_REAL_SECTOR_MAP,
    DEFAULT_REAL_UNIVERSE,
    seed_universe_membership,
    universe_with_benchmark,
)
from database import repository as repo
from features.feature_store import load_feature_matrix, store_feature_matrix
from learning.champion_challenger_v2 import run_promotion_cycle_v2
from models.registry import ModelPeriods, get_champion, register_model
from models.train import compute_excess_return_targets, get_feature_columns, prepare_training_frame
from portfolio.allocation import AllocationConfig
from portfolio.risk import RiskLimits

logger = get_logger(__name__)

REAL_FEATURE_VERSION = "real_fv1"
DEFAULT_UNIVERSE_NAME = "real_default"
PRIMARY_TARGET = "excess_return_20d"


def ingest_prices_step(con: duckdb.DuckDBPyConnection, symbols: list[str], start: datetime, end: datetime) -> dict:
    return ingest_prices(con, universe_with_benchmark(symbols), start, end)


def ingest_fundamentals_step(con: duckdb.DuckDBPyConnection, symbols: list[str]) -> dict:
    return ingest_fundamentals(con, symbols)


def ingest_macro_step(con: duckdb.DuckDBPyConnection, start: datetime, end: datetime) -> dict:
    return ingest_macro(con, start, end)


def ingest_news_step(con: duckdb.DuckDBPyConnection, symbols: list[str], start: datetime, end: datetime) -> dict:
    return ingest_news(con, symbols, start, end)


def build_real_features_step(
    con: duckdb.DuckDBPyConnection,
    symbols: list[str] | None = None,
    universe_name: str = DEFAULT_UNIVERSE_NAME,
    universe_start: datetime | None = None,
    sector_map: dict[str, str] | None = None,
    use_llm: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Seeds point-in-time universe membership (idempotent-ish: harmless
    to call again, it just appends another membership row), ingests the
    read-only event-probability signal, builds the point-in-time feature
    matrix, and persists it under ``REAL_FEATURE_VERSION`` via V0.1's
    existing ``feature_snapshots`` table."""
    symbols = symbols or DEFAULT_REAL_UNIVERSE
    sector_map = sector_map or {s: DEFAULT_REAL_SECTOR_MAP.get(s, "OTHER") for s in symbols}
    universe_start = universe_start or datetime(2020, 1, 1)

    seed_universe_membership(con, universe_name, symbols, universe_start)
    event_summary = ingest_event_probabilities(con, symbols, sector_map)

    matrix = build_real_feature_matrix(con, universe_name, symbols, sector_map, use_llm=use_llm)
    n_stored = store_feature_matrix(con, REAL_FEATURE_VERSION, matrix)
    return matrix, {"event_probabilities": event_summary, "matrix_shape": list(matrix.shape), "rows_stored": n_stored}


def _sector_map_for(symbols: list[str], sector_map: dict[str, str] | None) -> dict[str, str]:
    return sector_map or {s: DEFAULT_REAL_SECTOR_MAP.get(s, "OTHER") for s in symbols}


@dataclass
class RealEvaluationResult:
    fold_results: list[PurgedFoldResult]
    development_df: pd.DataFrame
    holdout_df: pd.DataFrame
    feature_cols: list[str]
    champion_model_version: str | None
    promoted: bool
    promotion_rationale: str
    fold_metrics_summary: dict = field(default_factory=dict)
    robustness: dict = field(default_factory=dict)
    post_holdout_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    incumbent_champion_version_before_evaluation: str | None = None
    sharpe_audit: dict = field(default_factory=dict)


def evaluate_real_step(
    con: duckdb.DuckDBPyConnection,
    symbols: list[str] | None = None,
    feature_version: str = REAL_FEATURE_VERSION,
    holdout: HoldoutConfig | None = None,
    horizon_days: int = MAX_TARGET_HORIZON_DAYS,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    initial_train_fraction: float = 0.6,
    validation_fraction: float = 0.15,
    window_mode: str = "expanding",
) -> RealEvaluationResult:
    """Loads the stored real feature matrix, joins to excess-return
    targets against the real benchmark, splits into the three temporal
    regions (PRE-HOLDOUT development / holdout / post-holdout -- holdout
    and post-holdout are NEVER touched here), runs purged+embargoed
    walk-forward on pre-holdout development only, and routes the latest
    fold's model through the V0.2 champion/challenger gate.

    Real ingestion routinely runs past the holdout end (e.g. ``real-demo``
    ingesting through "today"), so ``df`` may well contain rows after
    ``holdout.end_date`` -- those become ``post_holdout_df`` and take no
    part in anything below; they are never merged into development."""
    symbols = symbols or DEFAULT_REAL_UNIVERSE
    holdout = holdout or default_holdout_config()

    matrix = load_feature_matrix(con, feature_version, symbols=symbols)
    if matrix.empty:
        raise ValueError(f"no stored feature matrix for feature_version={feature_version!r} -- run build-real-features first")

    market = repo.get_market_observations(con, symbols=symbols)
    benchmark = repo.get_market_observations(con, symbols=[REAL_BENCHMARK_SYMBOL])
    if benchmark.empty:
        raise ValueError(f"no benchmark ({REAL_BENCHMARK_SYMBOL}) price data -- run ingest-prices first")
    targets = compute_excess_return_targets(market, benchmark)
    df = prepare_training_frame(matrix, targets)
    feature_cols = get_feature_columns(df)

    partition = split_temporal_partitions(df, holdout, horizon_days, embargo_days)
    development_df, holdout_df, post_holdout_df = (
        partition.development_df, partition.holdout_df, partition.post_holdout_df
    )
    if development_df.empty:
        raise ValueError("development set is empty after purge/embargo -- not enough history before the holdout period")

    # A feature column that is (effectively) ENTIRELY missing in the
    # development set -- e.g. fundamentals/macro/news/events were never
    # ingested for this run, or a provider reported UNAVAILABLE for the
    # whole period -- carries no real information and, worse, would make
    # every row fail the per-row dropna used at training time
    # (models/train.py::train_single_target), which would otherwise fail
    # the ENTIRE run over one absent data source. This is "skip feature"
    # (explicitly allowed by the no-fabrication rule), never "fabricate a
    # value" -- a column with meaningfully-often-present data (e.g. a
    # long-lookback technical indicator with an ordinary warm-up period)
    # is left untouched and handled correctly by the existing per-row
    # dropna. The 99% threshold (rather than a stricter ==100%) also
    # catches near-total-but-not-literally-100% missingness -- e.g. a
    # cross-sectional percentile rank that is technically defined for a
    # handful of boundary rows even though its underlying source column
    # (here, an un-ingested fundamentals field) is never actually present.
    max_missing_fraction = 0.99
    missing_fraction = development_df[feature_cols].isna().mean()
    effectively_missing = missing_fraction[missing_fraction > max_missing_fraction].index.tolist()
    if effectively_missing:
        logger.warning("dropping_effectively_missing_feature_columns", n_dropped=len(effectively_missing), columns=effectively_missing)
        feature_cols = [c for c in feature_cols if c not in effectively_missing]

    calendar = build_trading_calendar(development_df["timestamp"])
    n_days = len(calendar)
    initial_train_days = max(1, int(n_days * initial_train_fraction))
    validation_days = max(1, int(n_days * validation_fraction))
    folds = generate_purged_folds(calendar, initial_train_days, validation_days, window_mode=window_mode)
    assert_no_fold_touches_holdout(folds, holdout)

    fold_results = run_purged_walk_forward(development_df, folds, feature_cols, horizon_days, embargo_days)
    if not fold_results:
        return RealEvaluationResult(
            fold_results=[], development_df=development_df, holdout_df=holdout_df, feature_cols=feature_cols,
            champion_model_version=None, promoted=False,
            promotion_rationale="not enough development history to produce a single purged walk-forward fold",
            post_holdout_df=post_holdout_df,
        )

    fold_ics = [r.metrics.get(PRIMARY_TARGET, {}).get("information_coefficient", float("nan")) for r in fold_results]
    fold_ics = [v for v in fold_ics if v == v]
    fold_metrics_summary = {
        "n_folds": len(fold_results),
        "mean_information_coefficient": float(pd.Series(fold_ics).mean()) if fold_ics else float("nan"),
        "per_fold_information_coefficient": fold_ics,
    }

    last_fold = fold_results[-1]
    pred_col = TARGET_TO_PRED_COL[PRIMARY_TARGET]
    eval_frame = last_fold.predictions.merge(
        development_df[["symbol", "timestamp", PRIMARY_TARGET]], on=["symbol", "timestamp"], how="inner"
    )

    # An initial-champion candidate must clear a real backtest-level Sharpe,
    # not just IC (learning/initial_qualification.py) -- compute one from a
    # GENUINELY CHRONOLOGICAL DAILY portfolio return series (V0.3 Stage 2;
    # see backtesting/daily_portfolio.py's module docstring for the audit
    # of why the previous build_quantile_portfolio_returns-based Sharpe was
    # wrong: it sampled overlapping multi-day-forward targets at a daily
    # step and annualised with sqrt(252) as if they were independent daily
    # returns, inflating Sharpe by roughly sqrt(horizon_days)x).
    metrics = dict(last_fold.metrics)
    reg_metrics = dict(metrics.get(PRIMARY_TARGET, {}))
    sharpe_audit: dict = {}
    daily_portfolio_returns = build_daily_rebalanced_portfolio_returns(last_fold.predictions, market, pred_col)
    sharpe_audit = sharpe_audit_report(daily_portfolio_returns)
    if sharpe_audit["n_observations"] > 0:
        reg_metrics["sharpe_ratio"] = sharpe_audit["gross_sharpe"]
    metrics[PRIMARY_TARGET] = reg_metrics

    periods = ModelPeriods(
        training_start=last_fold.fold.train_start,
        training_end=last_fold.fold.validation_start - pd.Timedelta(days=1),
        validation_start=last_fold.fold.validation_start, validation_end=last_fold.fold.validation_end,
    )

    # The incumbent MUST be loaded before the challenger is registered --
    # never after (see champion_challenger_v2.run_promotion_cycle_v2's
    # docstring for why this ordering is a structural, not just
    # procedural, guarantee against self-comparison).
    champion_record = get_champion(con)
    model_version = register_model(con, last_fold.trained, feature_version, periods, metrics, role="CHALLENGER")

    promoted, rationale = run_promotion_cycle_v2(
        con, model_version, metrics, eval_frame, champion_record,
        challenger_validation_end=periods.validation_end, target_col=PRIMARY_TARGET, pred_col=pred_col,
    )

    robustness = {}
    if not eval_frame.empty:
        robustness["rank_ic"] = rank_ic_report(eval_frame, PRIMARY_TARGET, pred_col)
        robustness["permutation_test"] = permutation_test_ic(eval_frame[PRIMARY_TARGET], eval_frame[pred_col], n_permutations=500)

    return RealEvaluationResult(
        fold_results=fold_results, development_df=development_df, holdout_df=holdout_df, feature_cols=feature_cols,
        champion_model_version=model_version, promoted=promoted, promotion_rationale=rationale,
        fold_metrics_summary=fold_metrics_summary, robustness=robustness, post_holdout_df=post_holdout_df,
        incumbent_champion_version_before_evaluation=champion_record["model_version"] if champion_record else None,
        sharpe_audit=sharpe_audit,
    )


@dataclass
class RealDemoResult:
    evaluation: RealEvaluationResult
    n_fills: int
    n_rejected_orders: int
    rejection_reason_codes: list[str]
    backtest_period: tuple[str, str] | None
    holdout_evaluation: HoldoutEvaluationResult | None = None


def run_real_demo(
    con: duckdb.DuckDBPyConnection,
    symbols: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    initial_cash: float = 1_000_000.0,
    risk_limits: RiskLimits | None = None,
    allocation_config: AllocationConfig | None = None,
    use_llm: bool = False,
    skip_ingestion: bool = False,
) -> RealDemoResult:
    """End-to-end V0.2 real-data run: ingest -> build-real-features ->
    evaluate-real (pre-holdout development only) -> a diagnostic paper-
    trading backtest through the SAME unmodified Portfolio/Risk/Execution
    engine V0.1 uses, over the latest development-set walk-forward fold's
    validation window -- THEN, only once model selection is completely
    finished and the model is frozen, exactly one formal evaluation on the
    untouched final holdout via ``backtesting.holdout.evaluate_on_holdout``.
    The holdout result, not the development-fold backtest, is the genuine
    final out-of-sample result; nothing here ever tunes anything in
    response to it.

    Ingestion (``start``..``end``) may run through "today" regardless of
    where the holdout period falls -- ``evaluate_real_step`` partitions
    the resulting data into pre-holdout development / holdout / post-
    holdout (``backtesting.holdout.split_temporal_partitions``), and only
    the pre-holdout region is ever used for model selection.

    ``skip_ingestion=True`` lets a caller (tests, or a re-run against
    already-ingested data) skip the four ``ingest_*`` network calls and
    go straight to feature-building against whatever is already in the
    database -- this is how the risk-engine integration test exercises
    this pipeline without live network access.
    """
    symbols = symbols or DEFAULT_REAL_UNIVERSE
    sector_map = _sector_map_for(symbols, None)
    start = start or datetime(2020, 1, 1)
    end = end or datetime.now()

    if not skip_ingestion:
        ingest_prices_step(con, symbols, start, end)
        ingest_fundamentals_step(con, symbols)
        ingest_macro_step(con, start, end)
        ingest_news_step(con, symbols, start, end)

    build_real_features_step(con, symbols, DEFAULT_UNIVERSE_NAME, start, sector_map, use_llm)
    evaluation = evaluate_real_step(con, symbols)

    if evaluation.champion_model_version is None or not evaluation.fold_results:
        return RealDemoResult(evaluation=evaluation, n_fills=0, n_rejected_orders=0, rejection_reason_codes=[], backtest_period=None)

    champion = get_champion(con)
    model_version = champion["model_version"] if champion is not None else evaluation.champion_model_version
    last_fold = evaluation.fold_results[-1]

    # --- DIAGNOSTIC: paper-trading backtest over the last development
    # fold's own validation window. This is still pre-holdout data the
    # walk-forward process already saw the surrounding context of -- it
    # demonstrates the full Portfolio/Risk/Execution pipeline working end
    # to end (including the risk engine approving AND rejecting orders),
    # but it is NOT the final out-of-sample result. See the holdout
    # evaluation below for that.
    test_df = evaluation.development_df[
        (evaluation.development_df["timestamp"] >= last_fold.fold.validation_start)
        & (evaluation.development_df["timestamp"] <= last_fold.fold.validation_end)
    ]
    market = repo.get_market_observations(con, symbols=symbols)

    run_id = f"real_demo_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    result = run_ml_strategy_backtest(
        con, run_id, test_df, market, last_fold.trained.boosters, evaluation.feature_cols, model_version,
        REAL_FEATURE_VERSION, symbols, sector_map, initial_cash=initial_cash,
        risk_limits=risk_limits or RiskLimits(), allocation_config=allocation_config or AllocationConfig(),
    )
    reason_codes = sorted({code.value for order in result.rejected_orders for code in order.risk_reason_codes})
    backtest_period = (
        (str(test_df["timestamp"].min()), str(test_df["timestamp"].max())) if not test_df.empty else None
    )

    # --- FINAL: the model is frozen (it is exactly the model that just
    # went through champion qualification above -- nothing is retrained
    # or re-selected here) and evaluated on the untouched holdout exactly
    # once. This is the genuine final out-of-sample result. Every call is
    # logged to holdout_access_log regardless of how many times real-demo
    # itself is re-run -- the audit trail is the source of truth for how
    # many times the holdout was actually touched.
    holdout_evaluation = None
    if not evaluation.holdout_df.empty:
        holdout_evaluation = evaluate_on_holdout(
            con, last_fold.trained, evaluation.holdout_df, evaluation.feature_cols, model_version,
            purpose="real-demo CLI: final formal out-of-sample evaluation after model selection was completed",
            feature_version=REAL_FEATURE_VERSION,
        )

    return RealDemoResult(
        evaluation=evaluation, n_fills=len(result.fills), n_rejected_orders=len(result.rejected_orders),
        rejection_reason_codes=reason_codes, backtest_period=backtest_period, holdout_evaluation=holdout_evaluation,
    )
