"""market-agent-lab CLI entry point.

    uv run python main.py demo

runs the full Version 0.1 pipeline end to end: synthetic data generation,
feature engineering + Feature Store, walk-forward model training,
champion/challenger promotion, event-driven paper-trading backtest against
three benchmarks, outcome labelling, and a printed performance report.

PAPER-TRADING / SIMULATION ONLY. See docs/architecture.md for the
enumerated safety boundaries.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import typer

from backtesting.engine import (
    buy_and_hold_benchmark,
    equal_weight_benchmark,
    momentum_benchmark,
    run_ml_strategy_backtest,
)
from backtesting.metrics import compute_all_metrics
from backtesting.walk_forward import generate_expanding_folds, run_walk_forward
from core.config import settings
from core.logging import configure_logging, get_logger
from data import synthetic as synthetic_data
from data.market_data import get_benchmark, get_ohlcv, load_all_synthetic_data
from database.db import get_connection
from features.feature_store import (
    DEFAULT_FEATURE_VERSION,
    build_feature_matrix,
    load_feature_matrix,
    store_feature_matrix,
)
from learning.champion_challenger import run_promotion_cycle
from learning.outcomes import label_pending_outcomes
from models.registry import ModelPeriods, get_champion, load_model, register_model
from models.train import compute_excess_return_targets, get_feature_columns, prepare_training_frame
from portfolio.allocation import AllocationConfig
from portfolio.risk import RiskLimits

app = typer.Typer(help="market-agent-lab: paper-trading-only multi-agent research system (v0.1)")
logger = get_logger(__name__)

SECTOR_MAP = dict(synthetic_data.SYMBOLS)


def _section(title: str) -> None:
    typer.echo("\n" + "=" * 78)
    typer.echo(f"  {title}")
    typer.echo("=" * 78)


@app.command()
def demo(
    symbols: str = typer.Option(None, help="Comma-separated symbols; default = full synthetic universe (10 symbols)"),
    initial_train_end: str = typer.Option("2019-12-31", help="End of the initial walk-forward training window"),
    walk_forward_end: str = typer.Option("2022-12-31", help="Last validation year covered by walk-forward folds"),
    test_end: str = typer.Option("2023-12-31", help="End of the final held-out backtest / paper-trading period"),
    initial_cash: float = typer.Option(1_000_000.0, help="Starting paper-trading cash"),
    use_llm: bool = typer.Option(False, help="Enable optional LLM narrative enhancement for agents (requires OPENAI_API_KEY)"),
    db_path: str = typer.Option(None, help="Override the DuckDB file path"),
) -> None:
    """Run the full Version 0.1 pipeline: data -> features -> model -> walk-forward
    -> champion/challenger -> backtest -> paper execution -> report."""
    configure_logging()
    started_at = datetime.now(UTC)
    con = get_connection(db_path)
    symbol_list = symbols.split(",") if symbols else [s for s, _ in synthetic_data.SYMBOLS]

    _section("PHASE 1-2: Synthetic data + technical/fundamental/macro feature engineering")
    counts = load_all_synthetic_data(
        con, seed=settings.synthetic_seed, start_date=settings.synthetic_start_date,
        end_date=settings.synthetic_end_date, out_dir=settings.data_store_dir / "raw" / "synthetic",
    )
    typer.echo(f"Loaded synthetic data: {counts}")
    typer.echo(f"Universe: {symbol_list}")

    _section("PHASE 3: Research agents + PHASE 7: Feature Store")
    matrix = build_feature_matrix(con, symbols=symbol_list, use_llm=use_llm, persist_agent_reports=True)
    n_stored = store_feature_matrix(con, DEFAULT_FEATURE_VERSION, matrix)
    typer.echo(f"Stored {n_stored} feature rows under feature_version={DEFAULT_FEATURE_VERSION}")
    matrix = load_feature_matrix(con, DEFAULT_FEATURE_VERSION, symbols=symbol_list)

    market = get_ohlcv(con, symbols=symbol_list)
    benchmark = get_benchmark(con)
    targets = compute_excess_return_targets(market, benchmark)
    df = prepare_training_frame(matrix, targets)
    feature_cols = get_feature_columns(df)
    typer.echo(f"Joined feature+target frame: {df.shape}, {len(feature_cols)} features")

    _section("PHASE 4-5: LightGBM alpha model + expanding-window walk-forward validation")
    folds = generate_expanding_folds(
        data_start=settings.synthetic_start_date, initial_train_end=initial_train_end,
        overall_end=walk_forward_end, validation_years=1,
    )
    typer.echo(f"Generated {len(folds)} walk-forward folds:")
    for f in folds:
        typer.echo(f"  fold {f.fold_id}: train [{f.train_start.date()} -> {f.train_end.date()}]  validate [{f.validation_start.date()} -> {f.validation_end.date()}]")

    fold_results = run_walk_forward(df, folds, feature_cols, feature_version=DEFAULT_FEATURE_VERSION)

    _section("PHASE 10: Champion/challenger promotion (one candidate per walk-forward fold)")
    for fold_result in fold_results:
        periods = ModelPeriods(
            training_start=fold_result.fold.train_start, training_end=fold_result.fold.train_end,
            validation_start=fold_result.fold.validation_start, validation_end=fold_result.fold.validation_end,
        )
        model_version = register_model(
            con, fold_result.trained, DEFAULT_FEATURE_VERSION, periods, fold_result.metrics, role="CHALLENGER"
        )
        challenger_metrics = {
            "excess_return_5d": fold_result.metrics.get("excess_return_5d", {}),
            "positive_5d": fold_result.metrics.get("positive_5d", {}),
            "backtest": {"max_drawdown": 0.0},  # per-fold drawdown not computed at walk-forward stage; see docs/model_design.md
        }
        promoted, rationale = run_promotion_cycle(con, model_version, challenger_metrics)
        typer.echo(f"  fold {fold_result.fold.fold_id} -> {model_version}: {'PROMOTED' if promoted else 'rejected'} ({rationale})")

    champion = get_champion(con)
    if champion is None:
        typer.echo("No champion was promoted -- aborting demo.", err=True)
        raise typer.Exit(code=1)
    champion_version = champion["model_version"]
    boosters, champion_record = load_model(con, champion_version)
    typer.echo(f"\nChampion model: {champion_version}")

    _section("PHASE 6-9: Held-out backtest (Portfolio Decision Engine -> Risk Engine -> Paper Execution Engine)")
    test_df = df[(df["timestamp"] > pd.Timestamp(walk_forward_end)) & (df["timestamp"] <= pd.Timestamp(test_end))]
    typer.echo(f"Backtest period: {test_df['timestamp'].min()} -> {test_df['timestamp'].max()} ({test_df.shape[0]} rows)")

    run_id = f"demo_{started_at.strftime('%Y%m%dT%H%M%S')}"
    result = run_ml_strategy_backtest(
        con, run_id, test_df, market, boosters, feature_cols, champion_version, DEFAULT_FEATURE_VERSION,
        symbol_list, SECTOR_MAP, initial_cash=initial_cash,
        risk_limits=RiskLimits(), allocation_config=AllocationConfig(),
    )
    n_evaluated = len(result.fills) + len(result.rejected_orders)
    typer.echo(f"Orders evaluated by risk engine: {n_evaluated}  Fills: {len(result.fills)}  Risk-rejected: {len(result.rejected_orders)}")

    _section("PHASE 6: Benchmarks (buy-and-hold, equal-weight, simple momentum)")
    test_market = market[(market["timestamp"] > pd.Timestamp(walk_forward_end)) & (market["timestamp"] <= pd.Timestamp(test_end))]
    bh_equity = buy_and_hold_benchmark(test_market, symbol_list, initial_cash)
    ew_equity = equal_weight_benchmark(test_market, symbol_list, initial_cash)
    mom_equity = momentum_benchmark(test_market, symbol_list, initial_cash)

    ml_metrics = compute_all_metrics(
        result.equity_curve, bh_equity, result.trade_pnls, result.traded_notional,
        result.gross_exposure_series, result.holding_periods,
    )
    bh_metrics = compute_all_metrics(bh_equity, bh_equity, [], pd.Series(dtype=float), pd.Series(dtype=float), [])
    ew_metrics = compute_all_metrics(ew_equity, bh_equity, [], pd.Series(dtype=float), pd.Series(dtype=float), [])
    mom_metrics = compute_all_metrics(mom_equity, bh_equity, [], pd.Series(dtype=float), pd.Series(dtype=float), [])

    _section("PHASE 10 (cont.): Outcome labelling for retraining")
    labelled = label_pending_outcomes(con, market, benchmark, as_of=pd.Timestamp(test_end))
    typer.echo(f"Labelled {labelled} realised outcomes from this run's predictions")

    _section("RESULTS")
    report = {
        "run_id": run_id,
        "champion_model_version": champion_version,
        "universe": symbol_list,
        "backtest_period": [str(test_df["timestamp"].min()), str(test_df["timestamp"].max())],
        "strategy_metrics": ml_metrics,
        "buy_and_hold_metrics": bh_metrics,
        "equal_weight_metrics": ew_metrics,
        "momentum_metrics": mom_metrics,
        "n_fills": len(result.fills),
        "n_rejected_orders": len(result.rejected_orders),
        "n_predictions": len(result.predictions),
        "n_outcomes_labelled": labelled,
    }
    results_dir = settings.data_store_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{run_id}.json"
    results_path.write_text(json.dumps(report, indent=2, default=str))

    def _fmt(m: dict) -> str:
        return (
            f"total_return={m['total_return']:.2%}  cagr={m['cagr']:.2%}  sharpe={m['sharpe_ratio']:.2f}  "
            f"max_dd={m['max_drawdown']:.2%}  calmar={m['calmar_ratio']:.2f}"
        )

    typer.echo(f"Strategy   : {_fmt(ml_metrics)}")
    typer.echo(f"Buy&Hold   : {_fmt(bh_metrics)}")
    typer.echo(f"EqualWeight: {_fmt(ew_metrics)}")
    typer.echo(f"Momentum   : {_fmt(mom_metrics)}")
    typer.echo(f"\nFull report saved to {results_path}")

    if result.predictions:
        p = result.predictions[0]
        typer.echo(f"\nExample prediction: {p.model_dump_json(indent=2)}")
    if result.rejected_orders:
        r = result.rejected_orders[0]
        typer.echo(f"\nExample rejected order: symbol={r.symbol} side={r.side} reasons={[c.value for c in r.risk_reason_codes]}")
    if result.fills:
        fl = result.fills[0]
        typer.echo(f"\nExample fill: {fl.model_dump_json(indent=2)}")

    typer.echo(f"\nDemo completed in {(datetime.now(UTC) - started_at).total_seconds():.1f}s")


def _parse_date(value: str | None, default: datetime) -> datetime:
    return datetime.fromisoformat(value) if value else default


def _parse_symbols(value: str | None) -> list[str] | None:
    return value.split(",") if value else None


@app.command()
def ingest_prices(
    symbols: str = typer.Option(None, help="Comma-separated symbols; default = DEFAULT_REAL_UNIVERSE"),
    start: str = typer.Option(None, help="ISO start date, default 2020-01-01"),
    end: str = typer.Option(None, help="ISO end date, default today"),
    db_path: str = typer.Option(None, help="Override the DuckDB file path"),
) -> None:
    """V0.2: ingest real equity prices (Yahoo Finance + StockAnalysis.com, reconciled) for a universe."""
    import real_pipeline as rp

    con = get_connection(db_path)
    symbol_list = _parse_symbols(symbols) or rp.DEFAULT_REAL_UNIVERSE
    result = rp.ingest_prices_step(con, symbol_list, _parse_date(start, datetime(2020, 1, 1)), _parse_date(end, datetime.now()))
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def ingest_fundamentals(
    symbols: str = typer.Option(None, help="Comma-separated symbols; default = DEFAULT_REAL_UNIVERSE"),
    db_path: str = typer.Option(None, help="Override the DuckDB file path"),
) -> None:
    """V0.2: ingest real SEC EDGAR fundamentals (point-in-time, publication-timestamped) for a universe."""
    import real_pipeline as rp

    con = get_connection(db_path)
    symbol_list = _parse_symbols(symbols) or rp.DEFAULT_REAL_UNIVERSE
    result = rp.ingest_fundamentals_step(con, symbol_list)
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def ingest_macro(
    start: str = typer.Option(None, help="ISO start date, default 2020-01-01"),
    end: str = typer.Option(None, help="ISO end date, default today"),
    db_path: str = typer.Option(None, help="Override the DuckDB file path"),
) -> None:
    """V0.2: ingest real macro data (FRED, BLS, Treasury Fiscal Data; BEA reports UNAVAILABLE without an API key)."""
    import real_pipeline as rp

    con = get_connection(db_path)
    result = rp.ingest_macro_step(con, _parse_date(start, datetime(2020, 1, 1)), _parse_date(end, datetime.now()))
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def ingest_news(
    symbols: str = typer.Option(None, help="Comma-separated symbols; default = DEFAULT_REAL_UNIVERSE"),
    start: str = typer.Option(None, help="ISO start date, default 2020-01-01"),
    end: str = typer.Option(None, help="ISO end date, default today"),
    db_path: str = typer.Option(None, help="Override the DuckDB file path"),
) -> None:
    """V0.2: ingest real news (SEC 8-K item-classified events; other sources disabled by default -- see docs/data_sources.md)."""
    import real_pipeline as rp

    con = get_connection(db_path)
    symbol_list = _parse_symbols(symbols) or rp.DEFAULT_REAL_UNIVERSE
    result = rp.ingest_news_step(con, symbol_list, _parse_date(start, datetime(2020, 1, 1)), _parse_date(end, datetime.now()))
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command()
def build_real_features(
    symbols: str = typer.Option(None, help="Comma-separated symbols; default = DEFAULT_REAL_UNIVERSE"),
    universe_name: str = typer.Option("real_default", help="Point-in-time universe name to seed/use"),
    use_llm: bool = typer.Option(False, help="Enable optional LLM narrative enhancement for agents"),
    db_path: str = typer.Option(None, help="Override the DuckDB file path"),
) -> None:
    """V0.2: build the point-in-time real feature matrix from already-ingested data and store it."""
    import real_pipeline as rp

    con = get_connection(db_path)
    symbol_list = _parse_symbols(symbols) or rp.DEFAULT_REAL_UNIVERSE
    _matrix, summary = rp.build_real_features_step(con, symbol_list, universe_name)
    typer.echo(json.dumps(summary, indent=2, default=str))


@app.command()
def evaluate_real(
    symbols: str = typer.Option(None, help="Comma-separated symbols; default = DEFAULT_REAL_UNIVERSE"),
    db_path: str = typer.Option(None, help="Override the DuckDB file path"),
) -> None:
    """V0.2: purged+embargoed walk-forward evaluation on the development set, then the V0.2
    champion/challenger promotion decision. Never touches the final holdout period."""
    import real_pipeline as rp

    con = get_connection(db_path)
    symbol_list = _parse_symbols(symbols) or rp.DEFAULT_REAL_UNIVERSE
    evaluation = rp.evaluate_real_step(con, symbol_list)
    typer.echo(json.dumps(
        {
            "n_folds": len(evaluation.fold_results),
            "fold_metrics_summary": evaluation.fold_metrics_summary,
            "champion_model_version": evaluation.champion_model_version,
            "promoted": evaluation.promoted,
            "promotion_rationale": evaluation.promotion_rationale,
            "robustness": evaluation.robustness,
            "development_rows": len(evaluation.development_df),
            "holdout_rows_available_but_untouched": len(evaluation.holdout_df),
        },
        indent=2, default=str,
    ))


@app.command()
def real_demo(
    symbols: str = typer.Option(None, help="Comma-separated symbols; default = DEFAULT_REAL_UNIVERSE (20 liquid US large caps)"),
    start: str = typer.Option(None, help="ISO start date, default 2020-01-01"),
    end: str = typer.Option(None, help="ISO end date, default today"),
    initial_cash: float = typer.Option(1_000_000.0, help="Starting paper-trading cash"),
    use_llm: bool = typer.Option(False, help="Enable optional LLM narrative enhancement for agents"),
    skip_ingestion: bool = typer.Option(False, help="Skip live ingestion and use whatever is already in the database"),
    db_path: str = typer.Option(None, help="Override the DuckDB file path"),
) -> None:
    """V0.2: the full real-data pipeline end to end -- ingest -> build-real-features ->
    evaluate-real -> a genuine paper-trading backtest through the SAME Portfolio/Risk/
    Execution engine V0.1 uses, over real, out-of-sample data. PAPER-TRADING ONLY."""
    import real_pipeline as rp

    con = get_connection(db_path)
    symbol_list = _parse_symbols(symbols) or rp.DEFAULT_REAL_UNIVERSE
    result = rp.run_real_demo(
        con, symbol_list, _parse_date(start, datetime(2020, 1, 1)), _parse_date(end, datetime.now()),
        initial_cash=initial_cash, use_llm=use_llm, skip_ingestion=skip_ingestion,
    )
    typer.echo(json.dumps(
        {
            "champion_model_version": result.evaluation.champion_model_version,
            "promoted": result.evaluation.promoted,
            "promotion_rationale": result.evaluation.promotion_rationale,
            "fold_metrics_summary": result.evaluation.fold_metrics_summary,
            "backtest_period": result.backtest_period,
            "n_fills": result.n_fills,
            "n_rejected_orders": result.n_rejected_orders,
            "rejection_reason_codes": result.rejection_reason_codes,
        },
        indent=2, default=str,
    ))


@app.command()
def serve_api(host: str | None = None, port: int | None = None) -> None:
    """Start the FastAPI monitoring/inference API."""
    import uvicorn

    uvicorn.run("api.main:app", host=host or settings.api_host, port=port or settings.api_port)


@app.command()
def serve_dashboard() -> None:
    """Start the Streamlit monitoring dashboard."""
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).parent / "dashboard" / "app.py")])


if __name__ == "__main__":
    app()
