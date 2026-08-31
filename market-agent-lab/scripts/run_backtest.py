#!/usr/bin/env python
"""Run the event-driven paper-trading backtest for the current champion
model against the three comparison benchmarks (Phase 13, steps 5-11).
Assumes a champion model is already registered (see
``scripts/train_baseline.py`` or ``main.py demo``).

    uv run python scripts/run_backtest.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import typer

from backtesting.engine import (
    buy_and_hold_benchmark,
    run_ml_strategy_backtest,
)
from backtesting.metrics import compute_all_metrics
from core.logging import configure_logging, get_logger
from data import synthetic as synthetic_data
from data.market_data import get_ohlcv, get_symbols
from database.db import get_connection
from features.feature_store import DEFAULT_FEATURE_VERSION, load_feature_matrix
from models.registry import get_champion, load_model
from portfolio.allocation import AllocationConfig
from portfolio.risk import RiskLimits

logger = get_logger(__name__)


def main(
    test_start: str = typer.Option("2023-01-01"),
    test_end: str = typer.Option("2023-12-31"),
    initial_cash: float = typer.Option(1_000_000.0),
) -> None:
    configure_logging()
    con = get_connection()

    champion = get_champion(con)
    if champion is None:
        raise SystemExit("No champion model registered. Run scripts/train_baseline.py or main.py demo first.")
    boosters, record = load_model(con, champion["model_version"])
    feature_cols = record["feature_names"]

    symbols = get_symbols(con)
    sector_map = dict(synthetic_data.SYMBOLS)
    matrix = load_feature_matrix(con, DEFAULT_FEATURE_VERSION, symbols=symbols)
    market = get_ohlcv(con, symbols=symbols)

    test_df = matrix[(matrix["timestamp"] >= pd.Timestamp(test_start)) & (matrix["timestamp"] <= pd.Timestamp(test_end))]

    run_id = f"backtest_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    result = run_ml_strategy_backtest(
        con, run_id, test_df, market, boosters, feature_cols, champion["model_version"], DEFAULT_FEATURE_VERSION,
        symbols, sector_map, initial_cash=initial_cash, risk_limits=RiskLimits(), allocation_config=AllocationConfig(),
    )

    test_market = market[(market["timestamp"] >= pd.Timestamp(test_start)) & (market["timestamp"] <= pd.Timestamp(test_end))]
    bh = buy_and_hold_benchmark(test_market, symbols, initial_cash)
    metrics = compute_all_metrics(
        result.equity_curve, bh, result.trade_pnls, result.traded_notional, result.gross_exposure_series, result.holding_periods
    )
    logger.info("backtest_complete", run_id=run_id, **{k: round(v, 4) if v == v else None for k, v in metrics.items()})
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    typer.run(main)
