#!/usr/bin/env python
"""Build the Feature Store and train+register a baseline LightGBM model
(Phase 13, steps 2-4). Assumes ``scripts/generate_sample_data.py`` has
already been run (or runs it if the database is empty).

    uv run python scripts/train_baseline.py
"""

from __future__ import annotations

import pandas as pd

from core.config import settings
from core.logging import configure_logging, get_logger
from data.market_data import get_benchmark, get_ohlcv, get_symbols, load_all_synthetic_data
from database.db import get_connection
from features.feature_store import (
    DEFAULT_FEATURE_VERSION,
    build_feature_matrix,
    load_feature_matrix,
    store_feature_matrix,
)
from models.registry import ModelPeriods, register_model
from models.train import (
    compute_excess_return_targets,
    get_feature_columns,
    prepare_training_frame,
    train_all_targets,
)

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    con = get_connection()

    symbols = get_symbols(con)
    if not symbols:
        logger.info("no_data_found_generating_synthetic_universe")
        load_all_synthetic_data(
            con, seed=settings.synthetic_seed, start_date=settings.synthetic_start_date,
            end_date=settings.synthetic_end_date, out_dir=settings.data_store_dir / "raw" / "synthetic",
        )
        symbols = get_symbols(con)

    matrix = build_feature_matrix(con, symbols=symbols, use_llm=False, persist_agent_reports=True)
    store_feature_matrix(con, DEFAULT_FEATURE_VERSION, matrix)
    matrix = load_feature_matrix(con, DEFAULT_FEATURE_VERSION, symbols=symbols)

    market = get_ohlcv(con, symbols=symbols)
    benchmark = get_benchmark(con)
    targets = compute_excess_return_targets(market, benchmark)
    df = prepare_training_frame(matrix, targets)
    feature_cols = get_feature_columns(df)

    train_end = pd.Timestamp("2021-12-31")
    val_end = pd.Timestamp("2022-12-31")
    train_df = df[df["timestamp"] <= train_end]
    val_df = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)]

    trained = train_all_targets(train_df, val_df, feature_cols)
    periods = ModelPeriods(
        training_start=train_df["timestamp"].min(), training_end=train_df["timestamp"].max(),
        validation_start=val_df["timestamp"].min(), validation_end=val_df["timestamp"].max(),
    )
    model_version = register_model(con, trained, DEFAULT_FEATURE_VERSION, periods, metrics={}, role="CHALLENGER")
    logger.info("baseline_model_trained", model_version=model_version, n_features=len(feature_cols))


if __name__ == "__main__":
    main()
