"""LightGBM alpha-model training.

Targets (Phase 4, deliberately NOT buy/sell signals):

1. ``excess_return_5d``  -- regression, stock 5-day return minus benchmark
2. ``excess_return_20d`` -- regression, stock 20-day return minus benchmark
3. ``positive_5d``       -- binary classification, is excess_return_5d > 0
4. ``positive_20d``      -- binary classification, is excess_return_20d > 0

"Excess return" is always computed against the abstract synthetic
benchmark (``SYN_BENCH``), never against the symbol's own history, so the
model learns relative (alpha-seeking) behaviour rather than absolute
market direction.

Target leakage guards
----------------------
* Targets are built from ``close.shift(-horizon)`` on data *separate* from
  the feature matrix -- features for row ``t`` only ever use information
  available at or before ``t`` (see ``features/technical.py`` and
  ``features/historical.py`` leakage guards); targets for row ``t`` use
  information strictly after ``t``. The two are joined on
  ``(symbol, timestamp)``, never recomputed from each other.
* Rows whose target cannot yet be realised (the trailing
  ``horizon`` days of any period) are ``NaN`` and are dropped before
  training/evaluation -- never imputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from models import reproducibility

NON_FEATURE_COLUMNS = {
    "symbol", "timestamp",
    "excess_return_5d", "excess_return_20d", "positive_5d", "positive_20d",
}

DEFAULT_HYPERPARAMETERS: dict[str, dict] = {
    "regression": {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 30,
        "verbosity": -1,
        "seed": 42,
        # V0.3 Stage 13: LightGBM's default multi-threaded histogram build
        # is NOT bit-for-bit reproducible run-to-run even with a fixed
        # seed. ``deterministic`` (paired with a fixed row/col-wise mode)
        # trades a little training speed for a genuinely reproducible
        # artifact given the same data + seed, which the model registry's
        # reproducibility guarantee depends on.
        "deterministic": True,
        "force_row_wise": True,
    },
    "classification": {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 30,
        "verbosity": -1,
        "seed": 42,
        "deterministic": True,
        "force_row_wise": True,
    },
}

TARGET_KIND = {
    "excess_return_5d": "regression",
    "excess_return_20d": "regression",
    "positive_5d": "classification",
    "positive_20d": "classification",
}


def compute_excess_return_targets(market_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Build the four targets for every (symbol, timestamp) in ``market_df``."""
    bench = benchmark_df.sort_values("timestamp").copy()
    bench["bench_return_5d"] = bench["close"].shift(-5) / bench["close"] - 1.0
    bench["bench_return_20d"] = bench["close"].shift(-20) / bench["close"] - 1.0
    bench_lookup = bench.set_index("timestamp")[["bench_return_5d", "bench_return_20d"]]

    parts = []
    for _symbol, group in market_df.sort_values(["symbol", "timestamp"]).groupby("symbol", sort=False):
        g = group.sort_values("timestamp").copy()
        g["stock_return_5d"] = g["close"].shift(-5) / g["close"] - 1.0
        g["stock_return_20d"] = g["close"].shift(-20) / g["close"] - 1.0
        g = g.join(bench_lookup, on="timestamp")
        g["excess_return_5d"] = g["stock_return_5d"] - g["bench_return_5d"]
        g["excess_return_20d"] = g["stock_return_20d"] - g["bench_return_20d"]
        g["positive_5d"] = (g["excess_return_5d"] > 0).astype(float)
        g["positive_20d"] = (g["excess_return_20d"] > 0).astype(float)
        g.loc[g["excess_return_5d"].isna(), "positive_5d"] = np.nan
        g.loc[g["excess_return_20d"].isna(), "positive_20d"] = np.nan
        parts.append(g[["symbol", "timestamp", "excess_return_5d", "excess_return_20d", "positive_5d", "positive_20d"]])
    return pd.concat(parts, ignore_index=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def prepare_training_frame(feature_matrix: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Inner-join features to targets on (symbol, timestamp)."""
    merged = feature_matrix.merge(targets, on=["symbol", "timestamp"], how="inner")
    return merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


@dataclass
class TrainedModels:
    boosters: dict[str, lgb.Booster] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    hyperparameters: dict[str, dict] = field(default_factory=dict)
    # V0.3 Stage 13 reproducibility provenance -- see models/reproducibility.py.
    data_fingerprint: str | None = None
    target_definition_hash: str | None = None
    random_seed: int | None = None


def train_single_target(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    hyperparameters: dict | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
) -> lgb.Booster:
    kind = TARGET_KIND[target_col]
    params = dict(hyperparameters or DEFAULT_HYPERPARAMETERS[kind])

    train_rows = train_df.dropna(subset=[target_col, *feature_cols])
    val_rows = val_df.dropna(subset=[target_col, *feature_cols])
    if train_rows.empty or val_rows.empty:
        raise ValueError(f"insufficient non-NaN rows to train target={target_col}")

    train_set = lgb.Dataset(train_rows[feature_cols], label=train_rows[target_col])
    val_set = lgb.Dataset(val_rows[feature_cols], label=val_rows[target_col], reference=train_set)

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(period=0)],
    )
    return booster


def train_all_targets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    hyperparameters: dict[str, dict] | None = None,
) -> TrainedModels:
    feature_cols = feature_cols or get_feature_columns(train_df)
    hyperparameters = hyperparameters or DEFAULT_HYPERPARAMETERS
    boosters = {}
    for target_col in TARGET_KIND:
        kind = TARGET_KIND[target_col]
        boosters[target_col] = train_single_target(
            train_df, val_df, feature_cols, target_col, hyperparameters.get(kind)
        )
    data_fingerprint = reproducibility.compute_data_fingerprint(
        train_df, val_df, feature_cols, list(TARGET_KIND.keys())
    )
    target_definition_hash = reproducibility.hash_source(compute_excess_return_targets)
    random_seed = reproducibility.extract_seed(hyperparameters)
    return TrainedModels(
        boosters=boosters, feature_names=feature_cols, hyperparameters=hyperparameters,
        data_fingerprint=data_fingerprint, target_definition_hash=target_definition_hash, random_seed=random_seed,
    )
