"""Tests for LightGBM training, target construction, and prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.schemas import ModelPrediction
from models.predict import predict_batch, predict_one
from models.train import (
    NON_FEATURE_COLUMNS,
    compute_excess_return_targets,
    get_feature_columns,
    prepare_training_frame,
    train_all_targets,
    train_single_target,
)


def _synthetic_market(symbol: str, n: int, seed: int, drift: float = 0.0005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    return pd.DataFrame(
        {
            "symbol": symbol, "timestamp": dates, "open": close, "high": close * 1.001,
            "low": close * 0.999, "close": close, "adjusted_close": close, "volume": 1_000_000,
        }
    )


def test_excess_return_targets_positive_flag_matches_sign():
    market = pd.concat([_synthetic_market("SYN_A", 300, seed=1), _synthetic_market("SYN_B", 300, seed=2)])
    bench = _synthetic_market("SYN_BENCH", 300, seed=3, drift=0.0)
    targets = compute_excess_return_targets(market, bench)
    valid = targets.dropna(subset=["excess_return_5d"])
    assert ((valid["excess_return_5d"] > 0) == (valid["positive_5d"] == 1.0)).all()
    valid20 = targets.dropna(subset=["excess_return_20d"])
    assert ((valid20["excess_return_20d"] > 0) == (valid20["positive_20d"] == 1.0)).all()


def test_excess_return_targets_trailing_rows_are_nan():
    market = _synthetic_market("SYN_A", 100, seed=1)
    bench = _synthetic_market("SYN_BENCH", 100, seed=2)
    targets = compute_excess_return_targets(market, bench)
    tail_20 = targets.sort_values("timestamp").tail(20)
    assert tail_20["excess_return_20d"].isna().all()


def test_get_feature_columns_excludes_target_and_key_columns():
    df = pd.DataFrame(
        {
            "symbol": ["A"], "timestamp": [pd.Timestamp("2020-01-01")],
            "feat1": [1.0], "excess_return_5d": [0.1], "positive_5d": [1.0],
        }
    )
    cols = get_feature_columns(df)
    assert cols == ["feat1"]
    assert set(NON_FEATURE_COLUMNS).isdisjoint(cols)


def _training_frame(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    market = pd.concat(
        [_synthetic_market(f"SYN_{i}", n, seed=seed + i) for i in range(3)], ignore_index=True
    )
    bench = _synthetic_market("SYN_BENCH", n, seed=seed + 99, drift=0.0)
    targets = compute_excess_return_targets(market, bench)

    feature_rows = []
    for symbol in market["symbol"].unique():
        sym_dates = market[market["symbol"] == symbol]["timestamp"]
        for ts in sym_dates:
            feature_rows.append({"symbol": symbol, "timestamp": ts, "f1": rng.normal(), "f2": rng.normal()})
    features = pd.DataFrame(feature_rows)

    df = prepare_training_frame(features, targets)
    return df, ["f1", "f2"]


def test_train_single_target_produces_finite_predictions():
    df, feature_cols = _training_frame()
    train_df = df[df["timestamp"] < df["timestamp"].quantile(0.7)]
    val_df = df[df["timestamp"] >= df["timestamp"].quantile(0.7)]
    booster = train_single_target(train_df, val_df, feature_cols, "excess_return_5d", num_boost_round=20, early_stopping_rounds=5)
    preds = booster.predict(val_df.dropna(subset=feature_cols)[feature_cols])
    assert np.isfinite(preds).all()


def test_train_single_target_raises_on_empty_data():
    df, feature_cols = _training_frame()
    with pytest.raises(ValueError):
        train_single_target(df.iloc[0:0], df, feature_cols, "excess_return_5d")


def test_train_all_targets_trains_all_four_targets():
    df, feature_cols = _training_frame()
    train_df = df[df["timestamp"] < df["timestamp"].quantile(0.7)]
    val_df = df[df["timestamp"] >= df["timestamp"].quantile(0.7)]
    trained = train_all_targets(train_df, val_df, feature_cols)
    assert set(trained.boosters.keys()) == {"excess_return_5d", "excess_return_20d", "positive_5d", "positive_20d"}


def test_predict_one_returns_valid_immutable_prediction():
    df, feature_cols = _training_frame()
    train_df = df[df["timestamp"] < df["timestamp"].quantile(0.7)]
    val_df = df[df["timestamp"] >= df["timestamp"].quantile(0.7)]
    trained = train_all_targets(train_df, val_df, feature_cols)

    row = val_df.dropna(subset=feature_cols).iloc[0]
    pred = predict_one(trained.boosters, row, feature_cols, row["symbol"], row["timestamp"], "test_v1", "fv1")
    assert isinstance(pred, ModelPrediction)
    assert 0.0 <= pred.probability_positive_5d <= 1.0
    assert 0.0 <= pred.probability_positive_20d <= 1.0
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.predicted_volatility >= 0.0

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        pred.confidence = 0.99  # frozen model -- must not be mutable


def test_predict_batch_skips_rows_with_nan_features():
    df, feature_cols = _training_frame()
    train_df = df[df["timestamp"] < df["timestamp"].quantile(0.7)]
    val_df = df[df["timestamp"] >= df["timestamp"].quantile(0.7)].copy()
    trained = train_all_targets(train_df, val_df, feature_cols)

    val_df_with_nan = val_df.copy()
    val_df_with_nan.iloc[0, val_df_with_nan.columns.get_loc("f1")] = np.nan
    preds = predict_batch(trained.boosters, val_df_with_nan, feature_cols, "test_v1", "fv1")
    assert len(preds) == len(val_df_with_nan) - 1
