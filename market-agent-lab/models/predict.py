"""Inference: turn trained boosters + a feature row into an immutable ``ModelPrediction``.

Volatility note: v0.1 does not train a dedicated volatility model (the
brief specifies exactly four targets). ``predicted_volatility`` uses the
already-computed trailing 20-day realised volatility as a naive
persistence forecast -- this is a documented simplification, not a
learned estimate, and is called out in ``docs/model_design.md``.
"""

from __future__ import annotations

from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from core.schemas import ModelPrediction


def predict_one(
    boosters: dict[str, lgb.Booster],
    feature_row: pd.Series,
    feature_cols: list[str],
    symbol: str,
    timestamp: datetime,
    model_version: str,
    feature_version: str,
) -> ModelPrediction:
    x = feature_row[feature_cols].to_frame().T.astype(float)

    pred_5d = float(boosters["excess_return_5d"].predict(x)[0])
    pred_20d = float(boosters["excess_return_20d"].predict(x)[0])
    prob_5d = float(np.clip(boosters["positive_5d"].predict(x)[0], 0.0, 1.0))
    prob_20d = float(np.clip(boosters["positive_20d"].predict(x)[0], 0.0, 1.0))

    naive_vol = feature_row.get("raw_realised_vol_20d")
    predicted_volatility = float(naive_vol) if naive_vol == naive_vol and naive_vol is not None else 0.20

    # Confidence: how decisively the two probability heads lean away from
    # a coin flip, blended with the agent-composite confidence feature
    # (agent disagreement penalises this) when available.
    prob_conviction = float(np.mean([abs(prob_5d - 0.5), abs(prob_20d - 0.5)]) * 2.0)
    agent_confidence = feature_row.get("agent_composite_confidence")
    agent_confidence = float(agent_confidence) if agent_confidence == agent_confidence and agent_confidence is not None else 0.5
    confidence = float(np.clip(0.5 * prob_conviction + 0.5 * agent_confidence, 0.0, 1.0))

    return ModelPrediction(
        model_version=model_version,
        timestamp=timestamp,
        symbol=symbol,
        predicted_excess_return_5d=pred_5d,
        predicted_excess_return_20d=pred_20d,
        probability_positive_5d=prob_5d,
        probability_positive_20d=prob_20d,
        predicted_volatility=predicted_volatility,
        confidence=confidence,
        feature_version=feature_version,
    )


def predict_batch(
    boosters: dict[str, lgb.Booster],
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    model_version: str,
    feature_version: str,
) -> list[ModelPrediction]:
    predictions = []
    for _, row in feature_df.iterrows():
        if row[feature_cols].isna().any():
            continue
        predictions.append(
            predict_one(boosters, row, feature_cols, row["symbol"], row["timestamp"], model_version, feature_version)
        )
    return predictions
