"""Historical analogue similarity engine (mathematics only, no LLM).

Implements the standardised Euclidean distance the Historical Research
Agent is required to use:

    D_t = sqrt( sum_i( w_i * (x_i,t - x_i,current)^2 ) )

where ``x`` is a standardised feature vector (z-scored using only
history available as of the query date). The Historical Research Agent
(``agents/historical.py``) wraps this module's output in an
``AgentReport`` -- it never recomputes distances itself.

Leakage guard
-------------
For a query date ``as_of``, a historical row at date ``t'`` is only a
valid analogue if its *subsequent* return outcome was already fully
observable by ``as_of`` -- i.e. ``t' + horizon_days <= as_of``. This
module enforces that by computing forward returns on a feature history
that has itself already been truncated to ``timestamp <= as_of``: the
trailing ``horizon_days`` rows then naturally get ``NaN`` forward returns
(there is no future data past ``as_of`` to compute them from), and are
dropped before nearest-neighbour search. This makes the leakage guard a
structural property of the function rather than a manually-tuned cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimilarityResult:
    num_analogues: int
    avg_return_5d: float | None
    median_return_5d: float | None
    prob_positive_5d: float | None
    avg_return_20d: float | None
    median_return_20d: float | None
    prob_positive_20d: float | None
    p10_return_20d: float | None
    p90_return_20d: float | None
    similarity_confidence: float

    def as_dict(self) -> dict[str, float]:
        return {
            "num_analogues": float(self.num_analogues),
            "avg_return_5d": self.avg_return_5d if self.avg_return_5d is not None else float("nan"),
            "median_return_5d": self.median_return_5d if self.median_return_5d is not None else float("nan"),
            "prob_positive_5d": self.prob_positive_5d if self.prob_positive_5d is not None else float("nan"),
            "avg_return_20d": self.avg_return_20d if self.avg_return_20d is not None else float("nan"),
            "median_return_20d": self.median_return_20d if self.median_return_20d is not None else float("nan"),
            "prob_positive_20d": self.prob_positive_20d if self.prob_positive_20d is not None else float("nan"),
            "p10_return_20d": self.p10_return_20d if self.p10_return_20d is not None else float("nan"),
            "p90_return_20d": self.p90_return_20d if self.p90_return_20d is not None else float("nan"),
            "similarity_confidence": self.similarity_confidence,
        }


def _add_forward_returns(symbol_history: pd.DataFrame) -> pd.DataFrame:
    """Add forward_return_5d/20d columns. Trailing rows become NaN by
    construction if ``symbol_history`` has already been truncated to as_of."""
    df = symbol_history.sort_values("timestamp").copy()
    df["forward_return_5d"] = df["close"].shift(-5) / df["close"] - 1.0
    df["forward_return_20d"] = df["close"].shift(-20) / df["close"] - 1.0
    return df


def _standardize(feature_df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score ``feature_cols`` using only the rows present in ``feature_df``
    (caller is responsible for having already truncated to as-of history)."""
    values = feature_df[feature_cols].to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    z = (values - mean) / std_safe
    return z, mean, std_safe


def find_historical_analogues(
    feature_history_asof: pd.DataFrame,
    market_history_asof: pd.DataFrame,
    feature_cols: list[str],
    current_row_index: int | None = None,
    weights: dict[str, float] | None = None,
    k: int = 50,
    min_history: int = 60,
) -> SimilarityResult:
    """Find the ``k`` nearest historical analogues to the most recent feature row.

    Parameters
    ----------
    feature_history_asof:
        Technical feature rows for ONE symbol, already filtered to
        ``timestamp <= as_of`` and sorted ascending, with a ``timestamp``
        column aligned 1:1 to ``market_history_asof``.
    market_history_asof:
        OHLCV rows for the same symbol/date range, used to compute forward
        returns for analogue outcomes.
    feature_cols:
        Which columns of ``feature_history_asof`` form the feature vector.
    current_row_index:
        Row (positional) to treat as "current". Defaults to the last row.
    weights:
        Optional per-feature weight; defaults to equal weighting.
    """
    if len(feature_history_asof) < min_history:
        return SimilarityResult(0, None, None, None, None, None, None, None, None, 0.0)

    merged = feature_history_asof.merge(
        market_history_asof[["timestamp", "close"]], on="timestamp", how="inner"
    ).sort_values("timestamp").reset_index(drop=True)
    merged = _add_forward_returns(merged)

    if current_row_index is None:
        current_row_index = len(merged) - 1
    current_row = merged.iloc[current_row_index]

    usable_cols = [c for c in feature_cols if c in merged.columns]
    z, _, _ = _standardize(merged, usable_cols)

    current_pos_mask = merged["timestamp"] == current_row["timestamp"]
    current_idx = np.where(current_pos_mask.to_numpy())[0][0]
    current_vector = z[current_idx]

    w = np.array([weights.get(c, 1.0) if weights else 1.0 for c in usable_cols])

    # Eligible analogue rows: must have a fully-realised 20d forward return
    # (structurally excludes the trailing 20 rows and the current row itself)
    # and must not contain NaN features.
    valid_rows = ~np.isnan(z).any(axis=1)
    valid_rows &= merged["forward_return_20d"].notna().to_numpy()
    valid_rows[current_idx] = False

    if not valid_rows.any():
        return SimilarityResult(0, None, None, None, None, None, None, None, None, 0.0)

    diffs = z[valid_rows] - current_vector
    distances = np.sqrt(np.sum(w * diffs**2, axis=1))

    candidates = merged.loc[valid_rows].copy()
    candidates["distance"] = distances
    candidates = candidates.sort_values("distance").head(k)

    n = len(candidates)
    r5 = candidates["forward_return_5d"].dropna()
    r20 = candidates["forward_return_20d"].dropna()

    avg_distance = float(candidates["distance"].mean()) if n else float("nan")
    confidence = float(min(1.0, n / k) / (1.0 + avg_distance)) if n else 0.0

    return SimilarityResult(
        num_analogues=n,
        avg_return_5d=float(r5.mean()) if len(r5) else None,
        median_return_5d=float(r5.median()) if len(r5) else None,
        prob_positive_5d=float((r5 > 0).mean()) if len(r5) else None,
        avg_return_20d=float(r20.mean()) if len(r20) else None,
        median_return_20d=float(r20.median()) if len(r20) else None,
        prob_positive_20d=float((r20 > 0).mean()) if len(r20) else None,
        p10_return_20d=float(r20.quantile(0.10)) if len(r20) else None,
        p90_return_20d=float(r20.quantile(0.90)) if len(r20) else None,
        similarity_confidence=confidence,
    )


def compute_similarity_series(
    feature_history: pd.DataFrame,
    market_history: pd.DataFrame,
    feature_cols: list[str],
    weights: dict[str, float] | None = None,
    k: int = 50,
    min_history: int = 60,
) -> pd.DataFrame:
    """Vectorised batch version of :func:`find_historical_analogues`.

    Computes one :class:`SimilarityResult` per row of ``feature_history``
    for an entire symbol history efficiently, using an *expanding* z-score
    baseline (each row standardised only against data up to and including
    itself -- never against later rows) and numpy-vectorised distance
    broadcasting instead of recomputing a fresh standardisation per query
    date. This is what the Feature Store uses to build historical-analogue
    features for every training row; ``find_historical_analogues`` remains
    the single-query entry point used by the live orchestrator.
    """
    merged = feature_history.merge(
        market_history[["timestamp", "close"]], on="timestamp", how="inner"
    ).sort_values("timestamp").reset_index(drop=True)
    merged = _add_forward_returns(merged)

    usable_cols = [c for c in feature_cols if c in merged.columns]
    raw = merged[usable_cols].astype(float)

    expanding_mean = raw.expanding(min_periods=1).mean()
    expanding_std = raw.expanding(min_periods=1).std(ddof=0).replace(0.0, np.nan)
    z = ((raw - expanding_mean) / expanding_std).to_numpy()

    w = np.array([weights.get(c, 1.0) if weights else 1.0 for c in usable_cols])
    forward_20d = merged["forward_return_20d"].to_numpy()
    forward_5d = merged["forward_return_5d"].to_numpy()
    valid_feature_row = ~np.isnan(z).any(axis=1)
    valid_analogue = valid_feature_row & ~np.isnan(forward_20d)

    n = len(merged)
    results: list[SimilarityResult] = []
    for t in range(n):
        if t < min_history or not valid_feature_row[t]:
            results.append(SimilarityResult(0, None, None, None, None, None, None, None, None, 0.0))
            continue

        mask = valid_analogue[:t].copy()
        if not mask.any():
            results.append(SimilarityResult(0, None, None, None, None, None, None, None, None, 0.0))
            continue

        candidate_idx = np.where(mask)[0]
        diffs = z[candidate_idx] - z[t]
        distances = np.sqrt(np.sum(w * diffs**2, axis=1))

        top_n = min(k, len(distances))
        nearest_local = np.argpartition(distances, top_n - 1)[:top_n]
        nearest_idx = candidate_idx[nearest_local]
        nearest_dist = distances[nearest_local]

        r5 = pd.Series(forward_5d[nearest_idx]).dropna()
        r20 = pd.Series(forward_20d[nearest_idx]).dropna()
        avg_distance = float(nearest_dist.mean()) if top_n else float("nan")
        confidence = float(min(1.0, top_n / k) / (1.0 + avg_distance)) if top_n else 0.0

        results.append(
            SimilarityResult(
                num_analogues=top_n,
                avg_return_5d=float(r5.mean()) if len(r5) else None,
                median_return_5d=float(r5.median()) if len(r5) else None,
                prob_positive_5d=float((r5 > 0).mean()) if len(r5) else None,
                avg_return_20d=float(r20.mean()) if len(r20) else None,
                median_return_20d=float(r20.median()) if len(r20) else None,
                prob_positive_20d=float((r20 > 0).mean()) if len(r20) else None,
                p10_return_20d=float(r20.quantile(0.10)) if len(r20) else None,
                p90_return_20d=float(r20.quantile(0.90)) if len(r20) else None,
                similarity_confidence=confidence,
            )
        )

    out = pd.DataFrame([r.as_dict() for r in results])
    out.insert(0, "timestamp", merged["timestamp"].to_numpy())
    out.insert(0, "symbol", merged["symbol"].to_numpy() if "symbol" in merged.columns else None)
    return out
