"""Tests for outcome labelling, drift detection, and champion/challenger promotion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.schemas import ModelPrediction
from database import repository as repo
from database.db import fresh_connection
from learning.champion_challenger import PromotionCriteria, decide_promotion
from learning.drift import population_stability_index
from learning.outcomes import label_pending_outcomes


@pytest.fixture
def con():
    with fresh_connection(":memory:") as c:
        yield c


def _market(symbol: str, n: int, start_price: float = 100.0, daily_return: float = 0.001) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-01", periods=n)
    closes = start_price * (1 + daily_return) ** np.arange(n)
    return pd.DataFrame(
        {
            "symbol": symbol, "timestamp": dates, "open": closes, "high": closes * 1.001,
            "low": closes * 0.999, "close": closes, "adjusted_close": closes, "volume": 1_000_000,
        }
    )


def test_label_pending_outcomes_computes_correct_excess_return(con):
    market = _market("SYN_X", 60, daily_return=0.002)
    bench = _market("SYN_BENCH", 60, daily_return=0.0005)

    ts = market["timestamp"].iloc[10]
    prediction = ModelPrediction(
        model_version="mv1", timestamp=ts, symbol="SYN_X",
        predicted_excess_return_5d=0.01, predicted_excess_return_20d=0.02,
        probability_positive_5d=0.6, probability_positive_20d=0.6,
        predicted_volatility=0.2, confidence=0.5, feature_version="fv1",
    )
    repo.insert_prediction(con, prediction)

    labelled = label_pending_outcomes(con, market, bench, as_of=market["timestamp"].iloc[-1])
    assert labelled == 1

    outcomes = con.execute("SELECT * FROM outcomes").fetchdf()
    assert len(outcomes) == 1
    row = outcomes.iloc[0]

    expected_stock_5d = (1.002 ** 5) - 1
    expected_bench_5d = (1.0005 ** 5) - 1
    expected_excess_5d = expected_stock_5d - expected_bench_5d
    assert row["realised_excess_return_5d"] == pytest.approx(expected_excess_5d, rel=1e-6)

    expected_stock_20d = (1.002 ** 20) - 1
    expected_bench_20d = (1.0005 ** 20) - 1
    expected_excess_20d = expected_stock_20d - expected_bench_20d
    assert row["realised_excess_return_20d"] == pytest.approx(expected_excess_20d, rel=1e-6)


def test_label_pending_outcomes_skips_when_horizon_incomplete(con):
    market = _market("SYN_X", 60)
    bench = _market("SYN_BENCH", 60)
    ts = market["timestamp"].iloc[55]  # only 4 days of remaining data -- 20d horizon can't complete
    prediction = ModelPrediction(
        model_version="mv1", timestamp=ts, symbol="SYN_X",
        predicted_excess_return_5d=0.0, predicted_excess_return_20d=0.0,
        probability_positive_5d=0.5, probability_positive_20d=0.5,
        predicted_volatility=0.2, confidence=0.5, feature_version="fv1",
    )
    repo.insert_prediction(con, prediction)
    labelled = label_pending_outcomes(con, market, bench, as_of=market["timestamp"].iloc[-1])
    assert labelled == 0
    assert con.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0] == 0


def test_outcomes_are_immutable_cannot_double_label(con):
    market = _market("SYN_X", 60, daily_return=0.001)
    bench = _market("SYN_BENCH", 60)
    ts = market["timestamp"].iloc[5]
    prediction = ModelPrediction(
        model_version="mv1", timestamp=ts, symbol="SYN_X",
        predicted_excess_return_5d=0.0, predicted_excess_return_20d=0.0,
        probability_positive_5d=0.5, probability_positive_20d=0.5,
        predicted_volatility=0.2, confidence=0.5, feature_version="fv1",
    )
    repo.insert_prediction(con, prediction)
    first = label_pending_outcomes(con, market, bench, as_of=market["timestamp"].iloc[-1])
    second = label_pending_outcomes(con, market, bench, as_of=market["timestamp"].iloc[-1])
    assert first == 1
    assert second == 0  # already labelled -- not re-inserted


# --- champion/challenger promotion -----------------------------------------------


def test_first_challenger_auto_promotes_with_no_champion():
    promoted, rationale = decide_promotion(
        challenger_metrics={"excess_return_5d": {"information_coefficient": 0.05, "sharpe_ratio": 0.3}},
        champion_metrics=None,
    )
    assert promoted is True
    assert "no existing champion" in rationale


def test_challenger_rejected_for_low_information_coefficient():
    promoted, rationale = decide_promotion(
        challenger_metrics={"excess_return_5d": {"information_coefficient": -0.02, "sharpe_ratio": 1.0}},
        champion_metrics=None,
    )
    assert promoted is False
    assert "information_coefficient" in rationale


def test_challenger_rejected_for_worse_drawdown_despite_higher_return():
    challenger = {
        "excess_return_5d": {"information_coefficient": 0.10, "sharpe_ratio": 2.0},
        "positive_5d": {"brier_score": 0.20},
        "backtest": {"max_drawdown": -0.40, "total_return": 0.80},  # huge raw return...
    }
    champion = {
        "excess_return_5d": {"information_coefficient": 0.06, "sharpe_ratio": 1.0},
        "positive_5d": {"brier_score": 0.20},
        "backtest": {"max_drawdown": -0.10, "total_return": 0.10},  # ...but much safer
    }
    promoted, rationale = decide_promotion(
        challenger, champion, PromotionCriteria(max_drawdown_regression_tolerance=0.05)
    )
    assert promoted is False
    assert "drawdown" in rationale


def test_challenger_promoted_when_strictly_better():
    challenger = {
        "excess_return_5d": {"information_coefficient": 0.08, "sharpe_ratio": 1.5},
        "positive_5d": {"brier_score": 0.18},
        "backtest": {"max_drawdown": -0.08},
    }
    champion = {
        "excess_return_5d": {"information_coefficient": 0.05, "sharpe_ratio": 1.0},
        "positive_5d": {"brier_score": 0.22},
        "backtest": {"max_drawdown": -0.10},
    }
    promoted, rationale = decide_promotion(challenger, champion)
    assert promoted is True


# --- drift -------------------------------------------------------------------------


def test_psi_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 2000))
    cur = pd.Series(rng.normal(0, 1, 2000))
    psi = population_stability_index(ref, cur)
    assert psi < 0.05


def test_psi_high_for_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 2000))
    cur = pd.Series(rng.normal(5, 1, 2000))
    psi = population_stability_index(ref, cur)
    assert psi > 0.5
