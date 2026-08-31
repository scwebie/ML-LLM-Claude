"""Champion/challenger promotion gate (Phase 10).

A challenger is NEVER promoted merely because its raw return is higher.
Promotion requires passing every configured criterion: out-of-sample
Sharpe (not worse beyond a small tolerance), drawdown (not meaningfully
worse), information coefficient (a real, positive edge), and calibration
(Brier score not worse beyond tolerance). Every decision -- promote or
reject -- is written to the ``promotion_log`` table, including the
rationale, so the promotion history is fully auditable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import duckdb

from database import repository as repo
from models.registry import promote_to_champion


@dataclass(frozen=True)
class PromotionCriteria:
    min_sharpe_improvement: float = -0.05  # allow a small regression, but not much
    max_drawdown_regression_tolerance: float = 0.05  # challenger dd can be at most this much worse
    min_information_coefficient: float = 0.0  # must show *some* real edge
    max_brier_regression_tolerance: float = 0.02  # calibration can't meaningfully worsen


def _sharpe_of(metrics: dict) -> float:
    reg = metrics.get("excess_return_5d", {})
    return reg.get("sharpe_ratio", reg.get("information_coefficient", 0.0)) or 0.0


def _ic_of(metrics: dict) -> float:
    reg = metrics.get("excess_return_5d", {})
    return reg.get("information_coefficient", 0.0) or 0.0


def _brier_of(metrics: dict) -> float:
    cls = metrics.get("positive_5d", {})
    return cls.get("brier_score", 0.25) or 0.25


def _drawdown_of(metrics: dict) -> float:
    return metrics.get("backtest", {}).get("max_drawdown", 0.0) or 0.0


def decide_promotion(
    challenger_metrics: dict, champion_metrics: dict | None, criteria: PromotionCriteria | None = None
) -> tuple[bool, str]:
    criteria = criteria or PromotionCriteria()

    challenger_ic = _ic_of(challenger_metrics)
    if challenger_ic < criteria.min_information_coefficient or challenger_ic != challenger_ic:
        return False, f"challenger information_coefficient={challenger_ic:.4f} below minimum {criteria.min_information_coefficient}"

    if champion_metrics is None:
        return True, "no existing champion -- auto-promoting first viable challenger"

    champ_ic = _ic_of(champion_metrics)
    champ_brier = _brier_of(champion_metrics)
    chal_brier = _brier_of(challenger_metrics)
    champ_dd = _drawdown_of(champion_metrics)
    chal_dd = _drawdown_of(challenger_metrics)

    if chal_brier > champ_brier + criteria.max_brier_regression_tolerance:
        return False, f"challenger calibration worse: brier {chal_brier:.4f} vs champion {champ_brier:.4f}"

    if abs(chal_dd) > abs(champ_dd) + criteria.max_drawdown_regression_tolerance:
        return False, f"challenger drawdown worse: {chal_dd:.2%} vs champion {champ_dd:.2%}"

    champ_sharpe = _sharpe_of(champion_metrics)
    chal_sharpe = _sharpe_of(challenger_metrics)
    if chal_sharpe < champ_sharpe + criteria.min_sharpe_improvement:
        return False, f"challenger sharpe {chal_sharpe:.3f} not sufficiently close to/better than champion {champ_sharpe:.3f}"

    if challenger_ic < champ_ic - 0.01:
        return False, f"challenger information_coefficient {challenger_ic:.4f} worse than champion {champ_ic:.4f}"

    return True, (
        f"challenger passed all criteria: IC {challenger_ic:.4f} (champion {champ_ic:.4f}), "
        f"sharpe {chal_sharpe:.3f} (champion {champ_sharpe:.3f}), brier {chal_brier:.4f} (champion {champ_brier:.4f}), "
        f"drawdown {chal_dd:.2%} (champion {champ_dd:.2%})"
    )


def run_promotion_cycle(
    con: duckdb.DuckDBPyConnection,
    challenger_version: str,
    challenger_metrics: dict,
    criteria: PromotionCriteria | None = None,
) -> tuple[bool, str]:
    champion_record = repo.get_champion(con)
    champion_metrics = None
    champion_version = None
    if champion_record is not None:
        import json

        champion_metrics = json.loads(champion_record["metrics_json"])
        champion_version = champion_record["model_version"]

    promoted, rationale = decide_promotion(challenger_metrics, champion_metrics, criteria)

    repo.insert_promotion_log(
        con,
        {
            "id": uuid.uuid4().hex,
            "timestamp": datetime.now(UTC),
            "challenger_version": challenger_version,
            "champion_version": champion_version,
            "decision": "PROMOTED" if promoted else "REJECTED",
            "rationale": rationale,
            "metrics": {"challenger": challenger_metrics, "champion": champion_metrics},
        },
    )

    if promoted:
        promote_to_champion(con, challenger_version)

    return promoted, rationale
