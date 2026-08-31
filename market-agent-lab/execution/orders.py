"""Order routing: run proposed orders through the deterministic Risk Engine.

This module contains zero trading logic of its own -- it only routes each
proposed :class:`~core.schemas.PaperOrder` through
:meth:`~portfolio.risk.RiskEngine.evaluate_order` and partitions the
results. Nothing here (or anywhere else in ``execution/``) can approve an
order the Risk Engine rejected.
"""

from __future__ import annotations

from core.schemas import PaperOrder, RiskApprovalStatus
from portfolio.risk import PortfolioState, RiskEngine


def evaluate_and_route(
    orders: list[PaperOrder], risk_engine: RiskEngine, state: PortfolioState
) -> tuple[list[PaperOrder], list[PaperOrder]]:
    """Evaluate every order and split into (approved, rejected).

    Exposure-based checks (position/gross/net/sector/vol limits) are
    evaluated sequentially against the *same* ``state`` snapshot passed in
    -- i.e. against starting-of-batch exposure, not updated incrementally
    order-by-order. This is a deliberate v0.1 simplification (documented
    in ``docs/model_design.md``): it avoids order-arrival-order dependence
    within a single rebalance, at the cost of not catching exposure that
    only breaches a limit once several orders in the same batch are summed.
    """
    approved: list[PaperOrder] = []
    rejected: list[PaperOrder] = []
    for order in orders:
        evaluated = risk_engine.evaluate_order(order, state)
        if evaluated.risk_approval_status == RiskApprovalStatus.APPROVED:
            approved.append(evaluated)
        else:
            rejected.append(evaluated)
    return approved, rejected
