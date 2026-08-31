"""Portfolio Decision Engine (Phase 7).

Deterministically translates ML model predictions into target portfolio
weights, then into proposed :class:`~core.schemas.PaperOrder` objects. No
agent and no LLM has any influence here -- inputs are exactly the four
model outputs (plus their own predicted volatility/confidence), and the
function is pure arithmetic.

Kept deliberately simple for v0.1, as specified in the brief: a
volatility-scaled, confidence-weighted signal clipped to a fixed maximum
per-position weight, with a portfolio-level gross-exposure normalisation
pass so the engine doesn't routinely hand the Risk Engine proposals far
outside its limits (the Risk Engine still independently re-checks and can
reject every single order -- this is a courtesy, not a substitute).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from core.schemas import ModelPrediction, OrderSide, OrderType, PaperOrder


@dataclass(frozen=True)
class AllocationConfig:
    max_position_weight: float = 0.10
    # Kept comfortably under RiskLimits.max_net_exposure (0.60 default) so a
    # mostly one-directional book doesn't get stuck permanently bumping the
    # net-exposure ceiling once fully invested -- the Risk Engine still
    # independently enforces the real limit on every single order.
    target_gross_exposure: float = 0.50
    # Multiplies the horizon-adjusted risk signal before clipping to
    # max_position_weight; the clip is the primary sizing control, so this
    # is set close to 1.0 rather than used as an aggressive extra dampener.
    vol_target: float = 1.0
    min_confidence: float = 0.05
    min_trade_notional: float = 50.0
    volatility_floor: float = 0.05


def compute_target_weights(
    predictions: list[ModelPrediction], config: AllocationConfig | None = None
) -> dict[str, float]:
    """One deterministic weight per symbol in ``[-max_position_weight, max_position_weight]``."""
    config = config or AllocationConfig()
    raw_weights: dict[str, float] = {}

    for pred in predictions:
        if pred.confidence < config.min_confidence:
            continue
        directional_conviction = 2.0 * pred.probability_positive_5d - 1.0  # in [-1, 1]
        # Reinforce when the return forecast and the probability head agree
        # in sign; dampen (never flip) when they disagree.
        agreement = 1.0 if (pred.predicted_excess_return_5d >= 0) == (directional_conviction >= 0) else 0.4
        combined_signal = pred.predicted_excess_return_5d * pred.confidence * agreement

        # predicted_volatility is annualised; the signal is a 5-day excess
        # return, so rescale vol to the same horizon before using it to
        # risk-adjust the signal (dividing a 5-day return by an *annualised*
        # vol would systematically undersize every position by ~sqrt(252/5)).
        horizon_vol = max(pred.predicted_volatility * (5 / 252) ** 0.5, config.volatility_floor * (5 / 252) ** 0.5)
        risk_adjusted_signal = combined_signal / horizon_vol
        raw_weight = risk_adjusted_signal * config.vol_target
        raw_weights[pred.symbol] = max(-config.max_position_weight, min(config.max_position_weight, raw_weight))

    gross = sum(abs(w) for w in raw_weights.values())
    if gross > config.target_gross_exposure and gross > 0:
        scale = config.target_gross_exposure / gross
        raw_weights = {s: w * scale for s, w in raw_weights.items()}

    return raw_weights


def build_paper_orders(
    target_weights: dict[str, float],
    current_positions: dict[str, float],
    prices: dict[str, float],
    equity: float,
    timestamp: datetime,
    strategy_version: str,
    config: AllocationConfig | None = None,
) -> list[PaperOrder]:
    """Diff target weights against current holdings and emit rebalancing orders."""
    config = config or AllocationConfig()
    orders: list[PaperOrder] = []

    symbols = set(target_weights) | set(current_positions)
    for symbol in symbols:
        price = prices.get(symbol)
        if price is None or price <= 0:
            continue
        target_qty = (target_weights.get(symbol, 0.0) * equity) / price
        current_qty = current_positions.get(symbol, 0.0)
        delta_qty = target_qty - current_qty
        notional = abs(delta_qty) * price
        if notional < config.min_trade_notional:
            continue

        orders.append(
            PaperOrder(
                symbol=symbol,
                side=OrderSide.BUY if delta_qty > 0 else OrderSide.SELL,
                quantity=abs(delta_qty),
                order_type=OrderType.MARKET,
                proposed_price=price,
                timestamp=timestamp,
                strategy_version=strategy_version,
            )
        )
    return orders
