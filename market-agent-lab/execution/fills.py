"""Simulated fill mechanics for the Paper Execution Engine.

Pure, deterministic pricing math -- no live broker connectivity of any
kind. Given a reference execution price and volume, this module computes
a plausible fill price (spread + participation-scaled slippage),
commission, and partial-fill quantity, or a rejection reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.schemas import OrderSide, OrderType, PaperFill, PaperOrder


@dataclass(frozen=True)
class FillConfig:
    commission_per_share: float = 0.005
    commission_min: float = 1.0
    half_spread_bps: float = 5.0  # applied against the trader (buy up, sell down)
    slippage_coefficient: float = 0.02  # extra price impact at 100% of-volume participation
    max_participation_rate: float = 0.10  # max fraction of the bar's volume fillable


REJECT_NOT_MARKETABLE = "NOT_MARKETABLE_LIMIT_PRICE"
REJECT_NO_VOLUME = "NO_VOLUME_AVAILABLE"
REJECT_INSUFFICIENT_BUYING_POWER = "INSUFFICIENT_BUYING_POWER"


def simulate_fill(
    order: PaperOrder,
    execution_price: float,
    bar_volume: float,
    fill_timestamp,
    config: FillConfig | None = None,
    available_buying_power: float | None = None,
) -> tuple[PaperFill | None, str | None]:
    """Returns ``(fill, rejection_reason)`` -- exactly one is non-None."""
    config = config or FillConfig()
    side_sign = 1.0 if order.side == OrderSide.BUY else -1.0

    if order.order_type == OrderType.LIMIT and order.limit_price is not None:
        marketable = (
            execution_price <= order.limit_price if order.side == OrderSide.BUY else execution_price >= order.limit_price
        )
        if not marketable:
            return None, REJECT_NOT_MARKETABLE

    if bar_volume <= 0:
        return None, REJECT_NO_VOLUME

    participation = order.quantity / bar_volume
    filled_quantity = order.quantity
    if participation > config.max_participation_rate:
        filled_quantity = bar_volume * config.max_participation_rate
        participation = config.max_participation_rate

    spread_rate = config.half_spread_bps / 10_000.0
    slippage_rate = config.slippage_coefficient * participation
    fill_price = execution_price * (1.0 + side_sign * (spread_rate + slippage_rate))

    commission = max(config.commission_min, config.commission_per_share * filled_quantity)

    if order.side == OrderSide.BUY and available_buying_power is not None:
        required = fill_price * filled_quantity + commission
        if required > available_buying_power:
            affordable_quantity = max(0.0, (available_buying_power - commission) / fill_price)
            if affordable_quantity <= 0:
                return None, REJECT_INSUFFICIENT_BUYING_POWER
            filled_quantity = min(filled_quantity, affordable_quantity)
            commission = max(config.commission_min, config.commission_per_share * filled_quantity)

    slippage = fill_price - execution_price
    fill = PaperFill(
        order_id=order.order_id,
        fill_timestamp=fill_timestamp,
        fill_price=fill_price,
        quantity=filled_quantity,
        slippage=slippage,
        commission=commission,
    )
    return fill, None
