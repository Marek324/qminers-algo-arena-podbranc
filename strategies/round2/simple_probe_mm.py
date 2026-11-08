"""Price probe: buy a fixed position once, then never trade again.

The PnL is cash + the IOC mark-out of the held lots, so after the entry
PnL(t) = HOLD * (mark(t) - entry). PnL / HOLD + entry is the price path the
arena chart shows (mark ~ best bid for a small HOLD). Market making is gone on
purpose: spread capture and inventory swings would distort the curve.
"""

from __future__ import annotations

from simulator.models import (
    OrderBookSnapshot,
    OrderRequest,
    OrderStatus,
    OrderType,
    PrivateOrderEvent,
    Side,
    Trade,
)
from simulator.strategy import BaseStrategy

HOLD = 10  # lots held; small so the mark-out sweep stays at the best bid
SLIP = 3  # ticks over the best ask the entry IOC may pay


class SimpleProbeMM(BaseStrategy):
    """Buys HOLD lots with IOCs at the start and holds them to the end."""

    def __init__(self) -> None:
        super().__init__()
        self.inventory: int = 0
        self.pending: int | None = None

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        if self.pending is not None or self.inventory >= HOLD or not snapshot.asks:
            return
        self.pending = self.send_order(
            OrderRequest(
                side=Side.BUY,
                price=snapshot.asks[0].price + SLIP,
                quantity=HOLD - self.inventory,
                order_type=OrderType.IOC,
            )
        )

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        if event.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            self.inventory += event.filled_quantity
        if event.order_id == self.pending and event.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
        ):
            self.pending = None
