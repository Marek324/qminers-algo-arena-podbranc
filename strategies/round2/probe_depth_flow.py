"""Probe: buys 100 lots at the start and holds them to the end. Not a trading strategy."""

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


class ProbeDepthFlow(BaseStrategy):
    """Acquires exactly 100 lots at start and holds them to sweep book depth."""

    def __init__(self) -> None:
        super().__init__()
        self.inventory: int = 0
        self.order_in_flight: bool = False
        self.target_position: int = 100

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        # Once target position is reached, hold passively for the remainder of the session
        if self.inventory >= self.target_position:
            return

        if self.order_in_flight:
            return

        if not snapshot.asks:
            return

        needed = self.target_position - self.inventory
        if needed <= 0:
            return

        # Buy at best ask to fill inventory cleanly
        best_ask = snapshot.asks[0].price
        self.order_in_flight = True
        self.send_order(
            OrderRequest(
                side=Side.BUY,
                price=best_ask,
                quantity=needed,
                order_type=OrderType.IOC,
            )
        )

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        if event.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            self.inventory += event.filled_quantity

        if event.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            self.order_in_flight = False
