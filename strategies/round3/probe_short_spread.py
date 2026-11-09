"""Probe: sells 10 lots at the start and holds the short to the end. Not a trading strategy."""

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


class ProbeShortSpread(BaseStrategy):
    """Sells exactly 10 lots at start and holds a short position to sweep the Ask book."""

    def __init__(self) -> None:
        super().__init__()
        self.inventory: int = 0
        self.order_in_flight: bool = False
        self.target_position: int = -10

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        # Once target short position (-10 lots) is reached, hold passively for remainder of session
        if self.inventory <= self.target_position:
            return

        if self.order_in_flight:
            return

        if not snapshot.bids:
            return

        needed = self.inventory - self.target_position  # Positive quantity to sell
        if needed <= 0:
            return

        # Sell at best bid to fill short position cleanly
        best_bid = snapshot.bids[0].price
        self.order_in_flight = True
        self.send_order(
            OrderRequest(
                side=Side.SELL,
                price=best_bid,
                quantity=needed,
                order_type=OrderType.IOC,
            )
        )

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        if event.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            self.inventory -= event.filled_quantity

        if event.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            self.order_in_flight = False
