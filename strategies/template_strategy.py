"""Starter template: copy this file and implement your own strategy in it."""

from __future__ import annotations

from simulator.models import OrderBookSnapshot, PrivateOrderEvent, Trade
from simulator.strategy import BaseStrategy


class TemplateStrategy(BaseStrategy):
    """Template strategy: implement your algorithm by editing this class.

    Main callback: `on_order_book_change(snapshot, trades)`
    - called on every market event with the current public order book
    - `trades` lists every trade since the previous call (can be more than
      one, e.g. a taker sweeping several price levels)
    - use `self.send_order(OrderRequest(...))` to place orders

    Private callback: `on_private_order_update(event)`, the lifecycle of your orders
    - ACCEPTED: order entered the book
    - PARTIALLY_FILLED / FILLED: fill of `event.filled_quantity` at `event.fill_price`
    - CANCELED: IOC residual discarded, or a market cancel hit your order
    - REJECTED: order exceeded the position limit
    """

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        """React to the public book after every market event; `trades` are those since the previous call."""

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        """React to a state change of one of your own orders."""
