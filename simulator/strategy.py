"""Base class for participant strategies and their channel back into the engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from simulator.models import OrderBookSnapshot, OrderRequest, OrderType, PrivateOrderEvent, Trade


@dataclass(slots=True)
class StrategyContext:
    """Minimal context exposed to user strategy for placing orders."""

    submit_order: Callable[[OrderRequest], str]
    cancel_order: Callable[[str], None]
    modify_order: Callable[[str, int], None]
    now_us: Callable[[], int]


class BaseStrategy:
    """Base class of a strategy: override the callbacks and submit orders via `send_order`."""

    def __init__(self) -> None:
        self._ctx: StrategyContext | None = None

    def _attach_context(self, context: StrategyContext) -> None:
        self._ctx = context

    def send_order(self, order: OrderRequest) -> str:
        """Submit an order and return its id; it is accepted after the exchange latency."""
        if self._ctx is None:
            raise RuntimeError('Strategy context not attached')
        return self._ctx.submit_order(order)

    def cancel_order(self, order_id: str) -> None:
        """Cancel one of the strategy's resting orders."""
        if self._ctx is None:
            raise RuntimeError('Strategy context not attached')
        self._ctx.cancel_order(order_id)

    def modify_order(self, order_id: str, new_quantity: int) -> None:
        """Shrink one of the strategy's resting orders to `new_quantity`."""
        if self._ctx is None:
            raise RuntimeError('Strategy context not attached')
        self._ctx.modify_order(order_id, new_quantity)

    def now_us(self) -> int:
        """Return the current simulation time in microseconds."""
        if self._ctx is None:
            raise RuntimeError('Strategy context not attached')
        return self._ctx.now_us()

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        """Handle the public feed: runs after every market event once the book is updated.

        Args:
            snapshot: The visible book after the event.
            trades: Every trade since the previous call (possibly more than
                one, e.g. a taker sweeping several price levels).
        """

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        """Handle the private feed: runs on every state change of the strategy's own orders."""


class ExamplePassiveStrategy(BaseStrategy):
    """Tiny strategy example; users should replace this with their own logic."""

    def __init__(self) -> None:
        super().__init__()
        self._sent = False

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        """Join the best bid once."""
        if self._sent:
            return
        if not snapshot.bids or not snapshot.asks:
            return
        best_bid = snapshot.bids[0].price
        self.send_order(
            OrderRequest(
                side=snapshot.bids[0].side,
                price=best_bid,
                quantity=10,
                order_type=OrderType.LIMIT,
            )
        )
        self._sent = True
