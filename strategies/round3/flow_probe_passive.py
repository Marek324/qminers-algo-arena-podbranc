"""Flow probe (passive): rest QTY lots at the best bid and the best ask all the time.

Measures how often the touch is hit by market takers and how toxic those fills are (PnL per lot). Orders re-join the
best price whenever it moves; |inventory| is capped at MAX_INV by dropping the side that would grow it. Not a
trading strategy.
"""

from __future__ import annotations

from simulator.models import OrderBookSnapshot, OrderRequest, OrderStatus, OrderType, PrivateOrderEvent, Side, Trade
from simulator.strategy import BaseStrategy


class FlowProbePassive(BaseStrategy):
    QTY = 5
    MAX_INV = 50

    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        self.orders: dict[str, list] = {}   # order_id -> [sign, price, remaining]
        self.cancel_sent: set[str] = set()

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bb = next((o.price for o in snapshot.bids if o.owner == 'market'), None)
        ba = next((o.price for o in snapshot.asks if o.owner == 'market'), None)
        if bb is None or ba is None:
            return
        self._side(1, bb, self.inv < self.MAX_INV)
        self._side(-1, ba, self.inv > -self.MAX_INV)

    def _side(self, sign: int, price: int, allowed: bool) -> None:
        live = False
        for oid, o in self.orders.items():
            if o[0] != sign or oid in self.cancel_sent:
                continue
            if not allowed or o[1] != price:
                self.cancel_order(oid)
                self.cancel_sent.add(oid)
            else:
                live = True
        if allowed and not live:
            side = Side.BUY if sign > 0 else Side.SELL
            oid = self.send_order(OrderRequest(side=side, price=price, quantity=self.QTY, order_type=OrderType.LIMIT))
            self.orders[oid] = [sign, price, self.QTY]

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        st = event.status
        if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
            self.inv += o[0] * event.filled_quantity
            o[2] -= event.filled_quantity
            if o[2] <= 0:
                self._drop(event.order_id)
        elif st == OrderStatus.CANCELED or st == OrderStatus.REJECTED:
            self._drop(event.order_id)

    def _drop(self, oid: str) -> None:
        self.orders.pop(oid, None)
        self.cancel_sent.discard(oid)
