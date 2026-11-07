"""Queue-keeping market maker, skewed by touch imbalance.

The market: a deep book (hundreds of lots per level), a 1-2 tick spread, and
a heavy stream of mostly uninformed IOC orders (the mid barely moves after
them). Resting orders earn the spread, so profit is set by how many lots get
filled, and that depends on queue position.

Each event:
1. fair = mid + K_IMB * imbalance - K_INV * inventory, where
   imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty) at the touch.
2. Target bid: the highest price p <= best_bid + 1 (inside the spread when
   it is 2+ ticks wide), below the best ask, with fair - p >= EDGE. The ask
   mirrors it.
3. Orders are sticky: an order that ended up behind the target price keeps
   resting (and keeps its queue place) while it is within KEEP_LEVELS of the
   target. Orders priced more aggressively than the target, or deeper than
   KEEP_LEVELS, are cancelled.
4. New size goes to the target price, QTY per order. Inventory plus all our
   resting orders on a side stays within MAX_POS; when there is no room for
   the target order, the deepest resting order is cancelled to make room.

Inventory is counted from fills only.
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


class ImbalanceMMStrategy(BaseStrategy):
    QTY = 200
    MAX_POS = 100
    EDGE = 0.5           # min ticks between the target price and fair
    K_IMB = 0.3          # ticks of fair shift at full imbalance
    K_INV = 0.002        # ticks of fair shift per lot of inventory
    KEEP_LEVELS = 1      # how many ticks behind the target an order may stay
    MIN_NEW = 10         # don't send orders smaller than this

    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        # order_id -> [sign, price, remaining]
        self.orders: dict[str, list] = {}
        self.cancel_sent: set[str] = set()

    def _cancel(self, oid: str) -> None:
        if oid not in self.cancel_sent:
            self.cancel_order(oid)
            self.cancel_sent.add(oid)

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bids, asks = snapshot.bids, snapshot.asks
        if not bids or not asks:
            return
        # touch of market orders only (ours excluded)
        bb = ba = None
        bq = aq = 0
        for o in bids:
            if o.owner == 'market':
                if bb is None:
                    bb = o.price
                elif o.price != bb:
                    break
                bq += o.quantity
        for o in asks:
            if o.owner == 'market':
                if ba is None:
                    ba = o.price
                elif o.price != ba:
                    break
                aq += o.quantity
        if bb is None or ba is None:
            return

        fair = (bb + ba) / 2 + self.K_IMB * (bq - aq) / (bq + aq) - self.K_INV * self.inv

        pb = min(bb + 1, ba - 1)
        while fair - pb < self.EDGE:
            pb -= 1
        pa = max(ba - 1, bb + 1)
        while pa - fair < self.EDGE:
            pa += 1
        self._side(1, pb)
        self._side(-1, pa)

    def _side(self, sign: int, target: int) -> None:
        resting = 0
        at_target = 0
        keep = []
        for oid, o in self.orders.items():
            if o[0] != sign or oid in self.cancel_sent:
                if o[0] == sign:
                    resting += o[2]
                continue
            behind = (target - o[1]) * sign      # >0: less aggressive than target
            if behind < 0 or behind > self.KEEP_LEVELS:
                self._cancel(oid)
                resting += o[2]
                continue
            resting += o[2]
            if behind == 0:
                at_target += o[2]
            else:
                keep.append((behind, oid, o[2]))
        if at_target >= self.QTY:
            return
        room = self.MAX_POS - sign * self.inv - resting
        want = self.QTY - at_target
        if room < min(want, self.MIN_NEW) and keep:
            # free room: cancel the deepest kept order
            keep.sort(reverse=True)
            self._cancel(keep[0][1])
            return
        q = min(want, room)
        if q < self.MIN_NEW:
            return
        side = Side.BUY if sign > 0 else Side.SELL
        oid = self.send_order(OrderRequest(side=side, price=target, quantity=q, order_type=OrderType.LIMIT))
        self.orders[oid] = [sign, target, q]

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        oid = event.order_id
        o = self.orders.get(oid)
        if o is None:
            return
        st = event.status
        if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
            self.inv += o[0] * event.filled_quantity
            o[2] -= event.filled_quantity
            if o[2] <= 0:
                self._drop(oid)
        elif st == OrderStatus.CANCELED:
            self._drop(oid)
        elif st == OrderStatus.REJECTED and oid not in self.cancel_sent:
            self._drop(oid)

    def _drop(self, oid: str) -> None:
        self.orders.pop(oid, None)
        self.cancel_sent.discard(oid)
