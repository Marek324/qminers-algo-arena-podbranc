"""Take the stale book when the hidden fair price S has moved away from it, unwind passively.

Market (round2_public, docs/DATA_ANALYSIS.md): new limit orders are placed at floor(S - X) / ceil(S + X)
around a hidden price S, clamped so they never cross the book. S diffuses, is pushed by IOCs and
sometimes jumps 30-100 ticks; the visible book lags it by a few hundred ms. In volatile spells the lag is
several ticks, so asks below S (bids above S) are free money for whoever takes them first.

Each event:
1. S_hat: EMA over new market limit orders of the reading p + C (buy) or p - C (sell). A buy at
   best_ask - 1 or above (a sell at best_bid + 1 or below) may have been clamped, so it only says S is at
   least (at most) the reading: it moves S_hat up (down) but never back.
2. Take: if S_hat - best_ask >= TAKE_TH, IOC-buy up to price floor(S_hat - TAKE_PX) for the room left
   under MAX_POS (counting resting bids and IOCs in flight). Sells mirror it.
3. Quote: fair = S_hat - K_INV * inventory. Bid at the highest price <= best bid with fair - p >= EDGE,
   ask mirrors it. Orders keep their queue place while within KEEP_LEVELS behind the target; orders
   more aggressive than the target are cancelled.

Inventory is counted from fills only.
"""

from __future__ import annotations

import math

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

MAX_POS = 100
QTY = 100             # passive size per side
EDGE = 0.5            # min ticks between a passive quote and fair
K_INV = 0.01          # ticks of fair shift per lot of inventory
KEEP_LEVELS = 1
MIN_NEW = 5
C = 1.1               # gap between a new order's price and S
ALPHA = 0.2           # weight of each new order in S_hat
S_LEVELS = 3          # scan this many ticks behind the best price for new orders
TAKE_TH = 1.0         # take when S_hat is this far through the opposite best price
TAKE_PX = 0.5         # IOC limit: S_hat minus this (buy), so every lot taken has at least this edge
TAKE_COOLDOWN_US = 0  # min time between IOCs on one side
EXIT_ONLY = 1         # 1: rest passive orders only on the side that reduces inventory


class LagTaker(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        self.orders: dict[str, list] = {}     # resting limit orders: id -> [sign, price, remaining]
        self.iocs: dict[str, list] = {}       # IOCs in flight: id -> [sign, remaining]
        self.cancel_sent: set[str] = set()
        self.s = 0.0
        self.max_id = ''
        self.last_bids = None
        self.last_asks = None
        self.bb = 0
        self.ba = 0
        self.last_take = [0, 0]

    def _cancel(self, oid: str) -> None:
        if oid not in self.cancel_sent:
            self.cancel_order(oid)
            self.cancel_sent.add(oid)

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bids, asks = snapshot.bids, snapshot.asks
        if not bids or not asks:
            return
        bb = ba = None
        for o in bids:
            if o.owner == 'market':
                bb = o.price
                break
        for o in asks:
            if o.owner == 'market':
                ba = o.price
                break
        if bb is None or ba is None:
            return
        self.bb, self.ba = bb, ba
        if not self.s:
            self.s = (bb + ba) / 2
        if bids is not self.last_bids:
            self.last_bids = bids
            self._scan(bids, -1)
        if asks is not self.last_asks:
            self.last_asks = asks
            self._scan(asks, 1)
        s = self.s
        now = snapshot.timestamp_us

        # take the stale side
        if s - ba >= TAKE_TH and now - self.last_take[0] >= TAKE_COOLDOWN_US:
            room = MAX_POS - self.inv - self._exposure(1)
            if room > 0:
                px = max(ba, math.floor(s - TAKE_PX))
                oid = self.send_order(OrderRequest(side=Side.BUY, price=px, quantity=room, order_type=OrderType.IOC))
                self.iocs[oid] = [1, room]
                self.last_take[0] = now
        elif bb - s >= TAKE_TH and now - self.last_take[1] >= TAKE_COOLDOWN_US:
            room = MAX_POS + self.inv - self._exposure(-1)
            if room > 0:
                px = min(bb, math.ceil(s + TAKE_PX))
                oid = self.send_order(OrderRequest(side=Side.SELL, price=px, quantity=room, order_type=OrderType.IOC))
                self.iocs[oid] = [-1, room]
                self.last_take[1] = now

        fair = s - K_INV * self.inv
        pb = min(bb, math.floor(fair - EDGE))
        pa = max(ba, math.ceil(fair + EDGE))
        if EXIT_ONLY:
            if self.inv > 0:
                self._side(-1, pa, self.inv)
                self._side(1, pb, 0)
            elif self.inv < 0:
                self._side(1, pb, -self.inv)
                self._side(-1, pa, 0)
            else:
                self._side(1, pb, 0)
                self._side(-1, pa, 0)
        else:
            self._side(1, pb, QTY)
            self._side(-1, pa, QTY)

    def _exposure(self, sign: int) -> int:
        """Lots on `sign`'s side that could still fill: resting orders not being cancelled, plus IOCs in flight."""
        r = 0
        for o in self.orders.values():
            if o[0] == sign:
                r += o[2]
        for o in self.iocs.values():
            if o[0] == sign:
                r += o[1]
        return r

    def _scan(self, views, sign: int) -> None:
        first = views[0].price
        mx = self.max_id
        lm = len(mx)
        new_mx = mx
        s = self.s
        a = ALPHA
        # the opposite market best price, for the clamp check
        if sign < 0:
            lim = self.ba - 1
        else:
            lim = self.bb + 1
        for o in views:
            if (o.price - first) * sign > S_LEVELS:
                break
            if o.owner != 'market':
                continue
            oid = o.order_id
            if len(oid) > lm or (len(oid) == lm and oid > mx):
                p = o.price
                if sign < 0:
                    r = p + C
                    if p >= lim and r < s:
                        r = s
                else:
                    r = p - C
                    if p <= lim and r > s:
                        r = s
                s += a * (r - s)
                if len(oid) > len(new_mx) or (len(oid) == len(new_mx) and oid > new_mx):
                    new_mx = oid
        self.s = s
        self.max_id = new_mx

    def _side(self, sign: int, target: int, qty: int) -> None:
        resting = 0
        at_target = 0
        keep = []
        for oid, o in self.orders.items():
            if o[0] != sign:
                continue
            resting += o[2]
            if oid in self.cancel_sent:
                continue
            behind = (target - o[1]) * sign
            if behind < 0 or behind > KEEP_LEVELS:
                self._cancel(oid)
                continue
            if behind == 0:
                at_target += o[2]
            else:
                keep.append((behind, oid))
        if at_target >= qty:
            return
        inflight = 0
        for o in self.iocs.values():
            if o[0] == sign:
                inflight += o[1]
        room = MAX_POS - sign * self.inv - resting - inflight
        want = qty - at_target
        if room < min(want, MIN_NEW) and keep:
            keep.sort(reverse=True)
            self._cancel(keep[0][1])
            return
        q = min(want, room)
        if q < MIN_NEW:
            return
        side = Side.BUY if sign > 0 else Side.SELL
        oid = self.send_order(OrderRequest(side=side, price=target, quantity=q, order_type=OrderType.LIMIT))
        self.orders[oid] = [sign, target, q]

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        oid = event.order_id
        st = event.status
        o = self.orders.get(oid)
        if o is not None:
            if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
                self.inv += o[0] * event.filled_quantity
                o[2] -= event.filled_quantity
                if o[2] <= 0:
                    self._drop(oid)
            elif st == OrderStatus.CANCELED:
                self._drop(oid)
            elif st == OrderStatus.REJECTED and oid not in self.cancel_sent:
                self._drop(oid)
            return
        o = self.iocs.get(oid)
        if o is not None:
            if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
                self.inv += o[0] * event.filled_quantity
                o[1] -= event.filled_quantity
                if o[1] <= 0:
                    self.iocs.pop(oid, None)
            elif st == OrderStatus.CANCELED or st == OrderStatus.REJECTED:
                self.iocs.pop(oid, None)

    def _drop(self, oid: str) -> None:
        self.orders.pop(oid, None)
        self.cancel_sent.discard(oid)
