"""Queue-priority market maker that may go past the ±100 inventory limit for a few milliseconds.

The same quoting as queue_mm.py (read that docstring first), with one risk rule changed. The exchange charges
10,000 per second while |inventory| > 100, whatever the size and for the exact time. Briefly being over is cheap
(10 ms costs 100); staying over is ruinous. So this version lets inventory plus resting orders on a side reach
HARD_POS instead of 100, capturing more of the flow, and gets back under 100 fast:
- while |inventory| > SOFT_POS: no new orders on the side that adds to it, the unwinding side quotes at the touch
  (EDGE does not apply there), and once the excursion has lasted MAX_OVER_US it sends an IOC across the spread for the
  excess (one at a time);
- resting orders on the adding side beyond HARD_POS worst case are cancelled.

Whether the extra fills pay for the penalty is an empirical question: compare with queue_mm.py on the same sessions.

Round 1 (see docs/round1.md): the spread is 1 tick 93% of the time and about 900 lots wait at the best price, while
the median aggressive order is 60 lots. An order at the back of that queue is filled only by big sweeps, which are the
fills that lose (the price keeps moving through). An order near the front is filled by small orders and keeps its
edge. So this strategy is built around queue position; the fair price only decides where we must not be.

Fair price (same as imbalance_mm.py):
    fair = mid + K_S * (S_hat - mid) + K_IMB * imbalance + K_FLOW * flow / depth - K_INV * inventory
S_hat is the average of the bid-side and ask-side running means of newly arrived market limit order prices.

Quoting, per side (the ask mirrors the bid):
1. Target price: the best ask if fair - best_ask >= EDGE_TAKE (cross in), otherwise the highest price p <= best bid
   with fair - p >= EDGE. If the market queue at the best bid is thinner than THIN_QUEUE lots and we are already long
   more than THIN_INV lots, the target steps back one tick (hftbacktest's queue rule: a thin queue is about to be traded
   through, and we don't want to add to the side we are long).
2. Ladder: we want an order on each of LEVELS prices from the target outward (target, target - 1, ...). Deeper orders
   are pre-positioned: when the price moves onto them they are already near the front of the queue.
3. Keep, don't chase: an existing order stays as long as fair - price >= KEEP_EDGE (lower than EDGE, so small moves of
   the fair price don't throw away a queue place) and it is at most MAX_BEHIND ticks behind the target. Everything
   else is cancelled.
4. New size: up to QTY_TOP at the target and QTY_DEEP at deeper ladder prices (topped up after partial fills), in
   that order, while inventory plus all our
   resting orders on that side stays within MAX_POS (the worst case if all of them fill).

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


class PenaltyMMStrategy(BaseStrategy):
    MAX_POS = 250        # HARD_POS: worst-case inventory (inventory + resting orders) allowed per side
    SOFT_POS = 100       # the exchange's penalty threshold
    MAX_OVER_US = 10_000 # cross the spread to unwind once over SOFT_POS this long
    UNWIND_QTY_MIN = 1   # smallest unwind IOC
    LEVELS = 1           # prices we hold an order on, from the target outward (2+ lost: ties up room)
    QTY_TOP = 120        # size of a new order at the target price (beat 200 on synthetic round-1 sessions)
    QTY_DEEP = 60        # size of a new order at deeper ladder prices
    MIN_NEW = 10         # don't send orders smaller than this
    EDGE = 0.4           # min ticks between fair and a price to place a new order there
    KEEP_EDGE = 0.2      # an existing order stays while fair is at least this far on its good side
    MAX_BEHIND = 1       # cancel orders more than this many ticks behind the target
    EDGE_TAKE = 0.25     # cross the spread when fair is this far beyond the opposite best price
    THIN_QUEUE = 300     # market lots at the best price below which the queue counts as thin
    THIN_INV = 1000      # step back from a thin queue when this long (short); 1000 = off (30 cost about 1%)
    K_IMB = 0.15         # ticks of fair shift at full imbalance
    K_INV = 0.002        # ticks of fair shift per lot of inventory
    K_FLOW = 0.5         # ticks of fair shift per touch-depth of recent aggressive volume
    FLOW_TAU_US = 200_000
    DEPTH_EMA = 0.001
    K_S = 0.75           # weight of the hidden-price estimate
    S_ALPHA = 0.1        # weight of each new limit order in the per-side price average
    S_LEVELS = 2         # look for new orders this many ticks behind the best price

    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        self.orders: dict[str, list] = {}  # order_id -> [sign, price, remaining]
        self.cancel_sent: set[str] = set()
        self.flow = 0.0
        self.flow_t = 0
        self.depth = 0.0
        self.s_bid = 0.0
        self.s_ask = 0.0
        self.max_id = ''
        self.last_bids = None
        self.last_asks = None
        self.over_since = -1     # time inventory went past SOFT_POS, -1 when within
        self.unwind_id = ''      # unwind IOC in flight

    # ---------------------------------------------------------------- fair price
    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bids, asks = snapshot.bids, snapshot.asks
        if not bids or not asks:
            return
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

        mid = (bb + ba) / 2
        fair = mid + self.K_IMB * (bq - aq) / (bq + aq) - self.K_INV * self.inv
        if self.K_S:
            if bids is not self.last_bids:
                self.last_bids = bids
                self._scan_new(bids, -1)
            if asks is not self.last_asks:
                self.last_asks = asks
                self._scan_new(asks, 1)
            if self.s_bid and self.s_ask:
                fair += self.K_S * ((self.s_bid + self.s_ask) / 2 - mid)
        if self.K_FLOW:
            now = snapshot.timestamp_us
            self.flow *= math.exp((self.flow_t - now) / self.FLOW_TAU_US)
            self.flow_t = now
            for tr in trades:
                if tr.taker_owner == 'market':
                    self.flow += tr.quantity if tr.book_side == 'ask' else -tr.quantity
            half = (bq + aq) / 2
            self.depth = half if self.depth == 0 else self.depth + self.DEPTH_EMA * (half - self.depth)
            fair += self.K_FLOW * self.flow / self.depth

        # penalty zone: get back under SOFT_POS quickly
        now = snapshot.timestamp_us
        over = abs(self.inv) - self.SOFT_POS
        if over > 0:
            if self.over_since < 0:
                self.over_since = now
            if not self.unwind_id and now - self.over_since >= self.MAX_OVER_US and over >= self.UNWIND_QTY_MIN:
                if self.inv > 0:
                    self.unwind_id = self.send_order(OrderRequest(side=Side.SELL, price=bb, quantity=over, order_type=OrderType.IOC))
                    self.orders[self.unwind_id] = [-1, bb, over]
                else:
                    self.unwind_id = self.send_order(OrderRequest(side=Side.BUY, price=ba, quantity=over, order_type=OrderType.IOC))
                    self.orders[self.unwind_id] = [1, ba, over]
        else:
            self.over_since = -1
        long_over, short_over = self.inv > self.SOFT_POS, -self.inv > self.SOFT_POS

        # target prices
        if fair - ba >= self.EDGE_TAKE:
            pb = ba
        else:
            pb = bb
            while fair - pb < self.EDGE:
                pb -= 1
            if pb == bb and bq < self.THIN_QUEUE and self.inv > self.THIN_INV:
                pb -= 1
        if short_over:
            pb = max(pb, bb)     # unwinding side: quote at the touch
        if bb - fair >= self.EDGE_TAKE:
            pa = bb
        else:
            pa = ba
            while pa - fair < self.EDGE:
                pa += 1
            if pa == ba and aq < self.THIN_QUEUE and -self.inv > self.THIN_INV:
                pa += 1
        if long_over:
            pa = min(pa, ba)
        self._side(1, pb, fair, adding_blocked=long_over, unwinding=short_over)
        self._side(-1, pa, fair, adding_blocked=short_over, unwinding=long_over)

    def _scan_new(self, views, sign: int) -> None:
        """Fold market orders newer than max_id, within S_LEVELS of the best price, into the side's average."""
        first = views[0].price
        a = self.S_ALPHA
        for o in views:
            if (o.price - first) * sign > self.S_LEVELS:
                break
            if o.owner != 'market':
                continue
            oid = o.order_id
            if len(oid) > len(self.max_id) or (len(oid) == len(self.max_id) and oid > self.max_id):
                if sign < 0:
                    self.s_bid = o.price if not self.s_bid else self.s_bid + a * (o.price - self.s_bid)
                else:
                    self.s_ask = o.price if not self.s_ask else self.s_ask + a * (o.price - self.s_ask)
                self.max_id = oid

    # ---------------------------------------------------------------- quoting
    def _cancel(self, oid: str) -> None:
        if oid not in self.cancel_sent:
            self.cancel_order(oid)
            self.cancel_sent.add(oid)

    def _side(self, sign: int, target: int, fair: float, adding_blocked: bool = False, unwinding: bool = False) -> None:
        resting = 0
        have: dict[int, int] = {}  # price -> our live lots there
        for oid, o in self.orders.items():
            if o[0] != sign:
                continue
            resting += o[2]
            if oid in self.cancel_sent:
                continue
            price = o[1]
            edge = (fair - price) * sign            # how far fair is on the good side of this price
            behind = (target - price) * sign        # >0: deeper than the target
            if (edge < self.KEEP_EDGE and not unwinding) or behind > self.MAX_BEHIND:
                self._cancel(oid)
                continue
            have[price] = have.get(price, 0) + o[2]
        room = self.MAX_POS - sign * self.inv - resting
        if adding_blocked:
            return
        for k in range(self.LEVELS):
            price = target - sign * k
            want = (self.QTY_TOP if k == 0 else self.QTY_DEEP) - have.get(price, 0)   # top up after partial fills
            if want < self.MIN_NEW:
                continue
            q = min(want, room)
            if q < self.MIN_NEW:
                if k == 0:
                    self._free_room(sign, target)   # the order at the target earns the most: make room for it
                return
            side = Side.BUY if sign > 0 else Side.SELL
            oid = self.send_order(OrderRequest(side=side, price=price, quantity=q, order_type=OrderType.LIMIT))
            self.orders[oid] = [sign, price, q]
            room -= q

    def _free_room(self, sign: int, target: int) -> None:
        """Cancel our deepest live order behind the target on this side, so the next event can place at the target."""
        deepest, deepest_id = 0, ''
        for oid, o in self.orders.items():
            if o[0] != sign or oid in self.cancel_sent:
                continue
            behind = (target - o[1]) * sign
            if behind > deepest:
                deepest, deepest_id = behind, oid
        if deepest_id:
            self._cancel(deepest_id)

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        oid = event.order_id
        o = self.orders.get(oid)
        if o is None:
            return
        st = event.status
        if oid == self.unwind_id and st in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            self.unwind_id = ''
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
