"""Regime-switching market maker: make in quiet spells, only snipe the stale book in volatile ones.

Market (round2_public, docs/DATA_ANALYSIS.md): limit orders arrive at floor(S - X) / ceil(S + X) around a
hidden price S and are clamped so they never cross. The generator switches between two regimes:
- quiet (about 85% of the time): new buys and sells sit about 2 ticks apart, S moves about 1 tick/sqrt(s);
- volatile: they sit about 4.3 ticks apart, IOCs come 1.5x as often and S moves 4-6 ticks/sqrt(s), with
  most of the 30-100 tick jumps.
A market maker that makes about 235k in the quiet spells loses about 300k in the volatile ones.

Each event:
1. S_hat = average of the per-side EMAs of new market limit order prices (top S_LEVELS ticks).
   Regime: EMA over new orders (top REG_LEVELS ticks) of |price - mid|; volatile above REG_ON, back to quiet
   below REG_OFF, and never sooner than REG_HOLD_US after the last reading above REG_ON.
2. Quiet: imbalance_mm. fair = mid + K_IMB * imbalance + K_S * (S_hat - mid) + K_FLOW * flow / depth
   - K_INV * inventory; quote at the best price with at least EDGE to fair, cross when fair is EDGE_TAKE
   through the opposite best price.
3. Volatile: cancel the quotes. If S_hat - best_ask >= V_TAKE, IOC-buy up to floor(S_hat - V_TAKE_PX)
   (sells mirror it). Hold the position with one passive exit order at S_hat rounded away from the touch,
   and close what is left by IOC once S_hat says the position no longer has V_KEEP of edge, or after
   V_MAX_HOLD_US.

Inventory is counted from fills only.
"""

from __future__ import annotations

import math
from collections import deque

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

QTY = 200
MAX_POS = 100
EDGE = 0.3
K_IMB = 0.15
K_INV = 0.002
KEEP_LEVELS = 1
MIN_NEW = 10
EDGE_IN = 0.5
K_FLOW = 0.8
FLOW_TAU_US = 100_000
DEPTH_EMA = 0.001
K_S = 0.75
S_ALPHA = 0.1
S_LEVELS = 2
EDGE_TAKE = 0.25
FLOW_CHECK = True
FLOW_SLOPE_FULL = 0.5
FLOW_CHECK_TAU_US = 300_000_000
FLOW_CHECK_STEP_US = 50_000
FLOW_CHECK_H_US = 500_000
FLOW_CHECK_MIN = 200.0

REG_ON = 1.6          # regime score (ticks) that switches to volatile
REG_OFF = 1.35        # and back to quiet
REG_ALPHA = 0.03      # weight of each new order in the regime score
REG_LEVELS = 8        # scan depth for the regime score
REG_HOLD_US = 3_000_000
VOLATILE_MODE = 1     # 0: keep making (baseline), 1: flat + snipe
V_TAKE = 0.75          # S_hat past the opposite best price to take
V_TAKE_PX = 0.5       # IOC limit S_hat -/+ this
V_KEEP = -1.0          # close by IOC once S_hat - best_bid (long) drops below this
V_MAX_HOLD_US = 20_000_000
V_EXIT_OFF = 0.0      # passive exit price: S_hat + this (long), rounded up, never inside the touch
C = 1.1               # gap between a new order's price and S, for the clamp-aware estimate S2
S2_ALPHA = 0.25        # weight of each new order in S2
V_PASSIVE_EXIT = 0    # rest an exit order at S2 while holding
Q_TAKE = 99.0         # quiet regime: when S2 is this far through the opposite best price, bid up to S2 - Q_TAKE_PX
Q_TAKE_PX = 0.5
FAIR_S2 = 0             # quiet fair uses S2 instead of the per-side average
V_FLIP = 0.75         # volatile: threshold instead of V_TAKE when the trade would flip an opposite position
V_FOLLOW = 1e9        # volatile: market IOC volume (lots, this event) that triggers taking its side


class RegimeMM(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        self.orders: dict[str, list] = {}   # resting: id -> [sign, price, remaining]
        self.iocs: dict[str, list] = {}     # in flight: id -> [sign, remaining]
        self.cancel_sent: set[str] = set()
        self.flow = 0.0
        self.flow_t = 0
        self.depth = 0.0
        self.s_bid = 0.0
        self.s_ask = 0.0
        self.max_id = ''
        self.last_bids = None
        self.last_asks = None
        self.fc_pending: deque = deque()
        self.fc_next = 0
        self.fc_sxx = 0.0
        self.fc_sxy = 0.0
        self.fc_n = 0.0
        self.fc_decay = math.exp(-FLOW_CHECK_STEP_US / FLOW_CHECK_TAU_US)
        self.flow_scale = 0.0 if FLOW_CHECK else 1.0
        self.reg = 1.0
        self.volatile = False
        self.reg_t = 0
        self.mid = 0.0
        self.pos_t = 0          # when the current volatile position was opened
        self.s2 = 0.0           # clamp-aware estimate of S over all scanned new orders
        self.bb = 0
        self.ba = 0

    def _cancel(self, oid: str) -> None:
        if oid not in self.cancel_sent:
            self.cancel_order(oid)
            self.cancel_sent.add(oid)

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
        now = snapshot.timestamp_us
        mid = (bb + ba) / 2
        self.mid = mid
        self.bb, self.ba = bb, ba
        if not self.s2:
            self.s2 = mid
        if bids is not self.last_bids:
            self.last_bids = bids
            self._scan_new(bids, -1)
        if asks is not self.last_asks:
            self.last_asks = asks
            self._scan_new(asks, 1)
        if self.reg > REG_ON:
            self.reg_t = now
            self.volatile = True
        elif self.volatile and self.reg < REG_OFF and now - self.reg_t >= REG_HOLD_US:
            self.volatile = False
        s_hat = (self.s_bid + self.s_ask) / 2 if self.s_bid and self.s_ask else mid

        # flow (kept up to date in both regimes)
        self.flow *= math.exp((self.flow_t - now) / FLOW_TAU_US)
        self.flow_t = now
        agg = 0
        for tr in trades:
            if tr.taker_owner == 'market':
                agg += tr.quantity if tr.book_side == 'ask' else -tr.quantity
        self.flow += agg
        half = (bq + aq) / 2
        self.depth = half if self.depth == 0 else self.depth + DEPTH_EMA * (half - self.depth)
        x = self.flow / self.depth
        if FLOW_CHECK and now >= self.fc_next:
            self._flow_check(now, x, mid)

        if self.volatile and VOLATILE_MODE:
            self._volatile(now, bb, ba, self.s2, agg)
            return

        fair = mid + K_IMB * (bq - aq) / (bq + aq) - K_INV * self.inv
        fair += K_S * ((self.s2 if FAIR_S2 else s_hat) - mid)
        fair += K_FLOW * self.flow_scale * x
        if fair - ba >= EDGE_TAKE:
            pb = ba
        elif ba - bb >= 2 and fair - (bb + 1) >= EDGE_IN:
            pb = bb + 1
        else:
            pb = bb
            while fair - pb < EDGE:
                pb -= 1
        if bb - fair >= EDGE_TAKE:
            pa = bb
        elif ba - bb >= 2 and (ba - 1) - fair >= EDGE_IN:
            pa = ba - 1
        else:
            pa = ba
            while pa - fair < EDGE:
                pa += 1
        s2 = self.s2
        if s2 - ba >= Q_TAKE:
            pb = max(pb, math.floor(s2 - Q_TAKE_PX))
        elif bb - s2 >= Q_TAKE:
            pa = min(pa, math.ceil(s2 + Q_TAKE_PX))
        self._side(1, pb, QTY)
        self._side(-1, pa, QTY)

    def _volatile(self, now: int, bb: int, ba: int, s: float, agg: int) -> None:
        inv = self.inv
        busy = bool(self.iocs)
        # entries
        if not busy:
            if (s - ba >= (V_FLIP if inv < 0 else V_TAKE) or agg >= V_FOLLOW) and inv < MAX_POS:
                room = MAX_POS - inv - self._resting(1)
                if room > 0:
                    if inv <= 0:
                        self.pos_t = now
                    self._ioc(1, max(ba, math.floor(s - V_TAKE_PX)), room)
                    busy = True
            elif (bb - s >= (V_FLIP if inv > 0 else V_TAKE) or agg <= -V_FOLLOW) and inv > -MAX_POS:
                room = MAX_POS + inv - self._resting(-1)
                if room > 0:
                    if inv >= 0:
                        self.pos_t = now
                    self._ioc(-1, min(bb, math.ceil(s + V_TAKE_PX)), room)
                    busy = True
        # exits
        if inv > 0:
            if not busy and (s - bb < V_KEEP or now - self.pos_t >= V_MAX_HOLD_US):
                self._cancel_side(-1)
                self._cancel_side(1)
                q = min(inv, MAX_POS + inv - self._resting(-1))
                if q > 0:
                    self._ioc(-1, bb, q)
                return
            if V_PASSIVE_EXIT:
                self._side(-1, max(ba, math.ceil(s + V_EXIT_OFF)), inv)
            self._cancel_side(1)
        elif inv < 0:
            if not busy and (ba - s < V_KEEP or now - self.pos_t >= V_MAX_HOLD_US):
                self._cancel_side(1)
                self._cancel_side(-1)
                q = min(-inv, MAX_POS - inv - self._resting(1))
                if q > 0:
                    self._ioc(1, ba, q)
                return
            if V_PASSIVE_EXIT:
                self._side(1, min(bb, math.floor(s - V_EXIT_OFF)), -inv)
            self._cancel_side(-1)
        else:
            self._cancel_side(1)
            self._cancel_side(-1)

    def _ioc(self, sign: int, px: int, q: int) -> None:
        side = Side.BUY if sign > 0 else Side.SELL
        oid = self.send_order(OrderRequest(side=side, price=px, quantity=q, order_type=OrderType.IOC))
        self.iocs[oid] = [sign, q]

    def _resting(self, sign: int) -> int:
        r = 0
        for o in self.orders.values():
            if o[0] == sign:
                r += o[2]
        for o in self.iocs.values():
            if o[0] == sign:
                r += o[1]
        return r

    def _cancel_side(self, sign: int) -> None:
        for oid, o in self.orders.items():
            if o[0] == sign and oid not in self.cancel_sent:
                self._cancel(oid)

    def _flow_check(self, now: int, x: float, mid: float) -> None:
        self.fc_next = now + FLOW_CHECK_STEP_US
        pend = self.fc_pending
        d = self.fc_decay
        while pend and now - pend[0][0] >= FLOW_CHECK_H_US:
            _, x0, m0 = pend.popleft()
            y = mid - m0
            self.fc_sxx = self.fc_sxx * d + x0 * x0
            self.fc_sxy = self.fc_sxy * d + x0 * y
            self.fc_n = self.fc_n * d + 1.0
        pend.append((now, x, mid))
        if self.fc_n < FLOW_CHECK_MIN or self.fc_sxx <= 0:
            self.flow_scale = 0.0
            return
        slope = self.fc_sxy / self.fc_sxx
        self.flow_scale = min(1.0, max(0.0, slope / FLOW_SLOPE_FULL))

    def _scan_new(self, views, sign: int) -> None:
        first = views[0].price
        mx = self.max_id
        lm = len(mx)
        new_mx = mx
        a = S_ALPHA
        ra = REG_ALPHA
        mid = self.mid
        s2 = self.s2
        a2 = S2_ALPHA
        for o in views:
            d = (o.price - first) * sign
            if d > REG_LEVELS:
                break
            if o.owner != 'market':
                continue
            oid = o.order_id
            if len(oid) > lm or (len(oid) == lm and oid > mx):
                p = o.price
                self.reg += ra * (abs(p - mid) - self.reg)
                if sign < 0:
                    r = p + C
                    if p >= self.ba - 1 and r < s2:
                        r = s2
                else:
                    r = p - C
                    if p <= self.bb + 1 and r > s2:
                        r = s2
                s2 += a2 * (r - s2)
                if d <= S_LEVELS:
                    if sign < 0:
                        self.s_bid = p if not self.s_bid else self.s_bid + a * (p - self.s_bid)
                    else:
                        self.s_ask = p if not self.s_ask else self.s_ask + a * (p - self.s_ask)
                if len(oid) > len(new_mx) or (len(oid) == len(new_mx) and oid > new_mx):
                    new_mx = oid
        self.max_id = new_mx
        self.s2 = s2

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
        for o in self.iocs.values():
            if o[0] == sign:
                resting += o[1]
        room = MAX_POS - sign * self.inv - resting
        want = qty - at_target
        if room < min(want, MIN_NEW) and keep:
            keep.sort(reverse=True)
            self._cancel(keep[0][1])
            return
        q = min(want, room)
        if q < min(MIN_NEW, qty):
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
