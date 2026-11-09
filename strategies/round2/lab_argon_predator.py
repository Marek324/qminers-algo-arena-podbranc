"""Regime-switching market maker with jump snapping.

S estimate s2 from new limit orders. A new buy order at s2 + JUMP_SNAP_TICKS or above (sells mirror it), or a taker
trade that far past s2, means S has jumped: s2 snaps to the new level, volatile mode starts, all quotes are
cancelled and an IOC takes up to MAX_POS in the jump direction at the stale touch, without waiting for the cancels.
Volatile: holds the position (no forced exit, no passive exit).
Quiet: ladder market maker (KEEP_LEVELS levels, KEEP_EDGE hysteresis), room freed by shrinking orders with
modify_order. A side whose book empties has its quotes cancelled at once.
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

# Quoting & Risk Parameters
QTY = 200
MAX_POS = 100
MIN_NEW = 10

# Maker quote pricing & queue seniority
EDGE = 0.28
EDGE_IN = 0.45
EDGE_TAKE = 0.25
KEEP_EDGE = 0.22
KEEP_LEVELS = 2
USE_SHRINK = True

# Signals & Fair Value Parameters
K_IMB = 0.15
K_INV = 0.0014
K_S = 0.82
S_ALPHA = 0.10
S_LEVELS = 2
K_FLOW = 0.80
FLOW_TAU_US = 100_000
DEPTH_EMA = 0.001
FLOW_CHECK = True
FLOW_SLOPE_FULL = 0.5
FLOW_CHECK_TAU_US = 300_000_000
FLOW_CHECK_STEP_US = 50_000
FLOW_CHECK_H_US = 500_000
FLOW_CHECK_MIN = 200.0

# Non-HMM Regime Filter (EMA of |price - mid|)
REG_ON = 1.60
REG_OFF = 1.35
REG_ALPHA = 0.03
REG_LEVELS = 16
REG_HOLD_US = 2_000_000
VOLATILE_MODE = 1

# Volatile Execution & Holding Policy
V_TAKE = 0.75
V_TAKE_PX = 0.50
V_FLIP = 0.75
V_FOLLOW = 1e9
V_KEEP = -999.0        # Strict HOLD policy
V_MAX_HOLD_US = 999_999_999
V_PASSIVE_EXIT = 0

# S-Estimation & Jump Snapping
C = 1.10
S2_ALPHA = 0.25
JUMP_SNAP_TICKS = 3.0
IOC_SNAP = 1           # 1: enable instant jump snap on IOC trade sweeps
FAIR_S2 = 0            # 0: quiet fair uses per-side s_hat, 1: uses S2

# Fast unblocked jump snipe
UNBLOCKED_SNIPE = 1    # 1: compute room without waiting 5ms for stale cancels

# Quiet opportunistic take
Q_TAKE = 99.0
Q_TAKE_PX = 0.50


class LabArgonPredator(BaseStrategy):
    """Regime-switching market maker with jump snapping."""

    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        self.orders: dict[str, list] = {}   # oid -> [sign, price, remaining, modifying]
        self.iocs: dict[str, list] = {}     # in flight: oid -> [sign, remaining]
        self.cancel_sent: set[str] = set()

        self.flow = 0.0
        self.flow_t = 0
        self.depth = 0.0
        self.s_bid = 0.0
        self.s_ask = 0.0
        self.max_id = ""
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
        self.pos_t = 0
        self.s2 = 0.0
        self.bb = 0
        self.ba = 0

    def _cancel(self, oid: str) -> None:
        if oid not in self.cancel_sent:
            self.cancel_order(oid)
            self.cancel_sent.add(oid)

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bids, asks = snapshot.bids, snapshot.asks

        # Empty book safety: cancel resting quotes on empty side immediately
        if bids and not asks:
            self._cancel_side(-1)
            return
        if asks and not bids:
            self._cancel_side(1)
            return
        if not bids or not asks:
            return

        bb = ba = None
        bq = aq = 0
        for o in bids:
            if o.owner == "market":
                if bb is None:
                    bb = o.price
                elif o.price != bb:
                    break
                bq += o.quantity
        for o in asks:
            if o.owner == "market":
                if ba is None:
                    ba = o.price
                elif o.price != ba:
                    break
                aq += o.quantity
        if bb is None or ba is None:
            return

        now = snapshot.timestamp_us
        mid = (bb + ba) / 2.0
        self.mid = mid
        self.bb, self.ba = bb, ba
        if not self.s2:
            self.s2 = mid

        # Scan new limit orders with instant jump snapping
        if bids is not self.last_bids:
            self.last_bids = bids
            self._scan_new(bids, -1, now)
        if asks is not self.last_asks:
            self.last_asks = asks
            self._scan_new(asks, 1, now)

        # Process trades into flow and IOC bounds / jump snapping
        agg = 0
        max_buy_trade = 0
        min_sell_trade = 0
        for tr in trades:
            if tr.taker_owner == "market":
                p = tr.price
                if tr.book_side == "ask":
                    agg += tr.quantity
                    if p > max_buy_trade:
                        max_buy_trade = p
                else:
                    agg -= tr.quantity
                    if min_sell_trade == 0 or p < min_sell_trade:
                        min_sell_trade = p

        # IOC-implied S adjustments & fast IOC jump snapping
        if max_buy_trade > 0:
            ioc_bound = max_buy_trade - 1.0
            if IOC_SNAP and ioc_bound >= self.s2 + JUMP_SNAP_TICKS:
                self.s2 = ioc_bound
                self.s_bid = ioc_bound - C
                self.s_ask = ioc_bound + C
                self.reg = max(self.reg, REG_ON + 2.0)
                self.reg_t = now
                self.volatile = True
            elif ioc_bound > self.s2:
                self.s2 += 0.35 * (ioc_bound - self.s2)

        if min_sell_trade > 0:
            ioc_bound = min_sell_trade + 1.0
            if IOC_SNAP and ioc_bound <= self.s2 - JUMP_SNAP_TICKS:
                self.s2 = ioc_bound
                self.s_bid = ioc_bound - C
                self.s_ask = ioc_bound + C
                self.reg = max(self.reg, REG_ON + 2.0)
                self.reg_t = now
                self.volatile = True
            elif ioc_bound < self.s2:
                self.s2 += 0.35 * (ioc_bound - self.s2)

        # Regime switch
        if self.reg > REG_ON:
            self.reg_t = now
            self.volatile = True
        elif self.volatile and self.reg < REG_OFF and now - self.reg_t >= REG_HOLD_US:
            self.volatile = False

        s_hat = (self.s_bid + self.s_ask) / 2.0 if (self.s_bid and self.s_ask) else mid

        # Process flow
        self.flow *= math.exp((self.flow_t - now) / FLOW_TAU_US)
        self.flow_t = now
        self.flow += agg

        half = (bq + aq) / 2.0
        self.depth = half if self.depth == 0 else self.depth + DEPTH_EMA * (half - self.depth)
        x = self.flow / self.depth
        if FLOW_CHECK and now >= self.fc_next:
            self._flow_check(now, x, mid)

        # Volatile regime: cancel passive quotes & snipe
        if self.volatile and VOLATILE_MODE:
            self._volatile(now, bb, ba, self.s2, agg)
            return

        # Quiet regime market making
        s_target = self.s2 if FAIR_S2 else s_hat
        fair = mid + K_IMB * (bq - aq) / (bq + aq) - K_INV * self.inv
        fair += K_S * (s_target - mid)
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

        if Q_TAKE < 90.0:
            if self.s2 - ba >= Q_TAKE:
                pb = max(pb, math.floor(self.s2 - Q_TAKE_PX))
            elif bb - self.s2 >= Q_TAKE:
                pa = min(pa, math.ceil(self.s2 + Q_TAKE_PX))

        self._side(1, pb, QTY, fair)
        self._side(-1, pa, QTY, fair)

    def _volatile(self, now: int, bb: int, ba: int, s: float, agg: int) -> None:
        inv = self.inv
        busy = bool(self.iocs)

        # In volatile regime, cancel resting quotes to avoid being run over
        self._cancel_side(1)
        self._cancel_side(-1)

        # Sniping entries in jump direction
        if not busy:
            if (s - ba >= (V_FLIP if inv < 0 else V_TAKE) or agg >= V_FOLLOW) and inv < MAX_POS:
                resting = self._resting_unblocked(1) if UNBLOCKED_SNIPE else self._resting(1)
                room = MAX_POS - inv - resting
                if room > 0:
                    if inv <= 0:
                        self.pos_t = now
                    px = max(ba, math.floor(s - V_TAKE_PX))
                    self._ioc(1, px, room)
                    busy = True
            elif (bb - s >= (V_FLIP if inv > 0 else V_TAKE) or agg <= -V_FOLLOW) and inv > -MAX_POS:
                resting = self._resting_unblocked(-1) if UNBLOCKED_SNIPE else self._resting(-1)
                room = MAX_POS + inv - resting
                if room > 0:
                    if inv >= 0:
                        self.pos_t = now
                    px = min(bb, math.ceil(s + V_TAKE_PX))
                    self._ioc(-1, px, room)
                    busy = True

        # Volatile exit: HOLD position unless explicit exit threshold configured
        if V_KEEP > -900:
            if inv > 0 and (s - bb < V_KEEP or now - self.pos_t >= V_MAX_HOLD_US):
                if not busy:
                    q = min(inv, MAX_POS + inv - self._resting(-1))
                    if q > 0:
                        self._ioc(-1, bb, q)
            elif inv < 0 and (ba - s < V_KEEP or now - self.pos_t >= V_MAX_HOLD_US):
                if not busy:
                    q = min(-inv, MAX_POS - inv - self._resting(1))
                    if q > 0:
                        self._ioc(1, ba, q)

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

    def _resting_unblocked(self, sign: int) -> int:
        r = 0
        for oid, o in self.orders.items():
            if o[0] == sign and oid not in self.cancel_sent:
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

    def _scan_new(self, views, sign: int, now: int) -> None:
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
            if o.owner != "market":
                continue
            oid = o.order_id
            if len(oid) > lm or (len(oid) == lm and oid > mx):
                p = o.price
                self.reg += ra * (abs(p - mid) - self.reg)

                if sign < 0:
                    r = p + C
                    # Instant Jump Snapping: buy order way above current S2 is proof of up-jump
                    if p >= s2 + JUMP_SNAP_TICKS:
                        s2 = r
                        self.reg = max(self.reg, REG_ON + 2.0)
                        self.reg_t = now
                        self.volatile = True
                        # Re-anchor s_bid and s_ask so s_hat never lags
                        self.s_bid = r - C
                        self.s_ask = r + C
                    elif p >= self.ba - 1 and r < s2:
                        r = s2
                else:
                    r = p - C
                    # Instant Jump Snapping: sell order way below current S2 is proof of down-jump
                    if p <= s2 - JUMP_SNAP_TICKS:
                        s2 = r
                        self.reg = max(self.reg, REG_ON + 2.0)
                        self.reg_t = now
                        self.volatile = True
                        # Re-anchor s_bid and s_ask so s_hat never lags
                        self.s_bid = r - C
                        self.s_ask = r + C
                    elif p <= self.bb + 1 and r > s2:
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

    def _side(self, sign: int, target: int, qty: int, fair: float) -> None:
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
            # Queue hysteresis
            edge_left = fair - o[1] if sign > 0 else o[1] - fair
            if behind < 0 and edge_left >= KEEP_EDGE:
                at_target += o[2]
            elif behind < 0 or behind > KEEP_LEVELS:
                self._cancel(oid)
            elif behind == 0:
                at_target += o[2]
            elif not o[3]:
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
            deepest_oid = keep[0][1]
            deepest = self.orders.get(deepest_oid)
            if deepest is not None:
                needed = min(want, MAX_POS - sign * self.inv - at_target) - room
                if USE_SHRINK and deepest[2] > needed and deepest[2] - needed >= MIN_NEW:
                    new_qty = deepest[2] - needed
                    deepest[2] = new_qty
                    deepest[3] = True
                    self.modify_order(deepest_oid, new_qty)
                    room += needed
                else:
                    self._cancel(deepest_oid)
                    return

        q = min(want, room)
        if q < MIN_NEW:
            return
        side = Side.BUY if sign > 0 else Side.SELL
        oid = self.send_order(OrderRequest(side=side, price=target, quantity=q, order_type=OrderType.LIMIT))
        self.orders[oid] = [sign, target, q, False]

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
            elif st == OrderStatus.MODIFIED:
                o[2] = event.remaining_quantity
                o[3] = False
            elif st == OrderStatus.CANCELED:
                self._drop(oid)
            elif st == OrderStatus.REJECTED:
                if oid in self.cancel_sent:
                    self._drop(oid)
                else:
                    o[3] = False
            return

        ioc = self.iocs.get(oid)
        if ioc is not None:
            if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
                self.inv += ioc[0] * event.filled_quantity
                ioc[1] -= event.filled_quantity
                if ioc[1] <= 0:
                    self.iocs.pop(oid, None)
            elif st == OrderStatus.CANCELED or st == OrderStatus.REJECTED:
                self.iocs.pop(oid, None)

    def _drop(self, oid: str) -> None:
        self.orders.pop(oid, None)
        self.cancel_sent.discard(oid)
