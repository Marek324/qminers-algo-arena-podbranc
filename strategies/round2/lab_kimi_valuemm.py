"""Value-taking market maker v3 (lab_kimi worker kimiw4).

v2 fill drill (findings/lab_kimi/kimiw4/fill_drill.py on kimiw4_vm2_full): maker execution edge
was +317k but drift -289k; the maker BUY fills were toxic (mk1 -0.35 / mk5 -0.75 per lot) and
jump-window fills lost -40.8k, because the 200 ms-average jump detector lags and the post-jump
IOC surge lifts the stale bid. The predator's fill drill (kimi_pred_full) shows the fix: fills
with symmetric positive markout and positive jump-window markout, from instant jump snapping.

v3 = my framework (ladder maker with queue hysteresis, quiet value taker, fill-based inventory)
plus the validated instant-jump machinery:
1. Instant jump snap: a new market limit >= s2 + JUMP_SNAP_TICKS (or <= s2 - ...) is proof of a
   jump; snap s2, re-anchor s_hat, go volatile. IOC trade sweeps snap too (aggressive trade
   prices are round(S +- 1)).
2. Volatile regime (reg-score EMA of |new price - mid|, snap-forced on): cancel both sides,
   snipe with one IOC in the jump direction up to unblocked room, then strictly HOLD the
   position (no forced exits - team sweeps show exits hurt).
3. Quiet regime: fair = mid + K_IMB*imb + K_S*(s_hat - mid) + K_FLOW*flow_scale*x - K_INV*inv;
   ladder maker with KEEP_EDGE hysteresis; value taker at TAKE_EDGE beyond the opposite touch.
4. Empty-book safety: a drained side cancels that side's quotes at once.
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

# Fair value (predator-validated quiet config)
K_S = 0.82  # weight of the hidden-price estimate gap
S_ALPHA = 0.10  # per-side EMA weight of each new order (s_hat)
S_LEVELS = 2  # ticks behind the side's best a new order still folds into s_hat
K_IMB = 0.15
K_INV = 0.0014
K_FLOW = 0.8
FLOW_TAU_US = 100_000
DEPTH_EMA = 0.001
FLOW_CHECK = True
FLOW_SLOPE_FULL = 0.5
FLOW_CHECK_TAU_US = 300_000_000
FLOW_CHECK_STEP_US = 50_000
FLOW_CHECK_H_US = 500_000
FLOW_CHECK_MIN = 200.0

# Hidden-price estimate s2 (clamp-aware) + instant jump snapping
C = 1.1  # assumed gap between a new order's price and S
S2_ALPHA = 0.25  # weight of each new order in s2
SCAN_LEVELS = 16  # ticks behind the side's best to scan for new orders
JUMP_SNAP_TICKS = 3.0  # a new order this far beyond s2 is proof of a jump
IOC_SNAP = True  # snap on aggressive trade prices too

# Regime filter
REG_ON = 1.60
REG_OFF = 1.35
REG_ALPHA = 0.03
REG_HOLD_US = 2_000_000

# Volatile policy: snipe in the jump direction, then hold
V_TAKE = 0.75  # s2 past the opposite best to snipe
V_TAKE_PX = 0.5  # IOC limit s2 -/+ this
V_FLIP = 0.75  # threshold when the trade would flip an opposite position
V_KEEP = -999.0  # strict hold: never force an exit

# Value taker (quiet regime)
TAKE_EDGE = 0.75  # fair must be this far beyond the opposite touch to take (0: off)
TAKE_MAX = 50  # lots per take IOC
TAKE_CAP = 95  # taker inventory cap per side
TAKE_COOL_US = 25_000  # min gap between take IOCs

# Maker
QUOTE_SIZE = 200  # lots wanted at the target price
EDGE = 0.28  # min ticks between a passive quote and fair
EDGE_IN = 0.45  # min ticks of edge to quote inside a 2+ tick spread
EDGE_TAKE = 9.0  # cross with a passive order when fair is this far through (taker handles it)
LADDER_LEVELS = 2
MIN_NEW = 10
KEEP_EDGE = 0.15
SHRINK = True

# Risk
MAX_POS = 100  # worst case |inventory| if every open order on one side fills

BUY = Side.BUY
SELL = Side.SELL
LIMIT = OrderType.LIMIT
IOC = OrderType.IOC


def _touch(levels: list) -> tuple[int | None, int]:
    """Best market price on one side and the market quantity at it, ignoring our own orders."""
    best = None
    qty = 0
    for order in levels:
        if order.owner != 'market':
            continue
        if best is None:
            best = order.price
        elif order.price != best:
            break
        qty += order.quantity
    return best, qty


class ValueMarketMaker(BaseStrategy):
    """See the module docstring."""

    def __init__(self) -> None:
        super().__init__()
        self._inventory = 0
        # order_id -> [side, price, remaining_quantity, is_ioc, cancelling, shrinking, accepted]
        self._orders: dict[str, list] = {}
        self._take_id: str | None = None

        # flow signal
        self._flow = 0.0
        self._flow_depth = 0.0
        self._flow_ts = 0
        self._flow_scale = 0.0 if FLOW_CHECK else 1.0
        self._fc_pending: deque = deque()
        self._fc_next = 0
        self._fc_sxx = 0.0
        self._fc_sxy = 0.0
        self._fc_n = 0.0
        self._fc_decay = math.exp(-FLOW_CHECK_STEP_US / FLOW_CHECK_TAU_US)

        # hidden-price estimates
        self._s2 = 0.0
        self._s_bid = 0.0
        self._s_ask = 0.0
        self._max_id = ''
        self._last_bids: list | None = None
        self._last_asks: list | None = None

        # regime
        self._reg = 1.0
        self._vol = False
        self._reg_t = 0
        self._mid = 0.0
        self._bb = 0
        self._ba = 0

        self._take_ok_at = 0
        self._now = 0
        self._side_of: dict[str, Side] = {}  # side of every order ever sent, so no fill is missed

    # ---------- signals ----------

    def _scan_new(self, views: list, sign: int) -> None:
        """Fold new market orders into s2 / s_hat / the regime score, with instant jump snapping.

        Order ids only grow, so a market order id above the highest one seen so far is new.
        A new bid at p implies S >= p + C; a new ask implies S <= p - C. Orders at the touch are
        clamped (never cross), so their implied bound is only applied when it would move s2 the
        right way. A new order JUMP_SNAP_TICKS beyond s2 is proof of a jump: snap instantly.
        """
        first = views[0].price
        top_id = self._max_id
        new_top = top_id
        a = S_ALPHA
        ra = REG_ALPHA
        mid = self._mid
        s2 = self._s2
        a2 = S2_ALPHA
        now = self._now
        for o in views:
            if (o.price - first) * sign > SCAN_LEVELS:
                break
            if o.owner != 'market':
                continue
            oid = o.order_id
            if len(oid) > len(top_id) or (len(oid) == len(top_id) and oid > top_id):
                p = o.price
                self._reg += ra * (abs(p - mid) - self._reg)
                if sign < 0:
                    r = p + C
                    if p >= s2 + JUMP_SNAP_TICKS:
                        s2 = r
                        self._snap(now, r - C, r + C)
                    elif p >= self._ba - 1 and r < s2:
                        r = s2
                else:
                    r = p - C
                    if p <= s2 - JUMP_SNAP_TICKS:
                        s2 = r
                        self._snap(now, r - C, r + C)
                    elif p <= self._bb + 1 and r > s2:
                        r = s2
                s2 += a2 * (r - s2)
                if (o.price - first) * sign <= S_LEVELS:
                    if sign < 0:
                        self._s_bid = p if not self._s_bid else self._s_bid + a * (p - self._s_bid)
                    else:
                        self._s_ask = p if not self._s_ask else self._s_ask + a * (p - self._s_ask)
                if len(oid) > len(new_top) or (len(oid) == len(new_top) and oid > new_top):
                    new_top = oid
        self._max_id = new_top
        self._s2 = s2

    def _snap(self, now: int, s_bid: float, s_ask: float) -> None:
        """Instant jump snap: re-anchor the estimates and force the volatile regime."""
        self._s_bid = s_bid
        self._s_ask = s_ask
        self._reg = max(self._reg, REG_ON + 2.0)
        self._reg_t = now
        self._vol = True

    def _flow_check(self, now: int, x: float, mid: float) -> None:
        """Track the rolling slope of mid vs the flow signal; scale the flow term by it."""
        self._fc_next = now + FLOW_CHECK_STEP_US
        pend = self._fc_pending
        d = self._fc_decay
        while pend and now - pend[0][0] >= FLOW_CHECK_H_US:
            _, x0, m0 = pend.popleft()
            y = mid - m0
            self._fc_sxx = self._fc_sxx * d + x0 * x0
            self._fc_sxy = self._fc_sxy * d + x0 * y
            self._fc_n = self._fc_n * d + 1.0
        pend.append((now, x, mid))
        if self._fc_n < FLOW_CHECK_MIN or self._fc_sxx <= 0:
            self._flow_scale = 0.0
            return
        slope = self._fc_sxy / self._fc_sxx
        self._flow_scale = min(1.0, max(0.0, slope / FLOW_SLOPE_FULL))

    # ---------- orders ----------

    def _room(self, side: Side, unblocked: bool = False) -> int:
        """Lots that can still be added on `side` within MAX_POS worst case.

        With unblocked, cancelling orders do not count (used for the volatile snipe, which must
        not wait 5 ms for cancel confirmations).
        """
        open_qty = 0
        for order in self._orders.values():
            if order[0] is side and not order[3] and not (unblocked and order[4]):
                open_qty += order[2]
        if side is BUY:
            return MAX_POS - (self._inventory + open_qty)
        return MAX_POS + (self._inventory - open_qty)

    def _busy(self) -> bool:
        for order in self._orders.values():
            if order[3]:
                return True
        return False

    def _send(self, side: Side, price: int, qty: int, order_type: OrderType) -> str:
        order_id = self.send_order(OrderRequest(side=side, price=price, quantity=qty, order_type=order_type))
        self._orders[order_id] = [side, price, qty, order_type is IOC, False, False, False]
        self._side_of[order_id] = side
        return order_id

    def _forget(self, order_id: str, side: Side) -> None:
        self._orders.pop(order_id, None)
        if self._take_id == order_id:
            self._take_id = None

    def _cancel_side(self, side: Side) -> None:
        for order_id, order in self._orders.items():
            if order[0] is side and not order[3] and not order[4]:
                self.cancel_order(order_id)
                order[4] = True

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        """Update inventory from fills, and drop orders that are filled, cancelled or rejected."""
        order = self._orders.get(event.order_id)
        status = event.status
        if order is None:
            side = self._side_of.get(event.order_id)
            if side is not None and (status is OrderStatus.PARTIALLY_FILLED or status is OrderStatus.FILLED):
                self._inventory += event.filled_quantity if side is BUY else -event.filled_quantity
            return
        if status is OrderStatus.PARTIALLY_FILLED or status is OrderStatus.FILLED:
            self._inventory += event.filled_quantity if order[0] is BUY else -event.filled_quantity
            order[2] = event.remaining_quantity
            if status is OrderStatus.FILLED:
                self._forget(event.order_id, order[0])
        elif status is OrderStatus.ACCEPTED:
            order[6] = True
        elif status is OrderStatus.MODIFIED:
            order[2] = event.remaining_quantity
            order[5] = False
        elif status is OrderStatus.REJECTED and order[6]:
            # after ACCEPTED, a REJECTED is a cancel or shrink that lost the race: still live
            order[4] = False
            order[5] = False
        elif status is OrderStatus.CANCELED or status is OrderStatus.REJECTED:
            self._forget(event.order_id, order[0])

    # ---------- volatile regime ----------

    def _volatile(self, bb: int, ba: int) -> None:
        """Cancel all quotes, snipe once in the jump direction, then hold the position."""
        self._cancel_side(BUY)
        self._cancel_side(SELL)
        if self._busy():
            return
        s = self._s2
        inv = self._inventory
        if s - ba >= (V_FLIP if inv < 0 else V_TAKE) and inv < MAX_POS:
            room = self._room(BUY, unblocked=True)
            if room > 0:
                self._send(BUY, max(ba, math.floor(s - V_TAKE_PX)), room, IOC)
        elif bb - s >= (V_FLIP if inv > 0 else V_TAKE) and inv > -MAX_POS:
            room = self._room(SELL, unblocked=True)
            if room > 0:
                self._send(SELL, min(bb, math.ceil(s + V_TAKE_PX)), room, IOC)
        # strict hold: no forced exits (V_KEEP far below any reachable s-gap)

    # ---------- quiet maker ----------

    def _targets(self, fair: float, bid: int, ask: int) -> tuple[int, int]:
        """Target price per side, from fair."""
        if fair - ask >= EDGE_TAKE:
            pb = ask
        elif ask - bid >= 2 and fair - (bid + 1) >= EDGE_IN:
            pb = bid + 1
        else:
            pb = bid
            while fair - pb < EDGE:
                pb -= 1
        if bid - fair >= EDGE_TAKE:
            pa = bid
        elif ask - bid >= 2 and (ask - 1) - fair >= EDGE_IN:
            pa = ask - 1
        else:
            pa = ask
            while pa - fair < EDGE:
                pa += 1
        return pb, pa

    def _manage_ladder(self, side: Side, top: int | None, fair: float = 0.0) -> None:
        """Hold QUOTE_SIZE at `top`, and keep older orders up to LADDER_LEVELS - 1 ticks behind it."""
        step = -1 if side is BUY else 1
        at_top = 0
        kept = []
        for order_id, order in self._orders.items():
            if order[0] is not side or order[3] or order[4]:
                continue
            depth = (order[1] - top) * step if top is not None else -1
            if depth < 0 and top is not None and (fair - order[1] if side is BUY else order[1] - fair) >= KEEP_EDGE:
                at_top += order[2]  # queue hysteresis: a placed order with KEEP_EDGE left stands in
            elif depth < 0 or depth >= LADDER_LEVELS:
                self.cancel_order(order_id)
                order[4] = True
            elif depth == 0:
                at_top += order[2]
            elif not order[5]:
                kept.append((depth, order_id))
        if top is None or top <= 0 or at_top >= QUOTE_SIZE:
            return
        room = self._room(side)
        want = QUOTE_SIZE - at_top
        if room < min(want, MIN_NEW) and kept:
            kept.sort(reverse=True)
            order_id = kept[0][1]
            order = self._orders[order_id]
            left = order[2] - max(min(want, MAX_POS) - max(room, 0), MIN_NEW)
            if SHRINK and left >= MIN_NEW:
                self.modify_order(order_id, left)  # keeps its queue place
                order[5] = True
            else:
                self.cancel_order(order_id)
                order[4] = True
            return
        qty = min(want, room)
        if qty >= MIN_NEW:
            self._send(side, top, qty, LIMIT)

    # ---------- value taker (quiet) ----------

    def _run_taker(self, fair: float, bid: int, ask: int) -> None:
        """Take a resting quote that is mispriced versus fair (the S estimate leads the mid)."""
        if not TAKE_EDGE or self._take_id is not None or self._busy():
            return
        if self._now < self._take_ok_at:
            return
        inv = self._inventory
        if fair - ask >= TAKE_EDGE and inv < TAKE_CAP:
            qty = min(TAKE_MAX, self._room(BUY), TAKE_CAP - inv)
            if qty > 0:
                self._take_id = self._send(BUY, ask, qty, IOC)
                self._take_ok_at = self._now + TAKE_COOL_US
        elif bid - fair >= TAKE_EDGE and inv > -TAKE_CAP:
            qty = min(TAKE_MAX, self._room(SELL), TAKE_CAP + inv)
            if qty > 0:
                self._take_id = self._send(SELL, bid, qty, IOC)
                self._take_ok_at = self._now + TAKE_COOL_US

    # ---------- main ----------

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        """Per market event: scan/snap, regime switch, then volatile sniping or quiet making."""
        bids, asks = snapshot.bids, snapshot.asks
        # empty-book safety: a drained side's quotes are the only target left, cancel at once
        if bids and not asks:
            self._cancel_side(SELL)
            return
        if asks and not bids:
            self._cancel_side(BUY)
            return
        if not bids or not asks:
            return
        bid, bid_qty = _touch(bids)
        ask, ask_qty = _touch(asks)
        if bid is None or ask is None:
            return
        ts = snapshot.timestamp_us
        self._now = ts
        mid = (bid + ask) / 2
        self._mid = mid
        self._bb, self._ba = bid, ask
        if not self._s2:
            self._s2 = mid
        if bids is not self._last_bids:
            self._last_bids = bids
            self._scan_new(bids, -1)
        if asks is not self._last_asks:
            self._last_asks = asks
            self._scan_new(asks, 1)

        # trades: aggressive flow, and IOC sweep prices bound/snapping s2
        agg = 0
        max_buy = 0
        min_sell = 0
        for t in trades:
            if t.taker_owner == 'market':
                p = t.price
                if t.book_side == 'ask':
                    agg += t.quantity
                    if p > max_buy:
                        max_buy = p
                else:
                    agg -= t.quantity
                    if min_sell == 0 or p < min_sell:
                        min_sell = p
        if IOC_SNAP and max_buy and max_buy - 1.0 >= self._s2 + JUMP_SNAP_TICKS:
            self._s2 = max_buy - 1.0
            self._snap(ts, self._s2 - C, self._s2 + C)
        elif max_buy and max_buy - 1.0 > self._s2:
            self._s2 += 0.35 * (max_buy - 1.0 - self._s2)
        if IOC_SNAP and min_sell and min_sell + 1.0 <= self._s2 - JUMP_SNAP_TICKS:
            self._s2 = min_sell + 1.0
            self._snap(ts, self._s2 - C, self._s2 + C)
        elif min_sell and min_sell + 1.0 < self._s2:
            self._s2 += 0.35 * (min_sell + 1.0 - self._s2)

        # regime switch
        if self._reg > REG_ON:
            self._reg_t = ts
            self._vol = True
        elif self._vol and self._reg < REG_OFF and ts - self._reg_t >= REG_HOLD_US:
            self._vol = False

        # flow
        self._flow *= math.exp((self._flow_ts - ts) / FLOW_TAU_US)
        self._flow_ts = ts
        self._flow += agg
        half = (bid_qty + ask_qty) / 2
        self._flow_depth = half if self._flow_depth == 0 else self._flow_depth + DEPTH_EMA * (half - self._flow_depth)
        x = self._flow / self._flow_depth
        if FLOW_CHECK and ts >= self._fc_next:
            self._flow_check(ts, x, mid)

        if self._vol:
            self._volatile(bid, ask)
            return

        s_hat = (self._s_bid + self._s_ask) / 2 if self._s_bid and self._s_ask else mid
        fair = mid + K_IMB * (bid_qty - ask_qty) / (bid_qty + ask_qty)
        fair += K_S * (s_hat - mid)
        fair += K_FLOW * self._flow_scale * x
        fair -= K_INV * self._inventory
        self._run_taker(fair, bid, ask)
        pb, pa = self._targets(fair, bid, ask)
        self._manage_ladder(BUY, pb, fair)
        self._manage_ladder(SELL, pa, fair)
