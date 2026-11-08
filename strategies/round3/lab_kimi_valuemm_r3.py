"""Value-taking market maker for round 3 (lab_kimi worker kimiw4).

Round-3 tape (findings/lab_kimi/kimiw4/r3_tape.py on data/round3_public.csv):
- No calm/jump alternation like round 2: 278 one-second S moves >= 6 ticks; sustained swings of
  +/-300 ticks over ~1200-1500 s (S: 10000 -> 10104 -> 9777 -> 10081 -> 9872).
- Limit placement X: median 1.64 ticks from S (round 2: ~0.65) - a wider, looser book.
- Event rates like round 2 (G ~195/s, IOC ~9/s at round(S+-1), coin-flip side).
- Round-2 champion family transfers terribly (this framework's round-2 config: -725k with a
  breach; regime_mm: -680k): their quiet/volatile split sees 'volatile' all the time and the
  strict-hold policy rides reversals to the bottom.

Design (one regime, trend-aware):
1. Fair = s2, the clamp-aware EMA of new limit prices (C=1.1, S2_ALPHA=0.25) pulled by IOC
   trade prices (aggressive prints are round(S+-1)). At ~195 orders/s it converges in tens of
   ms, so the s2-mid gap tracks the swing velocity: fair LEADS the mid in a trend.
2. Swing rider: IOC entries when s2 is ENTER beyond the opposite touch (with the move), and
   reversal exits: when the s2 signal flips against held inventory by EXIT, unload it at the
   touch. Rides each swing, cuts the reversal - the round-3 replacement for round 2's hold.
3. Maker: always-on ladder around fair with KEEP_EDGE queue hysteresis; a side is paused when
   s2 has moved BURST_TICKS against it within BURST_US (the fast-move run-over defence).
4. Risk: room is always computed with cancel-pending orders counted (no unblocked snipes -
  that is what breached on this tape); MAX_POS caps |inventory| at 100 with margin.
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

# Hidden-price estimate
C = 1.1  # assumed gap between a new order's price and S
S2_ALPHA = 0.25  # weight of each new order in s2
SCAN_LEVELS = 12  # ticks behind the side's best to scan for new orders
IOC_PULL = 0.35  # pull of each IOC-implied bound on s2
K_S = 1.0  # fair = mid + K_S * (s2 - mid)

# Swing rider
ENTER = 1.25  # s2 beyond the opposite touch to enter with the move
EXIT = 1.0  # s2 signal flip against inventory to unload it
CHUNK = 50  # lots per rider IOC
RIDE_CAP = 90  # inventory cap for rider entries
RIDER_COOL_US = 100_000  # min gap between rider IOCs

# Maker
QUOTE_SIZE = 50  # lots wanted at the target price
EDGE = 0.3  # min ticks between a passive quote and fair (trend side)
EDGE_IN = 0.5  # min ticks of edge to quote inside a 2+ tick spread (trend side)
SKEW = 2.5  # extra ticks of edge on the anti-trend side (transition zone)
GAP_OFF = 0.5  # |gap| beyond this: the anti-trend side is not quoted at all
GAP_EMA = 0.02  # EMA weight of (s2 - mid) per event, for the swing direction
MAX_SKEW = 4.0  # cap on the skew contribution
LADDER_LEVELS = 2
MIN_NEW = 10
KEEP_EDGE = 0.15
SHRINK = True

# Burst defence (run-over protection)
BURST_TICKS = 1.5  # s2 move against a side within BURST_US that pauses it
BURST_US = 150_000
PAUSE_US = 300_000  # how long the run-over side stays paused

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
        self._rider_id: str | None = None

        # hidden-price estimate
        self._s2 = 0.0
        self._max_id = ''
        self._last_bids: list | None = None
        self._last_asks: list | None = None
        self._bb = 0
        self._ba = 0

        # burst defence
        self._s2_ref = 0.0
        self._s2_ref_ts = 0
        self._pause_buy_until = 0
        self._pause_sell_until = 0

        # swing direction (smoothed s2-mid gap) for the maker skew
        self._gap_ema = 0.0

        self._rider_ok_at = 0
        self._now = 0
        self._side_of: dict[str, Side] = {}  # side of every order ever sent, so no fill is missed

    # ---------- signals ----------

    def _scan_new(self, views: list, sign: int) -> None:
        """Fold new market orders into the clamp-aware s2 estimate.

        Order ids only grow, so a market order id above the highest one seen so far is new. A
        new bid at p implies S >= p + C; a new ask implies S <= p - C. Orders at the touch are
        clamped (cannot cross), so their implied bound only moves s2 the right way. sign: -1
        bids, +1 asks.
        """
        first = views[0].price
        top_id = self._max_id
        new_top = top_id
        s2 = self._s2
        a2 = S2_ALPHA
        for o in views:
            if (o.price - first) * sign > SCAN_LEVELS:
                break
            if o.owner != 'market':
                continue
            oid = o.order_id
            if len(oid) > len(top_id) or (len(oid) == len(top_id) and oid > top_id):
                p = o.price
                if sign < 0:
                    r = p + C
                    if p >= self._ba - 1 and r < s2:
                        r = s2
                else:
                    r = p - C
                    if p <= self._bb + 1 and r > s2:
                        r = s2
                s2 += a2 * (r - s2)
                if len(oid) > len(new_top) or (len(oid) == len(new_top) and oid > new_top):
                    new_top = oid
        self._max_id = new_top
        self._s2 = s2

    def _burst_defence(self) -> None:
        """Pause the side being run over when s2 moved fast against it."""
        if self._s2_ref_ts == 0:
            self._s2_ref = self._s2
            self._s2_ref_ts = self._now
            return
        if self._now - self._s2_ref_ts >= BURST_US:
            move = self._s2 - self._s2_ref
            self._s2_ref = self._s2
            self._s2_ref_ts = self._now
            if move <= -BURST_TICKS:
                self._pause_buy_until = self._now + PAUSE_US
            elif move >= BURST_TICKS:
                self._pause_sell_until = self._now + PAUSE_US

    # ---------- orders ----------

    def _room(self, side: Side) -> int:
        """Lots that can still be added on `side` while keeping the worst case within MAX_POS.

        Every open order counts, in-flight IOCs and cancel-pending ones included: an IOC that
        has not landed yet still buys inventory, and a cancel that loses the race can still
        fill. That is what keeps the inventory cap hard.
        """
        open_qty = 0
        for order in self._orders.values():
            if order[0] is side:
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
        if self._rider_id == order_id:
            self._rider_id = None

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

    # ---------- rider ----------

    def _run_rider(self, bid: int, ask: int) -> None:
        """Enter with the move when s2 is beyond the touch; exit on signal flip."""
        if self._rider_id is not None or self._busy():
            return
        if self._now < self._rider_ok_at:
            return
        s = self._s2
        inv = self._inventory
        # reversal exits first: the signal has flipped against held inventory
        if inv > 0 and bid - s >= EXIT:
            qty = min(inv, self._room(SELL))
            if qty > 0:
                self._rider_id = self._send(SELL, bid, qty, IOC)
                self._rider_ok_at = self._now + RIDER_COOL_US
            return
        if inv < 0 and s - ask >= EXIT:
            qty = min(-inv, self._room(BUY))
            if qty > 0:
                self._rider_id = self._send(BUY, ask, qty, IOC)
                self._rider_ok_at = self._now + RIDER_COOL_US
            return
        # entries with the move
        if s - ask >= ENTER and inv < RIDE_CAP:
            qty = min(CHUNK, self._room(BUY), RIDE_CAP - inv)
            if qty > 0:
                self._rider_id = self._send(BUY, ask, qty, IOC)
                self._rider_ok_at = self._now + RIDER_COOL_US
        elif bid - s >= ENTER and inv > -RIDE_CAP:
            qty = min(CHUNK, self._room(SELL), RIDE_CAP + inv)
            if qty > 0:
                self._rider_id = self._send(SELL, bid, qty, IOC)
                self._rider_ok_at = self._now + RIDER_COOL_US

    # ---------- maker ----------

    def _targets(self, fair: float, bid: int, ask: int) -> tuple[int, int]:
        """Target price per side: the trend side quotes tight, the anti-trend side is skewed deep.

        In a downswing (gap < 0) the sell side is the profitable one (sell into the falling
        flow); the buy side is adversely selected, so it is pushed SKEW ticks deeper. Mirror
        for upswings. The fill drill showed the anti-trend side was the whole maker bleed.
        """
        gap = self._gap_ema
        edge_buy = EDGE + min(SKEW * max(-gap, 0.0), MAX_SKEW)
        edge_sell = EDGE + min(SKEW * max(gap, 0.0), MAX_SKEW)
        pb = bid
        while fair - pb < edge_buy:
            pb -= 1
        if edge_buy <= EDGE_IN and ask - bid >= 2 and fair - (bid + 1) >= EDGE_IN:
            pb = bid + 1
        pa = ask
        while pa - fair < edge_sell:
            pa += 1
        if edge_sell <= EDGE_IN and ask - bid >= 2 and (ask - 1) - fair >= EDGE_IN:
            pa = ask - 1
        return pb, pa

    def _manage_ladder(self, side: Side, top: int | None, fair: float = 0.0) -> None:
        """Hold QUOTE_SIZE at `top`, and keep older orders up to LADDER_LEVELS - 1 ticks behind it."""
        step = -1 if side is BUY else 1
        if self._now < (self._pause_buy_until if side is BUY else self._pause_sell_until):
            self._cancel_side(side)
            return
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

    # ---------- main ----------

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        """Per market event: update s2, defend bursts, ride the swing, then quote around fair."""
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
        self._bb, self._ba = bid, ask
        if not self._s2:
            self._s2 = mid
            self._s2_ref = mid
        if bids is not self._last_bids:
            self._last_bids = bids
            self._scan_new(bids, -1)
        if asks is not self._last_asks:
            self._last_asks = asks
            self._scan_new(asks, 1)
        # IOC prints are round(S +- 1): pull s2 toward their implied bounds
        max_buy = 0
        min_sell = 0
        for t in trades:
            if t.taker_owner == 'market':
                p = t.price
                if t.book_side == 'ask':
                    if p > max_buy:
                        max_buy = p
                else:
                    if min_sell == 0 or p < min_sell:
                        min_sell = p
        if max_buy and max_buy - 1.0 > self._s2:
            self._s2 += IOC_PULL * (max_buy - 1.0 - self._s2)
        if min_sell and min_sell + 1.0 < self._s2:
            self._s2 += IOC_PULL * (min_sell + 1.0 - self._s2)

        self._burst_defence()
        self._gap_ema += GAP_EMA * ((self._s2 - mid) - self._gap_ema)
        fair = mid + K_S * (self._s2 - mid)
        self._run_rider(bid, ask)
        pb, pa = self._targets(fair, bid, ask)
        # one-sided trend making: in a clear swing the anti-trend side is not quoted at all
        if self._gap_ema > GAP_OFF:
            self._cancel_side(BUY)
        elif self._gap_ema < -GAP_OFF:
            self._cancel_side(SELL)
        self._manage_ladder(BUY, pb, fair)
        self._manage_ladder(SELL, pa, fair)
