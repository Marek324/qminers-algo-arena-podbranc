"""Adaptive market maker.

Maker leg (the money on round 1): quotes passively around a fair price.
- Fair = mid, moved K_S of the way toward a hidden-price estimate, plus small terms for touch
  imbalance, recent aggressive volume and inventory. The tape's limit orders are placed around a
  hidden price that random-walks, and the visible mid lags it by about 0.5 s; the estimate is the
  average of running means of newly arrived limit-order prices on each side. (Fair-value design
  and hidden-price estimate from strategies/round1/imbalance_mm.py.)
- Quote prices: the best price that keeps at least EDGE ticks from fair; cross the spread when fair
  is EDGE_TAKE beyond the opposite best price; inside a 2+ tick spread only with EDGE_IN of edge.
- Orders are as large as the position cap allows. An order the target moves a tick away from is
  kept rather than re-sent, and the deepest kept order is cancelled when room is needed. An order
  the target moves *behind* is also kept while it has KEEP_EDGE (0.25) of edge left, so a small
  dip in fair does not throw away its queue place (joining still needs EDGE, 0.4).

Taker leg (safety, idle on round 1): holds a directional position when order flow imbalance (OFI,
Cont, Kukanov & Stoikov) predicts a move, but only while a paper-traded 1-lot shadow copy of it is
making money, so on tapes without momentum it stays off.

Risk: every order is sized so that even if all our open orders on one side fill, |inventory| stays
within MAX_POS (the exchange penalty starts above 100). Inventory is counted from fills only, so a
cancel that loses the race against a fill cannot put it out of sync. Our own orders are ignored
when reading the book, so no signal reacts to us.

Hidden-price check: on tapes where orders are not placed around a hidden price, the estimate
misleads the quotes (control tapes: about -84k per hour). So its weight is scaled by how well it
has predicted the mid 0.5 s ahead over roughly the last 5 minutes, and drops to 0 when it hasn't.
That cut the control-tape losses to -6k ... -24k with no cost on round 1.

Run:  uv run exchange-simulator --strategy strategies/round1/adaptive_mm.py --backtest data/round1.csv --log-dir logs/adaptive --no-logs
Tune: python3 -I scripts/sweep.py strategies/round1/adaptive_mm.py data/round1.csv -p EDGE=0.25,0.3,0.35
Experiment history: docs/RESEARCH.md.
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

# Forecast
OFI_WEIGHT = 0.25  # forecast ticks per average touch depth of OFI
OFI_TAU_US = 1_000_000  # OFI decay time constant
DEPTH_EMA = 0.01  # per-event weight of the average touch size

# Taker
TAKE_ENABLED = True
TAKE_THRESHOLD = 1.0  # |forecast| in ticks that sets a direction
EXIT_THRESHOLD = 0.5  # |forecast| in ticks below which the direction is flat
TAKE_POS = 50  # lots held at full size
TAKE_SIZE = 50  # lots per IOC

# Momentum check
EDGE_CHECK = True
EDGE_TAU_US = 300_000_000  # decay of the shadow's PnL (about the last 5 minutes)
EDGE_FULL = 60.0  # shadow PnL (ticks per lot) needed for full TAKE_POS
LATENCY_US = 5_000  # exchange latency, applied to the shadow

# Fair value (maker)
K_S = 0.75  # fair moves this share of the way from the mid to the hidden-price estimate
S_ALPHA = 0.1  # weight of each newly arrived limit order in its side's running price average
S_LEVELS = 2  # new orders count up to this many ticks behind the best price
K_IMB = 0.15  # ticks of fair shift at full touch imbalance
K_FLOW = 0.8  # ticks of fair shift per average touch size of recent aggressive volume
FLOW_TAU_US = 100_000  # decay of the aggressive volume
FLOW_DEPTH_EMA = 0.001  # per-event weight of the touch size that normalises the flow
K_INV = 0.002  # ticks of fair shift per lot of inventory away from target

# Hidden-price check: K_S is scaled by how well the estimate has recently predicted the mid
HP_CHECK = True
HP_HORIZON_US = 500_000  # the mid move the estimate is checked against
HP_SAMPLE_US = 50_000  # one check sample per this much time
HP_TAU_US = 300_000_000  # decay of the check's statistics (about the last 5 minutes)
HP_CORR_FULL = 0.3  # correlation at which the full K_S is used; 0 or less turns the term off
HP_MIN_WEIGHT = 100.0  # samples needed before the term is used at all

# Maker
MAKE_ENABLED = True
QUOTE_SIZE = 200  # lots wanted at the target price (capped by the risk room)
EDGE = 0.4  # min ticks between a passive quote and fair (0.3 on round 1 alone; 0.4 holds up better in harder markets)
EDGE_IN = 0.5  # min ticks of edge to quote inside a 2+ tick spread
EDGE_TAKE = 0.25  # cross (bid at the best ask) when fair is at least this far through it
LADDER_LEVELS = 2  # an order the target moves away from is kept up to LADDER_LEVELS - 1 ticks behind it
MIN_NEW = 10  # smallest new order worth sending
KEEP_EDGE = 0.25  # an existing order priced better than the target is kept while it has this much edge

# Risk
MAX_POS = 100  # worst case |inventory| if every open order on one side fills (penalty only above 100)

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


class AdaptiveMarketMaker(BaseStrategy):
    """See the module docstring. State: inventory and target position, our open orders, the
    decayed signals (OFI, aggressive flow, average touch size) and the shadow taker."""

    def __init__(self) -> None:
        super().__init__()
        self._inventory = 0
        self._dir = 0
        self._target = 0
        # order_id -> [side, price, remaining_quantity, is_ioc, cancelling]
        self._orders: dict[str, list] = {}
        self._taker_id: str | None = None

        self._prev: tuple[int, int, int, int] | None = None
        self._ofi = 0.0
        self._avg_depth = 0.0
        self._last_ts = 0

        self._flow = 0.0
        self._flow_depth = 0.0
        self._flow_ts = 0
        self._s_bid = 0.0
        self._s_ask = 0.0
        self._max_id = ''
        self._last_bids: list | None = None
        self._last_asks: list | None = None
        self._hp_pending: deque = deque()  # (ts, delta, mid) samples waiting for their horizon
        self._hp_next = 0
        self._hp_ts = 0
        self._hp_sxy = 0.0
        self._hp_sxx = 0.0
        self._hp_syy = 0.0
        self._hp_w = 0.0

        self._sh_pos = 0
        self._sh_cash = 0.0
        self._sh_equity = 0.0
        self._sh_pending: tuple[int, int, int] | None = None  # (ready_ts, +1 buy / -1 sell, limit price)
        self._edge = 0.0

    def _update_signals(self, ts: int, bid: int, bid_qty: int, ask: int, ask_qty: int) -> None:
        """Decay the signals to `ts`, then add this event.

        OFI scores the change at the touch since the previous event: a bid queue that grows (or a best
        bid that moves up) is buying pressure, a bid queue that shrinks (or a best bid that moves down)
        is selling pressure, and mirror-wise for the ask.
        """
        dt = ts - self._last_ts
        self._last_ts = ts
        if dt > 0:
            self._ofi *= math.exp(-dt / OFI_TAU_US)
            self._edge *= math.exp(-dt / EDGE_TAU_US)

        if self._prev is not None:
            pb, pbq, pa, paq = self._prev
            e = 0
            if bid >= pb:
                e += bid_qty
            if bid <= pb:
                e -= pbq
            if ask <= pa:
                e -= ask_qty
            if ask >= pa:
                e += paq
            self._ofi += e
        self._prev = (bid, bid_qty, ask, ask_qty)

        depth = (bid_qty + ask_qty) / 2
        self._avg_depth = depth if self._avg_depth == 0.0 else self._avg_depth + DEPTH_EMA * (depth - self._avg_depth)

    def _forecast(self) -> float:
        """Expected mid move in ticks: OFI divided by the average touch size, so it is scale-free."""
        depth = self._avg_depth if self._avg_depth > 0 else 1.0
        return OFI_WEIGHT * self._ofi / depth

    def _room(self, side: Side) -> int:
        """Lots that can still be added on `side` while keeping the worst case within MAX_POS."""
        open_qty = 0
        for order in self._orders.values():
            if order[0] is side:
                open_qty += order[2]
        if side is BUY:
            return MAX_POS - (self._inventory + open_qty)
        return MAX_POS + (self._inventory - open_qty)

    def _send(self, side: Side, price: int, qty: int, order_type: OrderType) -> str:
        order_id = self.send_order(OrderRequest(side=side, price=price, quantity=qty, order_type=order_type))
        self._orders[order_id] = [side, price, qty, order_type is IOC, False]
        return order_id

    def _shadow_fill(self, ts: int) -> None:
        """Fill the shadow's pending IOC against the book it would have met on arrival (the previous touch)."""
        pending = self._sh_pending
        if pending is None or ts < pending[0] or self._prev is None:
            return
        self._sh_pending = None
        pb, _, pa, _ = self._prev
        if pending[1] > 0 and pa <= pending[2]:
            self._sh_pos += 1
            self._sh_cash -= pa
        elif pending[1] < 0 and pb >= pending[2]:
            self._sh_pos -= 1
            self._sh_cash += pb

    def _shadow_step(self, ts: int, bid: int, ask: int) -> None:
        """Paper-trade the 1-lot shadow toward the taker's direction and add its PnL change to `edge`.

        The shadow follows the real rules: its IOC lands LATENCY_US later and misses if the book moved
        away. It buys at the ask, sells at the bid and also unwinds by taking, so `edge` (its PnL marked
        at the touch, decayed over EDGE_TAU_US) is a conservative measure of the taker's momentum edge.
        """
        if self._sh_pending is None and self._sh_pos != self._dir:
            if self._dir > self._sh_pos:
                self._sh_pending = (ts + LATENCY_US, 1, ask)
            else:
                self._sh_pending = (ts + LATENCY_US, -1, bid)
        pos = self._sh_pos
        equity = self._sh_cash + (pos * bid if pos > 0 else pos * ask)
        self._edge += equity - self._sh_equity
        self._sh_equity = equity

    def _size(self) -> int:
        """Taker size: TAKE_POS * clamp(edge / EDGE_FULL, 0, 1). Zero while the shadow is losing."""
        if not EDGE_CHECK:
            return TAKE_POS
        scale = self._edge / EDGE_FULL
        if scale <= 0:
            return 0
        return round(TAKE_POS * min(1.0, scale))

    def _run_taker(self, forecast: float, bid: int, ask: int) -> None:
        """Set the direction and target position, and send an IOC toward the target if needed.

        The direction turns long or short beyond +-TAKE_THRESHOLD, goes flat inside +-EXIT_THRESHOLD and is
        otherwise kept. With size 0 nothing is sent, not even to flatten: that would only pay the spread
        to unwind maker inventory. With one IOC in flight at most, it buys at the ask or sells at the
        bid, and only in the direction the forecast points; unwinding against it is left to the quotes.
        """
        if forecast > TAKE_THRESHOLD:
            self._dir = 1
        elif forecast < -TAKE_THRESHOLD:
            self._dir = -1
        elif abs(forecast) < EXIT_THRESHOLD:
            self._dir = 0
        size = self._size()
        self._target = self._dir * size
        if size == 0 or self._taker_id is not None:
            return

        gap = self._target - self._inventory
        if gap > 0 and forecast > 0:
            side, price = BUY, ask
        elif gap < 0 and forecast < 0:
            side, price = SELL, bid
        else:
            return
        qty = min(abs(gap), TAKE_SIZE, self._room(side))
        if qty > 0:
            self._taker_id = self._send(side, price, qty, IOC)

    def _scan_new(self, views: list, sign: int) -> None:
        """Fold market orders that arrived since the last scan into their side's running price average.

        Order ids only grow, so a market order id above the highest one seen so far is new. Only orders
        within S_LEVELS ticks of the best price count (sign: -1 bids, +1 asks).
        """
        first = views[0].price
        top_id = self._max_id
        for o in views:
            if (o.price - first) * sign > S_LEVELS:
                break
            if o.owner != 'market':
                continue
            oid = o.order_id
            if len(oid) > len(top_id) or (len(oid) == len(top_id) and oid > top_id):
                if sign < 0:
                    self._s_bid = o.price if not self._s_bid else self._s_bid + S_ALPHA * (o.price - self._s_bid)
                else:
                    self._s_ask = o.price if not self._s_ask else self._s_ask + S_ALPHA * (o.price - self._s_ask)
                if len(oid) > len(self._max_id) or (len(oid) == len(self._max_id) and oid > self._max_id):
                    self._max_id = oid

    def _fair(self, snapshot: OrderBookSnapshot, trades: list[Trade], bid: int, bid_qty: int, ask: int, ask_qty: int) -> float:
        """Fair price: the mid, moved toward the hidden price and by imbalance, aggressive flow and inventory.

        The tape's limit orders are placed around a hidden price S that random-walks; the visible mid
        lags it by about 0.5 s. S is estimated as the average of the bid-side and ask-side running means
        of newly arrived limit-order prices. Aggressive volume (buys +, sells -), fading over FLOW_TAU_US,
        keeps moving the price for a fraction of a second after big IOCs.
        """
        mid = (bid + ask) / 2
        fair = mid + K_IMB * (bid_qty - ask_qty) / (bid_qty + ask_qty) - K_INV * (self._inventory - self._target)
        if K_S:
            if snapshot.bids is not self._last_bids:
                self._last_bids = snapshot.bids
                self._scan_new(snapshot.bids, -1)
            if snapshot.asks is not self._last_asks:
                self._last_asks = snapshot.asks
                self._scan_new(snapshot.asks, 1)
            if self._s_bid and self._s_ask:
                delta = (self._s_bid + self._s_ask) / 2 - mid
                fair += K_S * self._hp_scale(snapshot.timestamp_us, delta, mid) * delta
        if K_FLOW:
            ts = snapshot.timestamp_us
            self._flow *= math.exp((self._flow_ts - ts) / FLOW_TAU_US)
            self._flow_ts = ts
            for t in trades:
                if t.taker_owner == 'market':
                    self._flow += t.quantity if t.book_side == 'ask' else -t.quantity
            half = (bid_qty + ask_qty) / 2
            self._flow_depth = half if self._flow_depth == 0 else self._flow_depth + FLOW_DEPTH_EMA * (half - self._flow_depth)
            fair += K_FLOW * self._flow / self._flow_depth
        return fair

    def _hp_scale(self, ts: int, delta: float, mid: float) -> float:
        """Share of K_S to use, from how well delta = estimate - mid has predicted the mid recently.

        Every HP_SAMPLE_US a (delta, mid) sample is stored; once HP_HORIZON_US has passed, it is paired
        with the mid move since then. The correlation of delta with that move, over decayed sums
        (about the last HP_TAU_US), sets the scale: 0 while it is <= 0 or there are too few samples,
        1 from HP_CORR_FULL up. On tapes where orders are not placed around a hidden price, the
        estimate does not predict the mid and the term switches itself off.
        """
        if not HP_CHECK:
            return 1.0
        if ts > self._hp_ts:
            f = math.exp((self._hp_ts - ts) / HP_TAU_US)
            self._hp_sxy *= f
            self._hp_sxx *= f
            self._hp_syy *= f
            self._hp_w *= f
            self._hp_ts = ts
        pending = self._hp_pending
        while pending and pending[0][0] + HP_HORIZON_US <= ts:
            _, d0, m0 = pending.popleft()
            y = mid - m0
            self._hp_sxy += d0 * y
            self._hp_sxx += d0 * d0
            self._hp_syy += y * y
            self._hp_w += 1.0
        if ts >= self._hp_next:
            pending.append((ts, delta, mid))
            self._hp_next = ts + HP_SAMPLE_US
        if self._hp_w < HP_MIN_WEIGHT or self._hp_sxy <= 0 or self._hp_syy <= 0:
            return 0.0
        corr = self._hp_sxy / math.sqrt(self._hp_sxx * self._hp_syy)
        return min(1.0, corr / HP_CORR_FULL)

    def _targets(self, fair: float, bid: int, ask: int) -> tuple[int, int]:
        """Target price per side, from fair.

        Bid: the best ask if fair is EDGE_TAKE or more above it (cross; the quotes unwind it later);
        else one tick inside a 2+ tick spread if that still leaves EDGE_IN; otherwise the highest price
        at or below the best bid that is at least EDGE below fair. The ask mirrors it.
        """
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
        """Hold QUOTE_SIZE at `top`, and keep (never move) older orders up to LADDER_LEVELS - 1 ticks behind it.

        When the target moves a tick away from a resting order, the order stays where it is and keeps
        its queue place. An order priced better than `top` is also kept, and counts as the order at
        `top`, while it still has KEEP_EDGE of edge (joining needs EDGE); otherwise it is cancelled, as
        are orders deeper than the band. Cancelled orders count towards risk until confirmed. New size
        goes to `top`, capped by the risk room; when there is no room, the deepest kept order is
        cancelled to make some (without that, PnL fell by about 140k).
        """
        step = -1 if side is BUY else 1
        at_top = 0
        kept = []
        for order_id, order in self._orders.items():
            if order[0] is not side or order[3] or order[4]:
                continue
            depth = (order[1] - top) * step if top is not None else -1
            if depth < 0 and top is not None and (fair - order[1] if side is BUY else order[1] - fair) >= KEEP_EDGE:
                at_top += order[2]  # queue hysteresis: a placed order with KEEP_EDGE left stands in for the target order
            elif depth < 0 or depth >= LADDER_LEVELS:
                self.cancel_order(order_id)
                order[4] = True
            elif depth == 0:
                at_top += order[2]
            else:
                kept.append((depth, order_id))
        if top is None or top <= 0 or at_top >= QUOTE_SIZE:
            return
        room = self._room(side)
        want = QUOTE_SIZE - at_top
        if room < min(want, MIN_NEW) and kept:
            kept.sort(reverse=True)
            order_id = kept[0][1]
            self.cancel_order(order_id)
            self._orders[order_id][4] = True
            return
        qty = min(want, room)
        if qty >= MIN_NEW:
            self._send(side, top, qty, LIMIT)

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        """Per market event: read the touch, settle the shadow's pending IOC, update the signals,
        forecast, run the taker and the shadow, then place the quotes around fair."""
        bid, bid_qty = _touch(snapshot.bids)
        ask, ask_qty = _touch(snapshot.asks)
        if bid is None or ask is None:
            return
        ts = snapshot.timestamp_us
        self._shadow_fill(ts)
        self._update_signals(ts, bid, bid_qty, ask, ask_qty)
        forecast = self._forecast()
        if TAKE_ENABLED:
            self._run_taker(forecast, bid, ask)
            self._shadow_step(ts, bid, ask)
        if MAKE_ENABLED:
            fair = self._fair(snapshot, trades, bid, bid_qty, ask, ask_qty)
            pb, pa = self._targets(fair, bid, ask)
            self._manage_ladder(BUY, pb, fair)
            self._manage_ladder(SELL, pa, fair)

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        """Update inventory from fills, and drop orders that are filled, cancelled or rejected."""
        order = self._orders.get(event.order_id)
        if order is None:
            return
        status = event.status
        if status is OrderStatus.PARTIALLY_FILLED or status is OrderStatus.FILLED:
            self._inventory += event.filled_quantity if order[0] is BUY else -event.filled_quantity
            order[2] = event.remaining_quantity
            if status is OrderStatus.FILLED:
                self._forget(event.order_id, order[0])
        elif status is OrderStatus.CANCELED or status is OrderStatus.REJECTED:
            self._forget(event.order_id, order[0])

    def _forget(self, order_id: str, side: Side) -> None:
        self._orders.pop(order_id, None)
        if self._taker_id == order_id:
            self._taker_id = None
