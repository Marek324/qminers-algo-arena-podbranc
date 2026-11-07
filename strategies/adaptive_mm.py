"""Adaptive market maker.

Two legs share one short-horizon forecast:

- Maker leg: passive orders at the market's best bid and ask, earning the spread. On round 1 (a
  random walk above 1 s) this is where the money is. Orders are as large as the position cap
  allows and only step back from the touch when the forecast, a lopsided touch queue or our
  inventory clearly says so. When the touch moves a tick away, the order is kept one level behind
  instead of being cancelled; touch moves often revert, and the kept order is then near the front
  of the queue, where it is filled by small, profitable orders instead of only by the large sweeps
  that reach the back.
- Taker leg: holds a directional position when order flow predicts a move, crossing the spread
  with IOC orders. It only trades while a paper-traded 1-lot shadow copy of it is making money,
  so on tapes without momentum it stays off and the strategy is a pure market maker.

Forecast: order flow imbalance (OFI, Cont, Kukanov & Stoikov) at the touch, decayed over about
1 s and divided by the average touch size, giving an expected mid move in ticks.

Risk: every order is sized so that even if all our open orders on one side fill, |inventory|
stays within MAX_POS (the exchange penalty starts above 100). Inventory is counted from fills
only, so a cancel that loses the race against a fill cannot put it out of sync. Our own orders
are ignored when reading the book, so no signal reacts to us.

Run:  uv run exchange-simulator --strategy strategies/adaptive_mm.py --backtest data/round1.csv --log-dir logs/adaptive --no-logs
Tune: python3 -I scripts/sweep.py strategies/adaptive_mm.py data/round1.csv -p INV_FREE=40,60,80
Experiment history: docs/RESEARCH.md.
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

# Maker
MAKE_ENABLED = True
QUOTE_SIZE = 120  # lots at the top level (capped by the risk room)
BACKOFF_THRESHOLD = 2.0  # forecast ticks before the threatened quote steps back
INV_FREE = 80  # lots away from target before the adding side steps back
INV_STEP = 15  # lots per further tick of step-back
IMB_GUARD = 0.7  # touch imbalance beyond which the quote it points against steps back a tick
LADDER_LEVELS = 2  # an order the touch moves away from is kept up to LADDER_LEVELS - 1 ticks behind it

# Risk
MAX_POS = 99  # worst case |inventory| if every open order on one side fills

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

    def _desired_quotes(self, forecast: float, bid: int, bid_qty: int, ask: int, ask_qty: int) -> tuple[int, int]:
        """Top quote price per side, starting from the market's touch; each rule can only step it back.

        - Forecast: above BACKOFF_THRESHOLD the ask steps back ceil(forecast - threshold) ticks;
          below -BACKOFF_THRESHOLD the bid does.
        - Imbalance: when the touch queues are lopsided beyond IMB_GUARD (e.g. a much bigger bid queue,
          so the price is likely to tick up), the quote on the other side steps back a tick unless it
          unwinds inventory. A quote that is about to be run over is filled just before the move.
        - Inventory: once more than INV_FREE lots long of the target, the bid steps back one tick per
          INV_STEP lots beyond that (the ask when short). The side that unwinds stays at the touch.
        Quotes never go inside the spread: on round 1 and sample_v2 that cut PnL by a third or more.
        """
        qbid = bid
        qask = ask
        if forecast > BACKOFF_THRESHOLD:
            qask = ask + math.ceil(forecast - BACKOFF_THRESHOLD)
        elif forecast < -BACKOFF_THRESHOLD:
            qbid = bid - math.ceil(-forecast - BACKOFF_THRESHOLD)

        inv = self._inventory - self._target
        imb = (bid_qty - ask_qty) / (bid_qty + ask_qty)
        if imb > IMB_GUARD and inv <= 0:
            qask = max(qask, ask + 1)
        elif imb < -IMB_GUARD and inv >= 0:
            qbid = min(qbid, bid - 1)

        if inv > INV_FREE:
            qbid = min(qbid, bid - math.ceil((inv - INV_FREE) / INV_STEP))
        elif inv < -INV_FREE:
            qask = max(qask, ask + math.ceil((-inv - INV_FREE) / INV_STEP))
        return qbid, qask

    def _manage_ladder(self, side: Side, top: int | None) -> None:
        """Keep one fresh order at `top`, and keep (never move) older orders up to LADDER_LEVELS - 1 ticks behind it.

        When the touch moves a tick away from a resting order, the order stays where it is instead of
        being cancelled and re-sent at the back of the new queue. Touch moves often revert, and the kept
        order is then near the front of its queue. Orders priced better than `top` (the market moved
        away, or a step-back) or deeper than the band are cancelled; they count towards risk until the
        cancel is confirmed. New orders only go to `top`, sized QUOTE_SIZE capped by the risk room.
        """
        step = -1 if side is BUY else 1
        has_top = False
        for order_id, order in self._orders.items():
            if order[0] is not side or order[3] or order[4]:
                continue
            depth = (order[1] - top) * step if top is not None else -1
            if depth < 0 or depth >= LADDER_LEVELS:
                self.cancel_order(order_id)
                order[4] = True
            elif depth == 0:
                has_top = True
        if top is None or has_top or top <= 0:
            return
        qty = min(QUOTE_SIZE, self._room(side))
        if qty > 0:
            self._send(side, top, qty, LIMIT)

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        """Per market event: read the touch, settle the shadow's pending IOC, update the signals,
        forecast, run the taker and the shadow, then place the quotes."""
        bid, bid_qty = _touch(snapshot.bids)
        ask, ask_qty = _touch(snapshot.asks)
        if bid is None or ask is None:
            self._manage_ladder(BUY, None)
            self._manage_ladder(SELL, None)
            return
        ts = snapshot.timestamp_us
        self._shadow_fill(ts)
        self._update_signals(ts, bid, bid_qty, ask, ask_qty)
        forecast = self._forecast()
        if TAKE_ENABLED:
            self._run_taker(forecast, bid, ask)
            self._shadow_step(ts, bid, ask)
        if MAKE_ENABLED:
            qbid, qask = self._desired_quotes(forecast, bid, bid_qty, ask, ask_qty)
            self._manage_ladder(BUY, qbid)
            self._manage_ladder(SELL, qask)

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
