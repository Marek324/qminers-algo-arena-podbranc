"""Adaptive market maker: passive quotes at the touch, plus a momentum taker that only trades when a
paper-traded shadow copy of it is making money."""

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
OFI_TAU_US = 1_000_000
DEPTH_EMA = 0.01

# Taker
TAKE_ENABLED = True
TAKE_THRESHOLD = 1.0  # ticks
EXIT_THRESHOLD = 0.5  # ticks
TAKE_POS = 50  # lots
TAKE_SIZE = 50  # lots per IOC

# Momentum check
EDGE_CHECK = True
EDGE_TAU_US = 300_000_000
EDGE_FULL = 60.0  # shadow PnL (ticks per lot) needed for full TAKE_POS
LATENCY_US = 5_000

# Maker
MAKE_ENABLED = True
QUOTE_SIZE = 60  # lots
SKEW_PER_LOT = 0.1  # ticks per lot away from target
BACKOFF_THRESHOLD = 0.3  # ticks
FLOW_TAU_US = 500_000
PULL_RATIO = 1.0  # in average touch depths; 0 disables

# Risk
MAX_POS = 90  # worst case |inventory| if every open order on one side fills
MAX_QUOTES_PER_SIDE = 2

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
    def __init__(self) -> None:
        super().__init__()
        self._inventory = 0
        self._dir = 0
        self._target = 0
        # order_id -> [side, price, remaining_quantity, is_ioc]
        self._orders: dict[str, list] = {}
        self._quote: dict[Side, str | None] = {BUY: None, SELL: None}
        self._taker_id: str | None = None

        self._prev: tuple[int, int, int, int] | None = None
        self._ofi = 0.0
        self._flow = 0.0
        self._avg_depth = 0.0
        self._last_ts = 0

        self._sh_pos = 0
        self._sh_cash = 0.0
        self._sh_equity = 0.0
        self._sh_pending: tuple[int, int, int] | None = None  # (ready_ts, +1 buy / -1 sell, limit price)
        self._edge = 0.0

    def _update_signals(self, ts: int, bid: int, bid_qty: int, ask: int, ask_qty: int, trades: list[Trade]) -> None:
        dt = ts - self._last_ts
        self._last_ts = ts
        if dt > 0:
            self._ofi *= math.exp(-dt / OFI_TAU_US)
            self._flow *= math.exp(-dt / FLOW_TAU_US)
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

        for t in trades:
            if t.taker_owner == 'market':
                self._flow += t.quantity if t.book_side == 'ask' else -t.quantity

        depth = (bid_qty + ask_qty) / 2
        self._avg_depth = depth if self._avg_depth == 0.0 else self._avg_depth + DEPTH_EMA * (depth - self._avg_depth)

    def _forecast(self) -> float:
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

    def _count(self, side: Side) -> int:
        n = 0
        for order in self._orders.values():
            if order[0] is side and not order[3]:
                n += 1
        return n

    def _send(self, side: Side, price: int, qty: int, order_type: OrderType) -> str:
        order_id = self.send_order(OrderRequest(side=side, price=price, quantity=qty, order_type=order_type))
        self._orders[order_id] = [side, price, qty, order_type is IOC]
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
        if not EDGE_CHECK:
            return TAKE_POS
        scale = self._edge / EDGE_FULL
        if scale <= 0:
            return 0
        return round(TAKE_POS * min(1.0, scale))

    def _run_taker(self, forecast: float, bid: int, ask: int) -> None:
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

    def _desired_quotes(self, forecast: float, bid: int, ask: int) -> tuple[int | None, int | None]:
        signal = forecast - (self._inventory - self._target) * SKEW_PER_LOT
        qbid: int | None = bid
        qask: int | None = ask
        if signal > BACKOFF_THRESHOLD:
            qask = ask + math.ceil(signal - BACKOFF_THRESHOLD)
        elif signal < -BACKOFF_THRESHOLD:
            qbid = bid - math.ceil(-signal - BACKOFF_THRESHOLD)

        if PULL_RATIO > 0:
            depth = self._avg_depth if self._avg_depth > 0 else 1.0
            if self._flow > PULL_RATIO * depth and self._inventory <= self._target:
                qask = None
            elif self._flow < -PULL_RATIO * depth and self._inventory >= self._target:
                qbid = None
        return qbid, qask

    def _manage_quote(self, side: Side, price: int | None) -> None:
        quote_id = self._quote[side]
        if quote_id is not None:
            order = self._orders.get(quote_id)
            if order is None:
                self._quote[side] = None
            elif price is None or order[1] != price:
                # the cancelled order stays in _orders, and counts towards risk, until it is confirmed
                self.cancel_order(quote_id)
                self._quote[side] = None
            else:
                return
        if price is None or self._count(side) >= MAX_QUOTES_PER_SIDE:
            return
        qty = min(QUOTE_SIZE, self._room(side))
        if qty > 0:
            self._quote[side] = self._send(side, price, qty, LIMIT)

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bid, bid_qty = _touch(snapshot.bids)
        ask, ask_qty = _touch(snapshot.asks)
        if bid is None or ask is None:
            self._manage_quote(BUY, None)
            self._manage_quote(SELL, None)
            return
        ts = snapshot.timestamp_us
        self._shadow_fill(ts)
        self._update_signals(ts, bid, bid_qty, ask, ask_qty, trades)
        forecast = self._forecast()
        if TAKE_ENABLED:
            self._run_taker(forecast, bid, ask)
            self._shadow_step(ts, bid, ask)
        if MAKE_ENABLED:
            qbid, qask = self._desired_quotes(forecast, bid, ask)
            self._manage_quote(BUY, qbid)
            self._manage_quote(SELL, qask)

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
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
        if self._quote[side] == order_id:
            self._quote[side] = None
        if self._taker_id == order_id:
            self._taker_id = None
