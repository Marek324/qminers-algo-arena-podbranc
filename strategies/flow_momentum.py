"""Flow momentum: trade in the direction of aggressive (IOC) market orders.

In this market an aggressive order is followed by the quotes walking the same
way tick by tick for a few seconds, then giving part of the move back. The
strategy:

1. Signal: a trade where a market taker hits a market maker. Buy-side
   aggression (book_side 'ask') -> go long, sell-side -> go short.
2. Entry: IOC orders at the touch for a short window, while the price has
   not run more than ENTRY_MAX_TICKS past the price at the signal.
3. Exit: a ladder of resting take-profit limit orders TP_BASE, TP_BASE +
   TP_STEP, ... ticks beyond the signal price. The quotes walking into them
   fill them at our price (we are maker).
4. Time stop: after HOLD_US the take-profit is cancelled and the rest is
   closed with IOC orders at the touch.

Inventory is counted from fills only, and position + in-flight orders stay
within MAX_POS so the +-100 penalty band is never reached.
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


class FlowMomentumStrategy(BaseStrategy):
    MAX_POS = 95
    ENTRY_WINDOW_US = 1_500_000
    ENTRY_MAX_TICKS = 6
    TP_BASE = 14          # first take-profit level, ticks beyond the signal price
    TP_STEP = 6           # ticks between ladder levels
    TP_N = 5              # number of ladder levels
    HOLD_US = 6_000_000

    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        # order_id -> [side_sign, price, remaining, kind]; kind 'entry' | 'tp' | 'exit'
        self.orders: dict[str, list] = {}
        self.cancel_sent: set[str] = set()
        self.dir = 0
        self.sig_t = -1
        self.ref = 0
        self.ioc_busy = False

    # ---- helpers -------------------------------------------------------
    def _send(self, sign: int, price: int, qty: int, otype: OrderType, kind: str) -> None:
        side = Side.BUY if sign > 0 else Side.SELL
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=qty, order_type=otype))
        self.orders[oid] = [sign, price, qty, kind]
        if otype == OrderType.IOC:
            self.ioc_busy = True

    def _pending(self, sign: int) -> int:
        return sum(o[2] for o in self.orders.values() if o[0] == sign)

    def _cancel_kind(self, kind: str) -> None:
        for oid, o in self.orders.items():
            if o[3] == kind and oid not in self.cancel_sent:
                self.cancel_order(oid)
                self.cancel_sent.add(oid)

    # ---- callbacks -----------------------------------------------------
    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        if not snapshot.bids or not snapshot.asks:
            return
        now = snapshot.timestamp_us
        bids, asks = snapshot.bids, snapshot.asks
        best_bid, best_ask = bids[0].price, asks[0].price

        # 1. signal from market-vs-market aggression
        flow = 0
        for t in trades:
            if t.taker_owner == 'market' and t.maker_owner == 'market':
                flow += t.quantity if t.book_side == 'ask' else -t.quantity
        if flow != 0:
            d = 1 if flow > 0 else -1
            self.dir = d
            self.sig_t = now
            self.ref = best_ask if d > 0 else best_bid
            self._cancel_kind('tp')

        if self.dir == 0:
            return
        d = self.dir
        age = now - self.sig_t

        # 2. entry with IOCs
        if age < self.ENTRY_WINDOW_US:
            if not self.ioc_busy:
                px = best_ask if d > 0 else best_bid
                if (px - self.ref) * d <= self.ENTRY_MAX_TICKS:
                    room = self.MAX_POS - d * self.inv - self._pending(d)
                    lvl = asks if d > 0 else bids
                    avail = sum(o.quantity for o in lvl if o.price == px and o.owner == 'market')
                    q = min(room, avail)
                    if q > 0:
                        self._send(d, px, q, OrderType.IOC, 'entry')

        # 3. take-profit ladder, sized to the open position
        if age < self.HOLD_US:
            pos = self.inv * d
            if pos > 0:
                covered = sum(o[2] for oid, o in self.orders.items()
                              if o[3] == 'tp' and oid not in self.cancel_sent)
                need = pos - covered
                if need > 0:
                    # fill the ladder from the furthest level inwards
                    n = self.TP_N
                    per = max(1, -(-pos // n))
                    for i in range(n - 1, -1, -1):
                        if need <= 0:
                            break
                        price = self.ref + d * (self.TP_BASE + i * self.TP_STEP)
                        have = sum(o[2] for oid, o in self.orders.items()
                                   if o[3] == 'tp' and o[1] == price and oid not in self.cancel_sent)
                        q = min(need, per - have) if i > 0 else need
                        if q > 0:
                            self._send(-d, price, q, OrderType.LIMIT, 'tp')
                            need -= q
            return

        # 4. time stop: cancel take-profit, flatten with IOCs
        self._cancel_kind('tp')
        if self.inv == 0:
            self.dir = 0
            return
        if not self.ioc_busy:
            s = -1 if self.inv > 0 else 1
            px = best_bid if s < 0 else best_ask
            lvl = bids if s < 0 else asks
            avail = sum(o.quantity for o in lvl if o.price == px and o.owner == 'market')
            q = min(abs(self.inv) - self._pending(s), avail)
            if q > 0:
                self._send(s, px, q, OrderType.IOC, 'exit')

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        st = event.status
        if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
            self.inv += o[0] * event.filled_quantity
            o[2] -= event.filled_quantity
            if st == OrderStatus.FILLED or o[2] <= 0:
                self._done(event.order_id)
        elif st == OrderStatus.CANCELED or st == OrderStatus.REJECTED:
            if st == OrderStatus.REJECTED and event.order_id in self.cancel_sent:
                return  # cancel lost the race; the order is still live or filled
            self._done(event.order_id)

    def _done(self, oid: str) -> None:
        o = self.orders.pop(oid, None)
        self.cancel_sent.discard(oid)
        if o is not None and o[3] != 'tp':
            self.ioc_busy = False
