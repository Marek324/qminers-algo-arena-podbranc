"""Flow probe B (round trip): every PERIOD_US take SIZE lots through the book and flatten half a period later.

Measures the real cost of walking the book (spread plus depth) averaged over thousands of trips, on a tape we can't
see (the arena validation set); fills count the levels walked. Run the same file on a constructed twin: a gap points
at the twin's book depth, shape and spread. Trips alternate buy-first / sell-first so price drift cancels; one IOC in
flight at a time, priced THROUGH ticks past the opposite best. |inventory| <= SIZE. Not a trading strategy.
"""

from __future__ import annotations

from simulator.models import OrderBookSnapshot, OrderRequest, OrderStatus, OrderType, PrivateOrderEvent, Side, Trade
from simulator.strategy import BaseStrategy


class FlowProbeRoundtrip(BaseStrategy):
    SIZE = 30
    THROUGH = 3
    PERIOD_US = 5_000_000

    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        self.in_flight = False
        self.sides: dict[str, int] = {}   # order_id -> +1 buy / -1 sell
        self.next_t = 0
        self.cycle = 0

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bb = next((o.price for o in snapshot.bids if o.owner == 'market'), None)
        ba = next((o.price for o in snapshot.asks if o.owner == 'market'), None)
        now = snapshot.timestamp_us
        if bb is None or ba is None or self.in_flight or now < self.next_t:
            return
        if self.inv == 0:   # open a trip, alternating direction
            buy = self.cycle % 2 == 0
            self.cycle += 1
            qty = self.SIZE
        else:               # flatten
            buy = self.inv < 0
            qty = abs(self.inv)
        side = Side.BUY if buy else Side.SELL
        price = ba + self.THROUGH if buy else max(1, bb - self.THROUGH)
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=qty, order_type=OrderType.IOC))
        self.sides[oid] = 1 if buy else -1
        self.in_flight = True
        self.next_t = now + self.PERIOD_US // 2

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        st = event.status
        if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
            self.inv += self.sides.get(event.order_id, 0) * event.filled_quantity
        if st == OrderStatus.FILLED or st == OrderStatus.CANCELED or st == OrderStatus.REJECTED:
            self.in_flight = False
