"""Flow probe (impact): after a large market taker, take SIZE lots the same way and exit exactly HOLD_US later.

Measures how much of a big taker's price push is left after HOLD_US. Trigger: TRIG+ lots of market taker volume on
one side within one book event. Entry and exit are IOCs one tick through the touch, one position at a time,
|inventory| <= SIZE. Not a trading strategy.
"""

from __future__ import annotations

from simulator.models import OrderBookSnapshot, OrderRequest, OrderStatus, OrderType, PrivateOrderEvent, Side, Trade
from simulator.strategy import BaseStrategy


class FlowProbeImpact(BaseStrategy):
    TRIG = 200
    SIZE = 10
    HOLD_US = 5_000_000

    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        self.in_flight = False
        self.sides: dict[str, int] = {}   # order_id -> +1 buy / -1 sell
        self.exit_t = 0
        self.holding = False

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bb = next((o.price for o in snapshot.bids if o.owner == 'market'), None)
        ba = next((o.price for o in snapshot.asks if o.owner == 'market'), None)
        now = snapshot.timestamp_us
        if bb is None or ba is None or self.in_flight:
            return
        if self.holding or self.inv != 0:
            if now >= self.exit_t and self.inv != 0:
                buy = self.inv < 0
                side = Side.BUY if buy else Side.SELL
                price = ba + 1 if buy else max(1, bb - 1)
                oid = self.send_order(OrderRequest(side=side, price=price, quantity=abs(self.inv), order_type=OrderType.IOC))
                self.sides[oid] = 1 if buy else -1
                self.in_flight = True
            elif self.inv == 0 and now >= self.exit_t:
                self.holding = False
            return
        buy_vol = sell_vol = 0
        for t in trades:
            if t.taker_owner == 'market':
                if t.book_side == 'ask':
                    buy_vol += t.quantity
                else:
                    sell_vol += t.quantity
        if max(buy_vol, sell_vol) < self.TRIG:
            return
        buy = buy_vol >= sell_vol
        side = Side.BUY if buy else Side.SELL
        price = ba + 1 if buy else max(1, bb - 1)
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=self.SIZE, order_type=OrderType.IOC))
        self.sides[oid] = 1 if buy else -1
        self.in_flight = True
        self.holding = True
        self.exit_t = now + self.HOLD_US

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        st = event.status
        if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
            self.inv += self.sides.get(event.order_id, 0) * event.filled_quantity
        if st == OrderStatus.FILLED or st == OrderStatus.CANCELED or st == OrderStatus.REJECTED:
            self.in_flight = False
