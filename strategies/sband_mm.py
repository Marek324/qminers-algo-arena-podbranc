"""S-band market maker: rest orders just inside the band that new market orders draw around the hidden price S.

Generator (docs/DATA_ANALYSIS.md): a new limit buy is placed at floor(S - X), a sell at ceil(S + X), X > 0.05,
clamped so it never crosses; an IOC sell is priced round(S - 1), an IOC buy round(S + 1). So:
- every new market buy at p says S > p, every new sell at q says S < q (clamping keeps both true);
- a resting bid at b is hit by IOC sells only while S <= b + 1, and earns S - b per lot vs S, which is positive
  exactly when b < S.
Policy: L = highest recent new buy, U = lowest recent new sell (the band S lies in). Bid at L, ask at U; a bid
at L crosses any stale ask <= L (bought below S). A new sell at q <= L shows S fell below the old buys, so those
buys are dropped from the band at once (and the bid moves down), and vice versa. Optional second level at L - 1
/ U + 1 (filled only by IOCs that sweep the first level, at 1-2 ticks below S).
Inventory: |inventory| <= MAX_POS counting resting orders; beyond SKEW_AT the side that adds risk steps back a
tick.

Volatility gate: all quotes are pulled while an EMA of |mid change| exceeds VOL_MAX ticks/s.

Results (2026-11-08):
- Synthetic round-1 twin with true S (data/synthetic/exact_val_000): maker fills average +0.65 ticks/lot vs S at
  the fill, but the IOC that fills us pushes S, so it is +0.18 at 100 ms and +0.15 at 1 s.
- round3_public: 117,128 (WIN 100 ms, MAX_BAND 2, VOL_MAX 3). Ungated -195k to -390k, all lost in the most
  volatile quarter of minutes. Window 25 / 50 / 150 / 250 ms: 23k / 73k / 85k / 37k; second level always worse.
- Far below hmm_mm (~570k), which earns mostly by taking. Resting orders alone have a thin edge here: the push
  eats most of it and we queue behind the market orders placed at the same S bound.

Inventory is counted from fills only.
"""

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


class SBandMM(BaseStrategy):
    QTY = 100          # size per level
    MAX_POS = 100
    MIN_NEW = 5
    WIN_US = 100_000   # how long a new order keeps bounding S
    LEVELS = 1         # 1: only L / U; 2: also L - 1 / U + 1
    SKEW_AT = 60       # beyond this |inventory| the risk-adding side steps back one tick
    MAX_BAND = 2       # no quote when U - L is wider than this (no fresh information)
    VOL_TAU_US = 2_000_000
    VOL_MAX = 3.0      # pull all quotes while the mid moves faster than this (ticks/s, EMA of |mid change|)

    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        self.orders: dict[str, list] = {}   # id -> [sign, price, remaining]
        self.cancel_sent: set[str] = set()
        self.max_id = ''
        self.buys: deque = deque()   # (t, price), prices decreasing from the front: front = max
        self.sells: deque = deque()  # (t, price), prices increasing from the front: front = min
        self.last_bids = None
        self.last_asks = None
        self.vol = 0.0
        self.vol_t = 0
        self.mid = 0.0

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        now = snapshot.timestamp_us
        bids, asks = snapshot.bids, snapshot.asks
        if bids and asks:
            mid = (bids[0].price + asks[0].price) / 2
            dt = now - self.vol_t
            if self.mid and dt > 0:
                a = dt / self.VOL_TAU_US
                self.vol = self.vol * (1.0 - a) if a < 1.0 else 0.0
            self.vol += abs(mid - self.mid) * 1e6 / self.VOL_TAU_US if self.mid else 0.0
            self.mid = mid
            self.vol_t = now
        if bids is not self.last_bids or asks is not self.last_asks:
            self.last_bids = bids
            self.last_asks = asks
            self._scan(bids, asks, now)
        lo = now - self.WIN_US
        b, s = self.buys, self.sells
        while b and b[0][0] < lo:
            b.popleft()
        while s and s[0][0] < lo:
            s.popleft()
        if not b or not s:
            self._clear(1)
            self._clear(-1)
            return
        L = b[0][1]
        U = s[0][1]
        if U - L > self.MAX_BAND or self.vol > self.VOL_MAX:
            self._clear(1)
            self._clear(-1)
            return
        pb = L
        pa = U
        if self.inv > self.SKEW_AT:
            pb -= 1
        elif self.inv < -self.SKEW_AT:
            pa += 1
        self._quote(1, pb)
        self._quote(-1, pa)

    def _scan(self, bids, asks, now: int) -> None:
        mx = self.max_id
        lm = len(mx)
        new_max = mx
        news = []
        for views, sign in ((bids, 1), (asks, -1)):
            for o in views:
                if o.owner != 'market':
                    continue
                oid = o.order_id
                if len(oid) > lm or (len(oid) == lm and oid > mx):
                    news.append((oid, sign, o.price))
                    if len(oid) > len(new_max) or (len(oid) == len(new_max) and oid > new_max):
                        new_max = oid
        self.max_id = new_max
        if not news:
            return
        news.sort(key=lambda x: (len(x[0]), x[0]))
        b, s = self.buys, self.sells
        for _, sign, p in news:
            if sign > 0:
                # S > p: sells at or below p are stale
                while s and s[0][1] <= p:
                    s.popleft()
                while b and b[-1][1] <= p:
                    b.pop()
                b.append((now, p))
            else:
                while b and b[0][1] >= p:
                    b.popleft()
                while s and s[-1][1] >= p:
                    s.pop()
                s.append((now, p))

    def _cancel(self, oid: str) -> None:
        if oid not in self.cancel_sent:
            self.cancel_order(oid)
            self.cancel_sent.add(oid)

    def _clear(self, sign: int) -> None:
        for oid, o in self.orders.items():
            if o[0] == sign:
                self._cancel(oid)

    def _quote(self, sign: int, top: int) -> None:
        """Keep QTY at each of the LEVELS prices top, top - sign, ...; cancel anything else on this side."""
        want = {}
        for k in range(self.LEVELS):
            want[top - sign * k] = 0
        resting = 0
        for oid, o in self.orders.items():
            if o[0] != sign:
                continue
            resting += o[2]
            if oid in self.cancel_sent:
                continue
            if o[1] in want:
                want[o[1]] += o[2]
            else:
                self._cancel(oid)
        room = self.MAX_POS - sign * self.inv - resting
        side = Side.BUY if sign > 0 else Side.SELL
        for k in range(self.LEVELS):
            p = top - sign * k
            q = min(self.QTY - want[p], room)
            if q < self.MIN_NEW:
                continue
            oid = self.send_order(OrderRequest(side=side, price=p, quantity=q, order_type=OrderType.LIMIT))
            self.orders[oid] = [sign, p, q]
            room -= q

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        oid = event.order_id
        o = self.orders.get(oid)
        if o is None:
            return
        st = event.status
        if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
            self.inv += o[0] * event.filled_quantity
            o[2] -= event.filled_quantity
            if o[2] <= 0:
                self._drop(oid)
        elif st == OrderStatus.CANCELED:
            self._drop(oid)
        elif st == OrderStatus.REJECTED and oid not in self.cancel_sent:
            self._drop(oid)

    def _drop(self, oid: str) -> None:
        self.orders.pop(oid, None)
        self.cancel_sent.discard(oid)
