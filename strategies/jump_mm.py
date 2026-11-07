"""Market maker / stale-quote taker tuned for round 2 (fast hidden price with clustered jumps).

Round 2 (data/round2_public.csv): the hidden price S moves ~4.5x faster than in round 1 (sigma ~3.6 ticks/sqrt(s)),
mostly in jumps (1 s kurtosis 110-220, jumps cluster in time, sign is a coin flip, no post-jump drift).
After S jumps, new limit orders show the new level within milliseconds, while the visible book stays stale
for 50-500 ms. Our money comes from crossing to take those stale quotes (+1.4 ticks/lot after 1 s);
resting fills earn about 0.

Each event:
1. fair = mid + K_S * (S_hat - mid) + K_IMB * imbalance + K_FLOW * flow / depth
   S_hat = average of the per-side running means (weight S_ALPHA per order) of ALL newly arrived market limit
   orders (S_LEVELS = whole book: after a jump the informative orders sit several ticks behind the touch).
   flow = signed aggressive volume fading over FLOW_TAU_US, scaled live by FLOW_CHECK (see imbalance_mm.py).
2. Cross when fair - best_ask >= EDGE_TAKE, bidding up to floor(fair - EDGE_TAKE) (SWEEP), so all stale levels
   are taken in one order. Otherwise quote passively at the best price with edge >= EDGE + vol term
   (K_VOL * (recent mid path length / its slow average - 1), capped at VOL_CAP). The ask mirrors this.
3. Sticky orders, size capped so inventory + resting stays within MAX_POS (same as imbalance_mm.py).
No inventory skew (K_INV = 0): skewing made us cross to flatten and lost money on round 2.
JUMP_N (median-of-last-orders jump check) and K_PINV are kept but off: both lowered PnL on both halves.
Inventory is counted from fills only.
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


class JumpMMStrategy(BaseStrategy):
    QTY = 200
    MAX_POS = 100
    EDGE = 0.3           # min ticks between the target price and fair
    K_IMB = 0.15         # ticks of fair shift at full imbalance
    K_INV = 0.0        # ticks of fair shift per lot of inventory
    KEEP_LEVELS = 1      # how many ticks behind the target an order may stay
    MIN_NEW = 10         # don't send orders smaller than this
    EDGE_IN = 0.5        # min edge to quote inside a 2+ tick spread
    K_FLOW = 1.2         # ticks of fair shift per touch-depth of recent aggressive volume
    FLOW_TAU_US = 100_000
    DEPTH_EMA = 0.001
    K_S = 0.9           # weight of the hidden-price estimate: fair moves K_S of the way from mid to S_hat
    S_ALPHA = 0.3        # weight of each new limit order in the per-side price average
    S_LEVELS = 1000         # look for new orders this many ticks behind the best price
    EDGE_TAKE = 0.4     # cross the spread (bid at the best ask) when fair - best_ask >= this
    FLOW_CHECK = True    # scale K_FLOW by how well flow has predicted the mid lately
    FLOW_SLOPE_FULL = 0.5    # measured slope (mid ticks per unit of flow / depth) for full K_FLOW
    FLOW_CHECK_TAU_US = 300_000_000   # memory of the check (about 5 minutes)
    FLOW_CHECK_STEP_US = 50_000       # one sample every 50 ms
    FLOW_CHECK_H_US = 500_000         # paired with the mid move over the next 0.5 s
    FLOW_CHECK_MIN = 200.0            # samples (decayed) before the check trusts itself
    JUMP_N = 0           # jump check: median of the last JUMP_N new orders per side
    JUMP_TH = 1.5        # ticks between that median's S and mid that count as a jump
    JUMP_C0 = 1.3        # start value of the gap between a new order's price and S (learned online)
    JUMP_C_EMA = 0.001
    K_VOL = 1.0          # extra ticks of edge per unit of (recent vol / normal vol - 1)
    VOL_TAU_US = 3_000_000    # memory of the recent vol (sum of |mid moves|)
    VOL_SLOW = 0.0002         # per-event weight of the normal-vol average
    VOL_CAP = 3.0             # cap on the extra edge, ticks
    K_PINV = 0.0         # extra passive edge on the side that grows |inventory|, ticks at full position
    SWEEP = True         # when crossing, bid up to fair - EDGE_TAKE, not just the best ask

    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        # order_id -> [sign, price, remaining]
        self.orders: dict[str, list] = {}
        self.cancel_sent: set[str] = set()
        self.flow = 0.0          # signed aggressive volume, fading over FLOW_TAU_US
        self.flow_t = 0
        self.depth = 0.0         # running average touch size
        # hidden fair price: average price of newly arrived market limit orders, per side
        self.s_bid = 0.0
        self.s_ask = 0.0
        self.max_id = ''         # highest market order id seen so far (ids only grow)
        self.last_bids = None
        self.last_asks = None
        # flow check: decayed regression of the later mid move on flow / depth
        self.fc_pending: deque = deque()
        self.fc_next = 0
        self.fc_sxx = 0.0
        self.fc_sxy = 0.0
        self.fc_n = 0.0
        self.fc_decay = math.exp(-self.FLOW_CHECK_STEP_US / self.FLOW_CHECK_TAU_US)
        self.flow_scale = 0.0 if self.FLOW_CHECK else 1.0
        # jump check: recent new order prices per side, and the learned price-to-S gap
        self.rb: deque = deque(maxlen=self.JUMP_N)
        self.ra: deque = deque(maxlen=self.JUMP_N)
        self.jc = self.JUMP_C0
        self.vol = 0.0           # recent path length of the mid, fading over VOL_TAU_US
        self.vol_avg = 0.0       # slow average of vol
        self.vol_t = 0
        self.last_mid = 0.0

    def _cancel(self, oid: str) -> None:
        if oid not in self.cancel_sent:
            self.cancel_order(oid)
            self.cancel_sent.add(oid)

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bids, asks = snapshot.bids, snapshot.asks
        if not bids or not asks:
            return
        # touch of market orders only (ours excluded)
        bb = ba = None
        bq = aq = 0
        for o in bids:
            if o.owner == 'market':
                if bb is None:
                    bb = o.price
                elif o.price != bb:
                    break
                bq += o.quantity
        for o in asks:
            if o.owner == 'market':
                if ba is None:
                    ba = o.price
                elif o.price != ba:
                    break
                aq += o.quantity
        if bb is None or ba is None:
            return

        mid = (bb + ba) / 2
        fair = mid + self.K_IMB * (bq - aq) / (bq + aq) - self.K_INV * self.inv
        if self.K_S:
            # the side lists are cached by the engine: a new object means that side changed
            if bids is not self.last_bids:
                self.last_bids = bids
                self._scan_new(bids, -1)
            if asks is not self.last_asks:
                self.last_asks = asks
                self._scan_new(asks, 1)
            if self.s_bid and self.s_ask:
                fair += self.K_S * ((self.s_bid + self.s_ask) / 2 - mid)
        if self.K_FLOW:
            now = snapshot.timestamp_us
            self.flow *= math.exp((self.flow_t - now) / self.FLOW_TAU_US)
            self.flow_t = now
            for tr in trades:
                if tr.taker_owner == 'market':
                    self.flow += tr.quantity if tr.book_side == 'ask' else -tr.quantity
            half = (bq + aq) / 2
            self.depth = half if self.depth == 0 else self.depth + self.DEPTH_EMA * (half - self.depth)
            x = self.flow / self.depth
            if self.FLOW_CHECK and now >= self.fc_next:
                self._flow_check(now, x, mid)
            fair += self.K_FLOW * self.flow_scale * x
        if self.JUMP_N and len(self.ra) == self.JUMP_N and len(self.rb) == self.JUMP_N:
            if self.s_bid and self.s_ask:
                self.jc += self.JUMP_C_EMA * ((self.s_ask - self.s_bid) / 2 - self.jc)
            up = sorted(self.ra)[self.JUMP_N // 2] - self.jc
            dn = sorted(self.rb)[self.JUMP_N // 2] + self.jc
            # after a jump up, new bids are clamped under the ask, so only the asks show where S went
            if up - mid > self.JUMP_TH and mid - dn <= self.JUMP_TH:
                fair = max(fair, up)
            elif mid - dn > self.JUMP_TH and up - mid <= self.JUMP_TH:
                fair = min(fair, dn)

        edge = self.EDGE
        if self.K_VOL:
            now = snapshot.timestamp_us
            self.vol *= math.exp((self.vol_t - now) / self.VOL_TAU_US)
            self.vol_t = now
            if self.last_mid:
                self.vol += abs(mid - self.last_mid)
            self.last_mid = mid
            self.vol_avg = self.vol if self.vol_avg == 0 else self.vol_avg + self.VOL_SLOW * (self.vol - self.vol_avg)
            if self.vol_avg > 0:
                edge += min(self.VOL_CAP, self.K_VOL * max(0.0, self.vol / self.vol_avg - 1.0))

        if fair - ba >= self.EDGE_TAKE:
            pb = math.floor(fair - self.EDGE_TAKE) if self.SWEEP else ba
        elif ba - bb >= 2 and fair - (bb + 1) >= self.EDGE_IN:
            pb = bb + 1
        else:
            pb = bb
            eb = edge + self.K_PINV * max(0, self.inv) / self.MAX_POS
            while fair - pb < eb:
                pb -= 1
        if bb - fair >= self.EDGE_TAKE:
            pa = math.ceil(fair + self.EDGE_TAKE) if self.SWEEP else bb
        elif ba - bb >= 2 and (ba - 1) - fair >= self.EDGE_IN:
            pa = ba - 1
        else:
            pa = ba
            ea = edge + self.K_PINV * max(0, -self.inv) / self.MAX_POS
            while pa - fair < ea:
                pa += 1
        self._side(1, pb)
        self._side(-1, pa)

    def _flow_check(self, now: int, x: float, mid: float) -> None:
        """Every FLOW_CHECK_STEP_US: pair old (flow, mid) samples with the mid move since, update the slope."""
        self.fc_next = now + self.FLOW_CHECK_STEP_US
        pend = self.fc_pending
        d = self.fc_decay
        while pend and now - pend[0][0] >= self.FLOW_CHECK_H_US:
            _, x0, m0 = pend.popleft()
            y = mid - m0
            self.fc_sxx = self.fc_sxx * d + x0 * x0
            self.fc_sxy = self.fc_sxy * d + x0 * y
            self.fc_n = self.fc_n * d + 1.0
        pend.append((now, x, mid))
        if self.fc_n < self.FLOW_CHECK_MIN or self.fc_sxx <= 0:
            self.flow_scale = 0.0
            return
        slope = self.fc_sxy / self.fc_sxx
        self.flow_scale = min(1.0, max(0.0, slope / self.FLOW_SLOPE_FULL))

    def _scan_new(self, views, sign: int) -> None:
        """Fold market orders newer than max_id, within S_LEVELS of the best price, into the side's average."""
        first = views[0].price
        mx = self.max_id
        lm = len(mx)
        a = self.S_ALPHA
        for o in views:
            if (o.price - first) * sign > self.S_LEVELS:
                break
            if o.owner != 'market':
                continue
            oid = o.order_id
            if len(oid) > lm or (len(oid) == lm and oid > mx):
                (self.rb if sign < 0 else self.ra).append(o.price)
                if sign < 0:
                    self.s_bid = o.price if not self.s_bid else self.s_bid + a * (o.price - self.s_bid)
                else:
                    self.s_ask = o.price if not self.s_ask else self.s_ask + a * (o.price - self.s_ask)
                if len(oid) > len(self.max_id) or (len(oid) == len(self.max_id) and oid > self.max_id):
                    self.max_id = oid

    def _side(self, sign: int, target: int) -> None:
        resting = 0
        at_target = 0
        keep = []
        for oid, o in self.orders.items():
            if o[0] != sign or oid in self.cancel_sent:
                if o[0] == sign:
                    resting += o[2]
                continue
            behind = (target - o[1]) * sign      # >0: less aggressive than target
            if behind < 0 or behind > self.KEEP_LEVELS:
                self._cancel(oid)
                resting += o[2]
                continue
            resting += o[2]
            if behind == 0:
                at_target += o[2]
            else:
                keep.append((behind, oid, o[2]))
        if at_target >= self.QTY:
            return
        room = self.MAX_POS - sign * self.inv - resting
        want = self.QTY - at_target
        if room < min(want, self.MIN_NEW) and keep:
            # free room: cancel the deepest kept order
            keep.sort(reverse=True)
            self._cancel(keep[0][1])
            return
        q = min(want, room)
        if q < self.MIN_NEW:
            return
        side = Side.BUY if sign > 0 else Side.SELL
        oid = self.send_order(OrderRequest(side=side, price=target, quantity=q, order_type=OrderType.LIMIT))
        self.orders[oid] = [sign, target, q]

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
