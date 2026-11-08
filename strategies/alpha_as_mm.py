"""Avellaneda-Stoikov style market maker around an online-learned short-term fair value, with a toxicity breaker.

Ideas from the literature (2026-11-08 search): quote around a reservation price shifted by inventory and widen
with volatility (Avellaneda-Stoikov); value fills against a predicted price instead of the mid (Stoikov's
microprice, order-flow imbalance), which cuts adverse selection; pull quotes while flow is toxic (circuit
breaker) and take when the prediction is far through the book.

1. Features each event (no look-ahead): touch imbalance, 5-level imbalance, S-band centre ((L + U) / 2 - mid,
   where L / U are the highest new buy / lowest new sell of the last BAND_US: the generator puts buys below and
   sells above the hidden price S), new-order S EMA - mid, signed market taker flow / depth, mid - slow mid EMA.
2. Fair value: mid + RLS prediction of the mid change H_US ahead. RLS with forgetting LAM, updated every
   STEP_US with the realised (clipped) change, so it only learns from the past.
3. Quotes: r = fair - K_INV * inventory; half spread d = D0 + D1 * vol (vol = EMA of |mid change|, ticks/s).
   Bid at the highest price <= best bid with r - p >= d, ask mirrors it; cross (sweep up to r - EDGE_TAKE)
   when r is EDGE_TAKE through the opposite best price. Above VOL_MAX no resting quotes, only takes (threshold
   V_EDGE_TAKE). With SPLIT the calm and volatile periods get separate RLS models (regime-dependent alpha).

Tuning on round3_public only (2026-11-08): no breaker 173k; VOL_MAX 2 / 3 / 4 / 5 / 7: 521k / 535k / 520k /
495k / 419k. Flat around the optimum: D0 0.2-0.45, D1, H_US 300 ms-1 s, LAM, QTY, KEEP_LEVELS, Kalman feature
(all 507k-536k). Worse: EDGE_TAKE 0.3 / 0.8, K_INV 0 / 0.01, wider or tighter clips, volatility interactions,
split models, resting exit orders while volatile (138k-360k). PnL by minute volatility quartile: 44k calm,
59k, 104k, 328k wild, so most of it is taking during volatile spells.

Inventory is counted from fills only.
"""

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

NF = 8


class HiddenPriceKalman:
    """1-D Kalman filter for the hidden price S. See the module docstring for the model."""

    def __init__(self, q: float = 2.0, jump: float = 2e-4, jump_pow: float = 1.3, r0: float = 0.3,
                 r_alpha: float = 0.01, r_min: float = 0.05, m0: float = 1.0, m_alpha: float = 0.01,
                 push: float = 0.0, push_pow: float = 0.65) -> None:
        self.q = q  # ticks^2 per second
        self.jump = jump
        self.jump_pow = jump_pow
        self.r = r0
        self.r_alpha = r_alpha
        self.r_min = r_min
        self.m = m0
        self.m_alpha = m_alpha
        self.push = push
        self.push_pow = push_pow
        self.s = None
        self.p = 1.0
        self.ts = 0

    def _predict(self, ts_us: int) -> None:
        if ts_us > self.ts:
            self.p += self.q * (ts_us - self.ts) * 1e-6
            self.ts = ts_us

    def trade(self, ts_us: int, sign: int, qty: float) -> None:
        """Aggressive market volume `qty` in one event; sign +1 buyers lifted the ask, -1 sellers hit the bid."""
        if qty <= 0:
            return
        self._predict(ts_us)
        self.p += self.jump * qty ** self.jump_pow
        if self.push and self.s is not None:
            self.s += sign * self.push * qty ** self.push_pow

    def observe(self, ts_us: int, price: float, sign: int, capped: bool = False) -> None:
        """A new limit order at `price`; sign +1 buy, -1 sell. `capped`: it sits at the most aggressive price
        the generator allows (one tick from the opposite best), so it is only a bound."""
        if self.s is None:
            self.s = price + sign * self.m
            self.ts = ts_us
            return
        self._predict(ts_us)
        innov = price + sign * self.m - self.s
        if capped and innov * sign < 0:
            return  # consistent with the bound, nothing learned
        p = self.p
        self.m += self.m_alpha * (sign * (self.s - price) - self.m)
        self.r += self.r_alpha * (innov * innov - p - self.r)
        if self.r < self.r_min:
            self.r = self.r_min
        k = p / (p + self.r)
        self.s += k * innov
        self.p = (1.0 - k) * p

    def bound_low(self, lo: float) -> None:
        if self.s is not None and self.s < lo:
            self.s = lo

    def bound_high(self, hi: float) -> None:
        if self.s is not None and self.s > hi:
            self.s = hi


class AlphaASMM(BaseStrategy):
    QTY = 200
    MAX_POS = 100
    MIN_NEW = 10
    KEEP_LEVELS = 1
    # quoting
    D0 = 0.3
    D1 = 0.0
    K_INV = 0.004
    EDGE_TAKE = 0.5
    SWEEP = True
    VOL_TAU_US = 2_000_000
    VOL_MAX = 3.0
    SPLIT = False        # separate RLS models for calm (vol <= VOL_MAX) and volatile periods
    V_EDGE_TAKE = 0.5    # take threshold while volatile
    V_EXIT = 1e9         # while volatile, rest an inventory-reducing order this far from r (1e9: none)
    KF_LEVELS = 4        # new orders within this many ticks of their own best price feed the Kalman filter
    # learning
    H_US = 500_000
    STEP_US = 50_000
    LAM = 0.99995
    Y_CLIP = 4.0
    WARM = 1000          # RLS updates before the prediction is used
    PRED_CLIP = 6.0
    # features
    BAND_US = 100_000
    S_ALPHA = 0.2
    S_C = 1.1
    FLOW_TAU_US = 100_000
    DEPTH_EMA = 0.001
    SLOW_TAU_US = 1_000_000

    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        self.orders: dict[str, list] = {}
        self.cancel_sent: set[str] = set()
        self.max_id = ''
        self.last_bids = None
        self.last_asks = None
        self.buys: deque = deque()
        self.sells: deque = deque()
        self.s_ema = 0.0
        self.flow = 0.0
        self.flow_t = 0
        self.depth = 0.0
        self.slow = 0.0
        self.mid = 0.0
        self.t = 0
        self.vol = 0.0
        # RLS
        self.ws = [[0.0] * NF, [0.0] * NF]
        self.Ps = [[[(100.0 if i == j else 0.0) for j in range(NF)] for i in range(NF)] for _ in range(2)]
        self.ns = [0, 0]
        self.n_upd = 0
        self.pend: deque = deque()
        self.next_s = 0
        self.kf = HiddenPriceKalman()

    # ---------- learning
    def _rls(self, m: int, x, y) -> None:
        P = self.Ps[m]
        w = self.ws[m]
        self.ns[m] += 1
        Px = [sum(P[i][j] * x[j] for j in range(NF)) for i in range(NF)]
        den = self.LAM + sum(x[i] * Px[i] for i in range(NF))
        k = [v / den for v in Px]
        e = y - sum(w[i] * x[i] for i in range(NF))
        for i in range(NF):
            w[i] += k[i] * e
        il = 1.0 / self.LAM
        for i in range(NF):
            ki = k[i]
            row = P[i]
            for j in range(NF):
                row[j] = (row[j] - ki * Px[j]) * il

    # ---------- main
    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bids, asks = snapshot.bids, snapshot.asks
        now = snapshot.timestamp_us
        bb = ba = None
        bq = aq = 0
        b5 = a5 = 0
        for o in bids:
            if o.owner != 'market':
                continue
            if bb is None:
                bb = o.price
            if o.price == bb:
                bq += o.quantity
            if bb - o.price < 5:
                b5 += o.quantity
            else:
                break
        for o in asks:
            if o.owner != 'market':
                continue
            if ba is None:
                ba = o.price
            if o.price == ba:
                aq += o.quantity
            if o.price - ba < 5:
                a5 += o.quantity
            else:
                break
        if bb is None or ba is None:
            return
        mid = (bb + ba) / 2
        if not self.mid:
            self.mid = mid
            self.slow = mid
            self.s_ema = mid
            self.t = now
        dt = now - self.t
        if dt > 0:
            a = dt / self.VOL_TAU_US
            self.vol = self.vol * (1.0 - a) if a < 1.0 else 0.0
            a = dt / self.SLOW_TAU_US
            self.slow += (mid - self.slow) * (a if a < 1.0 else 1.0)
        self.vol += abs(mid - self.mid) * 1e6 / self.VOL_TAU_US
        self.mid = mid
        self.t = now

        if bids is not self.last_bids or asks is not self.last_asks:
            self.last_bids = bids
            self.last_asks = asks
            self._scan(bids, asks, now)
        lo = now - self.BAND_US
        b, s = self.buys, self.sells
        while b and b[0][0] < lo:
            b.popleft()
        while s and s[0][0] < lo:
            s.popleft()
        if b and s:
            band = (b[0][1] + s[0][1]) / 2 - mid
        elif b:
            band = max(0.0, b[0][1] + 1 - mid)
        elif s:
            band = min(0.0, s[0][1] - 1 - mid)
        else:
            band = 0.0

        self.flow *= math.exp((self.flow_t - now) / self.FLOW_TAU_US)
        self.flow_t = now
        tb = ts_ = 0
        for tr in trades:
            if tr.taker_owner == 'market':
                self.flow += tr.quantity if tr.book_side == 'ask' else -tr.quantity
                if tr.book_side == 'ask':
                    tb += tr.quantity
                else:
                    ts_ += tr.quantity
        if tb:
            self.kf.trade(now, 1, tb)
        if ts_:
            self.kf.trade(now, -1, ts_)
        half = (bq + aq) / 2
        self.depth = half if self.depth == 0 else self.depth + self.DEPTH_EMA * (half - self.depth)

        fl = self.flow / self.depth if self.depth > 0 else 0.0
        se = self.s_ema - mid
        kd = (self.kf.s - mid) if self.kf.s is not None else 0.0
        x = [1.0, (bq - aq) / (bq + aq), (b5 - a5) / (b5 + a5), band, se, fl, mid - self.slow, kd]

        reg = 1 if (self.SPLIT and self.vol > self.VOL_MAX) else 0
        # delayed learning
        if now >= self.next_s:
            self.next_s = now + self.STEP_US
            pend = self.pend
            while pend and now - pend[0][0] >= self.H_US:
                _, x0, m0, r0 = pend.popleft()
                y = mid - m0
                if y > self.Y_CLIP:
                    y = self.Y_CLIP
                elif y < -self.Y_CLIP:
                    y = -self.Y_CLIP
                self._rls(r0, x0, y)
                self.n_upd += 1
            pend.append((now, x, mid, reg))

        if self.n_upd < self.WARM:
            return
        m = reg if self.ns[reg] >= self.WARM else 0
        w = self.ws[m]
        pred = sum(w[i] * x[i] for i in range(NF))
        if pred > self.PRED_CLIP:
            pred = self.PRED_CLIP
        elif pred < -self.PRED_CLIP:
            pred = -self.PRED_CLIP
        r = mid + pred - self.K_INV * self.inv
        d = self.D0 + self.D1 * self.vol
        rest = self.vol <= self.VOL_MAX
        et = self.EDGE_TAKE if rest else self.V_EDGE_TAKE

        if r - ba >= et:
            self._side(1, math.floor(r - et) if self.SWEEP else ba)
        elif rest:
            pb = bb
            while r - pb < d:
                pb -= 1
            self._side(1, pb)
        elif self.inv < 0 and self.V_EXIT < 100:
            pb = bb
            while r - pb < self.V_EXIT:
                pb -= 1
            self._side(1, pb)
        else:
            self._clear(1)
        if bb - r >= et:
            self._side(-1, math.ceil(r + et) if self.SWEEP else bb)
        elif rest:
            pa = ba
            while pa - r < d:
                pa += 1
            self._side(-1, pa)
        elif self.inv > 0 and self.V_EXIT < 100:
            pa = ba
            while pa - r < self.V_EXIT:
                pa += 1
            self._side(-1, pa)
        else:
            self._clear(-1)

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
        news.sort(key=lambda z: (len(z[0]), z[0]))
        b, s = self.buys, self.sells
        al = self.S_ALPHA
        bb0 = bids[0].price if bids else None
        ba0 = asks[0].price if asks else None
        kl = self.KF_LEVELS
        for _, sign, p in news:
            self.s_ema += al * (p + sign * self.S_C - self.s_ema)
            if sign > 0:
                if bb0 is not None and bb0 - p < kl:
                    self.kf.observe(now, p, 1, ba0 is not None and p >= ba0 - 1)
            elif ba0 is not None and p - ba0 < kl:
                self.kf.observe(now, p, -1, bb0 is not None and p <= bb0 + 1)
            if sign > 0:
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

    # ---------- orders
    def _cancel(self, oid: str) -> None:
        if oid not in self.cancel_sent:
            self.cancel_order(oid)
            self.cancel_sent.add(oid)

    def _clear(self, sign: int) -> None:
        for oid, o in self.orders.items():
            if o[0] == sign:
                self._cancel(oid)

    def _side(self, sign: int, target: int) -> None:
        resting = 0
        at_target = 0
        keep = []
        for oid, o in self.orders.items():
            if o[0] != sign:
                continue
            resting += o[2]
            if oid in self.cancel_sent:
                continue
            behind = (target - o[1]) * sign
            if behind < 0 or behind > self.KEEP_LEVELS:
                self._cancel(oid)
            elif behind == 0:
                at_target += o[2]
            else:
                keep.append((behind, oid))
        if at_target >= self.QTY:
            return
        room = self.MAX_POS - sign * self.inv - resting
        want = self.QTY - at_target
        if room < min(want, self.MIN_NEW) and keep:
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
