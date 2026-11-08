"""Round 3 strategy: hmm_mm (two-regime HMM market maker, unchanged) plus leverage on jumps in the hidden price.

Round 3 (round3_public): same engine as round 2, with jumps in S up to ~120 ticks, higher volatility between them
(4.5 ticks/sqrt(s) at 1 s) and a thinner touch. hmm_mm scored 568,382 there, the best of our strategies, but it
sits at |inventory| = 100 half the time, so it is capacity-bound.

Leverage. The exchange charges a flat 10,000 per second while |inventory| > 100: the same for 101 lots as for 1000
(orders are rejected past |I + s(R + q)| = 1000). So the only leverage is a short, large overshoot when the edge is big:
1. Trigger: a new market order lands LEV_FAR or more ticks behind the opposite best price. S has jumped there:
   new S = price + c (bid) / price - c (ask), c the wider of the learned quiet / volatile order-to-S gaps (a smaller,
   safer jump estimate). A far ask means buy the stale asks,
   a far bid means sell the stale bids. The most conservative new S per snapshot is used.
2. Gate (big jumps only): walk the stale side down to new S -+ LEV_MARGIN, up to LEV_POS lots of position. For the
   lots past MAX_POS only, gain = sum((new S - price) - EXIT_COST). Go past MAX_POS only if
   gain >= LEV_HOLD_S * 10,000 * LEV_SAFETY (the expected penalty); otherwise take only what fits inside MAX_POS.
   With EXIT_COST 4.2 (measured) and a 1.5 s hold, 800 lots past the limit need about 23 ticks of edge per lot.
   One IOC, one in flight, LEV_COOLDOWN_US.
3. Unwind: while |inventory| > MAX_POS, IOC the excess limited at new S -+ UNWIND_GAP, which hits every order that
   has already caught up; after LEV_MAX_OVER_US, IOC priced through the visible book deep enough to clear it all.
Reused from strategies/round2/hmm_mm_plus.py and sniper_mm.py: one unwind IOC in flight, the caught-up-or-late rule.
Changed: they wait for the best price itself to catch up and then send one IOC a fixed slip through the touch;
strategies/round1/penalty_mm.py unwinds at the touch only, stays over on partial fills, and lost 2.6M to the
penalty on round 3. Here the late unwind is sized against the visible depth instead.
Against strategies/round2/sniper_mm.py (trigger at 20 ticks, no gate; it barely fired on round 3): the 10-tick trigger
catches the many 10-20 tick jumps of round 3, and the gate sizes the overshoot by the stale volume actually there
instead of overshooting into a thin stale side, where the penalty eats the edge.
Round 3 history (hmm_mm 568,382): best-price unwind, 1 s limit: 442,686, penalty 172,280. Depth-sized unwind,
0.5 s limit, EXIT_COST 1.5: 485,962, penalty 98,162. Its full-log breakdown: 17 overshoot entries of ~800 lots at
+8.9 ticks/lot against the price 1-2 s later (+124k), but 19 of 20 overshoots hit the forced unwind and paid
4.2 ticks/lot to get out, so +15.7k before the penalty. Hence the big-jumps-only gate above: 569,499, penalty 0
(no round 3 jump passes the gate; the +1.1k comes from the jump takes that fit inside MAX_POS).
LEV_FAR = 0 turns leverage off (plain hmm_mm).

Base (hmm_mm), docstring below.

Two-regime market maker: an online HMM picks the regime, a Kalman filter tracks the hidden price.

Market (round2_public, docs/DATA_ANALYSIS.md): limit orders arrive at floor(S - X) / ceil(S + X) around a hidden
price S and never cross the book. The generator switches between a quiet regime (new orders about 1 tick from the
mid, S about 1 tick/sqrt(s)) and a volatile one (about 2 ticks, S 4-6 ticks/sqrt(s), the big jumps). Resting
quotes earn the spread in quiet spells and get swept in volatile ones; taking the stale book pays in volatile ones.

1. Regime: a 2-state HMM with one observation per new market limit order: n = half-ticks between the order and
   the mid, beyond the first half tick. n is geometric in each state; the state means are learned online
   (soft EM, weight H_LR) from the start values H_N_Q0 / H_N_V0. Forward filter with switch probabilities
   H_PQV (quiet -> volatile) and H_PVQ per order gives p_vol.
2. Hidden price: a 1-D Kalman filter. Between orders the variance grows by sigma^2 * dt, sigma mixed by p_vol
   between SIG_Q and SIG_V. Each new order within KF_LEVELS (volatile: KF_LEVELS_V) ticks of its best price is a reading
   price + c (bid) or price - c (ask); c (learned online per state) is the average distance of new orders from
   S_hat, and the reading noise comes from the state's geometric spread.
3. Quiet (p_vol < P_ON): imbalance_mm. fair = mid + K_S * (S_hat - mid) + K_IMB * imbalance
   + K_FLOW * flow / depth - K_INV * inventory; cross at the opposite best price when fair is EDGE_TAKE through
   it, otherwise quote at the best price with EDGE to fair. Sticky orders, size capped by MAX_POS.
4. Volatile: fair from the V_ settings. Cross (sweeping up to fair - V_EDGE_TAKE) when the book is stale.
   No quote on the side that would grow |inventory|; the other side rests one exit order V_EXIT_EDGE from fair
   (100 = no exit order: pure taker, which was best on both halves of round 2).
Off by default (no gain on round 2): Q_SWEEP (sweep in quiet), H_IOC (IOC gaps as a second HMM observation).

Round 2 (round2_public): 502k with KF depth 1000; depth 4 in both regimes gave +26k on both halves.
Tuned on the two halves of round2_public, keeping values both halves agree on; most settings sit on flat plateaus.

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


class Round3Strategy(BaseStrategy):
    QTY = 200
    MAX_POS = 100
    MIN_NEW = 10
    KEEP_LEVELS = 1
    # quiet regime (round 1 market maker settings)
    EDGE = 0.3
    EDGE_IN = 0.5
    EDGE_TAKE = 0.25
    K_IMB = 0.15
    K_INV = 0.002
    K_S = 0.75
    K_FLOW = 0.8
    Q_SWEEP = False      # quiet: when crossing, bid up to fair - EDGE_TAKE instead of only the best ask
    # volatile regime
    V_EDGE_TAKE = 0.55
    V_EXIT_EDGE = 100.0
    V_K_S = 1.0
    V_K_FLOW = 1.2
    V_K_INV = 0.0
    P_ON = 0.5           # p_vol above this: volatile policy
    # HMM
    H_N_Q0 = 1.0         # start mean of n (half ticks beyond the first) in the quiet state
    H_N_V0 = 2.7         # and in the volatile state
    H_PQV = 0.0001       # per new order: P(quiet -> volatile)
    H_PVQ = 0.0003       # P(volatile -> quiet)
    H_LR = 0.0003        # soft-EM weight of each order in the state means
    H_NMAX = 20
    H_IOC = False         # also feed the gaps between market IOCs to the HMM (exponential, rate learned per state)
    H_IOC_RQ0 = 8.0      # start IOC rate (per s) in the quiet state
    H_IOC_RV0 = 13.0     # and in the volatile state
    H_IOC_LR = 0.003
    # Kalman filter for S
    SIG_Q = 0.5          # ticks / sqrt(s) in the quiet state
    SIG_V = 10.0          # in the volatile state
    KF_LEVELS = 4        # quiet: use new orders up to this many ticks behind their best price
    KF_LEVELS_V = 4   # volatile: same
    C_LR = 0.0005        # weight of each order in the learned price-to-S gap c
    R_MIN = 0.1
    # flow
    FLOW_TAU_US = 100_000
    DEPTH_EMA = 0.001
    FLOW_CHECK = True
    FLOW_SLOPE_FULL = 0.5
    FLOW_CHECK_TAU_US = 300_000_000
    FLOW_CHECK_STEP_US = 50_000
    FLOW_CHECK_H_US = 500_000
    FLOW_CHECK_MIN = 200.0
    # leverage: short overshoots past MAX_POS on jumps in S (penalty is flat per second outside +-MAX_POS)
    LEV_FAR = 10            # a new market order this many ticks behind the opposite best means S jumped (0: off)
    LEV_MARGIN = 2.0        # only take stale orders at least this far through the new S
    EXIT_COST = 4.2         # cost per lot of unwinding the overshoot (ticks; measured on round 3)
    LEV_HOLD_S = 1.5        # expected time outside +-MAX_POS per overshoot (s; the price settles in 1-2 s)
    LEV_SAFETY = 1.0        # overshoot only if gain >= LEV_HOLD_S * LEV_PENALTY * LEV_SAFETY
    LEV_PENALTY = 10_000.0  # exchange penalty per second outside +-MAX_POS
    LEV_POS = 900           # position an overshoot may reach (hard limit 1000 incl. resting orders)
    LEV_COOLDOWN_US = 500_000
    LEV_MAX_OVER_US = 1_500_000   # unwind at any price after this long outside +-MAX_POS (= LEV_HOLD_S)
    HARD_POS = 1000         # exchange rejects orders with |I + s(R + q)| above this
    UNWIND_GAP = 2.0        # unwind once the opposite best is within this of the new S
    UNWIND_SLIP = 2         # unwind IOC priced this many ticks through the opposite best
    JUMP_RESET = False      # on a trigger, also move S_hat to the new S and widen its variance
    JUMP_P = 4.0

    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        self.orders: dict[str, list] = {}   # order_id -> [sign, price, remaining]
        self.cancel_sent: set[str] = set()
        self.max_id = ''
        self.last_bids = None
        self.last_asks = None
        self.mid = 0.0
        # HMM
        self.pv = 0.0
        self.nq = self.H_N_Q0
        self.nv = self.H_N_V0
        self.iq = self.H_IOC_RQ0
        self.iv = self.H_IOC_RV0
        self.ioc_t = 0
        self.last_taker = ''
        # Kalman
        self.s = 0.0
        self.P = 1.0
        self.kf_t = 0
        self.cq = 0.5 + 0.5 * self.H_N_Q0
        self.cv = 0.5 + 0.5 * self.H_N_V0
        # flow
        self.flow = 0.0
        self.flow_t = 0
        self.depth = 0.0
        self.fc_pending: deque = deque()
        self.fc_next = 0
        self.fc_sxx = 0.0
        self.fc_sxy = 0.0
        self.fc_n = 0.0
        self.fc_decay = math.exp(-self.FLOW_CHECK_STEP_US / self.FLOW_CHECK_TAU_US)
        self.flow_scale = 0.0 if self.FLOW_CHECK else 1.0
        # leverage
        self.bb = 0
        self.ba = 0
        self.lev_dir = 0          # trigger found in this snapshot: +1 buy stale asks, -1 sell stale bids
        self.lev_new_s = 0.0      # its new S
        self.lev_bad = False      # both directions triggered in one snapshot
        self.lev_id = None        # leverage IOC in flight
        self.lev_t = -10**12
        self.lev_s = 0.0          # new S of the last leverage trade
        self.unw_id = None        # unwind IOC in flight
        self.over_since = 0

    def _cancel(self, oid: str) -> None:
        if oid not in self.cancel_sent:
            self.cancel_order(oid)
            self.cancel_sent.add(oid)

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bids, asks = snapshot.bids, snapshot.asks
        if not bids or not asks:
            return
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
        self.mid = mid
        now = snapshot.timestamp_us
        if not self.s:
            self.s = mid
            self.kf_t = now
        self.bb, self.ba = bb, ba
        if bids is not self.last_bids:
            self.last_bids = bids
            self._scan_new(bids, -1, now)
        if asks is not self.last_asks:
            self.last_asks = asks
            self._scan_new(asks, 1, now)
        if self.lev_dir:
            self._leverage(bids, asks, now)
        self._unwind(now, bids, asks, bb, ba)

        # flow
        self.flow *= math.exp((self.flow_t - now) / self.FLOW_TAU_US)
        self.flow_t = now
        for tr in trades:
            if tr.taker_owner == 'market':
                self.flow += tr.quantity if tr.book_side == 'ask' else -tr.quantity
                if self.H_IOC and tr.taker_order_id != self.last_taker:
                    self.last_taker = tr.taker_order_id
                    self._observe_ioc(now)
        half = (bq + aq) / 2
        self.depth = half if self.depth == 0 else self.depth + self.DEPTH_EMA * (half - self.depth)
        x = self.flow / self.depth
        if self.FLOW_CHECK and now >= self.fc_next:
            self._flow_check(now, x, mid)
        fx = self.flow_scale * x
        imb = (bq - aq) / (bq + aq)

        if self.pv < self.P_ON:
            fair = mid + self.K_S * (self.s - mid) + self.K_IMB * imb + self.K_FLOW * fx - self.K_INV * self.inv
            if fair - ba >= self.EDGE_TAKE:
                pb = math.floor(fair - self.EDGE_TAKE) if self.Q_SWEEP else ba
            elif ba - bb >= 2 and fair - (bb + 1) >= self.EDGE_IN:
                pb = bb + 1
            else:
                pb = bb
                while fair - pb < self.EDGE:
                    pb -= 1
            if bb - fair >= self.EDGE_TAKE:
                pa = math.ceil(fair + self.EDGE_TAKE) if self.Q_SWEEP else bb
            elif ba - bb >= 2 and (ba - 1) - fair >= self.EDGE_IN:
                pa = ba - 1
            else:
                pa = ba
                while pa - fair < self.EDGE:
                    pa += 1
            self._side(1, pb)
            self._side(-1, pa)
            return

        fair = mid + self.V_K_S * (self.s - mid) + self.V_K_FLOW * fx - self.V_K_INV * self.inv
        if fair - ba >= self.V_EDGE_TAKE:
            self._side(1, math.floor(fair - self.V_EDGE_TAKE))
        elif self.inv < 0:
            pb = bb
            while fair - pb < self.V_EXIT_EDGE:
                pb -= 1
            self._side(1, pb)
        else:
            self._clear(1)
        if bb - fair >= self.V_EDGE_TAKE:
            self._side(-1, math.ceil(fair + self.V_EDGE_TAKE))
        elif self.inv > 0:
            pa = ba
            while pa - fair < self.V_EXIT_EDGE:
                pa += 1
            self._side(-1, pa)
        else:
            self._clear(-1)

    def _observe(self, sign: int, price: int, depth: int, now: int) -> None:
        """One new market limit order: HMM forward step, then (if near the touch) a Kalman update."""
        # HMM on the order's distance from the mid
        n = min(self.H_NMAX, max(0, round(2 * (sign * (price - self.mid) - 0.5))))
        rq = self.nq / (1.0 + self.nq)
        rv = self.nv / (1.0 + self.nv)
        lq = (1.0 - rq) * rq ** n
        lv = (1.0 - rv) * rv ** n
        pv = self.pv * (1.0 - self.H_PVQ) + (1.0 - self.pv) * self.H_PQV
        a = pv * lv
        pv = a / (a + (1.0 - pv) * lq)
        self.pv = pv
        lr = self.H_LR
        self.nq += lr * (1.0 - pv) * (n - self.nq)
        self.nv += lr * pv * (n - self.nv)
        if self.nv < self.nq + 0.5:
            self.nv = self.nq + 0.5
        if depth > (self.KF_LEVELS_V if pv >= self.P_ON else self.KF_LEVELS):
            return
        # Kalman: predict
        dt = (now - self.kf_t) / 1e6
        self.kf_t = now
        sig2 = (1.0 - pv) * self.SIG_Q * self.SIG_Q + pv * self.SIG_V * self.SIG_V
        self.P += sig2 * max(dt, 0.0)
        # learned gap between new orders and S, per state
        d = -sign * (self.s - price)   # outward distance from S_hat
        self.cq += self.C_LR * (1.0 - pv) * (d - self.cq)
        self.cv += self.C_LR * pv * (d - self.cv)
        c = (1.0 - pv) * self.cq + pv * self.cv
        r = 0.25 * ((1.0 - pv) * rq / ((1.0 - rq) ** 2) + pv * rv / ((1.0 - rv) ** 2)) + self.R_MIN
        y = price - sign * c
        k = self.P / (self.P + r)
        self.s += k * (y - self.s)
        self.P *= 1.0 - k

    def _observe_ioc(self, now: int) -> None:
        """A new market IOC: the gap since the previous one is evidence for the regime."""
        if self.ioc_t:
            g = min(5.0, (now - self.ioc_t) / 1e6)
            lq = self.iq * math.exp(-self.iq * g)
            lv = self.iv * math.exp(-self.iv * g)
            pv = self.pv
            a = pv * lv
            pv = a / (a + (1.0 - pv) * lq)
            self.pv = pv
            # learn the rates: running mean of the gap per state
            mq = 1.0 / self.iq + self.H_IOC_LR * (1.0 - pv) * (g - 1.0 / self.iq)
            mv = 1.0 / self.iv + self.H_IOC_LR * pv * (g - 1.0 / self.iv)
            self.iq = 1.0 / max(mq, 0.01)
            self.iv = max(self.iq * 1.05, 1.0 / max(mv, 0.01))
        self.ioc_t = now

    def _scan_new(self, views, sign: int, now: int) -> None:
        """Feed market orders newer than max_id to the filters."""
        first = views[0].price
        mx = self.max_id
        lm = len(mx)
        new_max = mx
        for o in views:
            if o.owner != 'market':
                continue
            oid = o.order_id
            if len(oid) > lm or (len(oid) == lm and oid > mx):
                self._observe(sign, o.price, (o.price - first) * sign, now)
                if self.LEV_FAR:
                    self._jump_check(sign, o.price)
                if len(oid) > len(new_max) or (len(oid) == len(new_max) and oid > new_max):
                    new_max = oid
        self.max_id = new_max

    def _jump_check(self, sign: int, price: int) -> None:
        """A new market order far behind the opposite best: S jumped there. Keep the most conservative new S.

        A bid (sign -1) far below the best ask means S fell: sell the stale bids (direction -1). An ask (sign +1)
        far above the best bid means S rose: buy the stale asks (direction +1). So direction = sign.
        """
        gap = (self.ba - 1 - price) if sign < 0 else (price - self.bb - 1)
        if gap < self.LEV_FAR:
            return
        c = max(self.cq, self.cv)   # the wider (volatile) gap: a smaller, safer jump estimate
        s_new = price + c if sign < 0 else price - c
        if not self.lev_dir:
            self.lev_dir = sign
            self.lev_new_s = s_new
        elif self.lev_dir != sign:
            self.lev_bad = True
        elif sign > 0:
            self.lev_new_s = min(self.lev_new_s, s_new)
        else:
            self.lev_new_s = max(self.lev_new_s, s_new)

    def _leverage(self, bids, asks, now: int) -> None:
        """Sweep the stale side after a jump; go past MAX_POS only if the expected gain beats the penalty."""
        d = self.lev_dir
        s_new = self.lev_new_s
        bad = self.lev_bad
        self.lev_dir = 0
        self.lev_bad = False
        if bad:
            return
        if self.JUMP_RESET:
            self.s = s_new
            self.P = max(self.P, self.JUMP_P)
        if self.lev_id is not None or now - self.lev_t < self.LEV_COOLDOWN_US:
            return
        inv = self.inv
        resting = 0
        for o in self.orders.values():
            if o[0] == d:
                resting += o[2]
        qmax = min(self.LEV_POS - d * inv, self.HARD_POS - d * inv - resting)
        room = self.MAX_POS - d * inv - resting
        if d > 0:
            limit = math.floor(s_new - self.LEV_MARGIN)
            views = asks
        else:
            limit = math.ceil(s_new + self.LEV_MARGIN)
            views = bids
        free = max(0, room)   # lots that fit inside MAX_POS: no penalty
        qty = 0
        gain_over = 0.0       # expected gain of the lots past MAX_POS only (the worst-priced ones, filled last)
        for o in views:
            if o.owner != 'market':
                continue
            if d * (o.price - limit) > 0:
                break
            take = min(o.quantity, qmax - qty)
            if take <= 0:
                break
            over = max(0, qty + take - free) - max(0, qty - free)
            qty += take
            gain_over += (d * (s_new - o.price) - self.EXIT_COST) * over
        if qty > free and gain_over < self.LEV_HOLD_S * self.LEV_PENALTY * self.LEV_SAFETY:
            qty = min(qty, free)
        if qty < self.MIN_NEW:
            return
        side = Side.BUY if d > 0 else Side.SELL
        oid = self.send_order(OrderRequest(side=side, price=limit, quantity=qty, order_type=OrderType.IOC))
        self.orders[oid] = [d, limit, qty]
        self.lev_id = oid
        self.lev_t = now
        self.lev_s = s_new

    def _unwind(self, now: int, bids, asks, bb: int, ba: int) -> None:
        """Bring |inventory| back to MAX_POS: sell into whatever has caught up with the new S, all of it when late.

        Not late: IOC for the excess limited at new S -+ UNWIND_GAP, so every bid that has already caught up is hit
        at once instead of waiting for the best bid to get there. Late (LEV_MAX_OVER_US over the limit): IOC priced
        deep enough to clear the whole excess against the visible book (at least UNWIND_SLIP through the touch).
        penalty_mm unwound at the touch only and stayed over the limit on partial fills (round 3: 2.6M penalty).
        """
        inv = self.inv
        if -self.MAX_POS <= inv <= self.MAX_POS:
            self.over_since = 0
            return
        if not self.over_since:
            self.over_since = now
        if self.unw_id is not None:
            return
        late = now - self.over_since >= self.LEV_MAX_OVER_US
        s = self.lev_s if now - self.lev_t < 5_000_000 else self.s
        if inv > 0:
            q = inv - self.MAX_POS
            if late:
                px = min(bb - self.UNWIND_SLIP, self._clear_price(bids, q))
            else:
                px = math.ceil(s - self.UNWIND_GAP)
                if bb < px:
                    return
            px = max(1, px)
            oid = self.send_order(OrderRequest(side=Side.SELL, price=px, quantity=q, order_type=OrderType.IOC))
            self.orders[oid] = [-1, px, q]
        else:
            q = -inv - self.MAX_POS
            if late:
                px = max(ba + self.UNWIND_SLIP, self._clear_price(asks, q))
            else:
                px = math.floor(s + self.UNWIND_GAP)
                if ba > px:
                    return
            oid = self.send_order(OrderRequest(side=Side.BUY, price=px, quantity=q, order_type=OrderType.IOC))
            self.orders[oid] = [1, px, q]
        self.unw_id = oid

    def _clear_price(self, views, q: int) -> int:
        """Worst price needed to trade q lots against the visible market orders on one side (best first)."""
        px = views[0].price
        for o in views:
            if o.owner != 'market':
                continue
            px = o.price
            q -= o.quantity
            if q <= 0:
                break
        return px

    def _flow_check(self, now: int, x: float, mid: float) -> None:
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
        self.flow_scale = min(1.0, max(0.0, (self.fc_sxy / self.fc_sxx) / self.FLOW_SLOPE_FULL))

    def _clear(self, sign: int) -> None:
        for oid, o in self.orders.items():
            if o[0] == sign:
                self._cancel(oid)

    def _side(self, sign: int, target: int) -> None:
        resting = 0
        at_target = 0
        keep = []
        for oid, o in self.orders.items():
            if o[0] != sign or oid in self.cancel_sent:
                if o[0] == sign:
                    resting += o[2]
                continue
            behind = (target - o[1]) * sign
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
        if oid == self.lev_id:
            self.lev_id = None
        if oid == self.unw_id:
            self.unw_id = None
