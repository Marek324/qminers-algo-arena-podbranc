"""Round 3: confidence-gated stale-book sniper. Takes only when the edge beats how unsure the price estimate is.

Built from the round 3 evidence (round3_public, full-log run of round3_strategy at 569,499):
- PnL is taking: 507k taker lots at +0.97 ticks/lot by 5 s (about 85% of PnL). The edge is the book lagging the
  hidden price S. Passive fills that close a position earn +0.09 ticks/lot: cheap capacity recycling, not alpha.
- The leak is whipsaw. Taker orders followed by an opposite take within 0.5 s: 719 orders, 92k lots, -3.3 ticks/lot by
  5 s, about -306k (partly selection bias: a take that goes wrong is what triggers the reversal). Taker orders held
  over 1 s: +1.8 to +5.1 ticks/lot, +830k.
- Leverage (past +-100) is worth +1.7M with perfect foresight (max PnL 1.99M -> 3.69M with the 1000 cap and the
  10,000/s penalty; taking only 1.17M -> 2.26M), but it lives in ~300 bursts of ~1000 lots on ~8-tick moves in ~0.3 s
  with no observable precursor. Big IOC sweeps lose -0.5 to -1.1 ticks/lot after the spread; jump signals (a new order
  15+ ticks behind the opposite best) make +17 to +36 ticks/lot but leave only 30-50 lots at the touch. So leverage
  stays a narrow, gated jump snipe (unchanged from round3_strategy; it never passes the gate on round3_public).

Price estimate (kept from round3_strategy: two-regime HMM on how far new orders land from the mid, Kalman filter on S
from new orders near the touch, learned order-to-S gap per regime, flow-to-price slope check):
- quiet (p_vol < P_ON): fair = mid + K_S (S - mid) + K_IMB imbalance + K_FLOW flow - K_INV inventory
- volatile: fair = mid + V_K_S (S - mid) + V_K_FLOW flow - V_K_INV inventory

Decision layer (new):
1. Confidence gate (CONF_K). Each Kalman reading gives z^2 = innovation^2 / (P + r), about 1 while the filter is
   consistent with the data and large after jumps and in chop. zbar is its EWMA (CONF_A per reading). Every take
   threshold gets + CONF_K * max(0, sqrt(zbar) - CONF_Z0). The filter's noise model is conservative (sqrt(zbar) is
   0.58 at the median), so CONF_Z0 1 only reacts to the top ~3% (jumps); 0.8 reacts earlier.
2. Hysteresis (REV_EDGE, REV_T_US). A take against the direction of our last taker fill within REV_T_US needs
   REV_EDGE more edge, which breaks the +100 / -100 flip chains.
3. Edge-sized takes (SIZE_W, SIZE_MIN). A take may build the position to SIZE_MIN * MAX_POS at the threshold, rising to
   MAX_POS at threshold + SIZE_W. SIZE_W 0: all in, as before.
4. Censored readings (KF_CENSOR). A new bid at best ask - 1 (or ask at best bid + 1) sits at the no-cross cap, so it
   only bounds S from one side; it updates the filter only when it moves S_hat in the bound's direction.
5. Passive exits in the volatile regime: V_EXIT_EDGE (100 = none, pure taker; try 0.5-2 with hysteresis on).
6. Slow-flow mean reversion (MR_W; FlowReversion, copied from tools/flow_reversion.py). x = EWMA of signed market
   taker volume (MR_TAU_US) over the touch depth EMA. Online regression (through the origin, memory MR_MEM_US) of the
   next MR_H_US mid move on x, used only when its slope is negative and after MR_MIN samples. The quote / exit fair
   gets + MR_W * slope * x, capped at +-MR_CAP ticks; the take fair too only with MR_TAKE. Probe on the first
   1500 s of round 3: the slope settles at -0.3 to -0.48 (one dip to -0.03 at 700 s), predictions p10 / p90 -1.5 /
   +1.6 ticks; MR_W 1 with MR_TAKE (it chases a 5 s forecast across the spread and overrides the sub-second S edge)
   made 55k there against ~111k without; MR_W 1 on quotes only made 108k (neutral within noise).
Quiet regime: take at the touch when fair is the take threshold through it, otherwise quote at the touch with EDGE.
Volatile: sweep up to fair - threshold; no quote on the side that would grow |inventory|.

Off values (all of them together should reproduce round3_strategy, 569,499 on round3_public): CONF_K 0, REV_EDGE 0,
SIZE_W 0, KF_CENSOR False, V_EXIT_EDGE 100, MR_W 0 (checked bit for bit on the first 300 s). Defaults: confidence
gate and hysteresis on, the rest off.

Round 3 result of the defaults: 556,965 (round3_strategy 569,499; pnl_se of one run is about 30k, so this is noise
level). Full-log comparison (logs/r3-sample vs logs/r2-sample): hysteresis cut taker orders reversed within 0.5 s from
719 (-306k by 5 s) to 187 (-171k), but the reversals moved to 0.5-1 s (639 -> 796 orders, -34k -> -122k) and the
1-5 s takes lost 84k. All takes: +490k -> +484k. The first take was wrong; delaying the reversal only makes the exit
dearer. Losing takes can't be picked out at take time either: split by spread, levels swept, 200 ms or 2 s momentum or
price paid over the mid, every bucket is positive on average (bigger visible edge, better take). So this decision
layer sits on the plateau of its signal; gains need new information. Only lead left in the round 3 data: slow mean
reversion. Takes against the 2 s move earn +1.2-1.5 ticks/lot by 5 s, takes with it +0.45. The extreme deciles of
10 s IOC flow see -2.2 / +2.6 ticks over the next 5 s, and the strategy only uses 100 ms flow, with the momentum sign.

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


class FlowReversion:
    """Online slope of the future mid move on slow signed flow (copied from tools/flow_reversion.py)."""

    def __init__(self, tau_us: int = 10_000_000, h_us: int = 5_000_000, step_us: int = 100_000,
                 mem_us: int = 3_000_000_000, min_n: float = 3000.0, only_rev: bool = True) -> None:
        self.tau_us = tau_us
        self.h_us = h_us
        self.step_us = step_us
        self.min_n = min_n
        self.only_rev = only_rev
        self.decay = math.exp(-step_us / mem_us)
        self.flow = 0.0
        self.t = 0
        self.pend: deque = deque()
        self.next = 0
        self.sxx = 0.0
        self.sxy = 0.0
        self.n = 0.0
        self.beta = 0.0

    def _decay_to(self, now_us: int) -> None:
        if now_us > self.t:
            self.flow *= math.exp((self.t - now_us) / self.tau_us)
            self.t = now_us

    def add(self, now_us: int, signed_qty: float) -> None:
        self._decay_to(now_us)
        self.flow += signed_qty

    def step(self, now_us: int, mid: float, scale: float) -> float:
        self._decay_to(now_us)
        x = self.flow / scale if scale > 0 else 0.0
        if now_us >= self.next:
            self.next = now_us + self.step_us
            pend = self.pend
            d = self.decay
            while pend and now_us - pend[0][0] >= self.h_us:
                _, x0, m0 = pend.popleft()
                self.sxx = self.sxx * d + x0 * x0
                self.sxy = self.sxy * d + x0 * (mid - m0)
                self.n = self.n * d + 1.0
            pend.append((now_us, x, mid))
            b = self.sxy / self.sxx if self.n >= self.min_n and self.sxx > 0 else 0.0
            self.beta = min(b, 0.0) if self.only_rev else b
        return self.beta * x


class ConfSniperStrategy(BaseStrategy):
    QTY = 200
    MAX_POS = 100
    MIN_NEW = 10
    KEEP_LEVELS = 1
    # quiet regime
    EDGE = 0.3
    EDGE_IN = 0.5
    EDGE_TAKE = 0.3  
    K_IMB = 0.15
    K_INV = 0.002
    K_S = 0.75
    K_FLOW = 0.8
    # volatile regime
    V_EDGE_TAKE = 0.55
    V_EXIT_EDGE = 100.0   # resting exit order this far from fair (100: none, pure taker)
    V_K_S = 1.0
    V_K_FLOW = 1.2
    V_K_INV = 0.0
    P_ON = 0.5            # p_vol above this: volatile policy
    # decision layer (see docstring; the off value is in brackets)
    CONF_K = 0.0  # extra take edge per unit of sqrt(zbar) - CONF_Z0 [0]
    CONF_Z0 = 1.0         # sqrt(zbar) below this adds nothing (round 3: median 0.58, p90 0.86, p99 1.21, 5.7 after
                          # the 121-tick jump at t = 4130)
    CONF_A = 0.05         # EWMA weight of each Kalman reading's z^2
    CONF_ZMAX = 400.0     # cap on one reading's z^2
    REV_EDGE = 0.0  # extra take edge against the last taker fill's direction [0]
    REV_T_US = 500_000    # ... for this long after that fill
    SIZE_W = 0.0          # edge (ticks past the threshold) for a full-size take [0]
    SIZE_MIN = 0.5        # share of MAX_POS a take may build at the threshold
    KF_CENSOR = True  # readings at the no-cross cap are one-sided bounds [False]
    MR_W = 0.0            # weight of the slow-flow mean-reversion prediction in fair [0]
    MR_TAU_US = 10_000_000      # time constant of the slow flow
    MR_H_US = 5_000_000         # horizon of the predicted mid move
    MR_STEP_US = 100_000        # regression sample interval
    MR_MEM_US = 3_000_000_000   # regression memory
    MR_MIN = 3000.0             # samples (decayed count) before the slope is used (~5 min)
    MR_CAP = 3.0                # max shift of fair (ticks)
    MR_TAKE = False             # also shift the take decisions (False: only the resting quotes / exit orders)
    # HMM
    H_N_Q0 = 1.0          # start mean of n (half ticks beyond the first) in the quiet state
    H_N_V0 = 2.7          # and in the volatile state
    H_PQV = 0.0001        # per new order: P(quiet -> volatile)
    H_PVQ = 0.0003        # P(volatile -> quiet)
    H_LR = 0.0003         # soft-EM weight of each order in the state means
    H_NMAX = 20
    # Kalman filter for S
    SIG_Q = 0.5           # ticks / sqrt(s) in the quiet state
    SIG_V = 10.0          # in the volatile state
    KF_LEVELS = 4         # quiet: use new orders up to this many ticks behind their best price
    KF_LEVELS_V = 4       # volatile: same
    C_LR = 0.0005         # weight of each order in the learned price-to-S gap c
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
    # leverage: gated jump snipe past MAX_POS (penalty is flat per second outside +-MAX_POS)
    LEV_FAR = 10            # a new market order this many ticks behind the opposite best means S jumped (0: off)
    LEV_MARGIN = 2.0        # only take stale orders at least this far through the new S
    EXIT_COST = 4.2         # cost per lot of unwinding the overshoot (ticks)
    LEV_HOLD_S = 1.5        # expected time outside +-MAX_POS per overshoot (s)
    LEV_SAFETY = 1.0        # overshoot only if gain >= LEV_HOLD_S * LEV_PENALTY * LEV_SAFETY
    LEV_PENALTY = 10_000.0  # exchange penalty per second outside +-MAX_POS
    LEV_POS = 900           # position an overshoot may reach (hard limit 1000 incl. resting orders)
    LEV_COOLDOWN_US = 500_000
    LEV_MAX_OVER_US = 1_500_000   # unwind at any price after this long outside +-MAX_POS
    HARD_POS = 1000         # exchange rejects orders with |I + s(R + q)| above this
    UNWIND_GAP = 2.0        # unwind once the opposite best is within this of the new S
    UNWIND_SLIP = 2         # late unwind IOC priced at least this many ticks through the opposite best
    JUMP_RESET = False      # on a trigger, also move S_hat to the new S and widen its variance
    JUMP_P = 4.0

    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        self.orders: dict[str, list] = {}   # order_id -> [sign, price, remaining, is_take]
        self.cancel_sent: set[str] = set()
        self.max_id = ''
        self.last_bids = None
        self.last_asks = None
        self.mid = 0.0
        # HMM
        self.pv = 0.0
        self.nq = self.H_N_Q0
        self.nv = self.H_N_V0
        # Kalman
        self.s = 0.0
        self.P = 1.0
        self.kf_t = 0
        self.cq = 0.5 + 0.5 * self.H_N_Q0
        self.cv = 0.5 + 0.5 * self.H_N_V0
        self.zbar = 1.0           # EWMA of the readings' z^2
        # decision layer
        self.last_take_dir = 0    # direction of our last taker fill
        self.last_take_t = -10**12
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
        self.mr = FlowReversion(self.MR_TAU_US, self.MR_H_US, self.MR_STEP_US, self.MR_MEM_US, self.MR_MIN, True)
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

    # ------------------------------------------------------------------ main loop

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
                q = tr.quantity if tr.book_side == 'ask' else -tr.quantity
                self.flow += q
                if self.MR_W:
                    self.mr.add(now, q)
        half = (bq + aq) / 2
        self.depth = half if self.depth == 0 else self.depth + self.DEPTH_EMA * (half - self.depth)
        x = self.flow / self.depth
        if self.FLOW_CHECK and now >= self.fc_next:
            self._flow_check(now, x, mid)
        fx = self.flow_scale * x
        conf = self.CONF_K * max(0.0, math.sqrt(self.zbar) - self.CONF_Z0) if self.CONF_K else 0.0
        mr = 0.0
        if self.MR_W:
            mr = max(-self.MR_CAP, min(self.MR_CAP, self.MR_W * self.mr.step(now, mid, self.depth)))

        if self.pv < self.P_ON:
            fair = mid + self.K_S * (self.s - mid) + self.K_IMB * (bq - aq) / (bq + aq) + self.K_FLOW * fx \
                - self.K_INV * self.inv
            fq = fair + mr                               # quotes
            ft = fq if self.MR_TAKE else fair            # takes
            tb = self._threshold(1, self.EDGE_TAKE + conf, now)
            ta = self._threshold(-1, self.EDGE_TAKE + conf, now)
            cap_b = cap_a = self.MAX_POS
            take_b = ft - ba >= tb
            if take_b:
                pb = ba
                cap_b = self._take_cap(ft - ba - tb)
            elif ba - bb >= 2 and fq - (bb + 1) >= self.EDGE_IN:
                pb = bb + 1
            else:
                pb = bb
                while fq - pb < self.EDGE:
                    pb -= 1
            take_a = bb - ft >= ta
            if take_a:
                pa = bb
                cap_a = self._take_cap(bb - ft - ta)
            elif ba - bb >= 2 and (ba - 1) - fq >= self.EDGE_IN:
                pa = ba - 1
            else:
                pa = ba
                while pa - fq < self.EDGE:
                    pa += 1
            self._side(1, pb, cap_b, take_b)
            self._side(-1, pa, cap_a, take_a)
            return

        fair = mid + self.V_K_S * (self.s - mid) + self.V_K_FLOW * fx - self.V_K_INV * self.inv
        fq = fair + mr                                   # exit orders
        ft = fq if self.MR_TAKE else fair                # takes
        tb = self._threshold(1, self.V_EDGE_TAKE + conf, now)
        ta = self._threshold(-1, self.V_EDGE_TAKE + conf, now)
        if ft - ba >= tb:
            self._side(1, math.floor(ft - tb), self._take_cap(ft - ba - tb), True)
        elif self.inv < 0:
            pb = bb
            while fq - pb < self.V_EXIT_EDGE:
                pb -= 1
            self._side(1, pb, self.MAX_POS, False)
        else:
            self._clear(1)
        if bb - ft >= ta:
            self._side(-1, math.ceil(ft + ta), self._take_cap(bb - ft - ta), True)
        elif self.inv > 0:
            pa = ba
            while pa - fq < self.V_EXIT_EDGE:
                pa += 1
            self._side(-1, pa, self.MAX_POS, False)
        else:
            self._clear(-1)

    # ------------------------------------------------------------------ decision layer

    def _threshold(self, sign: int, base: float, now: int) -> float:
        """Take threshold for direction sign: base (+ confidence term) + REV_EDGE against a recent opposite take."""
        if self.REV_EDGE and self.last_take_dir == -sign and now - self.last_take_t < self.REV_T_US:
            return base + self.REV_EDGE
        return base

    def _take_cap(self, excess: float) -> int:
        """Largest |position| a take may build when its edge is `excess` ticks past the threshold."""
        if self.SIZE_W <= 0:
            return self.MAX_POS
        f = self.SIZE_MIN + (1.0 - self.SIZE_MIN) * excess / self.SIZE_W
        return int(self.MAX_POS * min(1.0, f))

    # ------------------------------------------------------------------ estimator

    def _observe(self, sign: int, price: int, depth: int, now: int) -> None:
        """One new market limit order (sign -1 bid, +1 ask): HMM step, then (if near the touch) a Kalman update."""
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
        if self.KF_CENSOR and (price == self.ba - 1 if sign < 0 else price == self.bb + 1):
            # at the no-cross cap: S is at least (bid) / at most (ask) the reading; skip it if that is no news
            c0 = (1.0 - pv) * self.cq + pv * self.cv
            if (price - sign * c0 - self.s) * sign > 0:
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
        innov = y - self.s
        z2 = innov * innov / (self.P + r)
        self.zbar += self.CONF_A * (min(z2, self.CONF_ZMAX) - self.zbar)
        k = self.P / (self.P + r)
        self.s += k * innov
        self.P *= 1.0 - k

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

    def _flow_check(self, now: int, x: float, mid: float) -> None:
        """Scale the flow term by the measured flow -> 0.5 s mid move slope (0 until enough samples)."""
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

    # ------------------------------------------------------------------ leverage (gated jump snipe)

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
        self.orders[oid] = [d, limit, qty, True]
        self.lev_id = oid
        self.lev_t = now
        self.lev_s = s_new

    def _unwind(self, now: int, bids, asks, bb: int, ba: int) -> None:
        """Bring |inventory| back to MAX_POS: sell into whatever has caught up with the new S, all of it when late."""
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
            self.orders[oid] = [-1, px, q, False]
        else:
            q = -inv - self.MAX_POS
            if late:
                px = max(ba + self.UNWIND_SLIP, self._clear_price(asks, q))
            else:
                px = math.floor(s + self.UNWIND_GAP)
                if ba > px:
                    return
            oid = self.send_order(OrderRequest(side=Side.BUY, price=px, quantity=q, order_type=OrderType.IOC))
            self.orders[oid] = [1, px, q, False]
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

    # ------------------------------------------------------------------ orders

    def _clear(self, sign: int) -> None:
        for oid, o in self.orders.items():
            if o[0] == sign:
                self._cancel(oid)

    def _side(self, sign: int, target: int, cap: int, is_take: bool) -> None:
        """Keep up to QTY lots at `target` on one side without letting the position pass `cap` (|inventory|)."""
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
        room = cap - sign * self.inv - resting
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
        self.orders[oid] = [sign, target, q, is_take]

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        oid = event.order_id
        o = self.orders.get(oid)
        if o is None:
            return
        st = event.status
        if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
            self.inv += o[0] * event.filled_quantity
            o[2] -= event.filled_quantity
            if o[3]:
                self.last_take_dir = o[0]
                self.last_take_t = self.now_us()
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
