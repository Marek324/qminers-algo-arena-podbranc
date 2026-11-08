"""Round 3 candidate: r3_cand_nm6 with H_NMAX 5, the best point of the 3-6 plateau.

Re-tested 2026-11-08 (sweep_nm_fine + sweep_sq on the two compute servers): round3_public 652,005 vs nm6 643,075
(+1.4%), quarters 133,471 / 191,016 / 160,695 / 165,290 vs 129,216 / 191,981 / 159,262 / 161,758 (4 of 4 that move;
q2 -0.5%), SIG_Q 0.45/0.8 stay within -1%/-3.7%: the family is not cliff-sensitive around this point. Matches the
earlier sweeps (hmm_617_fit history: 5 650,306 > 6 645,040 at EDGE_IN 0.3). Base docstring (r3_cand_nm6): the final
submission (tuned hmm_mm) plus SIG_Q 0.6 and H_NMAX 6, jump sniper off.

H_NMAX caps how far (in half-ticks) one new order can push the regime HMM toward storm; a lower cap keeps scattered
far placements from flipping the strategy out of calm market making. round3_public by H_NMAX: 20 (final) 618,001,
12 625,074, 10 630,959, 8 637,639, 6 643,075, 5 652,005, 4 633,862, 3 645,293, 2 338,049 (cliff). 6 is taken from the
3-6 plateau, away from the cliff. Quarters at 6: 129,216 / 191,981 / 159,262 / 161,758 (3 of 4 above H_NMAX 10).
H_NMAX 10 version (r3_cand_tune): 630,959; quarters 122,673 / 187,574 / 159,350 / 160,485, all above the final.
robustness.py: mirrored 631,233, 8/8 reshuffled sessions profitable. Each change came from a quarter-guarded
coordinate-descent tune (+0.5% and 3 of 4 quarters). SNIPE_FAR = 100000 disables the inherited sniper.

Previous docstring (r3_sniper_mm): hmm_mm plus millisecond inventory overshoots on big jumps in the hidden price.

The exchange charges 10 per millisecond while |inventory| > 100, the same for 101 or 1000 lots, so a short trip is
cheap: 20 ms over costs 200. When a new market limit order lands SNIPE_FAR or more ticks behind the opposite best,
S has just jumped about that far (round 3: far orders lead the mid by about 1 s). One IOC sweeps the stale opposite
side at least SNIPE_MARGIN through the new S, up to SNIPE_POS lots of position (the 1000-lot hard limit counts
resting orders). The overshoot is closed in two stages: as soon as the near side has caught up with the new S (best
bid >= S_new - UNWIND_GAP for a long position) by an IOC priced at S_new - UNWIND_GAP, so it never dumps below that
level; and after MAX_OVER_US by an IOC through the touch. Between snipes this is hmm_mm unchanged.

Docstring of hmm_mm.py:

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


class R3CandNm5Strategy(BaseStrategy):
    QTY = 150
    MAX_POS = 100
    MIN_NEW = 10
    KEEP_LEVELS = 1
    # quiet regime (round 1 market maker settings)
    EDGE = 0.2
    EDGE_IN = 0.4
    EDGE_TAKE = 0.25
    K_IMB = 0.15
    K_INV = 0.002
    K_S = 0.75
    K_FLOW = 1.0
    Q_SWEEP = False      # quiet: when crossing, bid up to fair - EDGE_TAKE instead of only the best ask
    # volatile regime
    V_EDGE_TAKE = 0.55
    V_EXIT_EDGE = 100.0
    V_K_S = 0.8
    V_K_FLOW = 1.2
    V_K_INV = 0.0
    P_ON = 0.5           # p_vol above this: volatile policy
    # HMM
    H_N_Q0 = 1.0         # start mean of n (half ticks beyond the first) in the quiet state
    H_N_V0 = 3.2         # and in the volatile state
    H_PQV = 0.00003       # per new order: P(quiet -> volatile)
    H_PVQ = 0.0003       # P(volatile -> quiet)
    H_LR = 0.0003        # soft-EM weight of each order in the state means
    H_NMAX = 5
    H_IOC = False         # also feed the gaps between market IOCs to the HMM (exponential, rate learned per state)
    H_IOC_RQ0 = 8.0      # start IOC rate (per s) in the quiet state
    H_IOC_RV0 = 13.0     # and in the volatile state
    H_IOC_LR = 0.003
    # Kalman filter for S
    SIG_Q = 0.6  # ticks / sqrt(s) in the quiet state
    SIG_V = 10.0          # in the volatile state
    KF_LEVELS = 4        # quiet: use new orders up to this many ticks behind their best price
    KF_LEVELS_V = 4   # volatile: same
    C_LR = 0.0005        # weight of each order in the learned price-to-S gap c
    R_MIN = 0.1
    # jump sniper
    SNIPE_FAR = 100000  # a new order this many ticks behind the opposite best triggers a snipe (0: off)
    SNIPE_POS = 1000     # position a snipe may reach
    SNIPE_MARGIN = 3.0   # sweep stale levels at least this far through the new S
    X_MEAN = 0.8         # a new order sits about this far outside S
    UNWIND_GAP = 2.0     # stage 1: close the overshoot at S_new -+ UNWIND_GAP once the book reaches it
    MAX_OVER_US = 100_000  # stage 2: close the rest through the touch after this long over 100
    UNWIND_SLIP = 2
    SNIPE_COOLDOWN_US = 1_000_000
    CONFIRM_N = 1        # far orders on the same side needed (2+: independent confirmation of the jump)
    CONFIRM_US = 50_000  # ... within this long
    CONFIRM_TOL = 3      # ... agreeing on the new level to this many ticks
    DEPTH_SHRINK = 1.0   # new S = opposite best + DEPTH_SHRINK * (far price - opposite best); round 3 far orders overstate
                         # the jump by about 1.41x, so 0.6-0.7 places the sweep limit right (1.0: price -+ X_MEAN)
    EXIT_PASSIVE = False # rest the overshoot as one limit order at the new S for the post-jump taker surge to lift
    PASSIVE_OFF = 0.0    # its price: S_new + PASSIVE_OFF for a long position (mirrored for a short)
    SNIPE_QUIET_ONLY = False
    KILL_RATE_N = 4              # kill switch: more than this many snipes within KILL_RATE_US stops sniping for good
    KILL_RATE_US = 600_000_000
    KILL_OVER_US = 3_000_000     # kill switch: total time over MAX_POS (3 s = 30k of penalty) stops sniping for good  # snipe only while hmm_mm's HMM still says quiet (it is not sweeping the stale side itself)
    # flow
    FLOW_TAU_US = 100_000
    DEPTH_EMA = 0.001
    FLOW_CHECK = True
    FLOW_SLOPE_FULL = 0.5
    FLOW_CHECK_TAU_US = 300_000_000
    FLOW_CHECK_STEP_US = 50_000
    FLOW_CHECK_H_US = 500_000
    FLOW_CHECK_MIN = 200.0

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
        self.bb = 0
        self.ba = 0
        self.snipe_s = 0.0
        self.snipe_t = -10**12
        self.snipe_id = None
        self.unwind_id = None
        self.over_since = 0
        self.far_seen: list = []  # (time, sign, price) of recent far orders
        self.pexit_id = None      # resting passive exit order
        self.killed = False       # kill switch tripped: no more snipes
        self.snipe_times: deque = deque()
        self.over_total = 0
        self.over_last = 0

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
        self.bb, self.ba = bb, ba
        now = snapshot.timestamp_us
        self._unwind(now, bb, ba)
        if not self.s:
            self.s = mid
            self.kf_t = now
        if bids is not self.last_bids:
            self.last_bids = bids
            self._scan_new(bids, -1, now)
        if asks is not self.last_asks:
            self.last_asks = asks
            self._scan_new(asks, 1, now)

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
                if self.SNIPE_FAR and self.ba:
                    behind = (self.ba - 1 - o.price) if sign < 0 else (o.price - self.bb - 1)
                    if behind >= self.SNIPE_FAR:
                        self._far(sign, o.price, now)
                self._observe(sign, o.price, (o.price - first) * sign, now)
                if len(oid) > len(new_max) or (len(oid) == len(new_max) and oid > new_max):
                    new_max = oid
        self.max_id = new_max

    def _far(self, sign: int, price: int, now: int) -> None:
        """Snipe once CONFIRM_N far orders on this side agree on the new level within CONFIRM_US."""
        if self.CONFIRM_N <= 1:
            self._snipe(sign, price, now)
            return
        seen = [f for f in self.far_seen if now - f[0] <= self.CONFIRM_US]
        agree = [f for f in seen if f[1] == sign and abs(f[2] - price) <= self.CONFIRM_TOL]
        seen.append((now, sign, price))
        self.far_seen = seen
        if len(agree) + 1 >= self.CONFIRM_N:
            # the least extreme of the agreeing orders: a conservative estimate of the new level
            p = max([f[2] for f in agree] + [price]) if sign < 0 else min([f[2] for f in agree] + [price])
            self._snipe(sign, p, now)

    def _snipe(self, sign: int, price: int, now: int) -> None:
        """A market order far behind the touch: S jumped. Sweep the stale opposite side with one IOC."""
        if self.snipe_id is not None or self.unwind_id is not None or now - self.snipe_t < self.SNIPE_COOLDOWN_US:
            return
        if self.SNIPE_QUIET_ONLY and self.pv >= self.P_ON:
            return
        if self.killed or self.over_total > self.KILL_OVER_US:
            self.killed = True
            return
        while self.snipe_times and now - self.snipe_times[0] > self.KILL_RATE_US:
            self.snipe_times.popleft()
        if len(self.snipe_times) >= self.KILL_RATE_N:
            self.killed = True   # far orders far more often than on the real tapes: the trigger is not seeing jumps
            return
        if sign < 0:   # a buy far below: S fell to about price + X_MEAN, the bids above it are stale
            s_new = price + self.X_MEAN if self.DEPTH_SHRINK >= 1.0 else self.ba - 1 - self.DEPTH_SHRINK * (self.ba - 1 - price)
            limit_px = math.ceil(s_new + self.SNIPE_MARGIN)
            if self.bb < limit_px:
                return
            resting = sum(o[2] for o in self.orders.values() if o[0] < 0)
            q = min(self.SNIPE_POS + self.inv, 1000 + self.inv - resting)
            side, sg = Side.SELL, -1
        else:          # a sell far above: S rose, the asks below it are stale
            s_new = price - self.X_MEAN if self.DEPTH_SHRINK >= 1.0 else self.bb + 1 + self.DEPTH_SHRINK * (price - self.bb - 1)
            limit_px = math.floor(s_new - self.SNIPE_MARGIN)
            if self.ba > limit_px:
                return
            resting = sum(o[2] for o in self.orders.values() if o[0] > 0)
            q = min(self.SNIPE_POS - self.inv, 1000 - self.inv - resting)
            side, sg = Side.BUY, 1
        if q < self.MIN_NEW:
            return
        self.snipe_times.append(now)
        self.snipe_id = self.send_order(OrderRequest(side=side, price=limit_px, quantity=q, order_type=OrderType.IOC))
        self.orders[self.snipe_id] = [sg, limit_px, q]
        self.snipe_s = s_new
        self.snipe_t = now

    def _unwind(self, now: int, bb: int, ba: int) -> None:
        """Close |inventory| above MAX_POS: at S_new -+ UNWIND_GAP once the book is there, through the touch when late."""
        inv = self.inv
        if abs(inv) <= self.MAX_POS:
            self.over_since = 0
            self.over_last = 0
            return
        if self.over_last:
            self.over_total += now - self.over_last
        self.over_last = now
        if not self.over_since:
            self.over_since = now
        if self.unwind_id is not None:
            return
        late = now - self.over_since >= self.MAX_OVER_US
        if self.EXIT_PASSIVE:
            if not late:
                if self.pexit_id is None:
                    sg = -1 if inv > 0 else 1
                    px = max(1, round(self.snipe_s + self.PASSIVE_OFF)) if inv > 0 else max(1, round(self.snipe_s - self.PASSIVE_OFF))
                    q = abs(inv) - self.MAX_POS
                    side = Side.SELL if inv > 0 else Side.BUY
                    self.pexit_id = self.send_order(OrderRequest(side=side, price=px, quantity=q, order_type=OrderType.LIMIT))
                    self.orders[self.pexit_id] = [sg, px, q]
                return
            if self.pexit_id is not None and self.pexit_id not in self.cancel_sent:
                self._cancel(self.pexit_id)  # late: drop the passive exit, IOC the rest below
        if inv > 0:
            target = math.floor(self.snipe_s - self.UNWIND_GAP)
            if late or bb >= target:
                px = max(1, bb - self.UNWIND_SLIP) if late else target
                q = inv - self.MAX_POS
                self.unwind_id = self.send_order(OrderRequest(side=Side.SELL, price=px, quantity=q, order_type=OrderType.IOC))
                self.orders[self.unwind_id] = [-1, px, q]
        else:
            target = math.ceil(self.snipe_s + self.UNWIND_GAP)
            if late or ba <= target:
                px = ba + self.UNWIND_SLIP if late else target
                q = -inv - self.MAX_POS
                self.unwind_id = self.send_order(OrderRequest(side=Side.BUY, price=px, quantity=q, order_type=OrderType.IOC))
                self.orders[self.unwind_id] = [1, px, q]

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
            if o[0] == sign and oid != self.pexit_id:
                self._cancel(oid)

    def _side(self, sign: int, target: int) -> None:
        resting = 0
        at_target = 0
        keep = []
        for oid, o in self.orders.items():
            if o[0] != sign or oid in self.cancel_sent or oid == self.pexit_id:
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
        if oid == self.snipe_id:
            self.snipe_id = None
        if oid == self.unwind_id:
            self.unwind_id = None
        if oid == self.pexit_id:
            self.pexit_id = None
        self.orders.pop(oid, None)
        self.cancel_sent.discard(oid)
