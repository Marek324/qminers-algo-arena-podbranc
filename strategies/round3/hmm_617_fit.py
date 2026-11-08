"""Two-regime market maker: an online HMM picks the regime, a Kalman filter tracks the hidden price.

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

Round 3 (round3_public): 617,400 (no breach penalty). Starts from hmm_mm_r3_tuned (586,613: EDGE 0.2, V_K_S 0.8, H_PQV 0.00003) and
adds four changes from a coordinate-descent tune on the four quarters of round3_public only (scripts/tune.py; a change
had to add 0.5% and win 3 of 4 quarters): EDGE_IN 0.5 -> 0.4, K_FLOW 0.8 -> 1.0, QTY 200 -> 150, H_N_V0 2.7 -> 3.2.
Quarters 119.7k / 184.6k / 159.1k / 154.6k (hmm_mm_r3_tuned: 114.7k / 181.9k / 137.8k / 155.6k). In-sample: the
arena scores on a private tape. EDGE_IN 0.4 -> 0.3: 625,338 (logs/r3-sample, full logs).

Iteration notes (2026-11-08), from a fill audit of the 625,338 run against the 1 s mid:
- Takes: 440k lots at +1.24 ticks/lot by 5 s (+544k). Passive: 646k lots at +0.19 (+123k). 72% of PnL (451k) is made
  in the [-2 s, +6 s] windows around 1 s moves of 5+ ticks, a third of the time; moves of 20+ ticks only +35k.
- Whipsaw: 643 takes are reversed by an opposite take within 0.5 s and lose -145k by 5 s, but the reversing takes
  make +78k. No pre-trade feature isolates the losers (their share rises from 7% to 20% with the size of the last 1 s
  move, yet every bucket still earns), so the confidence gate and reversal hysteresis of conf_sniper.py are not ported.
CURRENT DEFAULTS: H_NMAX 6 -> 5, EDGE_IN 0.3 -> 0.4, IDG_P 0 -> 25 (id-gap jump detector on); deep readings off.
round3_public 656,091, penalty 0. Decided on round3_public alone: the eval twin's order flow comes from our own
generator, so it rewards switches that match the generator's assumptions (see FLOW_POW below).
Public steps: 625,338 (H_NMAX 20, EDGE_IN 0.3) -> H_NMAX 5: 650,306 -> + IDG_P 25: 654,597; + EDGE_IN 0.4: 652,005;
both: 656,091.
On 656,091 (all within +-0.3% or worse, none adopted): IDG_P 100: 656,257; IDG_MISSING 2: 658,222; IDG_PV 0.9
(detection also forces the volatile policy): 656,115; KEEP_LEVELS 2: 656,406; V_EXIT_EDGE 1.5 (passive exit in the
volatile regime): 544,609, so pure taker in storms stays.
Queue ladder (L2_QTY lots one tick behind a top quote capped at TOP_QTY, quiet passive quotes only, never moved,
no room-freeing), the +31% idea of round 1: TOP/L2 60/40: 619,443; 50/50: 614,014; 80/40: 624,835; 100/50:
624,621. Rejected (-31k to -42k): on round 3 an order behind the touch fills only once the price has moved through
it, and round 3 moves are too big for that to pay. Left off (L2_QTY 0).

- Ported from hmm_deep617_r3, off: DEEP_R / JUMP_TH (deep-order readings and a variance boost on runs of far
  readings). Not ported (lost on round 3): K_INV / hold / recycling, leverage and the jump sniper, quote pull after
  far orders (r3_pull), 10 s flow reversion (r3_rev), censored readings.
- Backtests (pnl, penalty 0 everywhere), round3_public / r3_eval_twin (twin for reference only):
  H_NMAX 20: 625,338 / 645,104; H_NMAX 6: 645,040 / 735,536; DEEP_R 2 + JUMP_TH 6: 620,516 / 849,891;
  H_NMAX 6 + deep: 631,513 / 840,642; H_NMAX 5: 650,306 / 693,361; H_NMAX 5 + deep: 648,211 / 791,171.
- Mechanisms tried on H_NMAX 6 + deep (631,513 / 840,642):
  TB_OFF 1.5 + TB_P 1 (S_hat bounded by tape IOC trade prices): 619,475 / 816,096, worse on both.
  FLOW_POW 0.7 (concave IOC size in the flow): 625,513 / 947,218. The twin gain is not trustworthy: the twin
  generator (analysis/generator.py) pushes S by impact_k * size**impact_beta, the very shape this assumes, and the
  real public tape loses 6k.
  IDG_P 25 (tools/id_gap_jump.py detector boosts the filter): 635,310 / 840,642 (never fires on the twin: generated
  ids have no deep or hidden orders). Adopted: +4.3k again on the H_NMAX 5 base.
  Q_SWEEP True: 631,281 / 839,483, no change.
- H_NMAX history (r3_cand_nm6 at EDGE_IN 0.4): 20 -> 618k, 6 -> 643k, 5 -> 652k, 4 -> 634k, 3 -> 645k, cliff at 2.

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


class IdGapJump:
    """Jump detector from missing market order ids and clamped new orders (copied from tools/id_gap_jump.py)."""

    def __init__(self, window_us: int = 25_000, n_missing: int = 3) -> None:
        self.window_us = window_us
        self.n_missing = n_missing
        self.hi = 0
        self.gaps: deque = deque()
        self.cl_d: deque = deque()
        self.cl_u: deque = deque()
        self.nb: deque = deque()
        self.na: deque = deque()

    def seen(self, order_id: str, now: int) -> None:
        if not order_id.isdigit():
            return
        k = int(order_id)
        if k <= self.hi:
            return
        if self.hi:
            for _ in range(min(k - self.hi - 1, 5)):
                self.gaps.append(now)
        self.hi = k

    def new_order(self, sign: int, price: int, prev_bb: int, prev_ba: int, now: int) -> None:
        if sign > 0:
            if price == prev_bb + 1 and price < prev_ba:
                self.cl_d.append(now)
            else:
                self.na.append(now)
        else:
            if price == prev_ba - 1 and price > prev_bb:
                self.cl_u.append(now)
            else:
                self.nb.append(now)

    def check(self, now: int) -> int:
        lo = now - self.window_us
        for q in (self.gaps, self.cl_d, self.cl_u, self.nb, self.na):
            while q and q[0] < lo:
                q.popleft()
        if len(self.gaps) < self.n_missing:
            return 0
        if self.cl_d and not self.nb:
            return -1
        if self.cl_u and not self.na:
            return 1
        return 0


class Hmm617FitStrategy(BaseStrategy):
    QTY = 150
    MAX_POS = 100
    MIN_NEW = 10
    KEEP_LEVELS = 1
    L2_QTY = 0           # queue ladder (quiet passive quotes): lots resting one tick behind the top quote; 0: off
    TOP_QTY = 150        # ... and the top quote's cap while the ladder is on
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
    K_REV = 0.0          # reversion: fair moves K_REV of the way from mid to a slow EMA of the mid (round 3 reverts)
    REV_TAU_US = 30_000_000
    REV_CAP = 20.0       # cap on |slow EMA - mid| used, ticks
    P_ON = 0.5           # p_vol above this: volatile policy
    # HMM
    H_N_Q0 = 1.0         # start mean of n (half ticks beyond the first) in the quiet state
    H_N_V0 = 3.2         # and in the volatile state
    H_PQV = 0.00003       # per new order: P(quiet -> volatile)
    H_PVQ = 0.0003       # P(volatile -> quiet)
    H_LR = 0.0003        # soft-EM weight of each order in the state means
    H_NMAX = 5           # cap on one order's regime evidence (half ticks); r3_cand_nm6 plateau 3-6, cliff at 2
    H_IOC = False         # also feed the gaps between market IOCs to the HMM (exponential, rate learned per state)
    H_IOC_RQ0 = 8.0      # start IOC rate (per s) in the quiet state
    H_IOC_RV0 = 13.0     # and in the volatile state
    H_IOC_LR = 0.003
    # Kalman filter for S
    SIG_Q = 0.6  # ticks / sqrt(s) in the quiet state
    SIG_V = 10.0          # in the volatile state
    KF_LEVELS = 4        # quiet: use new orders up to this many ticks behind their best price
    KF_LEVELS_V = 4   # volatile: same
    # deep readings and jump boost (from hmm_deep617_r3; off: they only helped on the flawed eval twin)
    DEEP_R = 0.0         # > 0: orders deeper than KF_LEVELS(_V) also update S_hat, noise r * DEEP_R * extra depth
    DEEP_MAX = 1000      # ... up to this many ticks behind their best price
    JUMP_TH = 0.0        # > 0: a reading this many ticks from S_hat is a jump vote
    JUMP_N = 3           # consecutive same-side votes that trigger the boost
    JUMP_P = 25.0        # variance added to the filter on a trigger, so S_hat snaps to the new level
    C_LR = 0.0005        # weight of each order in the learned price-to-S gap c
    R_MIN = 0.1
    # trade bounds on S: tape IOCs are priced round(S +- 1), so a market buy that traded at P proves S >= P - 1.5
    TB_OFF = 0.0         # > 0: after a market buy at P, S_hat >= P - TB_OFF (sell mirrored); 0: off
    TB_P = 0.0           # variance added to the filter when the bound moves S_hat
    # id-gap jump detector (tools/id_gap_jump.py): on a detection, add IDG_P to the filter variance; 0: off
    IDG_P = 25.0
    IDG_WINDOW_US = 25_000
    IDG_MISSING = 3
    IDG_PV = 0.0         # > 0: a detection also lifts p_vol to at least this (switch to the volatile taker policy now)
    # flow
    FLOW_POW = 1.0       # each market taker order counts sign * q**FLOW_POW * FLOW_REF**(1 - FLOW_POW); 1: linear
    FLOW_REF = 60.0      # size (lots) that counts the same either way
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
        self.slow = 0.0
        self.slow_t = 0
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
        self.jv = 0          # jump votes: + for readings above S_hat, - below
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
        # id-gap jump detector
        self.idg = IdGapJump(self.IDG_WINDOW_US, self.IDG_MISSING) if self.IDG_P else None
        self.idg_t = 0
        self.pbb = 0
        self.pba = 0

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
        if bids is not self.last_bids:
            self.last_bids = bids
            self._scan_new(bids, -1, now)
        if asks is not self.last_asks:
            self.last_asks = asks
            self._scan_new(asks, 1, now)
        self.pbb, self.pba = bb, ba

        # flow, trade bounds on S
        self.flow *= math.exp((self.flow_t - now) / self.FLOW_TAU_US)
        self.flow_t = now
        idg = self.idg
        agg = None if self.FLOW_POW == 1.0 else {}
        tb_hi = tb_lo = 0
        for tr in trades:
            if tr.taker_owner == 'market':
                sgn = 1 if tr.book_side == 'ask' else -1
                if agg is None:
                    self.flow += sgn * tr.quantity
                else:
                    agg[tr.taker_order_id] = agg.get(tr.taker_order_id, 0) + sgn * tr.quantity
                if sgn > 0:
                    if tr.price > tb_hi:
                        tb_hi = tr.price
                elif tb_lo == 0 or tr.price < tb_lo:
                    tb_lo = tr.price
                if idg is not None:
                    idg.seen(tr.taker_order_id, now)
                if self.H_IOC and tr.taker_order_id != self.last_taker:
                    self.last_taker = tr.taker_order_id
                    self._observe_ioc(now)
        if agg:
            ref = self.FLOW_REF ** (1.0 - self.FLOW_POW)
            for v in agg.values():
                w = ref * abs(v) ** self.FLOW_POW
                self.flow += w if v > 0 else -w
        if self.TB_OFF:
            if tb_hi and self.s < tb_hi - self.TB_OFF:
                self.s = tb_hi - self.TB_OFF
                self.P += self.TB_P
            if tb_lo and self.s > tb_lo + self.TB_OFF:
                self.s = tb_lo + self.TB_OFF
                self.P += self.TB_P
        if idg is not None and now - self.idg_t > self.IDG_WINDOW_US and idg.check(now):
            self.P += self.IDG_P
            self.idg_t = now
            if self.pv < self.IDG_PV:
                self.pv = self.IDG_PV
        half = (bq + aq) / 2
        self.depth = half if self.depth == 0 else self.depth + self.DEPTH_EMA * (half - self.depth)
        x = self.flow / self.depth
        if self.FLOW_CHECK and now >= self.fc_next:
            self._flow_check(now, x, mid)
        fx = self.flow_scale * x
        imb = (bq - aq) / (bq + aq)
        rev = 0.0
        if self.K_REV:
            if not self.slow:
                self.slow = mid
            else:
                self.slow += (1.0 - math.exp((self.slow_t - now) / self.REV_TAU_US)) * (mid - self.slow)
            self.slow_t = now
            rev = self.slow - mid
            if rev > self.REV_CAP:
                rev = self.REV_CAP
            elif rev < -self.REV_CAP:
                rev = -self.REV_CAP
            rev *= self.K_REV

        if self.pv < self.P_ON:
            fair = mid + self.K_S * (self.s - mid) + self.K_IMB * imb + self.K_FLOW * fx - self.K_INV * self.inv + rev
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
            self._side(1, pb, pb < ba)
            self._side(-1, pa, pa > bb)
            return

        fair = mid + self.V_K_S * (self.s - mid) + self.V_K_FLOW * fx - self.V_K_INV * self.inv + rev
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
        extra = depth - (self.KF_LEVELS_V if pv >= self.P_ON else self.KF_LEVELS)
        if extra > 0 and (not self.DEEP_R or depth > self.DEEP_MAX):
            return
        # Kalman: predict
        dt = (now - self.kf_t) / 1e6
        self.kf_t = now
        sig2 = (1.0 - pv) * self.SIG_Q * self.SIG_Q + pv * self.SIG_V * self.SIG_V
        self.P += sig2 * max(dt, 0.0)
        # learned gap between new orders and S, per state (near-touch orders only)
        d = -sign * (self.s - price)   # outward distance from S_hat
        if extra <= 0:
            self.cq += self.C_LR * (1.0 - pv) * (d - self.cq)
            self.cv += self.C_LR * pv * (d - self.cv)
        c = (1.0 - pv) * self.cq + pv * self.cv
        r = 0.25 * ((1.0 - pv) * rq / ((1.0 - rq) ** 2) + pv * rv / ((1.0 - rv) ** 2)) + self.R_MIN
        if extra > 0:
            r *= self.DEEP_R * extra
        y = price - sign * c
        if self.JUMP_TH:
            e = y - self.s
            if e > self.JUMP_TH:
                self.jv = self.jv + 1 if self.jv > 0 else 1
            elif e < -self.JUMP_TH:
                self.jv = self.jv - 1 if self.jv < 0 else -1
            else:
                self.jv = 0
            if self.jv >= self.JUMP_N or -self.jv >= self.JUMP_N:
                self.P += self.JUMP_P
                self.jv = 0
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
                if self.idg is not None:
                    self.idg.seen(oid, now)
                    if self.pbb:
                        self.idg.new_order(sign, o.price, self.pbb, self.pba, now)
                if len(oid) > len(new_max) or (len(oid) == len(new_max) and oid > new_max):
                    new_max = oid
        self.max_id = new_max

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

    def _side(self, sign: int, target: int, passive: bool = False) -> None:
        """Keep up to QTY lots resting at `target` on side `sign` (orders up to KEEP_LEVELS behind are kept).

        Queue ladder (passive quiet quotes only, L2_QTY > 0): the top order is capped at TOP_QTY and up to L2_QTY lots
        rest one tick behind it. Neither is ever moved while inside the band, so when the touch steps onto the second
        order it is already near the front of that queue. No room-freeing cancels in this mode.
        """
        lad = passive and self.L2_QTY > 0
        qty = self.TOP_QTY if lad else self.QTY
        resting = 0
        at_target = 0
        l2 = 0
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
                if behind == 1:
                    l2 += o[2]
        if at_target >= qty and not lad:
            return
        room = self.MAX_POS - sign * self.inv - resting
        side = Side.BUY if sign > 0 else Side.SELL
        if at_target < qty:
            want = qty - at_target
            if not lad and room < min(want, self.MIN_NEW) and keep:
                keep.sort(reverse=True)
                self._cancel(keep[0][1])
                return
            q = min(want, room)
            if q >= self.MIN_NEW:
                oid = self.send_order(OrderRequest(side=side, price=target, quantity=q, order_type=OrderType.LIMIT))
                self.orders[oid] = [sign, target, q]
                room -= q
            elif not lad:
                return
        if lad:
            q = min(self.L2_QTY - l2, room)
            p2 = target - sign
            if q >= self.MIN_NEW and p2 > 0:
                oid = self.send_order(OrderRequest(side=side, price=p2, quantity=q, order_type=OrderType.LIMIT))
                self.orders[oid] = [sign, p2, q]

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
