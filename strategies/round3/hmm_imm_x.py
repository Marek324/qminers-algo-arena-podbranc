"""hmm_imm_x: hmm_imm_b2 plus switchable aggressive / leverage ideas (all off by default = hmm_imm_b2).

- LEV_POS > 100: jump-mode leverage. A volatile-policy cross may take the position up to LEV_POS while the IMM's jump
  mode has probability >= LEV_MUJ and fair is at least LEV_EDGE through the opposite best. The excess over 100 is
  closed by an IOC LEV_SLIP ticks through the touch once that no longer holds, or after LEV_US over 100 (the penalty
  is a flat 10 per ms over 100).
- Economics review (2026-11-08), all off by default:
  V_EDGE_SLACK > 0: a volatile take may pay one tick through the best (ba + 1) when fair is still V_EDGE_SLACK
  through that price (the best is often swept within the 5 ms latency).
  EDGE_TAKE_OPEN > 0: a quiet take only closes the position unless fair is EDGE_TAKE_OPEN through the touch (calm
  takes earn +0.77 t/lot at 5 s but -0.25 at 30 s, so the opening part of a weak take is held through the decay).
  AGE_SKEW > 0: inventory born from a quiet take skews the passive quotes by up to AGE_SKEW ticks towards closing,
  ramping from AGE_T0_US to AGE_T0_US + AGE_T1_US after the take.
  V_PASS_EDGE > 0: in the volatile policy, rest one quote at the touch on the side that reduces |inventory| when it is
  V_PASS_EDGE from fair (storm passive fills +0.6 t/lot at 5 s); takes then ignore orders behind their price.
  FLOW_CAP: upper clip of the online flow-impact scale (1.0 = hmm_mm).
- Mathematics review (2026-11-08), all off by default:
  R_SAME: the storm mode reads S with the calm noise r_q (the generator's X does not depend on the regime; the wide
  storm placements come from a stale book). Pair with SIG_V near the fitted storm sigma 4.2 (SIG_V 11 with r_v was
  the same gain: 11 * sqrt(r_q / r_v) = 4.2).
  J_PROPER: the jump mode reads every order with r_q, no deep inflation, offset cq + J_CDEEP for deep orders.
  MKT_BEST: an order's depth is measured from the best market order, not from a best that includes our quotes.
  NMAX_TAIL: the capped distance bin n = H_NMAX gets the tail probability r^n.
  SUPPORT: S_hat per mode stays above a new market bid and below a new market ask (X >= 0.05).
  IOC_K > 0: each market IOC shifts every mode by +-IOC_K * q^IOC_B (generator impact 0.0258 * q^0.49);
  IOC_BOUND > 0: a market buy that filled up to price p implies S >= p - IOC_BOUND (IOC priced at round(S +- 1)).
- Probability review (2026-11-08), all off by default:
  H_CAT: regime evidence from a learned 13-bin pmf of n per regime (real round3_public starting values), tempered by
  H_W (orders are strongly correlated, so each one is worth far less than an independent reading). Replaces the
  geometric likelihood in the mode update; the geometric means still set the reading noise.
  PJ_SPLIT: jump-mode entry probability per order differs by mode (IMM_PJ_Q from calm, IMM_PJ_V from storm).
  EXC: post-jump excitation of the storm variance: when the jump mode first passes 0.5, E += EXC_K * |innovation|,
  E decays with EXC_TAU_S, and the storm / jump modes predict with (SIG_V^2 + E) dt.
  H_IOC with IMM: the IOC-gap evidence now updates the IMM mode probabilities (weight H_IOC_W).
- ASYM > 1: quote size times ASYM on the side that reduces |inventory| (faster flips), quiet policy.

hmm_imm_b2: hmm_imm_pon02 retuned on round3_public: SIG_V 11, V_EDGE_TAKE 0.5, SIG_Q 0.7, K_FLOW 1.0, V_K_S 0.7,
DEEP_R 2.5, K_IMB 0.12. round3_public 728,501, 0 penalty (Lambda sweeps p1-p3, 2026-11-08). Every one-step neighbour
scores lower on public (SIG_V 10.5 / 12: 728,172 / 726,728; without DEEP_R 725,614; without SIG_V 723,718; K_IMB 0.1
728,117). Tuned on round3_public only, in-sample; not run on the eval twin.

hmm_imm_pon02: hmm_imm_r3 with P_ON 0.2 (volatile policy from p_vol 0.2), IMM_JVAR 25, jump rule off.

The best IMM on round3_public: 710,629 (r3_eval_twin 939,161), 0 penalty; Lambda sweep imm1, variant pon02
(IMM_WP 0, IMM_WP_J 1). hmm_deep_clear_ks: 713,244 / 925,207; hmm_imm_r3 defaults: 704,572 / 969,300.

hmm_imm_r3: hmm_deep_clear_ks with the HMM + Kalman + jump rule replaced by a 3-mode IMM filter (IMM = True).

The generator is a switching state-space model: S diffuses with a calm or storm volatility, the regime switches
(calm ~180 s, storm ~90 s on round3_public), Pareto jumps on top, and new limit orders are readings of S a gamma
distance X away (wider in storms). The IMM (interacting multiple model) runs one Kalman filter per mode - calm (Q),
storm (V), jump (J: storm noise plus a one-step variance IMM_JVAR) - mixes them by a Markov switch every order, and
weighs the modes by two kinds of evidence per new order:
- placement distance (exactly the tuned HMM likelihood: geometric in n half-ticks, capped at H_NMAX, per-mode means
  learned online), the jump mode uses the storm one;
- the price innovation y - m_i against its predicted variance P_i + r_i (Gaussian, tempered by IMM_WP so a single
  long-tailed X does not flip the mode; IMM_RMAX_SD > 0 also caps it).
Carried over from the HMM tuning: H_NMAX 5, H_PQV / H_PVQ, SIG_Q 0.6 / SIG_V 10, KF_LEVELS 2 / KF_LEVELS_V 3, deep
readings (DEEP_R 3, DEEP_MAX), learned order-to-S offsets, R_MIN; IMM_JVAR starts at the tuned JUMP_P 25.
p_vol = mu_V + mu_J and S_hat = sum mu_i m_i feed the unchanged decision layer. IMM = False is hmm_deep_clear_ks.

Results (Lambda 185.175.110.225, 2026-11-08; hmm_deep_clear_ks = IMM False: round3_public 713,244 / r3_eval_twin 925,207):
these defaults (IMM_WP 0, IMM_WP_J 1, IMM_JVAR 50, IMM_KEEP_JUMP on) 704,572 / 969,300 (jump rule off: 704,692 / 963,348). Every IMM variant beats hmm_deep_clear_ks on the
twin (+25k to +38k) and trails it on public (-8k to -10k); weighting the price evidence for calm vs storm (IMM_WP 0.1+)
costs 90-200k on the twin. P_ON 0.2 gives the best public IMM score (710,629 / 939,161).

hmm_deep_clear_ks: hmm_deep_clear_r3 retuned on Barbora (2026-11-08, sweeps s14/s15 and c1-c4).

Changes from hmm_deep_clear_r3: KF_LEVELS 3 -> 2, K_INV 0.002 -> 0.0015, H_NMAX 4 -> 5, K_FLOW 1.0 -> 0.9,
DEEP_R 2.0 -> 3.0, H_N_V0 3.2 -> 3.5, K_S 0.75 -> 0.85.
round3_public 713,244 (hmm_deep_clear_r3 693,522), r3_eval_twin 925,207 (815,273), 0 breach penalty.
Search: one step either side of 20 constants, then stacking the ones that beat the base on both tapes, then
leave-one-out (dropping H_PVQ 0.0004 helped both). Every one-step neighbour of the final point scores less on
round3_public (K_FLOW 0.8 / H_NMAX 6 add 21-24k on the twin but cost 6-12k on public).
Rejected: SIG_V 8-12, tighter quotes (EDGE 0.1-0.15, EDGE_IN 0.2), V_QTY 150/300, EDGE_TAKE 0.3, QTY 170, SIG_Q 0.7.

Docstring of hmm_deep_clear_r3.py:

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
   No quote on the side that would grow |inventory|; the other side is cleared (hmm_deep_clear_r3).
   (100 = no exit order: pure taker, which was best on both halves of round 2).
Off by default (no gain on round 2): Q_SWEEP (sweep in quiet), H_IOC (IOC gaps as a second HMM observation).

Round 2 (round2_public): 502k with KF depth 1000; depth 4 in both regimes gave +26k on both halves.
Tuned on the two halves of round2_public, keeping values both halves agree on; most settings sit on flat plateaus.

Round 3 (hmm_deep_r3) = hmm_mm plus jump-aware S readings:
- New orders deeper than KF_LEVELS(_V) = 2 ticks behind their best price are no longer dropped: they update the
  Kalman filter with noise r * DEEP_R * (extra depth) and do not train c. One stray far order (long-tailed X)
  barely moves S_hat; the run of far orders that follows a jump pulls it to the new level within a few orders.
- Jump boost: JUMP_N consecutive readings more than JUMP_TH ticks from S_hat on the same side add JUMP_P to the
  filter variance, so S_hat snaps to the new level.
- P_ON 0.3 (volatile policy a bit earlier).
Backtests (pnl): round3_public 590,474 (hmm_mm 568,382); r3_eval_twin 928,455 (hmm_mm 731,771). The twin's gain is
positive in every 20-minute chunk and comes mostly from the seconds around jumps.

hmm_deep617_r3 (2026-11-08) = hmm_deep_r3 with the hmm_617 constants (QTY 150, EDGE 0.2, EDGE_IN 0.4, K_FLOW 1.0,
V_K_S 0.8, H_N_V0 3.2, H_PQV 0.00003) and KF_LEVELS / KF_LEVELS_V 3. round3_public 620,473, 0 penalty (halves 309.4k /
310.2k; hmm_617 304.1k / 314.4k, so tied on public). Kept over hmm_617 because the deep-order readings handle big
jumps: +150k on r3_eval_twin in the untuned version. Settings near this point all score 605-620k on round3_public.

hmm_deep_nm4_r3 (2026-11-08) = hmm_deep617_r3 plus EDGE_IN 0.3, SIG_Q 0.6 (from hmm_617_fit) and H_NMAX 4. H_NMAX caps
how far one new order can push the regime HMM toward volatile, so scattered far placements no longer flip the strategy
out of quiet market making (idea from r3_cand_nm6). round3_public by H_NMAX on this base: 20 628,346, 6 650,525,
5 654,893, 4 664,205. 0 breach penalty.

hmm_deep_clear_r3 (2026-11-08) = hmm_deep_nm4_r3 plus the volatile-branch changes of hmm_vol_clear: no far exit order
on the reducing side (it used up position room, so flips were late), take at the best opposite price only (V_SWEEP
False), volatile order size V_QTY 200. round3_public 693,522, 0 penalty (H_NMAX 5: 682,933, 6: 679,350).

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


class HmmImmXStrategy(BaseStrategy):
    QTY = 150  
    MAX_POS = 100
    MIN_NEW = 10
    KEEP_LEVELS = 1
    # quiet regime (round 1 market maker settings)
    EDGE = 0.2  
    EDGE_IN = 0.3  
    EDGE_TAKE = 0.25
    K_IMB = 0.12
    K_INV = 0.0015
    K_S = 0.85
    K_FLOW = 1.0  
    Q_SWEEP = False      # quiet: when crossing, bid up to fair - EDGE_TAKE instead of only the best ask
    # volatile regime
    V_EDGE_TAKE = 0.5
    V_SWEEP = False      # True: sweep up to fair - V_EDGE_TAKE; False: take at the best price only
    V_QTY = 200          # order size in the volatile regime
    V_K_S = 0.7  
    V_K_FLOW = 1.2
    V_K_INV = 0.0
    P_ON = 0.2  # p_vol above this: volatile policy
    # HMM
    H_N_Q0 = 1.0         # start mean of n (half ticks beyond the first) in the quiet state
    H_N_V0 = 3.5  # and in the volatile state
    H_PQV = 0.00003  # per new order: P(quiet -> volatile)
    H_PVQ = 0.0003       # P(volatile -> quiet)
    H_LR = 0.0003        # soft-EM weight of each order in the state means
    H_NMAX = 5  
    H_IOC = False         # also feed the gaps between market IOCs to the HMM (exponential, rate learned per state)
    H_IOC_RQ0 = 8.0      # start IOC rate (per s) in the quiet state
    H_IOC_RV0 = 13.0     # and in the volatile state
    H_IOC_LR = 0.003
    # Kalman filter for S
    SIG_Q = 0.7  # ticks / sqrt(s) in the quiet state
    SIG_V = 11.0          # in the volatile state
    KF_LEVELS = 2  # quiet: use new orders up to this many ticks behind their best price
    KF_LEVELS_V = 3  # volatile: same
    DEEP_R = 2.5  # >0: use orders deeper than KF_LEVELS(_V) with noise r * DEEP_R * (extra depth)
    DEEP_MAX = 1000
    JUMP_TH = 6.0        # >0: innovation (ticks) that counts as a jump vote
    JUMP_N = 3           # consecutive same-direction votes that trigger
    JUMP_P = 25.0        # variance added to P on a trigger
    LEV_POS = 100        # >100: jump-mode leverage cap
    LEV_MUJ = 0.5
    LEV_EDGE = 2.0
    LEV_US = 200_000
    LEV_SLIP = 3
    ASYM = 1.0
    V_EDGE_SLACK = 0.0
    EDGE_TAKE_OPEN = 0.0
    AGE_SKEW = 0.0
    AGE_T0_US = 3_000_000
    AGE_T1_US = 7_000_000
    V_PASS_EDGE = 0.0
    FLOW_CAP = 1.0
    H_CAT = False
    H_W = 0.05
    PJ_SPLIT = False
    IMM_PJ_Q = 0.000033
    IMM_PJ_V = 0.00036
    EXC = False
    EXC_K = 3.5
    EXC_TAU_S = 3.4
    H_IOC_W = 0.5
    R_SAME = False
    J_PROPER = False
    J_CDEEP = 0.7
    MKT_BEST = False
    NMAX_TAIL = False
    SUPPORT = False
    IOC_K = 0.0
    IOC_B = 0.49
    IOC_BOUND = 0.0
    IMM = True
    IMM_WP = 0.0         # weight (tempering exponent) of the price-innovation likelihood in the mode update
    IMM_WP_J = 1.0       # weight of the price evidence for jump vs no jump (the storm filter is the no-jump reference)
    IMM_PJ = 0.0002      # per order: P(Q or V -> J)
    IMM_PJ_OUT = 0.5     # per order: P(J -> V) (J -> J otherwise); jumps come with storms
    IMM_JVAR = 25.0      # one-step S variance added in the jump mode
    IMM_KEEP_JUMP = False  # also run the tuned jump rule (JUMP_TH / JUMP_N / JUMP_P) on the IMM estimate
    IMM_RMAX_SD = 0.0    # >0: innovations beyond this many sd count as this many (robust likelihood)
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
        self.jv = 0
        self.lev_since = 0
        self.lev_id = None
        self.tk_t = 0
        self.tk_s = 0
        self.fq = [.488, .039, .344, .011, .077, .003, .011, .001, .007, .002, .004, .001, .013]
        self.fv = [.246, .070, .165, .086, .117, .050, .075, .022, .053, .014, .034, .009, .058]
        self.exc = 0.0
        self.exc_t = 0
        self.jflag = False
        self.mu = [1.0, 0.0, 0.0]          # IMM mode probabilities: Q, V, J
        self.mm = [0.0, 0.0, 0.0]          # per-mode S mean
        self.pp = [1.0, 1.0, 1.0]          # per-mode S variance
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
            self.mm = [mid, mid, mid]
        self.mbb, self.mba = bb, ba
        if bids is not self.last_bids:
            self.last_bids = bids
            self._scan_new(bids, -1, now)
        if asks is not self.last_asks:
            self.last_asks = asks
            self._scan_new(asks, 1, now)

        # flow
        self.flow *= math.exp((self.flow_t - now) / self.FLOW_TAU_US)
        self.flow_t = now
        ioc = {} if (self.IOC_K or self.IOC_BOUND) else None
        for tr in trades:
            if tr.taker_owner == 'market':
                self.flow += tr.quantity if tr.book_side == 'ask' else -tr.quantity
                if ioc is not None:
                    r_ = ioc.get(tr.taker_order_id)
                    if r_ is None:
                        ioc[tr.taker_order_id] = [1 if tr.book_side == 'ask' else -1, tr.quantity, tr.price]
                    else:
                        r_[1] += tr.quantity
                        r_[2] = tr.price
                if self.H_IOC and tr.taker_order_id != self.last_taker:
                    self.last_taker = tr.taker_order_id
                    self._observe_ioc(now)
        if ioc and self.IMM:
            mm = self.mm
            for sg, q, plast in ioc.values():
                if self.IOC_K:
                    dlt = sg * self.IOC_K * q ** self.IOC_B
                    mm[0] += dlt
                    mm[1] += dlt
                    mm[2] += dlt
                if self.IOC_BOUND:
                    lo = plast - sg * self.IOC_BOUND
                    for k in range(3):
                        if sg > 0 and mm[k] < lo:
                            mm[k] = lo
                        elif sg < 0 and mm[k] > lo:
                            mm[k] = lo
            mu = self.mu
            self.s = mu[0] * mm[0] + mu[1] * mm[1] + mu[2] * mm[2]
        half = (bq + aq) / 2
        self.depth = half if self.depth == 0 else self.depth + self.DEPTH_EMA * (half - self.depth)
        x = self.flow / self.depth
        if self.FLOW_CHECK and now >= self.fc_next:
            self._flow_check(now, x, mid)
        fx = self.flow_scale * x
        imb = (bq - aq) / (bq + aq)

        if self.pv < self.P_ON:
            fair = mid + self.K_S * (self.s - mid) + self.K_IMB * imb + self.K_FLOW * fx - self.K_INV * self.inv
            inv = self.inv
            if self.tk_s and self.tk_s * inv <= 0:
                self.tk_s = 0
            fq = fair
            if self.AGE_SKEW and self.tk_s:
                w = (now - self.tk_t - self.AGE_T0_US) / self.AGE_T1_US
                if w > 0:
                    fq = fair - self.tk_s * self.AGE_SKEW * (1.0 if w > 1.0 else w)
            qb = qa = 0
            take_b = fair - ba >= self.EDGE_TAKE
            take_a = bb - fair >= self.EDGE_TAKE
            if take_b and self.EDGE_TAKE_OPEN and fair - ba < self.EDGE_TAKE_OPEN:
                qb = -inv if inv < 0 else 0
                if qb < self.MIN_NEW:
                    take_b = False
            if take_a and self.EDGE_TAKE_OPEN and bb - fair < self.EDGE_TAKE_OPEN:
                qa = inv if inv > 0 else 0
                if qa < self.MIN_NEW:
                    take_a = False
            if take_b:
                pb = math.floor(fair - self.EDGE_TAKE) if self.Q_SWEEP else ba
                if self.tk_s <= 0:
                    self.tk_t, self.tk_s = now, 1
            elif ba - bb >= 2 and fq - (bb + 1) >= self.EDGE_IN:
                pb = bb + 1
            else:
                pb = bb
                while fq - pb < self.EDGE:
                    pb -= 1
            if take_a:
                pa = math.ceil(fair + self.EDGE_TAKE) if self.Q_SWEEP else bb
                if self.tk_s >= 0:
                    self.tk_t, self.tk_s = now, -1
            elif ba - bb >= 2 and (ba - 1) - fq >= self.EDGE_IN:
                pa = ba - 1
            else:
                pa = ba
                while pa - fq < self.EDGE:
                    pa += 1
            self._side(1, pb, qb)
            self._side(-1, pa, qa)
            return

        fair = mid + self.V_K_S * (self.s - mid) + self.V_K_FLOW * fx - self.V_K_INV * self.inv
        lev = self.LEV_POS > self.MAX_POS and self.mu[2] >= self.LEV_MUJ
        kp = 0 if self.V_PASS_EDGE else -1
        if fair - ba >= self.V_EDGE_TAKE:
            big = lev and fair - ba >= self.LEV_EDGE
            pb = math.floor(fair - self.V_EDGE_TAKE) if self.V_SWEEP else ba
            if self.V_EDGE_SLACK and fair - (ba + 1) >= self.V_EDGE_SLACK:
                pb = ba + 1
            self._side(1, pb, self.V_QTY + (self.LEV_POS - self.MAX_POS if big else 0), self.LEV_POS if big else 0, kp)
        elif self.V_PASS_EDGE and self.inv < 0 and fair - bb >= self.V_PASS_EDGE:
            self._side(1, bb, min(-self.inv, self.V_QTY))
        else:
            self._clear(1)
        if bb - fair >= self.V_EDGE_TAKE:
            big = lev and bb - fair >= self.LEV_EDGE
            pa = math.ceil(fair + self.V_EDGE_TAKE) if self.V_SWEEP else bb
            if self.V_EDGE_SLACK and (bb - 1) - fair >= self.V_EDGE_SLACK:
                pa = bb - 1
            self._side(-1, pa, self.V_QTY + (self.LEV_POS - self.MAX_POS if big else 0), self.LEV_POS if big else 0, kp)
        elif self.V_PASS_EDGE and self.inv > 0 and ba - fair >= self.V_PASS_EDGE:
            self._side(-1, ba, min(self.inv, self.V_QTY))
        else:
            self._clear(-1)
        if self.LEV_POS > self.MAX_POS:
            self._unlever(now, bb, ba, lev and (fair - ba >= self.LEV_EDGE or bb - fair >= self.LEV_EDGE))

    def _observe(self, sign: int, price: int, depth: int, now: int) -> None:
        """One new market limit order: HMM forward step, then (if near the touch) a Kalman update."""
        if self.IMM:
            self._imm(sign, price, depth, now)
            return
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
        lim = self.KF_LEVELS_V if pv >= self.P_ON else self.KF_LEVELS
        extra = depth - lim
        if extra > 0 and (not self.DEEP_R or depth > self.DEEP_MAX):
            return
        # Kalman: predict
        dt = (now - self.kf_t) / 1e6
        self.kf_t = now
        sig2 = (1.0 - pv) * self.SIG_Q * self.SIG_Q + pv * self.SIG_V * self.SIG_V
        self.P += sig2 * max(dt, 0.0)
        # learned gap between new orders and S, per state
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

    def _imm(self, sign: int, price: int, depth: int, now: int) -> None:
        """IMM step for one new market limit order: mix, predict, update the three mode filters, reweigh the modes."""
        mu = self.mu
        pq, pv_, pj = mu
        a, b, pjo, pjin = self.H_PQV, self.H_PVQ, self.IMM_PJ_OUT, self.IMM_PJ
        pjq = self.IMM_PJ_Q if self.PJ_SPLIT else pjin
        pjv = self.IMM_PJ_V if self.PJ_SPLIT else pjin
        # Markov switch per order: w_k = (Pi_Qk mu_Q, Pi_Vk mu_V, Pi_Jk mu_J), c_k = sum
        wq = (pq * (1.0 - a - pjq), pv_ * b, 0.0)
        wv = (pq * a, pv_ * (1.0 - b - pjv), pj * pjo)
        wj = (pq * pjq, pv_ * pjv, pj * (1.0 - pjo))
        mm, pp = self.mm, self.pp
        m0, p0, cc = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        for k, w in enumerate((wq, wv, wj)):
            c = w[0] + w[1] + w[2]
            cc[k] = c
            if c <= 1e-300:
                m0[k], p0[k] = mm[k], pp[k]
                continue
            m = (w[0] * mm[0] + w[1] * mm[1] + w[2] * mm[2]) / c
            p = (w[0] * (pp[0] + (mm[0] - m) ** 2) + w[1] * (pp[1] + (mm[1] - m) ** 2)
                 + w[2] * (pp[2] + (mm[2] - m) ** 2)) / c
            m0[k], p0[k] = m, p
        # placement-distance likelihood (the tuned HMM's)
        n = min(self.H_NMAX, max(0, round(2 * (sign * (price - self.mid) - 0.5))))
        rq = self.nq / (1.0 + self.nq)
        rv = self.nv / (1.0 + self.nv)
        if self.NMAX_TAIL and n >= self.H_NMAX:
            lq = rq ** n
            lv = rv ** n
        else:
            lq = (1.0 - rq) * rq ** n
            lv = (1.0 - rv) * rv ** n
        pvol = cc[1] + cc[2]
        if self.H_CAT:
            n2 = min(12, max(0, round(2 * (sign * (price - self.mid) - 0.5))))
            fq, fv = self.fq, self.fv
            aq_, av_ = self.H_LR * (1.0 - pvol), self.H_LR * pvol
            for i in range(13):
                fq[i] *= 1.0 - aq_
                fv[i] *= 1.0 - av_
            fq[n2] += aq_
            fv[n2] += av_
            lq = fq[n2] ** self.H_W
            lv = fv[n2] ** self.H_W
        lr = self.H_LR
        self.nq += lr * (1.0 - pvol) * (n - self.nq)
        self.nv += lr * pvol * (n - self.nv)
        if self.nv < self.nq + 0.5:
            self.nv = self.nq + 0.5
        # gate on the depth behind its own best, as in the tuned Kalman
        lim = self.KF_LEVELS_V if pvol >= self.P_ON else self.KF_LEVELS
        extra = depth - lim
        use_px = not (extra > 0 and (not self.DEEP_R or depth > self.DEEP_MAX))
        like = [lq * cc[0], lv * cc[1], lv * cc[2]]
        if use_px:
            dt = max((now - self.kf_t) / 1e6, 0.0)
            self.kf_t = now
            sv2 = self.SIG_V * self.SIG_V * dt
            if self.EXC and self.exc:
                self.exc *= math.exp(-dt / self.EXC_TAU_S)
                sv2 += self.exc * dt
            p0[0] += self.SIG_Q * self.SIG_Q * dt
            p0[1] += sv2
            p0[2] += sv2 + self.IMM_JVAR
            d = -sign * (self.s - price)
            if extra <= 0:
                self.cq += self.C_LR * (1.0 - pvol) * (d - self.cq)
                self.cv += self.C_LR * pvol * (d - self.cv)
            r_q = 0.25 * rq / ((1.0 - rq) ** 2) + self.R_MIN
            r_v = r_q if self.R_SAME else 0.25 * rv / ((1.0 - rv) ** 2) + self.R_MIN
            r_j = r_q
            if extra > 0:
                r_q *= self.DEEP_R * extra
                r_v *= self.DEEP_R * extra
            if self.IMM_KEEP_JUMP and self.JUMP_TH:
                e0 = price - sign * self.cv - self.s
                if e0 > self.JUMP_TH:
                    self.jv = self.jv + 1 if self.jv > 0 else 1
                elif e0 < -self.JUMP_TH:
                    self.jv = self.jv - 1 if self.jv < 0 else -1
                else:
                    self.jv = 0
                if self.jv >= self.JUMP_N or -self.jv >= self.JUMP_N:
                    p0[0] += self.JUMP_P
                    p0[1] += self.JUMP_P
                    p0[2] += self.JUMP_P
                    self.jv = 0
            wp = self.IMM_WP
            wpj = self.IMM_WP_J
            cap2 = self.IMM_RMAX_SD * self.IMM_RMAX_SD
            lp = [1.0, 1.0, 1.0]
            for k in range(3):
                r = r_q if k == 0 else r_v
                y = price - sign * (self.cq if k == 0 else self.cv)
                if k == 2 and self.J_PROPER:
                    r = r_j
                    y = price - sign * (self.cq + (self.J_CDEEP if extra > 0 else 0.0))
                e = y - m0[k]
                if k == 2:
                    e_j = e
                sv = p0[k] + r
                g = p0[k] / sv
                mm[k] = m0[k] + g * e
                pp[k] = p0[k] * (1.0 - g)
                if self.SUPPORT:
                    if sign < 0 and mm[k] < price:
                        mm[k] = price
                    elif sign > 0 and mm[k] > price:
                        mm[k] = price
                z2 = e * e / sv
                if cap2 and z2 > cap2:
                    z2 = cap2
                lp[k] = math.exp(-0.5 * z2) / math.sqrt(sv) + 1e-300
            # calm vs storm: price evidence with weight wp; jump vs no jump: weight wpj against the storm filter
            ref = lp[1] ** wpj
            like[0] *= lp[0] ** wp * ref
            like[1] *= lp[1] ** wp * ref
            like[2] *= lp[2] ** wpj * lp[1] ** wp
        else:
            # no price reading: carry the mixed estimates (time is propagated at the next reading)
            for k in range(3):
                mm[k], pp[k] = m0[k], p0[k]
        tot = like[0] + like[1] + like[2]
        if not tot > 0:
            like, tot = cc, cc[0] + cc[1] + cc[2]
        mu[0], mu[1], mu[2] = like[0] / tot, like[1] / tot, like[2] / tot
        if self.EXC and use_px:
            if mu[2] >= 0.5 and not self.jflag:
                self.exc += self.EXC_K * abs(e_j)
                self.jflag = True
            elif mu[2] < 0.5:
                self.jflag = False
        self.pv = mu[1] + mu[2]
        s_hat = mu[0] * mm[0] + mu[1] * mm[1] + mu[2] * mm[2]
        self.s = s_hat
        self.P = (mu[0] * (pp[0] + (mm[0] - s_hat) ** 2) + mu[1] * (pp[1] + (mm[1] - s_hat) ** 2)
                  + mu[2] * (pp[2] + (mm[2] - s_hat) ** 2))

    def _observe_ioc(self, now: int) -> None:
        """A new market IOC: the gap since the previous one is evidence for the regime."""
        if self.ioc_t:
            g = min(5.0, (now - self.ioc_t) / 1e6)
            lq = self.iq * math.exp(-self.iq * g)
            lv = self.iv * math.exp(-self.iv * g)
            pv = self.pv
            if self.IMM:
                w = self.H_IOC_W
                mu = self.mu
                mu[0] *= lq ** w
                mu[1] *= lv ** w
                mu[2] *= lv ** w
                t_ = mu[0] + mu[1] + mu[2]
                mu[0], mu[1], mu[2] = mu[0] / t_, mu[1] / t_, mu[2] / t_
                pv = mu[1] + mu[2]
            else:
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
        if self.MKT_BEST:
            first = self.mbb if sign < 0 else self.mba
        mx = self.max_id
        lm = len(mx)
        new_max = mx
        for o in views:
            if o.owner != 'market':
                continue
            oid = o.order_id
            if len(oid) > lm or (len(oid) == lm and oid > mx):
                self._observe(sign, o.price, (o.price - first) * sign, now)
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
        self.flow_scale = min(self.FLOW_CAP, max(0.0, (self.fc_sxy / self.fc_sxx) / self.FLOW_SLOPE_FULL))

    def _clear(self, sign: int) -> None:
        for oid, o in self.orders.items():
            if o[0] == sign:
                self._cancel(oid)

    def _unlever(self, now: int, bb: int, ba: int, keep: bool) -> None:
        """Close |inventory| above MAX_POS by IOC once the jump signal is gone or after LEV_US."""
        inv = self.inv
        if abs(inv) <= self.MAX_POS:
            self.lev_since = 0
            return
        if not self.lev_since:
            self.lev_since = now
        if self.lev_id is not None or (keep and now - self.lev_since < self.LEV_US):
            return
        q = abs(inv) - self.MAX_POS
        if inv > 0:
            px = max(1, bb - self.LEV_SLIP)
            self.lev_id = self.send_order(OrderRequest(side=Side.SELL, price=px, quantity=q, order_type=OrderType.IOC))
            self.orders[self.lev_id] = [-1, px, q]
        else:
            px = ba + self.LEV_SLIP
            self.lev_id = self.send_order(OrderRequest(side=Side.BUY, price=px, quantity=q, order_type=OrderType.IOC))
            self.orders[self.lev_id] = [1, px, q]

    def _side(self, sign: int, target: int, qty: int = 0, cap: int = 0, keep: int = -1) -> None:
        qty = qty or self.QTY
        kl = self.KEEP_LEVELS if keep < 0 else keep
        if self.ASYM != 1.0 and sign * self.inv < 0:
            qty = int(qty * self.ASYM)
        resting = 0
        at_target = 0
        keep = []
        for oid, o in self.orders.items():
            if o[0] != sign or oid in self.cancel_sent:
                if o[0] == sign:
                    resting += o[2]
                continue
            behind = (target - o[1]) * sign
            if behind < 0 or behind > kl:
                self._cancel(oid)
                resting += o[2]
                continue
            resting += o[2]
            if behind == 0:
                at_target += o[2]
            else:
                keep.append((behind, oid, o[2]))
        if at_target >= qty:
            return
        room = (cap or self.MAX_POS) - sign * self.inv - resting
        want = qty - at_target
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
        if oid == self.lev_id:
            self.lev_id = None
        self.orders.pop(oid, None)
        self.cancel_sent.discard(oid)
