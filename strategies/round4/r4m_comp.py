"""r4m_comp: r4m_risk_a plus quiet quote hysteresis (EDGE_NEW, from Jaro's r4_best_snipe), for component tests.

r4m_risk_a: round 4 risk candidate = r4m_alt with SNIPE_FAR 15, IDG_SNIPE 1, K_MOM / V_K_MOM 0.1 over 0.5 s.
round4_public 1,156,545 (penalty 7,963; same base without sniper/momentum 1,073,369).

r4m_alt: r4m_snipe plus switches for structural alternatives (momentum lean, quiet taker-only / reduce-only).

K_MOM / V_K_MOM: fair += K * (mid - mid MOM_US ago) in the quiet / storm branch (round 4: past 1 s return predicts
the next 1-5 s, Spearman +0.31 / +0.20). Q_PASSIVE: 0 = quote both sides in quiet (as before), 1 = only quote the
side that reduces |inventory|, 2 = no passive quotes at all in quiet (pure taker; passive at the touch earns about 0 by
30 s on round 4).

r4m_snipe: hmm_deep_clear_r3 retuned on round 4 plus a far-order jump sniper (leverage > 100).

Base: hmm_deep_clear_r3 with the round-4 sweep winners K_S 0.95, SIG_Q 0.8, V_K_S 0.9 (r4 quarters 1,073k vs 1,044k).
Sniper (ported from strategies/round3/r3_fire.py): CONFIRM_N new market limit orders on one side, SNIPE_FAR+ ticks
behind the opposite best and agreeing to CONFIRM_TOL ticks within CONFIRM_US, mean S has jumped (they show it about
6 ms after the jump, before the touch moves). One IOC sweeps the stale opposite side at least SNIPE_MARGIN through
the new S, up to SNIPE_POS lots of position, past the 100 cap (penalty 10/ms while over). The excess is closed at
S_new -+ UNWIND_GAP once the book gets there, else through the touch after MAX_OVER_US. Kill switches: too many
snipes in a window, or too much total time over 100.
Tape-only estimate for round4_public (scripts/lever_econ.py, conf 2, margin 10, hold 500 ms): about 10 trades,
+85k net. Jakub's docs/LEVERAGE.md: reactive chasing after the mid moves loses; this trigger fires before.

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

Iteration (2026-11-08), round3_public:
- Rejected: quiet-side take priority (a take cancels same-side passive orders at once and ignores up to 100 lots
  whose cancel is in flight, with an over-limit unwind): 682,664 incl. 6,047 breach penalty. Not kept in the code.
- IDG_P (off): the id-gap jump detector (tools/id_gap_jump.py) adds IDG_P to the filter variance on a detection;
  +4.3k on hmm_617_fit; here IDG_P 25: 694,244 (+722, +0.1%, 0 penalty), too small to matter, left off.
- Fill audit of the 693,522 run (full logs, markout against the 1 s mid): takes 479k lots at +1.18 ticks/lot by
  5 s (+567k), passive 697k lots at +0.22 (+151k). |inventory| >= 95 for 46% of the time; while there, the mid moves
  +1.04 ticks/s in the position's direction (+287k of carry), against ~0 when not at the cap. The leak: 2,428 full
  flips (+100 -> -100 in one take), 19% reversed within 0.5 s, those cost -281k by 5 s; flips overall +445k.
- Rejected, staged flips (a normal-strength take against the position only closes it; the other side waits for a
  confirmation window unless the edge is strong): 50 ms / strong at +0.5: 674,167; 100 ms / +1: 657,309;
  300 ms / +1: 642,618; 100 ms always staged: 604,574. Median |inventory| falls from 85 to 60-72: reaching the cap
  fast is worth more than the false flips cost. Not kept in the code.
- Take decomposition (scratch recorder logging the fair-price parts and Kalman state at each of 5,135 filled takes):
  reversals within 0.5 s are a volatile-regime effect (28% of volatile takes vs 4% quiet; -299k of -308k). Bad takes
  follow an S_hat jump (1.4 vs 0.6 ticks in 50 ms) at higher filter variance (P 0.90 vs 0.42). But every volatile
  edge bucket still earns at 5 s (edge within 0.1 of V_EDGE_TAKE: +0.69 ticks/lot, +77k), so a higher or
  P-scaled take threshold would cut net profit. The whipsaw is the cost of a profitable fast signal, not a filterable
  leak.
- Rejected, quiet takes sized 200 (one order for a full flip; QTY 150 otherwise): 692,949; with EDGE_TAKE 0.2:
  682,066. Not kept in the code.

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


class R4mCompStrategy(BaseStrategy):
    EDGE_NEW = 0.0       # >0: a new quiet quote needs this edge, resting ones stay until below EDGE (hysteresis, Jaro)
    K_MOM = 0.1
    V_K_MOM = 0.1
    MOM_US = 500_000
    Q_PASSIVE = 0
    QTY = 150  
    MAX_POS = 100
    MIN_NEW = 10
    KEEP_LEVELS = 1
    # quiet regime (round 1 market maker settings)
    EDGE = 0.2  
    EDGE_IN = 0.3  
    EDGE_TAKE = 0.25
    K_IMB = 0.15
    K_INV = 0.002
    K_S = 0.95
    K_FLOW = 1.0  
    Q_SWEEP = False      # quiet: when crossing, bid up to fair - EDGE_TAKE instead of only the best ask
    # volatile regime
    V_EDGE_TAKE = 0.55
    V_SWEEP = False      # True: sweep up to fair - V_EDGE_TAKE; False: take at the best price only
    V_QTY = 200          # order size in the volatile regime
    V_K_S = 0.9  
    V_K_FLOW = 1.2
    V_K_INV = 0.0
    P_ON = 0.3  # p_vol above this: volatile policy
    # HMM
    H_N_Q0 = 1.0         # start mean of n (half ticks beyond the first) in the quiet state
    H_N_V0 = 3.2  # and in the volatile state
    H_PQV = 0.00003  # per new order: P(quiet -> volatile)
    H_PVQ = 0.0003       # P(volatile -> quiet)
    H_LR = 0.0003        # soft-EM weight of each order in the state means
    H_NMAX = 4  
    H_IOC = False         # also feed the gaps between market IOCs to the HMM (exponential, rate learned per state)
    H_IOC_RQ0 = 8.0      # start IOC rate (per s) in the quiet state
    H_IOC_RV0 = 13.0     # and in the volatile state
    H_IOC_LR = 0.003
    # Kalman filter for S
    SIG_Q = 0.8  # ticks / sqrt(s) in the quiet state
    SIG_V = 10.0          # in the volatile state
    KF_LEVELS = 3  # quiet: use new orders up to this many ticks behind their best price
    KF_LEVELS_V = 3  # volatile: same
    DEEP_R = 2.0  # >0: use orders deeper than KF_LEVELS(_V) with noise r * DEEP_R * (extra depth)
    DEEP_MAX = 1000
    JUMP_TH = 6.0        # >0: innovation (ticks) that counts as a jump vote
    JUMP_N = 3           # consecutive same-direction votes that trigger
    JUMP_P = 25.0        # variance added to P on a trigger
    C_LR = 0.0005        # weight of each order in the learned price-to-S gap c
    R_MIN = 0.1
    # jump sniper (leverage past MAX_POS)
    SNIPE_FAR = 15       # a new order this many ticks behind the opposite best counts as far (0: sniper off)
    CONFIRM_N = 2        # far orders on the same side needed
    CONFIRM_US = 50_000  # ... within this long
    CONFIRM_TOL = 3      # ... agreeing on the new level to this many ticks
    X_MEAN = 0.8         # a new order sits about this far outside S
    SNIPE_POS = 1000     # position a snipe may reach
    SNIPE_MARGIN = 10.0  # sweep stale levels at least this far through the new S
    UNWIND_GAP = 2.0     # close the excess at S_new -+ UNWIND_GAP once the book reaches it
    MAX_OVER_US = 500_000  # ... or through the touch after this long over MAX_POS
    UNWIND_SLIP = 2
    SNIPE_COOLDOWN_US = 1_000_000
    KILL_RATE_N = 4              # more than this many snipes within KILL_RATE_US stops sniping for good
    KILL_RATE_US = 600_000_000
    KILL_OVER_US = 10_000_000    # total time over MAX_POS (10 s = 100k of penalty) stops sniping for good
    # id-gap jump detector (tools/id_gap_jump.py): on a detection add IDG_P to the filter variance; 0: off
    IDG_P = 0.0
    IDG_SNIPE = 1        # 1: an id-gap detection (a jump beyond the visible 20 ticks) also fires the sniper
    IDG_DEPTH = 10       # ... sweeping the stale side up to this many ticks through the old touch
    IDG_S_MIN = 20.0     # ... and assuming S moved at least this far (for the unwind target)
    IDG_WINDOW_US = 25_000
    IDG_MISSING = 3
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
        self.mom_q: deque = deque()
        self.snipe_s = 0.0
        self.snipe_t = -10**12
        self.snipe_id = None
        self.unwind_id = None
        self.over_since = 0
        self.far_seen: list = []
        self.killed = False
        self.snipe_times: deque = deque()
        self.over_total = 0
        self.over_last = 0
        # id-gap jump detector
        self.idg = IdGapJump(self.IDG_WINDOW_US, self.IDG_MISSING) if (self.IDG_P or self.IDG_SNIPE) else None
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
        self.bb, self.ba = bb, ba
        mq = self.mom_q
        mq.append((snapshot.timestamp_us, mid))
        while snapshot.timestamp_us - mq[0][0] > self.MOM_US:
            mq.popleft()
        mom = mid - mq[0][1]
        self._unwind(snapshot.timestamp_us, bb, ba)
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

        # flow
        self.flow *= math.exp((self.flow_t - now) / self.FLOW_TAU_US)
        self.flow_t = now
        idg = self.idg
        for tr in trades:
            if tr.taker_owner == 'market':
                self.flow += tr.quantity if tr.book_side == 'ask' else -tr.quantity
                if idg is not None:
                    idg.seen(tr.taker_order_id, now)
                if self.H_IOC and tr.taker_order_id != self.last_taker:
                    self.last_taker = tr.taker_order_id
                    self._observe_ioc(now)
        if idg is not None and now - self.idg_t > self.IDG_WINDOW_US:
            dj = idg.check(now)
            if dj:
                self.P += self.IDG_P
                self.idg_t = now
                if self.IDG_SNIPE:
                    self._snipe_idg(dj, bb, ba, now)
        half = (bq + aq) / 2
        self.depth = half if self.depth == 0 else self.depth + self.DEPTH_EMA * (half - self.depth)
        x = self.flow / self.depth
        if self.FLOW_CHECK and now >= self.fc_next:
            self._flow_check(now, x, mid)
        fx = self.flow_scale * x
        imb = (bq - aq) / (bq + aq)

        if self.pv < self.P_ON:
            fair = mid + self.K_S * (self.s - mid) + self.K_IMB * imb + self.K_FLOW * fx - self.K_INV * self.inv + self.K_MOM * mom
            if fair - ba >= self.EDGE_TAKE:
                pb = nb = math.floor(fair - self.EDGE_TAKE) if self.Q_SWEEP else ba
            elif ba - bb >= 2 and fair - (bb + 1) >= self.EDGE_IN:
                pb = nb = bb + 1
            else:
                pb = bb
                while fair - pb < self.EDGE:
                    pb -= 1
                nb = pb
                if self.EDGE_NEW:
                    while fair - nb < self.EDGE_NEW:
                        nb -= 1
            if bb - fair >= self.EDGE_TAKE:
                pa = na = math.ceil(fair + self.EDGE_TAKE) if self.Q_SWEEP else bb
            elif ba - bb >= 2 and (ba - 1) - fair >= self.EDGE_IN:
                pa = na = ba - 1
            else:
                pa = ba
                while pa - fair < self.EDGE:
                    pa += 1
                na = pa
                if self.EDGE_NEW:
                    while na - fair < self.EDGE_NEW:
                        na += 1
            qp = self.Q_PASSIVE
            if qp == 0 or pb >= ba or (qp == 1 and self.inv < 0):
                self._side(1, pb, 0, nb)
            else:
                self._clear(1)
            if qp == 0 or pa <= bb or (qp == 1 and self.inv > 0):
                self._side(-1, pa, 0, na)
            else:
                self._clear(-1)
            return

        fair = mid + self.V_K_S * (self.s - mid) + self.V_K_FLOW * fx - self.V_K_INV * self.inv + self.V_K_MOM * mom
        if fair - ba >= self.V_EDGE_TAKE:
            self._side(1, math.floor(fair - self.V_EDGE_TAKE) if self.V_SWEEP else ba, self.V_QTY)
        else:
            self._clear(1)
        if bb - fair >= self.V_EDGE_TAKE:
            self._side(-1, math.ceil(fair + self.V_EDGE_TAKE) if self.V_SWEEP else bb, self.V_QTY)
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
                if self.idg is not None:
                    self.idg.seen(oid, now)
                    if self.pbb:
                        self.idg.new_order(sign, o.price, self.pbb, self.pba, now)
                if len(oid) > len(new_max) or (len(oid) == len(new_max) and oid > new_max):
                    new_max = oid
        self.max_id = new_max

    def _far(self, sign: int, price: int, now: int) -> None:
        """Snipe once CONFIRM_N far orders on this side agree on the new level within CONFIRM_US."""
        seen = [f for f in self.far_seen if now - f[0] <= self.CONFIRM_US]
        agree = [f for f in seen if f[1] == sign and abs(f[2] - price) <= self.CONFIRM_TOL]
        seen.append((now, sign, price))
        self.far_seen = seen
        if len(agree) + 1 >= self.CONFIRM_N:
            p = max([f[2] for f in agree] + [price]) if sign < 0 else min([f[2] for f in agree] + [price])
            self._snipe(sign, p, now)

    def _snipe(self, sign: int, price: int, now: int) -> None:
        """S jumped: sweep the stale opposite side with one IOC, past the 100 cap."""
        if self.snipe_id is not None or self.unwind_id is not None or now - self.snipe_t < self.SNIPE_COOLDOWN_US:
            return
        if self.killed or self.over_total > self.KILL_OVER_US:
            self.killed = True
            return
        while self.snipe_times and now - self.snipe_times[0] > self.KILL_RATE_US:
            self.snipe_times.popleft()
        if len(self.snipe_times) >= self.KILL_RATE_N:
            self.killed = True
            return
        if sign < 0:   # a buy far below: S fell, the bids above it are stale
            s_new = price + self.X_MEAN
            limit_px = math.ceil(s_new + self.SNIPE_MARGIN)
            if self.bb < limit_px:
                return
            resting = sum(o[2] for o in self.orders.values() if o[0] < 0)
            q = min(self.SNIPE_POS + self.inv, 1000 + self.inv - resting)
            side, sg = Side.SELL, -1
        else:          # a sell far above: S rose, the asks below it are stale
            s_new = price - self.X_MEAN
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

    def _snipe_idg(self, d: int, bb: int, ba: int, now: int) -> None:
        """Id-gap detection: S jumped more than the visible depth in direction d. Sweep the stale side past the cap."""
        if self.snipe_id is not None or self.unwind_id is not None or now - self.snipe_t < self.SNIPE_COOLDOWN_US:
            return
        if self.killed or self.over_total > self.KILL_OVER_US:
            self.killed = True
            return
        while self.snipe_times and now - self.snipe_times[0] > self.KILL_RATE_US:
            self.snipe_times.popleft()
        if len(self.snipe_times) >= self.KILL_RATE_N:
            self.killed = True
            return
        if d > 0:
            limit_px = ba + self.IDG_DEPTH
            resting = sum(o[2] for o in self.orders.values() if o[0] > 0)
            q = min(self.SNIPE_POS - self.inv, 1000 - self.inv - resting)
            side, sg, s_new = Side.BUY, 1, ba + self.IDG_S_MIN
        else:
            limit_px = max(1, bb - self.IDG_DEPTH)
            resting = sum(o[2] for o in self.orders.values() if o[0] < 0)
            q = min(self.SNIPE_POS + self.inv, 1000 + self.inv - resting)
            side, sg, s_new = Side.SELL, -1, bb - self.IDG_S_MIN
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
            if o[0] == sign:
                self._cancel(oid)

    def _side(self, sign: int, target: int, qty: int = 0, new_target: int = 0) -> None:
        qty = qty or self.QTY
        if not new_target:
            new_target = target
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
            if behind == 0 or o[1] == new_target:
                at_target += o[2]
            else:
                keep.append((behind, oid, o[2]))
        if at_target >= qty:
            return
        room = self.MAX_POS - sign * self.inv - resting
        want = qty - at_target
        if room < min(want, self.MIN_NEW) and keep:
            keep.sort(reverse=True)
            self._cancel(keep[0][1])
            return
        q = min(want, room)
        if q < self.MIN_NEW:
            return
        side = Side.BUY if sign > 0 else Side.SELL
        oid = self.send_order(OrderRequest(side=side, price=new_target, quantity=q, order_type=OrderType.LIMIT))
        self.orders[oid] = [sign, new_target, q]

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
        self.orders.pop(oid, None)
        self.cancel_sent.discard(oid)
