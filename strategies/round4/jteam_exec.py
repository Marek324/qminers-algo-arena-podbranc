"""Two-regime market maker: an online HMM picks the regime, a Kalman filter tracks the hidden price S.

1. Regime: a 2-state HMM with one observation per new market limit order, n = half-ticks between the order and the
   mid beyond the first half tick, capped at H_NMAX. n is geometric in each state with means learned online (soft
   EM, H_LR); the forward filter with switch probabilities H_PQV / H_PVQ gives p_vol.
2. Price: a 1-D Kalman filter whose variance grows by sigma^2 * dt, sigma mixed between SIG_Q and SIG_V by p_vol.
   Each new order within KF_LEVELS (volatile: KF_LEVELS_V) ticks of its best price is a reading price + c (bid) or
   price - c (ask), with the gap c learned online. Deeper orders are readings too, with noise r * DEEP_R * (extra
   depth), so a run of far orders after a jump moves S_hat within a few orders. JUMP_N consecutive readings JUMP_TH+
   ticks off on one side add JUMP_P to the variance, so S_hat snaps to the new level.
3. Quiet (p_vol < P_ON): fair = mid + K_S * (S_hat - mid) + K_IMB * imbalance + K_FLOW * flow / depth - K_INV *
   inventory + K_MOM * (mid - mid MOM_US ago). Quote at the best price with EDGE_NEW (a resting quote stays while it
   keeps EDGE) to fair; cross at the opposite best when fair is EDGE_TAKE through it. Sticky orders, size capped by
   MAX_POS.
4. Volatile: fair from the V_ settings plus V_K_MOM * (mid - mid MOM_US ago); take the stale book when fair is
   V_EDGE_TAKE through it (at the best price only, V_QTY per order). No resting quotes.

K_FLOW is scaled by a live estimate of the flow -> mid slope (FLOW_CHECK). Inventory is counted from fills only.

Jump sniper (goes past the 100 cap; the penalty is a flat 10 per ms at any size): CONFIRM_N new orders SNIPE_FAR+
ticks behind the opposite best that agree within CONFIRM_US, or an id-gap detection (IdGapJump: missing market order
ids plus clamped orders inside the stale touch, IDG_SNIPE), mean S has jumped. One IOC sweeps the stale side up to
SNIPE_POS lots of position. The excess is closed at S_new -+ UNWIND_GAP once the book gets there, else through the
touch after MAX_OVER_US. Kill switches: too many snipes per KILL_RATE_US, or KILL_OVER_US total time over the cap.
Q_PASSIVE: 0 quotes both sides in quiet, 1 only the reducing side, 2 none.

Weak jump tier: an id gap of WEAK_MISSING ids takes toward +-WEAK_POS inside the cap at once, sweeping up to
WEAK_DEPTH ticks through the touch (WEAK_COOLDOWN_US between fires).
Storm depth imbalance: volatile fair += V_K_MLDI * imbalance of book depth over ML_DEPTH ticks, weighted
exp(-distance / ML_LAM). Every other signal switch in the class is off by default.

SHED (off): in storms, once a capped position has earned its edge (mid moved SHED_TICKS our way, or SHED_AGE_US old)
and no take adds to it, reduce it to SHED_TO, passively at the touch or as a take.

ADJ_SNIPE (off): a new order clamped one tick inside the stale touch whose id directly follows a missing id, with a
previous spread of ADJ_SP+ and no visible new order on the jump side within ADJ_QUIET_US, fires the id-gap sniper
1-5 ms into a big jump.
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
    """Jump detector from missing market order ids and clamped new orders."""

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


class R4mMldiStrategy(BaseStrategy):
    # weak jump tier: move to +-WEAK_POS inside the cap (a false fire costs only the spread, no penalty)
    WEAK_MISSING = 2     # id-gap detector with this many missing ids (0: off)
    WEAK_FAR = 0         # a single new order this far behind the opposite best (0: off)
    WEAK_POS = 100
    WEAK_DEPTH = 5       # sweep up to this many ticks through the touch
    WEAK_COOLDOWN_US = 300_000
    WEAK_STORM = 0       # 1: weak tier only when p_vol >= P_ON
    WEAK_FREE = 0        # 1: weak tier sizes to WEAK_POS ignoring our resting quotes on its side, and cancels them
    # order-book layer: order-flow imbalance at the touch (Cont et al., includes cancels) and depth imbalance over levels
    K_OFI = 0.0          # ticks per (decayed OFI / mean touch depth), quiet
    V_K_OFI = 0.0        # same, storm
    OFI_TAU_US = 500_000
    K_DIMB = 0.0         # ticks per unit of (bid depth - ask depth) / total over DIMB_LEVELS ticks, quiet
    V_K_DIMB = 0.0       # same, storm
    DIMB_LEVELS = 5
    # multi-level book features (market orders only, levels by tick distance from each side's best, up to ML_DEPTH)
    ML_DEPTH = 20
    K_MLDI = 0.0         # distance-weighted depth imbalance, weights exp(-d / ML_LAM), quiet
    V_K_MLDI = 0.6  # same, storm
    ML_LAM = 2.0
    ML_SPREAD = 0        # 1: the storm MLDI term is scaled by the half spread (microprice form: mid + k * imb * spread / 2)
    K_WMP = 0.0          # multi-level microprice minus mid (ticks per tick), quiet
    V_K_WMP = 0.0        # same, storm
    WMP_LEVELS = 5
    K_MLOFI = 0.0        # decayed change of (bid - ask) depth within MLOFI_LEVELS ticks, per mean touch depth, quiet
    V_K_MLOFI = 0.0      # same, storm
    MLOFI_LEVELS = 5
    MLOFI_TAU_US = 500_000
    K_FAR = 0.0          # depth imbalance of the band FAR_LO..ML_DEPTH ticks behind the best, quiet
    V_K_FAR = 0.0        # same, storm
    FAR_LO = 5
    # online volatility (EWMA of squared 1 s mid moves, VOL_TAU_S; volatility clusters for about 100 s)
    VOL_TAU_S = 60.0
    SIGV_K = 0.0         # >0: storm process noise SIG_V = SIGV_K * realized vol, clamped to [SIGV_MIN, SIGV_MAX]
    SIGV_MIN = 4.0
    SIGV_MAX = 25.0
    MLDI_VMIN = 0.0      # >0: storm depth-imbalance term only while realized vol >= this (ticks / sqrt s)
    VET_REF = 0.0        # >0: storm take edge scaled by sqrt(realized vol / VET_REF)
    # storm-onset defence (quiet regime): while P(storm) >= PW_LO or realized vol >= VW_VMIN, widen quiet quotes by
    # PW_EDGE and/or quote only the inventory-reducing side (PW_QP 1)
    PW_LO = 0.0
    VW_VMIN = 0.0
    PW_EDGE = 0.0
    PW_QP = 0
    # storm-end hazard: storms are regular (mean 84 s, CV 0.44), so lean against the position as the storm ages
    V_HZ = 0.0           # ticks per lot at full age
    HZ_US = 90_000_000   # age (since p_vol crossed P_ON) at which the lean is full
    # storm take size by edge: below ESZ_FULL ticks of edge take only V_QTY * ESZ_FRAC (0: off)
    ESZ_FULL = 0.0
    ESZ_FRAC = 0.5
    # jump-direction prior: the next jump repeats the last one's sign 61% of the time
    V_K_PJ = 0.0         # storm fair lean (ticks) towards the last jump's direction
    PJ_US = 120_000_000
    # pull our quiet quotes on the side a jump runs over, and keep it empty for JC_HOLD_US
    JUMP_CANCEL = 0      # 1: on id-gap fires (weak and strong)
    FAR_CANCEL = 0       # >0: also on a single new order this far behind the opposite best
    JC_HOLD_US = 50_000
    # pre-storm lean: fair += K_PRE * sign(market IOC net volume over PRE_US) while PRE_LO <= p_vol < P_ON
    PRE_LO = 0.1
    K_PRE = 0.0
    PRE_US = 2_000_000
    # quiet quote hysteresis: a new quote needs EDGE_NEW, resting ones stay until below EDGE (0: off)
    EDGE_NEW = 0.3
    K_MOM = 0.1
    V_K_MOM = 0.1
    MOM_US = 500_000
    Q_PASSIVE = 0
    QTY = 150  
    MAX_POS = 100
    MIN_NEW = 10
    KEEP_LEVELS = 1
    # quiet regime (round 1 market maker settings)
    EDGE = 0.15
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
    # H1 capacity recycling (storm only unless SHED_CALM)
    SHED = 0             # 1: on
    SHED_AT = 90         # |inventory| at which a position counts as capped
    SHED_TO = 0          # reduce down to this |inventory|
    SHED_TICKS = 0.0     # >0: shed once the mid moved this far our way since the cap was reached
    SHED_AGE_US = 3_000_000  # >0: ... or once the cap is this old
    SHED_MODE = 0        # 0: passive at the touch (inside the spread when it is 2+ wide); 1: take the touch if fair is within SHED_EDGE
    SHED_EDGE = 0.5      # mode 1: sell at the bid when bid >= fair - SHED_EDGE (buy at the ask likewise)
    SHED_CALM = 0        # 1: also in the quiet regime
    SHED_STORM = 1       # 0: not in the volatile regime (calm-only with SHED_CALM 1)

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
    # id-gap jump detector: on a detection add IDG_P to the filter variance; 0: off
    IDG_P = 0.0
    IDG_SNIPE = 1        # 1: an id-gap detection (a jump beyond the visible 20 ticks) also fires the sniper
    IDG_DEPTH = 10       # ... sweeping the stale side up to this many ticks through the old touch
    IDG_S_MIN = 20.0     # ... and assuming S moved at least this far (for the unwind target)
    IDG_WINDOW_US = 25_000
    IDG_MISSING = 3
    LOG_IDG = 0  # 1: print every id-gap detection and what the sniper did with it
    ADJ_SNIPE = 0        # 1: a clamped new order whose id directly follows a missing id, with a wide spread, fires the id-gap sniper
    ADJ_SP = 4           # ... previous spread (ba - bb) at least this many ticks
    ADJ_QUIET_US = 25_000  # ... and no visible new market limit order on the jump side within this long
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
        self.idg2 = IdGapJump(self.IDG_WINDOW_US, self.WEAK_MISSING) if self.WEAK_MISSING else None
        self.weak_t = -10**12
        self.adj_last_bid_t = -10**12
        self.adj_last_ask_t = -10**12
        self.adj_ids = set()
        self.adj_q = deque()
        self.adj_pend = []
        self.weak_id = None
        self.ofi = 0.0
        self.mlofi = 0.0
        self.in_storm = False
        self.vol2 = 0.0
        self.vol_t = 0
        self.vol_mid = 0.0
        self.vol = 0.0
        self.sig_v = self.SIG_V
        self.pre_q: deque = deque()
        self.pre_net = 0.0
        self.storm_t = 0
        self.pj_sign = 0
        self.pj_t = -10**12
        self.block = {1: 0, -1: 0}
        self.mlofi_t = 0
        self.mlofi_prev = None
        self.ml_on = any((self.K_MLDI, self.V_K_MLDI, self.K_WMP, self.V_K_WMP, self.K_MLOFI, self.V_K_MLOFI, self.K_FAR, self.V_K_FAR))
        self.ml_w = [math.exp(-d / self.ML_LAM) for d in range(self.ML_DEPTH)]
        self.ofi_t = 0
        self.pbq = 0
        self.paq = 0
        self.cap_sign = 0
        self.cap_t = 0
        self.cap_mid = 0.0

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
        ts_ = snapshot.timestamp_us
        if not self.vol_t:
            self.vol_t, self.vol_mid = ts_, mid
        elif ts_ - self.vol_t >= 1_000_000:
            dt_s = (ts_ - self.vol_t) / 1e6
            a = 1.0 - math.exp(-dt_s / self.VOL_TAU_S)
            dmv = mid - self.vol_mid
            self.vol2 += a * (dmv * dmv / dt_s - self.vol2)
            self.vol = math.sqrt(self.vol2)
            self.vol_t, self.vol_mid = ts_, mid
            if self.SIGV_K:
                self.sig_v = min(self.SIGV_MAX, max(self.SIGV_MIN, self.SIGV_K * self.vol))
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
        if self.K_OFI or self.V_K_OFI:
            if self.pbb:
                eb = bq if bb > self.pbb else (bq - self.pbq if bb == self.pbb else -self.pbq)
                ea = aq if ba < self.pba else (aq - self.paq if ba == self.pba else -self.paq)
                self.ofi = self.ofi * math.exp((self.ofi_t - now) / self.OFI_TAU_US) + eb - ea
                self.ofi_t = now
            self.pbq, self.paq = bq, aq
        self.pbb, self.pba = bb, ba

        # flow
        self.flow *= math.exp((self.flow_t - now) / self.FLOW_TAU_US)
        self.flow_t = now
        idg = self.idg
        for tr in trades:
            if tr.taker_owner == 'market':
                self.flow += tr.quantity if tr.book_side == 'ask' else -tr.quantity
                if self.ADJ_SNIPE:
                    self._adj_add(tr.taker_order_id)
                if idg is not None:
                    idg.seen(tr.taker_order_id, now)
                if self.idg2 is not None:
                    self.idg2.seen(tr.taker_order_id, now)
                if self.H_IOC and tr.taker_order_id != self.last_taker:
                    self.last_taker = tr.taker_order_id
                    self._observe_ioc(now)
        if self.K_PRE:
            seen_ioc = {}
            for tr in trades:
                if tr.taker_owner == 'market':
                    sg = 1 if tr.book_side == 'ask' else -1
                    seen_ioc[tr.taker_order_id] = seen_ioc.get(tr.taker_order_id, 0) + sg * tr.quantity
            for v in seen_ioc.values():
                self.pre_q.append((now, v))
                self.pre_net += v
            pq = self.pre_q
            while pq and now - pq[0][0] > self.PRE_US:
                self.pre_net -= pq.popleft()[1]
        if idg is not None and now - self.idg_t > self.IDG_WINDOW_US:
            dj = idg.check(now)
            if dj:
                if self.LOG_IDG:
                    print(f'IDG {now / 1e6:.3f} strong {dj} inv={self.inv} bb={bb} ba={ba}')
                self.P += self.IDG_P
                self.idg_t = now
                if self.IDG_SNIPE:
                    self._snipe_idg(dj, bb, ba, now)
                self._jump_seen(dj, now)
        if self.idg2 is not None:
            dj = self.idg2.check(now)
            if dj:
                if self.LOG_IDG:
                    print(f'IDG {now / 1e6:.3f} weak {dj} inv={self.inv} bb={bb} ba={ba}')
                self._weak(dj, bb, ba, now)
                self._jump_seen(dj, now)
        if self.adj_pend:
            for k, d in self.adj_pend:
                if self.adj_ids and (k - 1) not in self.adj_ids:
                    if self.LOG_IDG:
                        print(f'IDG {now / 1e6:.3f} adj {d} inv={self.inv} bb={bb} ba={ba}')
                    self._snipe_idg(d, bb, ba, now)
                    self._jump_seen(d, now)
                    break
            self.adj_pend = []
        half = (bq + aq) / 2
        self.depth = half if self.depth == 0 else self.depth + self.DEPTH_EMA * (half - self.depth)
        x = self.flow / self.depth
        if self.FLOW_CHECK and now >= self.fc_next:
            self._flow_check(now, x, mid)
        fx = self.flow_scale * x
        imb = (bq - aq) / (bq + aq)
        if self.K_OFI or self.V_K_OFI:
            ofi = self.ofi * math.exp((self.ofi_t - now) / self.OFI_TAU_US) / self.depth
        else:
            ofi = 0.0
        dimb = self._dimb(bids, asks, bb, ba) if (self.K_DIMB or self.V_K_DIMB) else 0.0
        mldi = wmp = mlofi = far = 0.0
        if self.ml_on:
            mldi, wmp, mlofi, far = self._ml(bids, asks, bb, ba, mid, now)
        ml_q = self.K_MLDI * mldi + self.K_WMP * wmp + self.K_MLOFI * mlofi + self.K_FAR * far
        kml = self.V_K_MLDI if (not self.MLDI_VMIN or self.vol >= self.MLDI_VMIN) else 0.0
        ml_v = kml * mldi * ((ba - bb) * 0.5 if self.ML_SPREAD else 1.0) + self.V_K_WMP * wmp + self.V_K_MLOFI * mlofi + self.V_K_FAR * far

        storm = self.pv >= self.P_ON
        if storm and not self.in_storm:
            self.storm_t = now
        self.in_storm = storm
        if self.pv < self.P_ON:
            fair = mid + self.K_S * (self.s - mid) + self.K_IMB * imb + self.K_FLOW * fx - self.K_INV * self.inv + self.K_MOM * mom + self.K_OFI * ofi + self.K_DIMB * dimb + ml_q
            if self.K_PRE and self.pv >= self.PRE_LO:
                fair += self.K_PRE * (1 if self.pre_net > 0 else (-1 if self.pre_net < 0 else 0))
            alert = (self.PW_LO and self.pv >= self.PW_LO) or (self.VW_VMIN and self.vol >= self.VW_VMIN)
            q_edge = self.EDGE + (self.PW_EDGE if alert else 0.0)
            q_new = (self.EDGE_NEW + (self.PW_EDGE if alert else 0.0)) if self.EDGE_NEW else 0.0
            if fair - ba >= self.EDGE_TAKE:
                pb = nb = math.floor(fair - self.EDGE_TAKE) if self.Q_SWEEP else ba
            elif ba - bb >= 2 and fair - (bb + 1) >= self.EDGE_IN:
                pb = nb = bb + 1
            else:
                pb = bb
                while fair - pb < q_edge:
                    pb -= 1
                nb = pb
                if q_new:
                    while fair - nb < q_new:
                        nb -= 1
            if bb - fair >= self.EDGE_TAKE:
                pa = na = math.ceil(fair + self.EDGE_TAKE) if self.Q_SWEEP else bb
            elif ba - bb >= 2 and (ba - 1) - fair >= self.EDGE_IN:
                pa = na = ba - 1
            else:
                pa = ba
                while pa - fair < q_edge:
                    pa += 1
                na = pa
                if q_new:
                    while na - fair < q_new:
                        na += 1
            sq_b = sq_a = 0
            if self.SHED and self.SHED_CALM:
                shed = self._shed_side(now, mid)
                if shed:
                    q_sh = abs(self.inv) - self.SHED_TO if self.SHED_TO else max(abs(self.inv), self.QTY)
                    if shed > 0:
                        px = bb + 1 if ba - bb >= 2 else bb
                        if self.SHED_MODE == 1 and ba <= fair + self.SHED_EDGE:
                            px = ba
                        if px > pb:
                            pb = nb = px
                            sq_b = q_sh
                    else:
                        px = ba - 1 if ba - bb >= 2 else ba
                        if self.SHED_MODE == 1 and bb >= fair - self.SHED_EDGE:
                            px = bb
                        if px < pa:
                            pa = na = px
                            sq_a = q_sh
            qp = self.PW_QP if (alert and self.PW_QP) else self.Q_PASSIVE
            if now < self.block[1] and pb < ba:
                self._clear(1)
            elif qp == 0 or pb >= ba or (qp == 1 and self.inv < 0):
                self._side(1, pb, sq_b, nb)
            else:
                self._clear(1)
            if now < self.block[-1] and pa > bb:
                self._clear(-1)
            elif qp == 0 or pa <= bb or (qp == 1 and self.inv > 0):
                self._side(-1, pa, sq_a, na)
            else:
                self._clear(-1)
            return

        fair = mid + self.V_K_S * (self.s - mid) + self.V_K_FLOW * fx - self.V_K_INV * self.inv + self.V_K_MOM * mom + self.V_K_OFI * ofi + self.V_K_DIMB * dimb + ml_v
        if self.V_HZ:
            fair -= self.V_HZ * min(1.0, (now - self.storm_t) / self.HZ_US) * self.inv
        if self.V_K_PJ and now - self.pj_t < self.PJ_US:
            fair += self.V_K_PJ * self.pj_sign
        vet = self.V_EDGE_TAKE * (math.sqrt(max(self.vol, 1e-9) / self.VET_REF) if self.VET_REF else 1.0)
        shed = self._shed_side(now, mid) if (self.SHED and self.SHED_STORM) else 0
        if fair - ba >= vet:
            q = self.V_QTY if (not self.ESZ_FULL or fair - ba >= self.ESZ_FULL) else int(self.V_QTY * self.ESZ_FRAC)
            self._side(1, math.floor(fair - self.V_EDGE_TAKE) if self.V_SWEEP else ba, q)
        elif shed > 0:
            self._shed(1, fair, bb, ba)
        else:
            self._clear(1)
        if bb - fair >= vet:
            q = self.V_QTY if (not self.ESZ_FULL or bb - fair >= self.ESZ_FULL) else int(self.V_QTY * self.ESZ_FRAC)
            self._side(-1, math.ceil(fair + self.V_EDGE_TAKE) if self.V_SWEEP else bb, q)
        elif shed < 0:
            self._shed(-1, fair, bb, ba)
        else:
            self._clear(-1)

    def _shed_side(self, now: int, mid: float) -> int:
        """+1 to buy back a capped short, -1 to sell a capped long, once its edge is earned; else 0."""
        inv = self.inv
        if abs(inv) < self.SHED_AT or abs(inv) > self.MAX_POS:
            self.cap_sign = 0
            return 0
        sg = 1 if inv > 0 else -1
        if self.cap_sign != sg:
            self.cap_sign, self.cap_t, self.cap_mid = sg, now, mid
            return 0
        if (self.SHED_TICKS and sg * (mid - self.cap_mid) >= self.SHED_TICKS) or \
                (self.SHED_AGE_US and now - self.cap_t >= self.SHED_AGE_US):
            return -sg
        return 0

    def _shed(self, sign: int, fair: float, bb: int, ba: int) -> None:
        """Reduce the capped position toward SHED_TO on side sign (passive at the touch, or take it in mode 1)."""
        q = abs(self.inv) - self.SHED_TO
        if q < self.MIN_NEW:
            self._clear(sign)
            return
        if sign > 0:
            if self.SHED_MODE == 1 and ba <= fair + self.SHED_EDGE:
                px = ba
            else:
                px = bb + 1 if ba - bb >= 2 else bb
        else:
            if self.SHED_MODE == 1 and bb >= fair - self.SHED_EDGE:
                px = bb
            else:
                px = ba - 1 if ba - bb >= 2 else ba
        self._side(sign, px, q)

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
        sig2 = (1.0 - pv) * self.SIG_Q * self.SIG_Q + pv * self.sig_v * self.sig_v
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
                self.pj_sign = 1 if self.jv > 0 else -1
                self.pj_t = now
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
                if self.ADJ_SNIPE:
                    self._adj_new(sign, o.price, oid, now)
                if self.idg is not None:
                    self.idg.seen(oid, now)
                    if self.pbb:
                        self.idg.new_order(sign, o.price, self.pbb, self.pba, now)
                if self.idg2 is not None:
                    self.idg2.seen(oid, now)
                    if self.pbb:
                        self.idg2.new_order(sign, o.price, self.pbb, self.pba, now)
                if self.FAR_CANCEL and self.ba:
                    behind = (self.ba - 1 - o.price) if sign < 0 else (o.price - self.bb - 1)
                    if behind >= self.FAR_CANCEL:
                        self._pull(sign, now)
                if self.WEAK_FAR and self.ba:
                    behind = (self.ba - 1 - o.price) if sign < 0 else (o.price - self.bb - 1)
                    if behind >= self.WEAK_FAR:
                        self._weak(sign, self.bb, self.ba, now)
                if len(oid) > len(new_max) or (len(oid) == len(new_max) and oid > new_max):
                    new_max = oid
        self.max_id = new_max

    def _adj_add(self, oid) -> None:
        """Remember the last 256 numeric market order ids."""
        s = str(oid)
        if not s.isdigit():
            return
        k = int(s)
        if k in self.adj_ids:
            return
        self.adj_ids.add(k)
        self.adj_q.append(k)
        if len(self.adj_q) > 256:
            self.adj_ids.discard(self.adj_q.popleft())

    def _adj_new(self, sign: int, price: int, oid, now: int) -> None:
        """Queue a clamped new order with a wide previous spread and a quiet jump side."""
        s = str(oid)
        if s.isdigit() and self.pbb and self.pba:
            d = 0
            if sign > 0 and price == self.pbb + 1 and price < self.pba:
                d = -1
            elif sign < 0 and price == self.pba - 1 and price > self.pbb:
                d = 1
            if d and self.pba - self.pbb >= self.ADJ_SP:
                last = self.adj_last_bid_t if d < 0 else self.adj_last_ask_t
                if now - last > self.ADJ_QUIET_US:
                    self.adj_pend.append((int(s), d))
        if sign > 0:
            self.adj_last_ask_t = now
        else:
            self.adj_last_bid_t = now
        self._adj_add(oid)

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
        if self.snipe_id is not None:
            if self.LOG_IDG:
                print(f'IDG {now / 1e6:.3f} skip pending snipe')
            return
        if self.unwind_id is not None:
            if self.LOG_IDG:
                print(f'IDG {now / 1e6:.3f} skip pending unwind')
            return
        if now - self.snipe_t < self.SNIPE_COOLDOWN_US:
            if self.LOG_IDG:
                print(f'IDG {now / 1e6:.3f} skip cooldown {(now - self.snipe_t) / 1e3:.1f}ms since last snipe')
            return
        if self.killed or self.over_total > self.KILL_OVER_US:
            self.killed = True
            if self.LOG_IDG:
                print(f'IDG {now / 1e6:.3f} skip killed over_total')
            return
        while self.snipe_times and now - self.snipe_times[0] > self.KILL_RATE_US:
            self.snipe_times.popleft()
        if len(self.snipe_times) >= self.KILL_RATE_N:
            self.killed = True
            if self.LOG_IDG:
                print(f'IDG {now / 1e6:.3f} skip killed rate')
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
            if self.LOG_IDG:
                print(f'IDG {now / 1e6:.3f} skip q<MIN_NEW q={q} resting={resting}')
            return
        if self.LOG_IDG:
            print(f'IDG {now / 1e6:.3f} sent side={side} limit_px={limit_px} q={q}')
        self.snipe_times.append(now)
        self.snipe_id = self.send_order(OrderRequest(side=side, price=limit_px, quantity=q, order_type=OrderType.IOC))
        self.orders[self.snipe_id] = [sg, limit_px, q]
        self.snipe_s = s_new
        self.snipe_t = now

    def _ml(self, bids, asks, bb: int, ba: int, mid: float, now: int):
        n = self.ML_DEPTH
        b = [0] * n
        a = [0] * n
        for o in bids:
            d = bb - o.price
            if d >= n:
                break
            if o.owner == 'market':
                b[d] += o.quantity
        for o in asks:
            d = o.price - ba
            if d >= n:
                break
            if o.owner == 'market':
                a[d] += o.quantity
        w = self.ml_w
        num = den = 0.0
        for d in range(n):
            num += w[d] * (b[d] - a[d])
            den += w[d] * (b[d] + a[d])
        mldi = num / den if den else 0.0
        L = self.WMP_LEVELS
        bq = sum(b[:L])
        aq = sum(a[:L])
        wmp = 0.0
        if bq and aq:
            bv = bb - sum(d * b[d] for d in range(L)) / bq
            av = ba + sum(d * a[d] for d in range(L)) / aq
            wmp = (bv * aq + av * bq) / (aq + bq) - mid
        L = self.MLOFI_LEVELS
        x = sum(b[:L]) - sum(a[:L])
        if self.mlofi_prev is not None:
            self.mlofi = self.mlofi * math.exp((self.mlofi_t - now) / self.MLOFI_TAU_US) + x - self.mlofi_prev
        self.mlofi_t = now
        self.mlofi_prev = x
        fb = sum(b[self.FAR_LO:])
        fa = sum(a[self.FAR_LO:])
        far = (fb - fa) / (fb + fa) if fb + fa else 0.0
        return mldi, wmp, self.mlofi / self.depth if self.depth else 0.0, far

    def _dimb(self, bids, asks, bb: int, ba: int) -> float:
        lo, hi = bb - self.DIMB_LEVELS, ba + self.DIMB_LEVELS
        b = 0
        for o in bids:
            if o.price <= lo:
                break
            if o.owner == 'market':
                b += o.quantity
        a = 0
        for o in asks:
            if o.price >= hi:
                break
            if o.owner == 'market':
                a += o.quantity
        return (b - a) / (b + a) if b + a else 0.0

    def _jump_seen(self, d: int, now: int) -> None:
        self.pj_sign = d
        self.pj_t = now
        if self.JUMP_CANCEL:
            self._pull(d, now)

    def _pull(self, d: int, now: int) -> None:
        """S moved in direction d: our quotes on the other side are about to be run over."""
        self.block[-d] = now + self.JC_HOLD_US
        for oid, o in self.orders.items():
            if o[0] == -d and oid not in self.cancel_sent:
                self._cancel(oid)

    def _weak(self, d: int, bb: int, ba: int, now: int) -> None:
        """Weak jump sign in direction d: take toward +-WEAK_POS inside the cap at once."""
        if self.weak_id is not None:
            if self.LOG_IDG:
                print(f'IDG {now / 1e6:.3f} weak skip pending')
            return
        if now - self.weak_t < self.WEAK_COOLDOWN_US:
            if self.LOG_IDG:
                print(f'IDG {now / 1e6:.3f} weak skip cooldown')
            return
        if abs(self.inv) > self.MAX_POS:
            if self.LOG_IDG:
                print(f'IDG {now / 1e6:.3f} weak skip over cap')
            return
        if self.WEAK_STORM and self.pv < self.P_ON:
            if self.LOG_IDG:
                print(f'IDG {now / 1e6:.3f} weak skip storm gate')
            return
        if self.WEAK_FREE:
            if d > 0:
                q = self.WEAK_POS - self.inv
                px, side = ba + self.WEAK_DEPTH, Side.BUY
            else:
                q = self.WEAK_POS + self.inv
                px, side = max(1, bb - self.WEAK_DEPTH), Side.SELL
            resting_d = sum(o[2] for o in self.orders.values() if o[0] == d)
            q = min(q, 1000 - d * self.inv - resting_d)
            if q >= self.MIN_NEW:
                for oid, o in list(self.orders.items()):
                    if o[0] == d and oid != self.snipe_id and oid != self.unwind_id and oid != self.weak_id:
                        self._cancel(oid)
        elif d > 0:
            q = self.WEAK_POS - self.inv - sum(o[2] for o in self.orders.values() if o[0] > 0)
            px, side = ba + self.WEAK_DEPTH, Side.BUY
        else:
            q = self.WEAK_POS + self.inv - sum(o[2] for o in self.orders.values() if o[0] < 0)
            px, side = max(1, bb - self.WEAK_DEPTH), Side.SELL
        if q < self.MIN_NEW:
            if self.LOG_IDG:
                print(f'IDG {now / 1e6:.3f} weak skip q<MIN_NEW q={q}')
            return
        if self.LOG_IDG:
            print(f'IDG {now / 1e6:.3f} weak sent side={side} px={px} q={q}')
        self.weak_t = now
        self.weak_id = self.send_order(OrderRequest(side=side, price=px, quantity=q, order_type=OrderType.IOC))
        self.orders[self.weak_id] = [d, px, q]
        self.cancel_sent.add(self.weak_id)

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
            if self.LOG_IDG:
                if oid == self.snipe_id:
                    print(f'IDG fill snipe qty={event.filled_quantity} inv={self.inv}')
                elif oid == self.weak_id:
                    print(f'IDG fill weak qty={event.filled_quantity} inv={self.inv}')
                elif oid == self.unwind_id:
                    print(f'IDG fill unwind qty={event.filled_quantity} inv={self.inv}')
            if o[2] <= 0:
                self._drop(oid)
        elif st == OrderStatus.CANCELED:
            self._drop(oid)
        elif st == OrderStatus.REJECTED and (oid not in self.cancel_sent or oid == self.weak_id):
            self._drop(oid)

    def _drop(self, oid: str) -> None:
        if oid == self.snipe_id:
            self.snipe_id = None
        if oid == self.unwind_id:
            self.unwind_id = None
        if oid == self.weak_id:
            self.weak_id = None
        self.orders.pop(oid, None)
        self.cancel_sent.discard(oid)
