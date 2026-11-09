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
   inventory. Quote at the best price with EDGE_NEW (a resting quote stays while it keeps EDGE) to fair; cross at
   the opposite best when fair is EDGE_TAKE through it. Sticky orders, size capped by MAX_POS.
4. Volatile: fair from the V_ settings plus V_K_MOM * (mid - 1 s EMA of mid) + V_K_MOM2 * (mid - 3 s EMA of mid);
   take the stale book when fair is V_EDGE_TAKE through it (at the best price only, V_QTY per order). No resting
   quotes.

K_FLOW is scaled by a live estimate of the flow -> mid slope (FLOW_CHECK). Inventory is counted from fills only.

Jump sniper (goes past the 100 cap; the penalty is a flat 10 per ms at any size): on an id-gap detection (IdGapJump:
missing market order ids plus clamped orders inside the stale touch) it cancels our quotes on the run-over side and
IOCs up to SN_POS lots of position, priced up to the stale touch + SN_REACH - SN_MARGIN. It unwinds to the cap once
the near touch is within SN_GAP of the new level, or after SN_TMAX_US.
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


class R4BestSnipeStrategy(BaseStrategy):
    QTY = 150  
    MAX_POS = 100
    MIN_NEW = 10
    KEEP_LEVELS = 1
    # quiet regime (round 1 market maker settings)
    EDGE = 0.1
    EDGE_IN = 0.3  
    EDGE_TAKE = 0.25
    K_IMB = 0.15
    K_INV = 0.002
    K_S = 0.9
    K_FLOW = 1.0  
    Q_SWEEP = False      # quiet: when crossing, bid up to fair - EDGE_TAKE instead of only the best ask
    # volatile regime
    V_EDGE_TAKE = 0.55
    V_SWEEP = False      # True: sweep up to fair - V_EDGE_TAKE; False: take at the best price only
    V_QTY = 200          # order size in the volatile regime
    V_K_S = 0.8  
    V_K_FLOW = 1.2
    V_K_INV = 0.0
    K_MOM = 0.0          # quiet: fair += K_MOM * (mid - EMA of mid over MOM_TAU_US), short-term momentum
    V_K_MOM = 0.2  # volatile: same
    MOM_TAU_US = 1_000_000
    # jump sniper: a new market order SN_FAR+ ticks behind its own best price shows S jumped away from the book;
    # take the stale opposite side up to SN_POS lots (beyond 100 costs a flat 10 per ms), unwind when the book catches up
    SN_FAR = 0           # >0: on
    SN_IDG = True  # also fire on the id-gap jump detector (deep, invisible far orders)
    SN_REACH = 10  # id-gap fire: assume new S is at least SN_REACH ticks past the stale touch
    SN_N = 1             # far orders in the same direction needed within SN_W_US
    SN_W_US = 30_000
    SN_X = 2.0           # new S = far order price -+ SN_X
    SN_MARGIN = 2.0      # entry IOC limit: new S -+ SN_MARGIN
    SN_POS = 990
    SN_GAP = 1.0         # unwind once the near touch is within SN_GAP of new S
    SN_TMAX_US = 400_000
    SN_SLIP = 2
    SN_ABORT_US = 0      # >0: unwind early if the near touch has not moved SN_CONF ticks toward new S by then
    SN_CONF = 3
    V_K_MOM2 = -0.05     # volatile: fair += V_K_MOM2 * (mid - EMA of mid over MOM2_TAU_US)
    MOM2_TAU_US = 3_000_000
    P_ON = 0.3  # p_vol above this: volatile policy
    # r4 quiet experiments (all off by default)
    EDGE_NEW = 0.25      # >0: a new quiet quote needs this edge; resting ones cancel below EDGE (hysteresis)
    A_MOM = 0.0          # widen the side against momentum: bid edge += A_MOM * max(0, -mom), ask += A_MOM * max(0, mom)
    A_SGN = 0.0          # widen the side against recent IOC aggressor signs (EMA of +-1, tau SGN_TAU_US)
    SGN_TAU_US = 1_000_000
    A_FX = 0.0           # widen the side against flow/depth: ask += A_FX * max(0, fx)
    PULL_MOM = 0.0       # >0: no quiet passive quote on the side against momentum when |mom| > PULL_MOM
    A_FILL = 0.0         # after a passive fill on a side, that side needs A_FILL more edge for FILL_COOL_US
    FILL_COOL_US = 300_000
    P_PASS = 0.0         # >0: no quiet passive quotes while p_vol > P_PASS (takes still allowed)
    KT_MOM = 0.0         # quiet: momentum term only in the take decision
    A_INV = 0.0          # widen the side that grows |inventory|: += A_INV * |inv| / MAX_POS
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
    SIG_Q = 0.6  # ticks / sqrt(s) in the quiet state
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
    # id-gap jump detector: on a detection add IDG_P to the filter variance; 0: off
    IDG_P = 0.0
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
        self.iocs: dict[str, list] = {}
        self.sn_hits: list = []
        self.sn_fire = None
        self.sn_s = 0.0
        self.sn_d = 0
        self.sn_t = 0
        self.sn_b0 = 0
        self.max_id = ''
        self.last_bids = None
        self.last_asks = None
        self.mid = 0.0
        self.mema = 0.0
        self.mema2 = 0.0
        self.mema_t = 0
        self.sgn = 0.0
        self.fill_t = {1: -10**12, -1: -10**12}
        self.now = 0
        self.sgn_t = 0
        self.last_tk = ''
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
        # id-gap jump detector
        self.idg = IdGapJump(self.IDG_WINDOW_US, self.IDG_MISSING) if (self.IDG_P or self.SN_IDG) else None
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
        self.now = now
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
        if self.A_SGN:
            self.sgn *= math.exp((self.sgn_t - now) / self.SGN_TAU_US)
            self.sgn_t = now
        for tr in trades:
            if tr.taker_owner == 'market':
                self.flow += tr.quantity if tr.book_side == 'ask' else -tr.quantity
                if self.A_SGN and tr.taker_order_id != self.last_tk:
                    self.last_tk = tr.taker_order_id
                    self.sgn += 1.0 if tr.book_side == 'ask' else -1.0
                if idg is not None:
                    idg.seen(tr.taker_order_id, now)
                if self.H_IOC and tr.taker_order_id != self.last_taker:
                    self.last_taker = tr.taker_order_id
                    self._observe_ioc(now)
        if idg is not None and now - self.idg_t > self.IDG_WINDOW_US:
            jd = idg.check(now)
            if jd:
                self.P += self.IDG_P
                self.idg_t = now
                if self.SN_IDG:
                    self.sn_fire = (jd, (ba + self.SN_REACH) if jd > 0 else (bb - self.SN_REACH))
        if (self.SN_FAR or self.SN_IDG) and self._snipe(bb, ba, now):
            return
        half = (bq + aq) / 2
        self.depth = half if self.depth == 0 else self.depth + self.DEPTH_EMA * (half - self.depth)
        x = self.flow / self.depth
        if self.FLOW_CHECK and now >= self.fc_next:
            self._flow_check(now, x, mid)
        fx = self.flow_scale * x
        imb = (bq - aq) / (bq + aq)

        mom = 0.0
        if self.K_MOM or self.V_K_MOM:
            if not self.mema:
                self.mema = mid
            else:
                self.mema += (1.0 - math.exp((self.mema_t - now) / self.MOM_TAU_US)) * (mid - self.mema)
            mom = mid - self.mema
        mom2 = 0.0
        if self.V_K_MOM2:
            if not self.mema2:
                self.mema2 = mid
            else:
                self.mema2 += (1.0 - math.exp((self.mema_t - now) / self.MOM2_TAU_US)) * (mid - self.mema2)
            mom2 = mid - self.mema2
        self.mema_t = now
        if self.pv < self.P_ON:
            fair = mid + self.K_S * (self.s - mid) + self.K_IMB * imb + self.K_FLOW * fx - self.K_INV * self.inv + self.K_MOM * mom
            eb = ea = self.EDGE
            if self.A_MOM:
                eb += self.A_MOM * max(0.0, -mom)
                ea += self.A_MOM * max(0.0, mom)
            if self.A_SGN:
                eb += self.A_SGN * max(0.0, -self.sgn)
                ea += self.A_SGN * max(0.0, self.sgn)
            if self.A_FX:
                eb += self.A_FX * max(0.0, -fx)
                ea += self.A_FX * max(0.0, fx)
            if self.A_FILL:
                if now - self.fill_t[1] < self.FILL_COOL_US:
                    eb += self.A_FILL
                if now - self.fill_t[-1] < self.FILL_COOL_US:
                    ea += self.A_FILL
            if self.A_INV:
                if self.inv > 0:
                    eb += self.A_INV * self.inv / self.MAX_POS
                else:
                    ea -= self.A_INV * self.inv / self.MAX_POS
            dn = self.EDGE_NEW - self.EDGE if self.EDGE_NEW else 0.0
            tf = fair + self.KT_MOM * mom
            if tf - ba >= self.EDGE_TAKE:
                pb = nb = math.floor(fair - self.EDGE_TAKE) if self.Q_SWEEP else ba
            elif ba - bb >= 2 and fair - (bb + 1) >= self.EDGE_IN + eb - self.EDGE:
                pb = nb = bb + 1
            else:
                pb = bb
                while fair - pb < eb:
                    pb -= 1
                nb = pb
                while fair - nb < eb + dn:
                    nb -= 1
            if bb - tf >= self.EDGE_TAKE:
                pa = na = math.ceil(fair + self.EDGE_TAKE) if self.Q_SWEEP else bb
            elif ba - bb >= 2 and (ba - 1) - fair >= self.EDGE_IN + ea - self.EDGE:
                pa = na = ba - 1
            else:
                pa = ba
                while pa - fair < ea:
                    pa += 1
                na = pa
                while na - fair < ea + dn:
                    na += 1
            if self.P_PASS and self.pv > self.P_PASS:
                if pb >= ba:
                    self._side(1, pb)
                else:
                    self._clear(1)
                if pa <= bb:
                    self._side(-1, pa)
                else:
                    self._clear(-1)
                return
            if self.PULL_MOM and -mom > self.PULL_MOM and pb < ba:
                self._clear(1)
            else:
                self._side(1, pb, 0, nb)
            if self.PULL_MOM and mom > self.PULL_MOM and pa > bb:
                self._clear(-1)
            else:
                self._side(-1, pa, 0, na)
            return

        fair = mid + self.V_K_S * (self.s - mid) + self.V_K_FLOW * fx - self.V_K_INV * self.inv + self.V_K_MOM * mom + self.V_K_MOM2 * mom2
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
                self._observe(sign, o.price, (o.price - first) * sign, now)
                if self.SN_FAR and (o.price - first) * sign >= self.SN_FAR:
                    h = self.sn_hits
                    h.append((now, sign, o.price))
                    while h and now - h[0][0] > self.SN_W_US:
                        h.pop(0)
                    k = 0
                    for x in h:
                        if x[1] == sign:
                            k += 1
                    if k >= self.SN_N:
                        self.sn_fire = (sign, o.price - sign * self.SN_X)
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

    def _snipe(self, bb: int, ba: int, now: int) -> bool:
        """Jump sniper. Returns True while it manages the position (skip the normal policy this event)."""
        inv = self.inv
        f = self.sn_fire
        if f is not None:
            self.sn_fire = None
            d, s_new = f
            if not self.iocs and d * inv < self.SN_POS - self.MAX_POS:
                self._clear(-d)
                rest = 0
                for o in self.orders.values():
                    if o[0] == d:
                        rest += o[2]
                q = self.SN_POS - d * inv - rest
                px = math.floor(s_new - self.SN_MARGIN) if d > 0 else math.ceil(s_new + self.SN_MARGIN)
                if q > 0 and ((d > 0 and px >= ba) or (d < 0 and px <= bb)):
                    self._ioc(d, px, q)
                    self.sn_s, self.sn_d, self.sn_t = s_new, d, now
                    self.sn_b0 = bb if d > 0 else ba
                    return True
        if inv > self.MAX_POS or inv < -self.MAX_POS:
            if self.iocs:
                return True
            d = 1 if inv > 0 else -1
            s_new = self.sn_s if self.sn_d == d else self.s
            near = (s_new - bb) if d > 0 else (ba - s_new)
            moved = (bb - self.sn_b0) if d > 0 else (self.sn_b0 - ba)
            abort = self.SN_ABORT_US and now - self.sn_t >= self.SN_ABORT_US and moved < self.SN_CONF
            if near <= self.SN_GAP or abort or now - self.sn_t >= self.SN_TMAX_US:
                px = bb - self.SN_SLIP if d > 0 else ba + self.SN_SLIP
                self._ioc(-d, px, d * inv - self.MAX_POS)
            return True
        return False

    def _ioc(self, sign: int, price: int, q: int) -> None:
        if q <= 0:
            return
        side = Side.BUY if sign > 0 else Side.SELL
        oid = self.send_order(OrderRequest(side=side, price=price, quantity=q, order_type=OrderType.IOC))
        self.iocs[oid] = [sign, price, q]

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
            bn = (new_target - o[1]) * sign
            if behind < 0 or bn > self.KEEP_LEVELS:
                self._cancel(oid)
                resting += o[2]
                continue
            resting += o[2]
            if bn <= 0:
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
            o = self.iocs.get(oid)
            if o is None:
                return
            st = event.status
            if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
                self.inv += o[0] * event.filled_quantity
                o[2] -= event.filled_quantity
                if o[2] <= 0:
                    self.iocs.pop(oid, None)
            elif st == OrderStatus.CANCELED or st == OrderStatus.REJECTED:
                self.iocs.pop(oid, None)
            return
        st = event.status
        if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
            self.inv += o[0] * event.filled_quantity
            self.fill_t[o[0]] = self.now
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
