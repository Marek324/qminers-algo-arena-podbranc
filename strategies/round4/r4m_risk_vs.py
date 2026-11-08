"""r4m_risk_vs: r4m_risk_a + quiet quote size by the vol forecast (QTY * clamp(QV_REF / vol, QV_MIN, QV_MAX), vol =
EWMA of squared 1 s mid moves over VOL_TAU_S). Bigger quotes when the next minutes look calm.
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


class R4mRiskVsStrategy(BaseStrategy):
    VOL_TAU_S = 60.0
    QV_REF = 1.5         # 0: off
    QV_MIN = 0.5
    QV_MAX = 1.5
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
        self.vol2 = 0.0
        self.vol = 0.0
        self.vol_t = 0
        self.vol_mid = 0.0

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
            qp = self.Q_PASSIVE
            qq = 0
            if self.QV_REF and self.vol > 0:
                qq = max(self.MIN_NEW, int(self.QTY * min(self.QV_MAX, max(self.QV_MIN, self.QV_REF / self.vol))))
            if qp == 0 or pb >= ba or (qp == 1 and self.inv < 0):
                self._side(1, pb, qq)
            else:
                self._clear(1)
            if qp == 0 or pa <= bb or (qp == 1 and self.inv > 0):
                self._side(-1, pa, qq)
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

    def _side(self, sign: int, target: int, qty: int = 0) -> None:
        qty = qty or self.QTY
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
        self.orders.pop(oid, None)
        self.cancel_sent.discard(oid)
