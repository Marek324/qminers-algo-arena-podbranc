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
arena scores on a private tape.

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


class HmmEstVStrategy(BaseStrategy):
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
    H_NMAX = 20
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
    # estimator experiments (0 / False = hmm_617_fit)
    ROB_NU = 0.0         # Student-t weighting of the Kalman reading noise (nu); 0 = Gaussian
    ROB_CLIP = 0.0       # clip the innovation at this many sds; 0 = off
    ROB_VOL = True       # apply the robust update in the volatile state too
    C_SPLIT = False      # learn the price-to-S gap separately for bids and asks
    R_DEPTH = 0.0        # reading noise *= 1 + R_DEPTH * ticks behind the best price
    H_FROM_S = False     # HMM distance from S_hat instead of the mid
    H_PREMID = False     # HMM distance from the mid before this order arrived
    H_CAP = 0            # orders at the no-cross cap: 0 as usual, 1 skip in the HMM, 2 count as n = H_CAP_N
    H_CAP_N = 8
    IOC_K = 0.0          # each market IOC pushes S_hat by IOC_K * executed^IOC_POW in its direction
    IOC_POW = 0.63
    BOUND_OFF = -1.0     # >= 0: after each Kalman reading keep S_hat at least this far inside the new order's price
    H_HIST = False       # HMM emission: learned histogram of n per state instead of a geometric
    H_HIST_LR = 0.0003
    H_TIME = False       # HMM switch probabilities per second (H_RQV, H_RVQ) instead of per order
    H_RQV = 0.006
    H_RVQ = 0.06
    H_GAP = False        # also use the gap since the previous new order (exponential, rate learned per state)
    H_GAP_LR = 0.001
    H_SIZE = False       # also use log order size (Gaussian per state, learned)
    H_SIZE_LR = 0.001
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
        self.cq = 0.5 + 0.5 * self.H_N_Q0
        self.cv = 0.5 + 0.5 * self.H_N_V0
        self.cqs = [self.cq, self.cq]
        self.cvs = [self.cv, self.cv]
        self.premid = 0.0
        self.bb = 0
        self.ba = 0
        rq0 = self.H_N_Q0 / (1.0 + self.H_N_Q0)
        rv0 = self.H_N_V0 / (1.0 + self.H_N_V0)
        self.hq = [(1.0 - rq0) * rq0 ** i for i in range(self.H_NMAX + 1)]
        self.hv = [(1.0 - rv0) * rv0 ** i for i in range(self.H_NMAX + 1)]
        self.hq[-1] = rq0 ** self.H_NMAX
        self.hv[-1] = rv0 ** self.H_NMAX
        self.ord_t = 0
        self.gq = 150.0
        self.gv = 150.0
        self.zq = [2.5, 1.0]
        self.zv = [2.5, 1.0]
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
        self.premid = self.mid if self.mid else mid
        self.mid = mid
        self.bb = bb
        self.ba = ba
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

        # flow
        self.flow *= math.exp((self.flow_t - now) / self.FLOW_TAU_US)
        self.flow_t = now
        if self.IOC_K and trades:
            agg = {}
            for tr in trades:
                if tr.taker_owner == 'market':
                    q = tr.quantity if tr.book_side == 'ask' else -tr.quantity
                    agg[tr.taker_order_id] = agg.get(tr.taker_order_id, 0) + q
            for q in agg.values():
                self.s += (1 if q > 0 else -1) * self.IOC_K * abs(q) ** self.IOC_POW
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
            self._side(1, pb)
            self._side(-1, pa)
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

    def _observe(self, sign: int, price: int, depth: int, now: int, qty: int = 1) -> None:
        """One new market limit order: HMM forward step, then (if near the touch) a Kalman update."""
        ref = self.s if self.H_FROM_S else (self.premid if self.H_PREMID else self.mid)
        n = min(self.H_NMAX, max(0, round(2 * (sign * (price - ref) - 0.5))))
        capped = (price >= self.ba - 1) if sign < 0 else (price <= self.bb + 1)   # sign -1 = bid
        skip = False
        if capped and self.H_CAP:
            if self.H_CAP == 1:
                skip = True
            else:
                n = self.H_CAP_N
        rq = self.nq / (1.0 + self.nq)
        rv = self.nv / (1.0 + self.nv)
        pv = self.pv
        if not skip:
            if self.H_HIST:
                lq = self.hq[n]
                lv = self.hv[n]
            else:
                lq = (1.0 - rq) * rq ** n
                lv = (1.0 - rv) * rv ** n
            g = 0.0
            if self.H_TIME or self.H_GAP:
                g = min(1.0, max(0.0, (now - self.ord_t) / 1e6)) if self.ord_t else 0.0
                self.ord_t = now
            if self.H_GAP:
                lq *= self.gq * math.exp(-self.gq * g)
                lv *= self.gv * math.exp(-self.gv * g)
            lz = 0.0
            if self.H_SIZE:
                lz = math.log(max(qty, 1))
                mq, vq = self.zq
                mv, vv = self.zv
                lq *= math.exp(-0.5 * (lz - mq) ** 2 / vq) / math.sqrt(vq)
                lv *= math.exp(-0.5 * (lz - mv) ** 2 / vv) / math.sqrt(vv)
            if self.H_TIME:
                pqv = 1.0 - math.exp(-self.H_RQV * g)
                pvq = 1.0 - math.exp(-self.H_RVQ * g)
            else:
                pqv = self.H_PQV
                pvq = self.H_PVQ
            pv = self.pv * (1.0 - pvq) + (1.0 - self.pv) * pqv
            a = pv * lv
            den = a + (1.0 - pv) * lq
            pv = a / den if den > 0 else pv
            self.pv = pv
            if self.H_HIST:
                hl = self.H_HIST_LR
                wq = hl * (1.0 - pv)
                wv = hl * pv
                hq = self.hq
                hv = self.hv
                for i in range(len(hq)):
                    hq[i] *= 1.0 - wq
                    hv[i] *= 1.0 - wv
                hq[n] += wq
                hv[n] += wv
            if self.H_GAP:
                gl = self.H_GAP_LR
                mq = 1.0 / self.gq + gl * (1.0 - pv) * (g - 1.0 / self.gq)
                mv = 1.0 / self.gv + gl * pv * (g - 1.0 / self.gv)
                self.gq = 1.0 / max(mq, 1e-4)
                self.gv = 1.0 / max(mv, 1e-4)
            if self.H_SIZE:
                sl = self.H_SIZE_LR
                for st, w in ((self.zq, sl * (1.0 - pv)), (self.zv, sl * pv)):
                    dz = lz - st[0]
                    st[0] += w * dz
                    st[1] = max(0.05, st[1] + w * (dz * dz - st[1]))
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
        if self.C_SPLIT:
            j = 0 if sign > 0 else 1
            self.cqs[j] += self.C_LR * (1.0 - pv) * (d - self.cqs[j])
            self.cvs[j] += self.C_LR * pv * (d - self.cvs[j])
            c = (1.0 - pv) * self.cqs[j] + pv * self.cvs[j]
            self.cq = 0.5 * (self.cqs[0] + self.cqs[1])
            self.cv = 0.5 * (self.cvs[0] + self.cvs[1])
        else:
            self.cq += self.C_LR * (1.0 - pv) * (d - self.cq)
            self.cv += self.C_LR * pv * (d - self.cv)
            c = (1.0 - pv) * self.cq + pv * self.cv
        r = 0.25 * ((1.0 - pv) * rq / ((1.0 - rq) ** 2) + pv * rv / ((1.0 - rv) ** 2)) + self.R_MIN
        if self.R_DEPTH:
            r *= 1.0 + self.R_DEPTH * depth
        y = price - sign * c
        e = y - self.s
        if self.ROB_VOL or pv < self.P_ON:
            if self.ROB_NU:
                z2 = e * e / (self.P + r)
                r *= (self.ROB_NU + z2) / (self.ROB_NU + 1.0)
            if self.ROB_CLIP:
                lim = self.ROB_CLIP * math.sqrt(self.P + r)
                if e > lim:
                    e = lim
                elif e < -lim:
                    e = -lim
        k = self.P / (self.P + r)
        self.s += k * e
        self.P *= 1.0 - k
        if self.BOUND_OFF >= 0:
            if sign < 0:
                if self.s < price + self.BOUND_OFF:
                    self.s = price + self.BOUND_OFF
            elif self.s > price - self.BOUND_OFF:
                self.s = price - self.BOUND_OFF

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
                self._observe(sign, o.price, (o.price - first) * sign, now, o.quantity)
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
