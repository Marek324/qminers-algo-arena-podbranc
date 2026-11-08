"""regime_mm.py with an exact Bayesian grid filter for the hidden price S and the regime (tools/grid_price.py).

Why: regime_mm with S replaced by the real mid 100 ms ahead (a local cheat) makes 888k on round2_public against
470k with our running averages or Kalman filter, so the estimate of S is the bottleneck (docs/RESEARCH.md).

The filter keeps a probability grid over S per regime and updates it with the generator's own rules: new orders at
floor(S - X) / ceil(S + X) with X fitted per regime on round2_public (quiet gamma(0.854, loc 0.025, theta 0.655);
volatile gamma(0.384, loc 2.594, theta 2.903): never closer than 2.6 ticks), clamped orders as bounds, trades as
bounds (IOCs at round(S +- 1)), regime switching, optional IOC push.
Result on round2_public: the grid loses. Grid in both regimes 262k, grid in the volatile regime only 324k, against
469,637 with the Kalman in the volatile regime (the defaults below, = regime_mm_kf.py). corr^2 with the mid move
300 ms later, quiet: Kalman 0.231, grid 0.215, S2 0.204; 100 ms later, volatile: S2 0.104, Kalman 0.057, grid 0.033.
Its P(volatile) averages 0.10 inside regime_mm's volatile spells, so the fitted volatile X does not match the tape.
S_VOL_SRC / S_QUIET_SRC pick the estimate per regime (0: regime_mm's own, 1: Kalman, 2: grid). GRID_REGIME uses the
grid's P(volatile) instead of regime_mm's score.

Original docstring of regime_mm_kf.py:

regime_mm.py plus a Kalman filter for the hidden price S (tools/kalman_price.py), ported from adaptive_mm_r2.py.

Round 2 public (regime_mm.py: 448,818): KF_VOL alone 464k, then KF_Q_VOL 10: 469,844; KF_QUIET 297k-311k (off).
Also tried on this file and left at regime_mm's values: EDGE_TAKE 0.12/0.18/0.35/0.5 (400k-437k), S_ALPHA 0.05/0.25
(412k/409k), V_PASSIVE_EXIT with V_EXIT_OFF -0.5..1 (345k-399k), V_KEEP 0/1 (460k/310k), V_MAX_HOLD 3 s/8 s (421k/469k),
KF_LEVELS 3/20 (449k/470k), KF_JUMP 3e-4/3e-3 (459k/459k).

KF_QUIET: the quiet-regime fair uses the Kalman estimate instead of the per-side averages (s_hat).
KF_VOL: the volatile-regime sniper uses it instead of the clamp-aware average S2.
The filter reads new market orders within KF_LEVELS ticks of their best price, treats orders one tick from the
opposite best as bounds (censored), inflates its variance after every aggressive market order (KF_JUMP x lots^1.3),
and diffuses at KF_Q_QUIET or KF_Q_VOL ticks^2/s depending on the regime. With both flags 0 this is regime_mm.py.

Original docstring of regime_mm.py:

Regime-switching market maker: make in quiet spells, only snipe the stale book in volatile ones.

Market (round2_public, docs/DATA_ANALYSIS.md): limit orders arrive at floor(S - X) / ceil(S + X) around a
hidden price S and are clamped so they never cross. The generator switches between two regimes:
- quiet (about 85% of the time): new buys and sells sit about 2 ticks apart, S moves about 1 tick/sqrt(s);
- volatile: they sit about 4.3 ticks apart, IOCs come 1.5x as often and S moves 4-6 ticks/sqrt(s), with
  most of the 30-100 tick jumps.
A market maker that makes about 235k in the quiet spells loses about 300k in the volatile ones.

Each event:
1. S_hat = average of the per-side EMAs of new market limit order prices (top S_LEVELS ticks).
   Regime: EMA over new orders (top REG_LEVELS ticks) of |price - mid|; volatile above REG_ON, back to quiet
   below REG_OFF, and never sooner than REG_HOLD_US after the last reading above REG_ON.
2. Quiet: imbalance_mm. fair = mid + K_IMB * imbalance + K_S * (S_hat - mid) + K_FLOW * flow / depth
   - K_INV * inventory; quote at the best price with at least EDGE to fair, cross when fair is EDGE_TAKE
   through the opposite best price.
3. Volatile: cancel the quotes. If S_hat - best_ask >= V_TAKE, IOC-buy up to floor(S_hat - V_TAKE_PX)
   (sells mirror it). Hold the position with one passive exit order at S_hat rounded away from the touch,
   and close what is left by IOC once S_hat says the position no longer has V_KEEP of edge, or after
   V_MAX_HOLD_US.

Inventory is counted from fills only.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy import stats

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

QTY = 200
MAX_POS = 100
EDGE = 0.3
K_IMB = 0.15
K_INV = 0.002
KEEP_LEVELS = 1
MIN_NEW = 10
EDGE_IN = 0.5
K_FLOW = 0.8
FLOW_TAU_US = 100_000
DEPTH_EMA = 0.001
K_S = 0.75
S_ALPHA = 0.1
S_LEVELS = 2
EDGE_TAKE = 0.25
FLOW_CHECK = True
FLOW_SLOPE_FULL = 0.5
FLOW_CHECK_TAU_US = 300_000_000
FLOW_CHECK_STEP_US = 50_000
FLOW_CHECK_H_US = 500_000
FLOW_CHECK_MIN = 200.0

REG_ON = 1.6          # regime score (ticks) that switches to volatile
REG_OFF = 1.35        # and back to quiet
REG_ALPHA = 0.03      # weight of each new order in the regime score
REG_LEVELS = 8        # scan depth for the regime score
REG_HOLD_US = 3_000_000
VOLATILE_MODE = 1     # 0: keep making (baseline), 1: flat + snipe
V_TAKE = 0.75          # S_hat past the opposite best price to take
V_TAKE_PX = 0.5       # IOC limit S_hat -/+ this
V_KEEP = -1.0          # close by IOC once S_hat - best_bid (long) drops below this
V_MAX_HOLD_US = 20_000_000
V_EXIT_OFF = 0.0      # passive exit price: S_hat + this (long), rounded up, never inside the touch
C = 1.1               # gap between a new order's price and S, for the clamp-aware estimate S2
S2_ALPHA = 0.25        # weight of each new order in S2
V_PASSIVE_EXIT = 0    # rest an exit order at S2 while holding
Q_TAKE = 99.0         # quiet regime: when S2 is this far through the opposite best price, bid up to S2 - Q_TAKE_PX
Q_TAKE_PX = 0.5
FAIR_S2 = 0             # quiet fair uses S2 instead of the per-side average
V_FLIP = 0.75         # volatile: threshold instead of V_TAKE when the trade would flip an opposite position
V_FOLLOW = 1e9
S_QUIET_SRC = 0       # quiet-regime estimate of S: 0 regime_mm (s_hat / S2), 1 Kalman, 2 grid
S_VOL_SRC = 1         # volatile-regime estimate of S: 0 regime_mm's S2, 1 Kalman, 2 grid
GRID_REGIME = 0       # 1: regime from the grid's P(volatile) (GRID_P_ON / GRID_P_OFF) instead of the order-spread score
GRID_P_ON = 0.6
GRID_P_OFF = 0.3
GRID_LEVELS = 20      # new orders within this many ticks of their best price feed the grid (the whole visible book)
GRID_Q_QUIET = 1.0    # diffusion of S per regime, ticks^2/s
GRID_Q_VOL = 10.0
GRID_TO_VOL = 0.03    # regime switches per second
GRID_TO_QUIET = 0.2
GRID_PUSH = 0.0       # IOC push of S: GRID_PUSH * lots**0.63 (fitted 0.0159 on round 2)
KF_QUIET = 0          # quiet fair uses the Kalman estimate of S
KF_VOL = 1            # volatile sniper uses the Kalman estimate of S
KF_Q_QUIET = 1.0      # diffusion of S in the quiet regime, ticks^2/s (S moves about 1 tick/sqrt(s))
KF_Q_VOL = 10.0       # volatile-regime diffusion, ticks^2/s (round 2: 3 454k, 6 465k, 10 470k, 25 464k, 60 374k)
KF_JUMP = 1e-3        # variance added per aggressive market order: KF_JUMP * lots**1.3
KF_LEVELS = 8         # new orders within this many ticks of their best price are observations
KF_CENSOR = 1         # an order one tick from the opposite best price is only a bound
KF_BOUND = 1          # a market buy that traded at P proves S >= P - 1.5 (IOCs priced at round(S +- 1))        # volatile: market IOC volume (lots, this event) that triggers taking its side


class HiddenPriceKalman:
    """1-D Kalman filter for the hidden price S (copy of tools/kalman_price.py; the model is in its docstring)."""

    def __init__(self, q: float = 2.0, jump: float = 2e-4, jump_pow: float = 1.3, r0: float = 0.3,
                 r_alpha: float = 0.01, r_min: float = 0.05, m0: float = 1.0, m_alpha: float = 0.01,
                 push: float = 0.0, push_pow: float = 0.65) -> None:
        self.q = q  # ticks^2 per second
        self.jump = jump
        self.jump_pow = jump_pow
        self.r = r0
        self.r_alpha = r_alpha
        self.r_min = r_min
        self.m = m0
        self.m_alpha = m_alpha
        self.push = push
        self.push_pow = push_pow
        self.s = None
        self.p = 1.0
        self.ts = 0

    def _predict(self, ts_us: int) -> None:
        if ts_us > self.ts:
            self.p += self.q * (ts_us - self.ts) * 1e-6
            self.ts = ts_us

    def trade(self, ts_us: int, sign: int, qty: float) -> None:
        """Aggressive market volume `qty` in one event; sign +1 buyers lifted the ask, -1 sellers hit the bid."""
        if qty <= 0:
            return
        self._predict(ts_us)
        self.p += self.jump * qty ** self.jump_pow
        if self.push and self.s is not None:
            self.s += sign * self.push * qty ** self.push_pow

    def observe(self, ts_us: int, price: float, sign: int, capped: bool = False) -> None:
        """A new limit order at `price`; sign +1 buy, -1 sell. `capped`: it sits at the most aggressive price
        the generator allows (one tick from the opposite best), so it is only a bound."""
        if self.s is None:
            self.s = price + sign * self.m
            self.ts = ts_us
            return
        self._predict(ts_us)
        innov = price + sign * self.m - self.s
        if capped and innov * sign < 0:
            return  # consistent with the bound, nothing learned
        p = self.p
        self.m += self.m_alpha * (sign * (self.s - price) - self.m)
        self.r += self.r_alpha * (innov * innov - p - self.r)
        if self.r < self.r_min:
            self.r = self.r_min
        k = p / (p + self.r)
        self.s += k * innov
        self.p = (1.0 - k) * p

    def bound_low(self, lo: float) -> None:
        if self.s is not None and self.s < lo:
            self.s = lo

    def bound_high(self, hi: float) -> None:
        if self.s is not None and self.s > hi:
            self.s = hi


class HiddenPriceGrid:
    """Joint Bayesian grid filter over (regime, S) (copy of tools/grid_price.py; the model is in its docstring)."""

    def __init__(self, x_quiet: tuple = (0.854, 0.025, 0.655), x_vol: tuple = (0.384, 2.594, 2.903),
                 q_quiet: float = 1.0, q_vol: float = 10.0, to_vol: float = 0.03, to_quiet: float = 0.2,
                 push_k: float = 0.0, push_pow: float = 0.63, h: float = 0.1, half: int = 80,
                 floor: float = 0.002, reset_lik: float = 0.005, surprises: int = 3) -> None:
        self.h = h
        self.half = half
        self.n = 2 * half + 1
        self.q_quiet = q_quiet
        self.q_vol = q_vol
        self.to_vol = to_vol
        self.to_quiet = to_quiet
        self.push_k = push_k
        self.push_pow = push_pow
        self.floor = floor
        self.reset_lik = reset_lik
        self.surprises = surprises
        self.n_surprise = 0
        self.step = int(round(1.0 / h))  # grid points per tick
        # tables over u = S - p (buy) / p - S (sell) at u = j * h, j in [-span, span]
        self.span = 40 * self.step
        u = np.arange(-self.span, self.span + 1) * h
        self.tabs = []
        for k, loc, theta in (x_quiet, x_vol):
            cdf = stats.gamma.cdf(u, a=k, loc=loc, scale=theta)
            pmf = np.empty_like(cdf)
            pmf[self.step:] = cdf[self.step:] - cdf[:-self.step]
            pmf[:self.step] = cdf[:self.step]
            self.tabs.append((pmf + floor, cdf + floor))
        self.offs = np.arange(-half, half + 1) * h  # grid point offsets from the centre
        self.c = None
        self.pq = None
        self.pv = None
        self.ts = 0
        self.var_q = 0.0
        self.var_v = 0.0
        self.switch_t = 0.0
        self._mean = None

    def _start(self, centre: int, p_vol: float = 0.1) -> None:
        self.c = centre
        self.pq = np.full(self.n, (1.0 - p_vol) / self.n)
        self.pv = np.full(self.n, p_vol / self.n)
        self._mean = None

    def _diffuse(self, p: np.ndarray, var: float) -> np.ndarray:
        s2 = var / (self.h * self.h)  # variance in grid points^2
        if s2 < 0.6:
            a = 0.5 * s2
            out = p * (1.0 - 2.0 * a)
            out[1:] += a * p[:-1]
            out[:-1] += a * p[1:]
            return out
        w = int(math.ceil(3.0 * math.sqrt(s2)))
        if w > self.half:
            w = self.half
        x = np.arange(-w, w + 1)
        ker = np.exp(-0.5 * x * x / s2)
        ker /= ker.sum()
        return np.convolve(p, ker, 'same')

    def _predict(self, ts_us: int) -> None:
        if ts_us > self.ts:
            dt = (ts_us - self.ts) * 1e-6
            self.ts = ts_us
            self.var_q += self.q_quiet * dt
            self.var_v += self.q_vol * dt
            self.switch_t += dt
        h2 = 0.25 * self.h * self.h
        if self.var_q >= h2 or self.var_v >= h2:
            self.pq = self._diffuse(self.pq, self.var_q)
            self.pv = self._diffuse(self.pv, self.var_v)
            self.var_q = 0.0
            self.var_v = 0.0
        if self.switch_t > 0.0:
            a = min(1.0, self.to_vol * self.switch_t)
            b = min(1.0, self.to_quiet * self.switch_t)
            pq, pv = self.pq, self.pv
            self.pq = pq * (1.0 - a) + pv * b
            self.pv = pv * (1.0 - b) + pq * a
            self.switch_t = 0.0
        self._mean = None

    def _shift(self, pts: int) -> None:
        """Move the probability mass `pts` grid points up (S rises) without moving the grid."""
        if pts == 0:
            return
        self.pq = self._moved(self.pq, pts)
        self.pv = self._moved(self.pv, pts)
        self._mean = None

    def _moved(self, p: np.ndarray, pts: int) -> np.ndarray:
        q = np.zeros_like(p)
        if pts >= self.n or -pts >= self.n:
            return q
        if pts > 0:
            q[pts:] = p[:-pts]
        else:
            q[:pts] = p[-pts:]
        return q

    def _recentre(self) -> None:
        m = self.mean()
        d = int(round(m - self.c))
        if abs(d) >= 2:
            self.c += d
            self._shift(-d * self.step)
            tot = self.pq.sum() + self.pv.sum()
            self.pq += 1e-9
            self.pv += 1e-9
            self.pq /= tot
            self.pv /= tot
            self._mean = None

    def trade(self, ts_us: int, sign: int, lots: float, price: float) -> None:
        """Aggressive market volume in one event on one side (sign +1: buyers lifted the ask), deepest price."""
        if self.c is None or lots <= 0:
            return
        self._predict(ts_us)
        if self.push_k:
            self._shift(int(round(sign * self.push_k * lots ** self.push_pow / self.h)))
        # bound: buy => S >= price - 1.5, sell => S <= price + 1.5
        bound = price - 1.5 * sign
        cut = (bound - self.c) / self.h + self.half
        if (sign > 0 and cut >= self.n - 1) or (sign < 0 and cut <= 0):
            self._start(int(round(bound + 2.0 * sign)), self.p_vol())  # S is past the edge of the grid
            return
        if sign > 0:
            i = int(math.ceil(cut))
            if 0 < i <= self.n:
                self.pq[:i] *= 0.01
                self.pv[:i] *= 0.01
        else:
            i = int(math.floor(cut)) + 1
            if 0 <= i < self.n:
                self.pq[i:] *= 0.01
                self.pv[i:] *= 0.01
        tot = self.pq.sum() + self.pv.sum()
        self.pq /= tot
        self.pv /= tot
        self._mean = None

    def observe(self, ts_us: int, price: int, sign: int, capped: bool = False) -> None:
        """A new market limit order at `price` (sign +1 buy, -1 sell); `capped`: one tick from the opposite best."""
        if self.c is None:
            self._start(int(price + sign))
            self.ts = ts_us
        self._predict(ts_us)
        # u = sign * (S - p) on the grid: index j = sign * (step * (c - p) + k), k = -half..half
        base = self.step * (self.c - price)
        col = 1 if capped else 0
        if sign > 0:
            j0 = base - self.half + self.span
            if j0 < 0 or j0 + self.n > 2 * self.span + 1:
                return
            lq = self.tabs[0][col][j0:j0 + self.n]
            lv = self.tabs[1][col][j0:j0 + self.n]
        else:
            j0 = -base - self.half + self.span  # u = p - S runs downwards along the grid
            if j0 < 0 or j0 + self.n > 2 * self.span + 1:
                return
            lq = self.tabs[0][col][j0:j0 + self.n][::-1]
            lv = self.tabs[1][col][j0:j0 + self.n][::-1]
        pq = self.pq * lq
        pv = self.pv * lv
        tot = pq.sum() + pv.sum()
        if tot < self.reset_lik:
            self.n_surprise += 1
            if self.n_surprise >= self.surprises:  # S jumped beyond the grid: restart around this order
                self.n_surprise = 0
                self._start(int(round(price + sign * 3.0)), 0.9)
            return
        self.n_surprise = 0
        self.pq = pq / tot
        self.pv = pv / tot
        self._mean = None
        self._recentre()

    def mean(self) -> float:
        if self._mean is None:
            w = self.pq + self.pv
            self._mean = self.c + float(np.dot(w, self.offs)) / float(w.sum())
        return self._mean

    def p_vol(self) -> float:
        s = float(self.pv.sum())
        return s / (s + float(self.pq.sum()))


class RegimeMMGrid(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        self.orders: dict[str, list] = {}   # resting: id -> [sign, price, remaining]
        self.iocs: dict[str, list] = {}     # in flight: id -> [sign, remaining]
        self.cancel_sent: set[str] = set()
        self.flow = 0.0
        self.flow_t = 0
        self.depth = 0.0
        self.s_bid = 0.0
        self.s_ask = 0.0
        self.max_id = ''
        self.last_bids = None
        self.last_asks = None
        self.fc_pending: deque = deque()
        self.fc_next = 0
        self.fc_sxx = 0.0
        self.fc_sxy = 0.0
        self.fc_n = 0.0
        self.fc_decay = math.exp(-FLOW_CHECK_STEP_US / FLOW_CHECK_TAU_US)
        self.flow_scale = 0.0 if FLOW_CHECK else 1.0
        self.reg = 1.0
        self.volatile = False
        self.reg_t = 0
        self.mid = 0.0
        self.pos_t = 0          # when the current volatile position was opened
        self.s2 = 0.0           # clamp-aware estimate of S over all scanned new orders
        self.bb = 0
        self.ba = 0
        self.now = 0
        self.kf = HiddenPriceKalman(q=KF_Q_QUIET, jump=KF_JUMP)
        self.gf = HiddenPriceGrid(q_quiet=GRID_Q_QUIET, q_vol=GRID_Q_VOL, to_vol=GRID_TO_VOL, to_quiet=GRID_TO_QUIET,
                                  push_k=GRID_PUSH)

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
        now = snapshot.timestamp_us
        mid = (bb + ba) / 2
        self.mid = mid
        self.bb, self.ba = bb, ba
        if not self.s2:
            self.s2 = mid
        kf = self.kf
        kf.q = KF_Q_VOL if self.volatile else KF_Q_QUIET
        hi_buy = 0
        lo_sell = 0
        if trades:
            bv = 0
            sv = 0
            for tr in trades:
                if tr.taker_owner == 'market':
                    if tr.book_side == 'ask':
                        bv += tr.quantity
                        if tr.price > hi_buy:
                            hi_buy = tr.price
                    else:
                        sv += tr.quantity
                        if not lo_sell or tr.price < lo_sell:
                            lo_sell = tr.price
            kf.trade(now, 1, bv)
            kf.trade(now, -1, sv)
            if bv:
                self.gf.trade(now, 1, bv, hi_buy)
            if sv:
                self.gf.trade(now, -1, sv, lo_sell)
        self.now = now
        if bids is not self.last_bids:
            self.last_bids = bids
            self._scan_new(bids, -1)
        if asks is not self.last_asks:
            self.last_asks = asks
            self._scan_new(asks, 1)
        if KF_BOUND:
            if hi_buy:
                kf.bound_low(hi_buy - 1.5)
            if lo_sell:
                kf.bound_high(lo_sell + 1.5)
        s_kf = kf.s if kf.s is not None else mid
        gf = self.gf
        s_g = gf.mean() if gf.c is not None else mid
        if GRID_REGIME and gf.c is not None:
            pv = gf.p_vol()
            if pv > GRID_P_ON:
                self.reg_t = now
                self.volatile = True
            elif self.volatile and pv < GRID_P_OFF and now - self.reg_t >= REG_HOLD_US:
                self.volatile = False
        elif self.reg > REG_ON:
            self.reg_t = now
            self.volatile = True
        elif self.volatile and self.reg < REG_OFF and now - self.reg_t >= REG_HOLD_US:
            self.volatile = False
        s_hat = (self.s_bid + self.s_ask) / 2 if self.s_bid and self.s_ask else mid

        # flow (kept up to date in both regimes)
        self.flow *= math.exp((self.flow_t - now) / FLOW_TAU_US)
        self.flow_t = now
        agg = 0
        for tr in trades:
            if tr.taker_owner == 'market':
                agg += tr.quantity if tr.book_side == 'ask' else -tr.quantity
        self.flow += agg
        half = (bq + aq) / 2
        self.depth = half if self.depth == 0 else self.depth + DEPTH_EMA * (half - self.depth)
        x = self.flow / self.depth
        if FLOW_CHECK and now >= self.fc_next:
            self._flow_check(now, x, mid)

        if self.volatile and VOLATILE_MODE:
            self._volatile(now, bb, ba, s_g if S_VOL_SRC == 2 else s_kf if S_VOL_SRC == 1 else self.s2, agg)
            return

        fair = mid + K_IMB * (bq - aq) / (bq + aq) - K_INV * self.inv
        s_q = s_g if S_QUIET_SRC == 2 else s_kf if S_QUIET_SRC == 1 else self.s2 if FAIR_S2 else s_hat
        fair += K_S * (s_q - mid)
        fair += K_FLOW * self.flow_scale * x
        if fair - ba >= EDGE_TAKE:
            pb = ba
        elif ba - bb >= 2 and fair - (bb + 1) >= EDGE_IN:
            pb = bb + 1
        else:
            pb = bb
            while fair - pb < EDGE:
                pb -= 1
        if bb - fair >= EDGE_TAKE:
            pa = bb
        elif ba - bb >= 2 and (ba - 1) - fair >= EDGE_IN:
            pa = ba - 1
        else:
            pa = ba
            while pa - fair < EDGE:
                pa += 1
        s2 = self.s2
        if s2 - ba >= Q_TAKE:
            pb = max(pb, math.floor(s2 - Q_TAKE_PX))
        elif bb - s2 >= Q_TAKE:
            pa = min(pa, math.ceil(s2 + Q_TAKE_PX))
        self._side(1, pb, QTY)
        self._side(-1, pa, QTY)

    def _volatile(self, now: int, bb: int, ba: int, s: float, agg: int) -> None:
        inv = self.inv
        busy = bool(self.iocs)
        # entries
        if not busy:
            if (s - ba >= (V_FLIP if inv < 0 else V_TAKE) or agg >= V_FOLLOW) and inv < MAX_POS:
                room = MAX_POS - inv - self._resting(1)
                if room > 0:
                    if inv <= 0:
                        self.pos_t = now
                    self._ioc(1, max(ba, math.floor(s - V_TAKE_PX)), room)
                    busy = True
            elif (bb - s >= (V_FLIP if inv > 0 else V_TAKE) or agg <= -V_FOLLOW) and inv > -MAX_POS:
                room = MAX_POS + inv - self._resting(-1)
                if room > 0:
                    if inv >= 0:
                        self.pos_t = now
                    self._ioc(-1, min(bb, math.ceil(s + V_TAKE_PX)), room)
                    busy = True
        # exits
        if inv > 0:
            if not busy and (s - bb < V_KEEP or now - self.pos_t >= V_MAX_HOLD_US):
                self._cancel_side(-1)
                self._cancel_side(1)
                q = min(inv, MAX_POS + inv - self._resting(-1))
                if q > 0:
                    self._ioc(-1, bb, q)
                return
            if V_PASSIVE_EXIT:
                self._side(-1, max(ba, math.ceil(s + V_EXIT_OFF)), inv)
            self._cancel_side(1)
        elif inv < 0:
            if not busy and (ba - s < V_KEEP or now - self.pos_t >= V_MAX_HOLD_US):
                self._cancel_side(1)
                self._cancel_side(-1)
                q = min(-inv, MAX_POS - inv - self._resting(1))
                if q > 0:
                    self._ioc(1, ba, q)
                return
            if V_PASSIVE_EXIT:
                self._side(1, min(bb, math.floor(s - V_EXIT_OFF)), -inv)
            self._cancel_side(-1)
        else:
            self._cancel_side(1)
            self._cancel_side(-1)

    def _ioc(self, sign: int, px: int, q: int) -> None:
        side = Side.BUY if sign > 0 else Side.SELL
        oid = self.send_order(OrderRequest(side=side, price=px, quantity=q, order_type=OrderType.IOC))
        self.iocs[oid] = [sign, q]

    def _resting(self, sign: int) -> int:
        r = 0
        for o in self.orders.values():
            if o[0] == sign:
                r += o[2]
        for o in self.iocs.values():
            if o[0] == sign:
                r += o[1]
        return r

    def _cancel_side(self, sign: int) -> None:
        for oid, o in self.orders.items():
            if o[0] == sign and oid not in self.cancel_sent:
                self._cancel(oid)

    def _flow_check(self, now: int, x: float, mid: float) -> None:
        self.fc_next = now + FLOW_CHECK_STEP_US
        pend = self.fc_pending
        d = self.fc_decay
        while pend and now - pend[0][0] >= FLOW_CHECK_H_US:
            _, x0, m0 = pend.popleft()
            y = mid - m0
            self.fc_sxx = self.fc_sxx * d + x0 * x0
            self.fc_sxy = self.fc_sxy * d + x0 * y
            self.fc_n = self.fc_n * d + 1.0
        pend.append((now, x, mid))
        if self.fc_n < FLOW_CHECK_MIN or self.fc_sxx <= 0:
            self.flow_scale = 0.0
            return
        slope = self.fc_sxy / self.fc_sxx
        self.flow_scale = min(1.0, max(0.0, slope / FLOW_SLOPE_FULL))

    def _scan_new(self, views, sign: int) -> None:
        first = views[0].price
        mx = self.max_id
        lm = len(mx)
        new_mx = mx
        a = S_ALPHA
        ra = REG_ALPHA
        mid = self.mid
        s2 = self.s2
        a2 = S2_ALPHA
        for o in views:
            d = (o.price - first) * sign
            if d > REG_LEVELS:
                if d > GRID_LEVELS:
                    break
                if o.owner == 'market':
                    oid = o.order_id
                    if len(oid) > lm or (len(oid) == lm and oid > mx):
                        self.gf.observe(self.now, o.price, -sign, False)
                        if len(oid) > len(new_mx) or (len(oid) == len(new_mx) and oid > new_mx):
                            new_mx = oid
                continue
            if o.owner != 'market':
                continue
            oid = o.order_id
            if len(oid) > lm or (len(oid) == lm and oid > mx):
                p = o.price
                self.gf.observe(self.now, p, -sign, p >= self.ba - 1 if sign < 0 else p <= self.bb + 1)
                self.reg += ra * (abs(p - mid) - self.reg)
                if sign < 0:
                    r = p + C
                    if p >= self.ba - 1 and r < s2:
                        r = s2
                else:
                    r = p - C
                    if p <= self.bb + 1 and r > s2:
                        r = s2
                s2 += a2 * (r - s2)
                if d <= KF_LEVELS:
                    capped = KF_CENSOR and (p >= self.ba - 1 if sign < 0 else p <= self.bb + 1)
                    self.kf.observe(self.now, p, -sign, capped)
                if d <= S_LEVELS:
                    if sign < 0:
                        self.s_bid = p if not self.s_bid else self.s_bid + a * (p - self.s_bid)
                    else:
                        self.s_ask = p if not self.s_ask else self.s_ask + a * (p - self.s_ask)
                if len(oid) > len(new_mx) or (len(oid) == len(new_mx) and oid > new_mx):
                    new_mx = oid
        self.max_id = new_mx
        self.s2 = s2

    def _side(self, sign: int, target: int, qty: int) -> None:
        resting = 0
        at_target = 0
        keep = []
        for oid, o in self.orders.items():
            if o[0] != sign:
                continue
            resting += o[2]
            if oid in self.cancel_sent:
                continue
            behind = (target - o[1]) * sign
            if behind < 0 or behind > KEEP_LEVELS:
                self._cancel(oid)
                continue
            if behind == 0:
                at_target += o[2]
            else:
                keep.append((behind, oid))
        if at_target >= qty:
            return
        for o in self.iocs.values():
            if o[0] == sign:
                resting += o[1]
        room = MAX_POS - sign * self.inv - resting
        want = qty - at_target
        if room < min(want, MIN_NEW) and keep:
            keep.sort(reverse=True)
            self._cancel(keep[0][1])
            return
        q = min(want, room)
        if q < min(MIN_NEW, qty):
            return
        side = Side.BUY if sign > 0 else Side.SELL
        oid = self.send_order(OrderRequest(side=side, price=target, quantity=q, order_type=OrderType.LIMIT))
        self.orders[oid] = [sign, target, q]

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        oid = event.order_id
        st = event.status
        o = self.orders.get(oid)
        if o is not None:
            if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
                self.inv += o[0] * event.filled_quantity
                o[2] -= event.filled_quantity
                if o[2] <= 0:
                    self._drop(oid)
            elif st == OrderStatus.CANCELED:
                self._drop(oid)
            elif st == OrderStatus.REJECTED and oid not in self.cancel_sent:
                self._drop(oid)
            return
        o = self.iocs.get(oid)
        if o is not None:
            if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
                self.inv += o[0] * event.filled_quantity
                o[1] -= event.filled_quantity
                if o[1] <= 0:
                    self.iocs.pop(oid, None)
            elif st == OrderStatus.CANCELED or st == OrderStatus.REJECTED:
                self.iocs.pop(oid, None)

    def _drop(self, oid: str) -> None:
        self.orders.pop(oid, None)
        self.cancel_sent.discard(oid)
