"""Bayesian grid filter for the hidden price S and the generator's quiet/volatile regime.

Copy the class (and `import math`, `import numpy as np`, `from scipy import stats`) into a strategy file.

Model (the arena generator, docs/DATA_ANALYSIS.md; X fitted on round2_public, docs/RESEARCH.md):
- Two regimes. New limit orders sit X from S: buy at floor(S - X), sell at ceil(S + X), clamped so they never
  cross the book. X ~ shifted gamma(k, loc, theta) with its own parameters per regime.
- S diffuses with variance q per second (q per regime); the regime switches at rates `to_vol` / `to_quiet` per second.
- Each aggressive market order pushes S by push_k * lots**push_pow in its direction (0 turns it off).
- A market buy that traded at P proves S >= P - 1.5 (IOCs are priced at round(S + 1)); sells mirror it.

State: two arrays over the same grid of S values (spacing `h` ticks, `half` points each side of the integer centre
`c`), one per regime; together they sum to 1. The likelihood of an order at price p under a regime is
F(S - p) - F(S - p - 1) for a buy (F = CDF of X) and F(S - p) for a clamped buy (only a bound), read from
precomputed tables by slicing. An order that no point of the grid explains is skipped as an outlier; `surprises`
of them in a row (S jumped beyond the grid) restart the grid around the last one. A trade bound beyond the edge of
the grid restarts it just past the bound.

Use: `trade(ts_us, sign, lots, price)` per side per event, `observe(ts_us, price, sign, capped)` per new market limit
order (sign +1 buy, -1 sell), then `mean()` and `p_vol()`. Cost: a handful of numpy operations on arrays of
2 * half + 1 points per call.
"""

import math

import numpy as np
from scipy import stats


class HiddenPriceGrid:
    """Joint grid filter over (regime, S). See the module docstring."""

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
