"""Online linear regressions, O(1) or O(k^2) per update, for use inside a strategy.

COPY-PASTE TOOL: copy the class you need, plus the imports below, into the
strategy file. Do not import this module from a strategy (only one .py file is
submitted). Every class is standalone and uses only the standard library and
builtins allowed by the submission checker.

    RollingLinReg   y = a + b*x over the last `window` samples (exact OLS)
    EwLinReg        y = a + b*x with exponential forgetting in x (e.g. time)
    RLS             y = w . features, recursive least squares with forgetting

Feed x in seconds (`now_us() / 1e6`), not raw microseconds: smaller numbers keep
the sums well conditioned.
"""

import math
from collections import deque


class RollingLinReg:
    """Exact OLS of y on x over the last `window` samples.

    Running sums make each update O(1). Sums are taken relative to an anchor and
    rebuilt from the buffer every `window` updates, so float error cannot drift.

        reg = RollingLinReg(200)
        reg.update(t_s, mid)
        if reg.ready(): slope = reg.slope
    """

    def __init__(self, window: int, min_n: int = 3) -> None:
        self.window = window
        self.min_n = max(2, min_n)
        self.buf: deque = deque()
        self.x0 = 0.0
        self.y0 = 0.0
        self.sx = self.sy = self.sxx = self.sxy = self.syy = 0.0
        self.since_rebuild = 0

    def _rebuild(self) -> None:
        self.x0, self.y0 = self.buf[0]
        self.sx = self.sy = self.sxx = self.sxy = self.syy = 0.0
        for x, y in self.buf:
            dx, dy = x - self.x0, y - self.y0
            self.sx += dx
            self.sy += dy
            self.sxx += dx * dx
            self.sxy += dx * dy
            self.syy += dy * dy
        self.since_rebuild = 0

    def update(self, x: float, y: float) -> None:
        if not self.buf:
            self.x0, self.y0 = x, y
        self.buf.append((x, y))
        dx, dy = x - self.x0, y - self.y0
        self.sx += dx
        self.sy += dy
        self.sxx += dx * dx
        self.sxy += dx * dy
        self.syy += dy * dy
        if len(self.buf) > self.window:
            ox, oy = self.buf.popleft()
            dx, dy = ox - self.x0, oy - self.y0
            self.sx -= dx
            self.sy -= dy
            self.sxx -= dx * dx
            self.sxy -= dx * dy
            self.syy -= dy * dy
        self.since_rebuild += 1
        if self.since_rebuild >= self.window:
            self._rebuild()

    @property
    def n(self) -> int:
        return len(self.buf)

    def ready(self) -> bool:
        return len(self.buf) >= self.min_n and self._vx() > 1e-12

    def _vx(self) -> float:
        n = len(self.buf)
        return self.sxx - self.sx * self.sx / n if n else 0.0

    @property
    def slope(self) -> float:
        n = len(self.buf)
        vx = self._vx()
        if n < 2 or vx <= 1e-12:
            return 0.0
        return (self.sxy - self.sx * self.sy / n) / vx

    @property
    def intercept(self) -> float:
        """Intercept in the original (un-anchored) coordinates."""
        n = len(self.buf)
        if not n:
            return 0.0
        b = self.slope
        return self.y0 + self.sy / n - b * (self.x0 + self.sx / n)

    def predict(self, x: float) -> float:
        n = len(self.buf)
        if not n:
            return 0.0
        return self.y0 + self.sy / n + self.slope * (x - self.x0 - self.sx / n)

    @property
    def r2(self) -> float:
        n = len(self.buf)
        vx = self._vx()
        vy = self.syy - self.sy * self.sy / n if n else 0.0
        if n < 2 or vx <= 1e-12 or vy <= 1e-12:
            return 0.0
        cxy = self.sxy - self.sx * self.sy / n
        return min(1.0, cxy * cxy / (vx * vy))

    @property
    def resid_std(self) -> float:
        """Standard deviation of residuals (n - 2 degrees of freedom)."""
        n = len(self.buf)
        if n < 3:
            return 0.0
        vy = self.syy - self.sy * self.sy / n
        sse = max(0.0, vy * (1.0 - self.r2))
        return math.sqrt(sse / (n - 2))

    @property
    def slope_t(self) -> float:
        """t-statistic of the slope; |t| > 2 is roughly significant."""
        n = len(self.buf)
        vx = self._vx()
        s = self.resid_std
        if n < 3 or vx <= 1e-12 or s <= 1e-12:
            return 0.0
        return self.slope / (s / math.sqrt(vx))


class EwLinReg:
    """Weighted OLS of y on x where a sample's weight halves every `half_life` units of x.

    Decay is driven by x itself (usually time in seconds), so irregular event
    spacing is handled correctly. O(1) per update, no buffer.

        reg = EwLinReg(half_life=30.0)
        reg.update(t_s, mid)
        drift_per_s = reg.slope
    """

    def __init__(self, half_life: float, min_weight: float = 3.0) -> None:
        self.lam = math.log(2.0) / half_life
        self.min_weight = min_weight
        self.started = False
        self.x_last = 0.0
        self.y0 = 0.0
        # sums of w, w*dx, w*dy, w*dx^2, w*dx*dy, w*dy^2 where dx = x - x_last
        self.sw = self.sx = self.sy = self.sxx = self.sxy = self.syy = 0.0

    def update(self, x: float, y: float) -> None:
        if not self.started:
            self.started = True
            self.x_last = x
            self.y0 = y
        dt = x - self.x_last
        if dt > 0:
            d = math.exp(-self.lam * dt)
            # decay, then shift the origin from x_last to x (dx_old -> dx_old - dt)
            sw, sx = self.sw * d, self.sx * d
            sxx, sxy = self.sxx * d, self.sxy * d
            self.sy *= d
            self.syy *= d
            self.sxx = sxx - 2.0 * dt * sx + dt * dt * sw
            self.sxy = sxy - dt * self.sy
            self.sx = sx - dt * sw
            self.sw = sw
            self.x_last = x
        dy = y - self.y0
        self.sw += 1.0
        self.sy += dy
        self.syy += dy * dy
        # dx = 0 for the new sample, so sx, sxx, sxy are unchanged

    def ready(self) -> bool:
        return self.sw >= self.min_weight and self._vx() > 1e-12

    def _vx(self) -> float:
        return self.sxx - self.sx * self.sx / self.sw if self.sw > 0 else 0.0

    @property
    def weight(self) -> float:
        """Effective number of samples."""
        return self.sw

    @property
    def slope(self) -> float:
        vx = self._vx()
        if vx <= 1e-12:
            return 0.0
        return (self.sxy - self.sx * self.sy / self.sw) / vx

    def predict(self, x: float) -> float:
        if self.sw <= 0:
            return 0.0
        mx, my = self.sx / self.sw, self.sy / self.sw
        return self.y0 + my + self.slope * (x - self.x_last - mx)

    @property
    def r2(self) -> float:
        if self.sw <= 0:
            return 0.0
        vx = self._vx()
        vy = self.syy - self.sy * self.sy / self.sw
        if vx <= 1e-12 or vy <= 1e-12:
            return 0.0
        cxy = self.sxy - self.sx * self.sy / self.sw
        return min(1.0, cxy * cxy / (vx * vy))


class RLS:
    """Recursive least squares: y ~ w . features, with forgetting factor `lam`.

    Pure-Python lists; O(k^2) per update, fine for k up to ~6 on every tick.
    Effective memory is about 1 / (1 - lam) updates (lam=0.999 -> ~1000).
    Include a constant 1.0 in the features if you want an intercept.

        rls = RLS(k=3, lam=0.999)
        pred = rls.predict([1.0, imbalance, spread])   # before the outcome is known
        rls.update([1.0, imbalance, spread], future_return)
    """

    def __init__(self, k: int, lam: float = 0.999, delta: float = 100.0) -> None:
        self.k = k
        self.lam = lam
        self.w = [0.0] * k
        self.P = [[delta if i == j else 0.0 for j in range(k)] for i in range(k)]
        self.n = 0
        self.err_ew = 0.0  # exponentially weighted mean squared prediction error

    def predict(self, x: list) -> float:
        w = self.w
        return sum(w[i] * x[i] for i in range(self.k))

    def update(self, x: list, y: float) -> float:
        """Update with one sample; returns the a-priori prediction error."""
        k, P, lam = self.k, self.P, self.lam
        Px = [sum(P[i][j] * x[j] for j in range(k)) for i in range(k)]
        denom = lam + sum(x[i] * Px[i] for i in range(k))
        if denom <= 1e-12:
            return 0.0
        g = [v / denom for v in Px]
        err = y - self.predict(x)
        for i in range(k):
            self.w[i] += g[i] * err
        inv_lam = 1.0 / lam
        for i in range(k):
            gi = g[i]
            row = P[i]
            for j in range(k):
                row[j] = (row[j] - gi * Px[j]) * inv_lam
        # keep P symmetric against float drift
        for i in range(k):
            for j in range(i + 1, k):
                v = 0.5 * (P[i][j] + P[j][i])
                P[i][j] = v
                P[j][i] = v
        self.n += 1
        self.err_ew = lam * self.err_ew + (1.0 - lam) * err * err
        return err

    def ready(self, min_n: int = 50) -> bool:
        return self.n >= min_n
