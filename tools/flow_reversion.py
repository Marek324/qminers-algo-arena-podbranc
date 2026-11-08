"""Slow-flow mean reversion: online prediction of the mid move over the next h from slow market-order flow.

Copy the class and these imports into a strategy file:
    import math
    from collections import deque

Signal: x = EWMA of signed aggressive market volume (time constant tau_us; + buyers lifting the ask) divided by a
depth scale the caller passes (e.g. an EMA of the touch size), so it is unitless across tapes.
Learning: every step_us it stores (t, x, mid); once h_us has passed it regresses y = mid(t + h) - mid(t) on x through
the origin, with exponential memory mem_us. beta is used only after `min_n` (decayed) samples, and with only_rev only
when it is negative (mean reversion), so a tape without the effect gets 0.
Round 3 (round3_public, analysis report): 10 s flow has Pearson -0.10 with the next 5 s mid move and -0.12 with the
next 30 s; its extreme deciles see +2.6 / -2.2 ticks over the next 5 s.
Use: `add(now_us, signed_qty)` for each market taker trade, then `pred = step(now_us, mid, scale)` once per event:
the expected mid move (ticks) over the next h.
Cost: O(1) amortized per call (the pending queue holds h_us / step_us samples).
"""

import math
from collections import deque


class FlowReversion:
    """Online slope of the future mid move on slow signed flow. See the module docstring."""

    def __init__(self, tau_us: int = 10_000_000, h_us: int = 5_000_000, step_us: int = 100_000,
                 mem_us: int = 3_000_000_000, min_n: float = 3000.0, only_rev: bool = True) -> None:
        self.tau_us = tau_us
        self.h_us = h_us
        self.step_us = step_us
        self.min_n = min_n
        self.only_rev = only_rev
        self.decay = math.exp(-step_us / mem_us)
        self.flow = 0.0
        self.t = 0
        self.pend: deque = deque()
        self.next = 0
        self.sxx = 0.0
        self.sxy = 0.0
        self.n = 0.0
        self.beta = 0.0

    def _decay_to(self, now_us: int) -> None:
        if now_us > self.t:
            self.flow *= math.exp((self.t - now_us) / self.tau_us)
            self.t = now_us

    def add(self, now_us: int, signed_qty: float) -> None:
        self._decay_to(now_us)
        self.flow += signed_qty

    def step(self, now_us: int, mid: float, scale: float) -> float:
        self._decay_to(now_us)
        x = self.flow / scale if scale > 0 else 0.0
        if now_us >= self.next:
            self.next = now_us + self.step_us
            pend = self.pend
            d = self.decay
            while pend and now_us - pend[0][0] >= self.h_us:
                _, x0, m0 = pend.popleft()
                self.sxx = self.sxx * d + x0 * x0
                self.sxy = self.sxy * d + x0 * (mid - m0)
                self.n = self.n * d + 1.0
            pend.append((now_us, x, mid))
            b = self.sxy / self.sxx if self.n >= self.min_n and self.sxx > 0 else 0.0
            self.beta = min(b, 0.0) if self.only_rev else b
        return self.beta * x
