"""Take gate: raise a taker's edge threshold when the price filter is inconsistent or the take reverses a recent one.

Copy the class into a strategy file (only `import math` needed).

Three pieces, each off at its off value:
- Confidence (k, z0). Feed every filter reading's innovation and predicted variance (P + r) to `reading`; zbar is the
  EWMA (weight a) of z^2 = innovation^2 / (P + r), about 1 while the filter is consistent with the data, large after
  jumps and in chop. The threshold gets + k * max(0, sqrt(zbar) - z0). Round 3 with round3_strategy's Kalman:
  sqrt(zbar) median 0.58, p90 0.86, p99 1.21, 5.7 right after a 121-tick jump. k = 0: off.
- Hysteresis (rev_edge, rev_us). Call `take_fill(sign, now_us)` on each taker fill (sign +1 buy, -1 sell). A take
  against the last taker fill's direction within rev_us needs rev_edge more. Round 3 (round3_strategy): taker orders
  reversed within 0.5 s lost -3.3 ticks/lot by 5 s, held over 1 s made +1.8 to +5.1. rev_edge = 0: off.
- Edge sizing (size_w, size_min). `cap(excess, max_pos)`: the |position| a take may build when its edge is `excess`
  ticks past the threshold, from size_min * max_pos at the threshold to max_pos at size_w past it. size_w = 0: max_pos.
Use: `th = gate.threshold(sign, base_edge, now_us)`; take if edge >= th, up to `gate.cap(edge - th, MAX_POS)`.
Cost: O(1) per call.
"""

import math


class TakeGate:
    """Confidence- and reversal-aware take threshold. See the module docstring."""

    def __init__(self, k: float = 0.5, z0: float = 1.0, a: float = 0.05, zmax: float = 400.0, rev_edge: float = 1.0,
                 rev_us: int = 500_000, size_w: float = 0.0, size_min: float = 0.5) -> None:
        self.k = k
        self.z0 = z0
        self.a = a
        self.zmax = zmax
        self.rev_edge = rev_edge
        self.rev_us = rev_us
        self.size_w = size_w
        self.size_min = size_min
        self.zbar = 1.0
        self.last_dir = 0
        self.last_us = -10**12

    def reading(self, innov: float, var: float) -> None:
        """One filter reading: innovation (reading - prior estimate) and its predicted variance (P + r)."""
        z2 = innov * innov / var if var > 0 else self.zmax
        self.zbar += self.a * (min(z2, self.zmax) - self.zbar)

    def take_fill(self, sign: int, now_us: int) -> None:
        self.last_dir = sign
        self.last_us = now_us

    def threshold(self, sign: int, base: float, now_us: int) -> float:
        th = base
        if self.k:
            th += self.k * max(0.0, math.sqrt(self.zbar) - self.z0)
        if self.rev_edge and self.last_dir == -sign and now_us - self.last_us < self.rev_us:
            th += self.rev_edge
        return th

    def cap(self, excess: float, max_pos: int) -> int:
        if self.size_w <= 0:
            return max_pos
        f = self.size_min + (1.0 - self.size_min) * excess / self.size_w
        return int(max_pos * min(1.0, f))
