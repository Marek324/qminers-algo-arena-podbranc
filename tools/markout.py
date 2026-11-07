"""Running markout of our own fills: how much each lot made against the mid a fixed time after the fill.

Copy the class into a strategy file (needs `from collections import deque`).
"""

from collections import deque


class FillMarkout:
    """Exponentially weighted markout per lot, in ticks, at `horizon_us` after each fill.

    `add_fill(ts_us, sign, price, qty)` records a fill (sign +1 buy, -1 sell). `update(ts_us, mid)` must be called
    on every event: fills older than the horizon are scored as sign * (mid - price) and folded into the average,
    weighted by their size. `value()` is None until `min_lots` lots have been scored.
    """

    def __init__(self, horizon_us: int = 1_000_000, n_eff_lots: float = 20_000.0, min_lots: float = 500.0) -> None:
        self.horizon_us = horizon_us
        self.n_eff_lots = n_eff_lots
        self.min_lots = min_lots
        self.pending: deque = deque()
        self.avg = 0.0
        self.lots = 0.0

    def add_fill(self, ts_us: int, sign: int, price: float, qty: int) -> None:
        self.pending.append((ts_us, sign, price, qty))

    def update(self, ts_us: int, mid: float) -> None:
        pending = self.pending
        while pending and pending[0][0] + self.horizon_us <= ts_us:
            _, sign, price, qty = pending.popleft()
            self.lots += qty
            w = min(1.0, qty / min(self.lots, self.n_eff_lots))
            self.avg += w * (sign * (mid - price) - self.avg)

    def value(self) -> float | None:
        if self.lots < self.min_lots:
            return None
        return self.avg
