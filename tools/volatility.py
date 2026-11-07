"""Online realised volatility of a price, in ticks per sqrt(second).

Copy the class into a strategy file (no imports needed besides `math`).
"""

import math


class RealizedVol:
    """Exponentially weighted variance of price changes sampled about once per `sample_us`.

    Call `update(ts_us, price)` on every event. The price is sampled the first time `ts_us` passes the next
    sample time; the change since the previous sample is divided by the elapsed seconds, so irregular gaps are
    handled. The average weight is 1/n for the first samples (fast warm-up) and 1/`n_eff` afterwards.
    `vol()` is None until `min_samples` samples have been seen.
    """

    def __init__(self, sample_us: int = 1_000_000, n_eff: float = 300.0, min_samples: int = 5) -> None:
        self.sample_us = sample_us
        self.n_eff = n_eff
        self.min_samples = min_samples
        self.var = 0.0
        self.n = 0
        self.last_ts = -1
        self.last_price = 0.0

    def update(self, ts_us: int, price: float) -> None:
        if self.last_ts < 0:
            self.last_ts = ts_us
            self.last_price = price
            return
        if ts_us - self.last_ts < self.sample_us:
            return
        dt = (ts_us - self.last_ts) / 1_000_000
        r2 = (price - self.last_price) ** 2 / dt
        self.n += 1
        w = max(1.0 / self.n_eff, 1.0 / self.n)
        self.var += w * (r2 - self.var)
        self.last_ts = ts_us
        self.last_price = price

    def vol(self) -> float | None:
        if self.n < self.min_samples:
            return None
        return math.sqrt(self.var)
