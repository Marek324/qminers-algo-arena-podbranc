"""Kalman filter for a hidden price S that the market's limit orders are placed around.

Copy the class into a strategy file (no imports needed).

Model (fits the arena generator, see docs/DATA_ANALYSIS.md):
- S is a random walk: between events its variance grows by `q` ticks^2 per second, and every aggressive market
  order of `qty` lots adds `jump * qty**jump_pow` (IOCs push S by about kappa * qty**0.65, so the variance of
  the push is about kappa^2 * qty**1.3). With `push` > 0 the mean is also moved by push * qty**push_pow.
- A new limit buy at price p says S ~ p + m, a new limit sell says S ~ p - m, with observation noise r.
  The offset m (mean distance of new orders from S, plus rounding) and r are both learned online: m from the
  prior distance of each new order to the estimate (buys and sells cancel each other's error), r from the
  innovations.
- Censoring: the generator never lets a new order cross, so a buy sitting at the highest allowed price
  (best ask - 1) only says S is at least p + m. Pass `capped=True` for such an order; it then updates only
  when it raises the estimate.

Use: `trade(ts_us, sign, qty)` once per side per event with the aggressive market volume, then
`observe(ts_us, price, sign, capped)` for each new limit order near the touch (sign +1 buy, -1 sell), then read
`s` (None until the first observation). `bound_low` / `bound_high` clip S to what swept trades prove: a buy
IOC is priced at round(S + 1), so a buy that traded at P means S >= P - 1.5.
Cost: O(1) per call.
"""


class HiddenPriceKalman:
    """1-D Kalman filter for the hidden price S. See the module docstring for the model."""

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
