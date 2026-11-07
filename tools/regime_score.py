"""Quiet / volatile regime detector from how far new limit orders land from the mid.

Copy the class into a strategy file (no imports needed).

round2_public switches between two regimes (docs/DATA_ANALYSIS.md): in quiet spells new buys and sells sit about
2 ticks apart and S moves about 1 tick/sqrt(s); in volatile spells they sit about 4.3 ticks apart, IOCs come
1.5x as often and S moves 4-6 ticks/sqrt(s), with most jumps. The switch is abrupt, and the placement width
shows it within a few dozen orders (about 0.2 s), long before realized volatility does.

score = EMA over new orders of |price - mid| (about 1.0 quiet, 2.1 volatile). Volatile above `on`, back to
quiet below `off`, and never sooner than `hold_us` after the last reading above `on`. Defaults are the ones
regime_mm uses (flat optimum 1.6-2.0 for `on`). A jump also trips it, since orders then land far from the mid.

Use: `observe(price, mid)` for each new market limit order within ~8 ticks of its own best price, then
`update(now_us)` once per event and read `volatile`. Cost: O(1).
"""


class RegimeScore:
    """Hysteresis switch on the EMA of new orders' distance from the mid."""

    def __init__(self, alpha: float = 0.03, on: float = 1.6, off: float = 1.35, hold_us: int = 3_000_000) -> None:
        self.alpha = alpha
        self.on = on
        self.off = off
        self.hold_us = hold_us
        self.score = 1.0
        self.volatile = False
        self.last_on_us = 0

    def observe(self, price: int, mid: float) -> None:
        self.score += self.alpha * (abs(price - mid) - self.score)

    def update(self, now_us: int) -> bool:
        if self.score > self.on:
            self.last_on_us = now_us
            self.volatile = True
        elif self.volatile and self.score < self.off and now_us - self.last_on_us >= self.hold_us:
            self.volatile = False
        return self.volatile
