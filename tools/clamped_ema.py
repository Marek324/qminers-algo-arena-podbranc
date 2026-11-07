"""Fast estimate of the hidden price S from new limit orders, aware of the generator's no-cross clamp.

Copy the class into a strategy file (no imports needed).

The generator places a new buy at floor(S - X) and a sell at ceil(S + X), but never lets one cross the book:
a buy that would cross is put at best_ask - 1 (a sell at best_bid + 1). So a buy at best_ask - 1 or above
only says S is at least p + c. During a jump the far side's new orders are often out of view (more than
20 ticks away), and those capped orders are the only sign of the jump.

One EMA over all new orders (both sides): reading p + c for a buy, p - c for a sell; a capped reading may only
move the estimate toward the cap side, never back. On round2_public, alpha 0.25-0.5 and c 0.8-1.1 give the
best R^2 for the next 0.25 s mid move (about 0.12 in volatile spells, 0.13 in quiet ones); medians of the last
N readings did worse. It reacts within a few orders, which is what an IOC taker needs; for passive quoting the
smoother per-side averages of imbalance_mm worked better.

Use: `observe(price, sign, best_bid, best_ask)` for each new market limit order (sign +1 buy, -1 sell; touch of
market orders after the insert), then read `s` (None until the first order). Cost: O(1).
"""


class ClampedPriceEMA:
    """EMA of new-order prices corrected by +-c, treating orders at the no-cross cap as one-sided bounds."""

    def __init__(self, alpha: float = 0.25, c: float = 1.1) -> None:
        self.alpha = alpha
        self.c = c
        self.s = None

    def observe(self, price: int, sign: int, best_bid: int, best_ask: int) -> None:
        if sign > 0:
            r = price + self.c
            capped = price >= best_ask - 1
        else:
            r = price - self.c
            capped = price <= best_bid + 1
        s = self.s
        if s is None:
            self.s = r
            return
        if capped and (r - s) * sign < 0:
            return  # a capped buy can't say S is lower (a capped sell can't say it is higher)
        self.s = s + self.alpha * (r - s)
