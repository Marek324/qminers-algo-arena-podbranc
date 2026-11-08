"""Jump detector from missing market order ids and clamped new orders.

Copy the class into a strategy file (needs `from collections import deque`).

When the hidden price S jumps 20+ ticks (round2/round3), new limit orders on the jump side land more than the
20 visible ticks away (we never see them) and new orders on the other side are clamped one tick inside the stale
touch. Market order ids rise by one per insert, so an id never seen (not in the book, not a trade's taker) is a
deep order or a non-trading IOC: 0.3% of inserts normally, about half during a jump.

Signal (down jump): at least `n_missing` missing ids and a clamped sell (exactly best bid + 1, improving the ask)
within `window_us`, and no other visible new bid; up jump mirrors it. On round3_public with n_missing 3:
7 of 12 jumps, 1 false alarm, but it fires 5-25 ms after the jump starts, when the stale side has mostly gone.

Use: `seen(order_id, now)` for every new market order seen in the book and every market trade's taker id;
`new_order(sign, price, prev_bb, prev_ba, now)` for each new visible market limit order (sign +1 ask, -1 bid,
book before the event); then `check(now)` returns -1 (down jump), +1 (up) or 0. Cost: O(1) amortised.
"""

from collections import deque


class IdGapJump:
    def __init__(self, window_us: int = 25_000, n_missing: int = 3) -> None:
        self.window_us = window_us
        self.n_missing = n_missing
        self.hi = 0
        self.gaps: deque = deque()
        self.cl_d: deque = deque()   # clamped sells: down signal
        self.cl_u: deque = deque()
        self.nb: deque = deque()     # other visible new bids / asks
        self.na: deque = deque()

    def seen(self, order_id: str, now: int) -> None:
        if not order_id.isdigit():
            return
        k = int(order_id)
        if k <= self.hi:
            return
        if self.hi:
            for _ in range(min(k - self.hi - 1, 5)):
                self.gaps.append(now)
        self.hi = k

    def new_order(self, sign: int, price: int, prev_bb: int, prev_ba: int, now: int) -> None:
        if sign > 0:
            if price == prev_bb + 1 and price < prev_ba:
                self.cl_d.append(now)
            else:
                self.na.append(now)
        else:
            if price == prev_ba - 1 and price > prev_bb:
                self.cl_u.append(now)
            else:
                self.nb.append(now)

    def check(self, now: int) -> int:
        lo = now - self.window_us
        for q in (self.gaps, self.cl_d, self.cl_u, self.nb, self.na):
            while q and q[0] < lo:
                q.popleft()
        if len(self.gaps) < self.n_missing:
            return 0
        if self.cl_d and not self.nb:
            return -1
        if self.cl_u and not self.na:
            return 1
        return 0
