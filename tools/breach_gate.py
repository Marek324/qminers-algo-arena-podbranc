from collections import deque
import math


class BreachGate:
    """Should we go over |I| = 100 right now? Paper-trades every excursion first and says yes only if they pay.

    Going over 100 is renting capacity at a flat PENALTY $/s (10,000), whatever the size of the excess. Taking Q
    extra lots at the touch and giving them back after a hold h pays

        net = Q * move - roundtrip_cost(Q) - PENALTY * (h + latency)

    so the rule is: breach iff E[net | signal] > 0. The gate estimates that expectation from this round's own tape:
    whenever the caller sees a signal (bucket = any small int, e.g. a jump-size class), it calls `open()` with the
    cash it would pay or receive for Q lots now (walking the visible book). When `due()` lists that shadow trade,
    the caller passes the cash of unwinding it in the book as it is then to `close()`. `allow(bucket)` is True once
    the bucket has MIN_N shadow trades whose mean net minus Z standard errors is above zero.

    Nothing is ever sent: the shadow trades only read the book, so the gate costs nothing to run and adapts to a
    new round's parameters. Their fills ignore our own impact and the race for stale lots, so a bucket that passes
    is a candidate to backtest, not a guarantee. On round3_public no bucket of "mid moved >= X ticks in 20 ms" has
    a positive mean at any hold from 2 ms to 1 s (see scripts/lever_oracle.py).
    """

    def __init__(self, hold_us=100_000, latency_us=5_000, penalty_per_s=10_000.0, min_n=30, z=2.0, n_buckets=8):
        self.hold_us = hold_us
        self.latency_us = latency_us
        self.penalty_per_s = penalty_per_s
        self.min_n = min_n
        self.z = z
        self.n = [0] * n_buckets
        self.s = [0.0] * n_buckets
        self.s2 = [0.0] * n_buckets
        self.pending = deque()  # (due_us, key)
        self.open_trades = {}   # key -> (bucket, side, entry_cash)
        self.next_key = 0

    def open(self, now_us, bucket, side, entry_cash):
        """Start a shadow excursion. side +1: bought Q lots for entry_cash; -1: sold Q lots for entry_cash."""
        key = self.next_key
        self.next_key += 1
        self.open_trades[key] = (bucket, side, entry_cash)
        self.pending.append((now_us + self.latency_us + self.hold_us, key))
        return key

    def due(self, now_us):
        """Shadow trades whose hold is over: list of (key, side). Unwind each with close(key, exit_cash)."""
        out = []
        while self.pending and self.pending[0][0] <= now_us:
            key = self.pending.popleft()[1]
            out.append((key, self.open_trades[key][1]))
        return out

    def close(self, key, exit_cash):
        """exit_cash: cash received selling Q (side +1) or paid buying Q back (side -1)."""
        bucket, side, entry_cash = self.open_trades.pop(key)
        gross = exit_cash - entry_cash if side > 0 else entry_cash - exit_cash
        net = gross - self.penalty_per_s * (self.hold_us + self.latency_us) / 1e6
        self.n[bucket] += 1
        self.s[bucket] += net
        self.s2[bucket] += net * net
        return net

    def mean(self, bucket):
        n = self.n[bucket]
        return self.s[bucket] / n if n else 0.0

    def allow(self, bucket):
        n = self.n[bucket]
        if n < self.min_n:
            return False
        m = self.s[bucket] / n
        var = max(self.s2[bucket] / n - m * m, 0.0)
        return m - self.z * math.sqrt(var / n) > 0.0
