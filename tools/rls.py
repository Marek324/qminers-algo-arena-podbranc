"""Recursive least squares with exponential forgetting, for an online-learned short-term price prediction.

Copy the class into a strategy file (no imports needed).

Use: `update(x, y)` with a feature list x (include a constant 1.0) and the realised target y, `predict(x)` for
the current features. To learn "mid change H ahead" without look-ahead, queue (t, x, mid) samples and call
update only once H has passed (see strategies/alpha_as_mm.py). Forgetting `lam` per update: memory about
1 / (1 - lam) updates. Cost: O(n^2) per update, O(n) per prediction (n = len(x), keep it under ~10).
"""


class RLS:
    def __init__(self, n: int, lam: float = 0.99995, p0: float = 100.0) -> None:
        self.n = n
        self.lam = lam
        self.w = [0.0] * n
        self.P = [[(p0 if i == j else 0.0) for j in range(n)] for i in range(n)]
        self.count = 0

    def predict(self, x) -> float:
        w = self.w
        return sum(w[i] * x[i] for i in range(self.n))

    def update(self, x, y: float) -> None:
        n = self.n
        P = self.P
        Px = [sum(P[i][j] * x[j] for j in range(n)) for i in range(n)]
        den = self.lam + sum(x[i] * Px[i] for i in range(n))
        k = [v / den for v in Px]
        e = y - self.predict(x)
        for i in range(n):
            self.w[i] += k[i] * e
        il = 1.0 / self.lam
        for i in range(n):
            ki = k[i]
            row = P[i]
            for j in range(n):
                row[j] = (row[j] - ki * Px[j]) * il
        self.count += 1
