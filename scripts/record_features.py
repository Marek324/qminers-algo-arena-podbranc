"""Replay a tape through the simulator with a passive recorder that computes strategy-visible features online.

Samples every GRID_US (default 20 ms) and saves an .npz with one row per sample. Used to fit and compare
fair-value predictors offline (scripts/fit_fair.py).

usage: uv run python scripts/record_features.py data/round1.csv logs/features/round1.npz [grid_us]
"""
import math
import sys
from pathlib import Path

import numpy as np

from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.strategy import BaseStrategy

ALPHAS = (0.05, 0.1, 0.2, 0.4)  # per-order weights of the new-order price averages
S_LEVELS = 2
TAUS = {'f100': 100_000, 'f300': 300_000, 'f1s': 1_000_000}  # aggressive-flow decays
DEPTH_EMA = 0.001

COLS = ['t', 'bb', 'ba', 'bq', 'aq', 'bq2', 'aq2', 'nb', 'na', 'depth',
        *[f'sb{a}' for a in ALPHAS], *[f'sa{a}' for a in ALPHAS],
        *TAUS, *[f'sw_{k}' for k in TAUS], 'ofi1s', 'nbid_new', 'nask_new', 'tlast_mid']


class Rec(BaseStrategy):
    def __init__(self, grid_us: int) -> None:
        super().__init__()
        self.grid = grid_us
        self.next = 0
        self.rows: list[tuple] = []
        self.max_id = ''
        self.sb = [0.0] * len(ALPHAS)
        self.sa = [0.0] * len(ALPHAS)
        self.flow = dict.fromkeys(TAUS, 0.0)
        self.sweep = dict.fromkeys(TAUS, 0.0)
        self.ofi = 0.0
        self.t_last = 0
        self.depth = 0.0
        self.prev = None
        self.new_b = 0.0  # decayed count of new orders joining/improving the bid (100 ms)
        self.new_a = 0.0
        self.last_mid = None
        self.t_mid = 0

    def _scan(self, views, sign):
        first = views[0].price
        top = self.max_id
        for o in views:
            if (o.price - first) * sign > S_LEVELS:
                break
            if o.owner != 'market':
                continue
            oid = o.order_id
            if len(oid) > len(top) or (len(oid) == len(top) and oid > top):
                for i, a in enumerate(ALPHAS):
                    if sign < 0:
                        self.sb[i] = o.price if not self.sb[i] else self.sb[i] + a * (o.price - self.sb[i])
                    else:
                        self.sa[i] = o.price if not self.sa[i] else self.sa[i] + a * (o.price - self.sa[i])
                if o.price == first:
                    if sign < 0:
                        self.new_b += 1
                    else:
                        self.new_a += 1
                if len(oid) > len(self.max_id) or (len(oid) == len(self.max_id) and oid > self.max_id):
                    self.max_id = oid

    def on_order_book_change(self, s, trades):
        if not s.bids or not s.asks:
            return
        ts = s.timestamp_us
        bb, ba = s.bids[0].price, s.asks[0].price
        bq = aq = bq2 = aq2 = nb = na = 0
        for o in s.bids:
            if o.price < bb - 1:
                break
            bq2 += o.quantity
            if o.price == bb:
                bq += o.quantity
                nb += 1
        for o in s.asks:
            if o.price > ba + 1:
                break
            aq2 += o.quantity
            if o.price == ba:
                aq += o.quantity
                na += 1
        dt = ts - self.t_last
        self.t_last = ts
        if dt > 0:
            for k, tau in TAUS.items():
                f = math.exp(-dt / tau)
                self.flow[k] *= f
                self.sweep[k] *= f
            self.ofi *= math.exp(-dt / 1_000_000)
            f = math.exp(-dt / 100_000)
            self.new_b *= f
            self.new_a *= f
        prev = self.prev
        for t in trades:
            if t.taker_owner != 'market':
                continue
            q = t.quantity if t.book_side == 'ask' else -t.quantity
            through = prev is not None and ((t.book_side == 'ask' and t.price > prev[2]) or (t.book_side == 'bid' and t.price < prev[0]))
            for k in TAUS:
                self.flow[k] += q
                if through:
                    self.sweep[k] += q
        if prev is not None:
            pb, pbq, pa, paq = prev
            e = 0
            if bb >= pb:
                e += bq
            if bb <= pb:
                e -= pbq
            if ba <= pa:
                e -= aq
            if ba >= pa:
                e += paq
            self.ofi += e
        self.prev = (bb, bq, ba, aq)
        half = (bq + aq) / 2
        self.depth = half if self.depth == 0 else self.depth + DEPTH_EMA * (half - self.depth)
        self._scan(s.bids, -1)
        self._scan(s.asks, 1)
        mid = bb + ba
        if mid != self.last_mid:
            self.last_mid = mid
            self.t_mid = ts
        if ts >= self.next:
            self.next = ts - ts % self.grid + self.grid
            self.rows.append((ts, bb, ba, bq, aq, bq2, aq2, nb, na, self.depth, *self.sb, *self.sa,
                              *self.flow.values(), *self.sweep.values(), self.ofi, self.new_b, self.new_a, ts - self.t_mid))


def main():
    grid = int(sys.argv[3]) if len(sys.argv) > 3 else 20_000
    r = Rec(grid)
    ExchangeSimulator(r, SimulationConfig()).run(Path(sys.argv[1]))
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, data=np.array(r.rows, dtype=np.float64), cols=np.array(COLS))
    print(out, len(r.rows), 'rows')


if __name__ == '__main__':
    main()
