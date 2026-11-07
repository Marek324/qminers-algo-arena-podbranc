"""Replay a tape through the simulator with a passive recorder; save top-of-book and trades as .npz.

usage: uv run python scripts/record_book.py data/x.csv out.npz
"""
import sys
from pathlib import Path
import numpy as np
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.strategy import BaseStrategy

class Rec(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.b = []; self.tr = []
    def on_order_book_change(self, s, trades):
        if s.bids and s.asks:
            bb, ba = s.bids[0].price, s.asks[0].price
            bq = sum(o.quantity for o in s.bids if o.price == bb)
            aq = sum(o.quantity for o in s.asks if o.price == ba)
            bq2 = sum(o.quantity for o in s.bids if o.price >= bb - 4)
            aq2 = sum(o.quantity for o in s.asks if o.price <= ba + 4)
            self.b.append((s.timestamp_us, bb, ba, bq, aq, bq2, aq2))
        for t in trades:
            self.tr.append((t.timestamp_us, t.price, t.quantity, 1 if t.book_side == 'ask' else -1))

r = Rec()
ExchangeSimulator(r, SimulationConfig()).run(Path(sys.argv[1]))
np.savez(sys.argv[2], book=np.array(r.b, dtype=np.int64), trades=np.array(r.tr, dtype=np.int64))
print(len(r.b), len(r.tr))
