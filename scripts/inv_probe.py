"""Sample a strategy's state every 10 ms: t, mid (market touch), s_hat (`s`), pv (`pv`), inventory (`inv`).

usage: .venv/bin/python scripts/inv_probe.py strategies/<x>.py data/<tape>.csv logs/features/<out>.npz
"""
import sys
from pathlib import Path

import numpy as np

from simulator.main import load_strategy_from_file
from simulator.simulator import ExchangeSimulator, SimulationConfig

strat = load_strategy_from_file(sys.argv[1])
cb = strat.on_order_book_change
rows = []
nxt = [0]


def wrapped(snap, trades):
    cb(snap, trades)
    now = snap.timestamp_us
    if now >= nxt[0]:
        nxt[0] = now + 10_000
        rows.append((now / 1e6, strat.mid, strat.s, strat.pv, strat.inv))


strat.on_order_book_change = wrapped
res = ExchangeSimulator(strat, SimulationConfig()).run(Path(sys.argv[2]))
print('pnl', res.pnl)
a = np.array(rows)
np.savez_compressed(sys.argv[3], t=a[:, 0], mid=a[:, 1], s=a[:, 2], pv=a[:, 3], inv=a[:, 4])
