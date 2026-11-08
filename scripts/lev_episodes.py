"""Per-episode breakdown of inventory overshoots (|inv| > 100) for a strategy with a `_lev` / `_ioc` leverage overlay.

usage: .venv/bin/python scripts/lev_episodes.py logs/r4/lev/v_6.py data/round4_public.csv out.npy
Per episode: start time, trigger edge (fair through the touch), peak |inv|, ms over 100, cash of the overshoot IOCs
marked at the exit, i.e. the gross gain of the extra lots. Prints gain per ms over 100 by trigger-edge bucket.
"""
import sys
from pathlib import Path

import numpy as np

from simulator.main import load_strategy_from_file
from simulator.simulator import ExchangeSimulator, SimulationConfig

s = load_strategy_from_file(sys.argv[1])
ou = s.on_private_order_update
ep = []
cur = {}
last_edge = [0.0]


def lev_w(fair, bb, ba, now):
    last_edge[0] = max(fair - ba, bb - fair)
    lev(fair, bb, ba, now)


def upd(e):
    lev_ioc = e.order_id in s.iocs
    before = s.inv
    ou(e)
    after = s.inv
    t = s.now_us()
    if abs(before) <= 100 < abs(after):
        cur.update(t0=t, edge=last_edge[0], peak=abs(after), cash=0.0, lots=0)
    if cur and lev_ioc and e.fill_price:
        q = after - before
        cur['cash'] -= q * e.fill_price
        cur['lots'] += q
        cur['peak'] = max(cur['peak'], abs(after))
    if cur and abs(before) > 100 >= abs(after):
        ep.append((cur['t0'] / 1e6, cur['edge'], cur['peak'], (t - cur['t0']) / 1e3, cur['cash'], cur['lots'], e.fill_price or 0))
        cur.clear()


if hasattr(s, '_lev'):
    lev = s._lev
    s._lev = lev_w
s.on_private_order_update = upd
r = ExchangeSimulator(s, SimulationConfig(max_position=100, position_breach_penalty_per_second=10000,
                                          max_potential_position=1000)).run(Path(sys.argv[2]))
print(f'pnl {r.pnl:,.0f} penalty {r.breach_penalty:,.0f} episodes {len(ep)}')
a = np.array(ep)
np.save(sys.argv[3], a)
# gross of the overshoot: cash + leftover lots marked at the last fill price
g = a[:, 4] + a[:, 5] * a[:, 6]
cost = a[:, 3] * 10.0
for x, gg in zip(a, g):
    print(f'  t {x[0]:8.2f}  peak {x[2]:4.0f}  ms {x[3]:7.0f}  gross {gg:>9,.0f}  net {gg - 10 * x[3]:>9,.0f}')
for lo, hi in ((0, 8), (8, 10), (10, 14), (14, 20), (20, 1e9)):
    m = (a[:, 1] >= lo) & (a[:, 1] < hi)
    if m.any():
        print(f'edge {lo:>4}-{hi:<5} n {m.sum():4d}  gross {g[m].sum():>10,.0f}  penalty {cost[m].sum():>10,.0f}  net {g[m].sum() - cost[m].sum():>10,.0f}  median ms {np.median(a[m, 3]):6.0f}')
