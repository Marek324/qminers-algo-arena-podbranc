"""List every jump-sniper fire of a strategy with an `_ioc` / `iocs` overlay: time, side, limit, lots filled, avg price,
and the markout of those lots against the market mid 1 s and 5 s later.

usage: .venv/bin/python scripts/sn_fires.py logs/r4/sn/sn2/v_2.py data/round4_train.csv
"""
import sys
from pathlib import Path

import numpy as np

from simulator.main import load_strategy_from_file
from simulator.simulator import ExchangeSimulator, SimulationConfig

s = load_strategy_from_file(sys.argv[1])
ioc0 = s._ioc
ou = s.on_private_order_update
cb = s.on_order_book_change
fires = []
fill = {}
mids = []


def ioc(sign, price, q):
    n0 = len(s.iocs)
    ioc0(sign, price, q)
    for oid in s.iocs:
        if oid not in fill:
            fill[oid] = [s.now / 1e6, sign, price, q, 0, 0.0, s.inv]
            fires.append(oid)


def upd(e):
    if e.order_id in fill and e.fill_price:
        f = fill[e.order_id]
        f[4] += e.filled_quantity
        f[5] += e.filled_quantity * e.fill_price
    ou(e)


def w(snap, trades):
    cb(snap, trades)
    mids.append((snap.timestamp_us / 1e6, s.mid))


s._ioc = ioc
s.on_private_order_update = upd
s.on_order_book_change = w
r = ExchangeSimulator(s, SimulationConfig(max_position=100, position_breach_penalty_per_second=10000,
                                          max_potential_position=1000)).run(Path(sys.argv[2]))
print(f'pnl {r.pnl:,.0f} penalty {r.breach_penalty:,.0f}  iocs {len(fires)}')
mt = np.array([m[0] for m in mids])
mv = np.array([m[1] for m in mids])
tot = {1: 0.0, 5: 0.0}
for oid in fires:
    t, sg, px, q, got, cash, inv0 = fill[oid]
    if not got:
        print(f'  t {t:8.2f} {"BUY " if sg > 0 else "SELL"} lim {px} q {q:4d}  filled 0   (inv before {inv0})')
        continue
    avg = cash / got
    row = f'  t {t:8.2f} {"BUY " if sg > 0 else "SELL"} lim {px} q {q:4d}  filled {got:4d} @ {avg:8.1f}  inv before {inv0:5d}'
    for h in (1, 5):
        m = mv[min(np.searchsorted(mt, t + h), len(mv) - 1)]
        mo = sg * (m - avg) * got
        tot[h] += mo
        row += f'  mo{h}s {mo:>9,.0f}'
    print(row)
print('total markout 1 s', round(tot[1]), ' 5 s', round(tot[5]))
