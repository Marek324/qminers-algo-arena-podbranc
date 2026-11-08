"""Infer rivals' trading style from leaderboard PnL curves (25 s) against the private tape's price path.

usage: .venv/bin/python scripts/lb_style.py logs/r4pb/lb/r3.json docs/results3/pnl.csv
The probe held +100 lots, so its pnl / 100 is the mid path (+ constant). Per team, regress the 25 s PnL change on:
dm (signed mid move: average exposure in lots), path (sum of |1 s moves|: earning from activity), const.
"""
import json
import sys

import numpy as np
import pandas as pd

lb = json.load(open(sys.argv[1]))
pr = pd.read_csv(sys.argv[2])
mid = np.interp(np.arange(0, 12001), pr.elapsed_s.values, pr.pnl.values / 100.0)
g = np.arange(0, 12001, 25)
dm = np.diff(mid[g])
d1 = np.abs(np.diff(mid))
path = np.array([d1[a:a + 25].sum() for a in g[:-1]])
X = np.c_[np.ones(len(dm)), dm, path]
print(f'{"team":<26}{"PnL":>10}{"const/25s":>10}{"lots exp":>9}{"$/tick path":>12}{"R2":>6}  calm(path<q25) busy(path>q75)  neg%')
q25, q75 = np.quantile(path, [0.25, 0.75])
for r in lb[:12]:
    s = r.get('pnlSeries') or []
    p = np.array([x['pnl'] for x in s], float)
    if len(p) < len(g):
        continue
    d = np.diff(p[:len(g)])
    b, res = np.linalg.lstsq(X, d, rcond=None)[:2]
    r2 = 1 - res[0] / np.sum((d - d.mean()) ** 2)
    print(f'{r["teamName"][:25]:<26}{r["pnl"]:>10,.0f}{b[0]:>10,.0f}{b[1]:>9.1f}{b[2]:>12.1f}{r2:>6.2f}  {d[path < q25].sum():>12,.0f} {d[path > q75].sum():>12,.0f}  {np.mean(d < 0):.1%}')
print(f'path per 25 s: q25 {q25:.0f}  median {np.median(path):.0f}  q75 {q75:.0f} ticks')
