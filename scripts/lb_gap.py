"""Where does a rival's lead over us build up? Leaderboard PnL curves (25 s), intervals ranked by market activity.

usage: .venv/bin/python scripts/lb_gap.py logs/r4pb/lb/r2.json [rival] [us]
Activity = median |dPnL| of the top-20 teams in that interval (all teams move together on jumps and storms).
"""
import json
import sys

import numpy as np

lb = json.load(open(sys.argv[1]))
rv = sys.argv[2] if len(sys.argv) > 2 else lb[0]['teamName']
us = sys.argv[3] if len(sys.argv) > 3 else 'Podbranč'
S = {r['teamName']: np.array([p['pnl'] for p in r['pnlSeries']], float) for r in lb if len(r.get('pnlSeries') or []) > 100}
n = min(len(S[k]) for k in S)
D = {k: np.diff(v[:n]) for k, v in S.items()}
top = [r['teamName'] for r in lb[:20] if r['teamName'] in D]
act = np.median(np.abs(np.array([D[k] for k in top])), axis=0)
gap = D[rv] - D[us]
print(f'{rv} - {us}: total gap {gap.sum():,.0f} over {len(gap)} intervals of 25 s')
o = np.argsort(-act)
for lo, hi in ((0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.0)):
    idx = o[int(lo * len(o)):int(hi * len(o))]
    print(f'activity top {lo:>4.0%}-{hi:<4.0%} n {len(idx):4d}  {rv[:10]} {D[rv][idx].sum():>10,.0f}  us {D[us][idx].sum():>10,.0f}  gap {gap[idx].sum():>10,.0f}  median team {np.median([D[k][idx].sum() for k in top]):>10,.0f}')
# the biggest single intervals for us (losses) and what the rival did there
w = np.argsort(D[us])[:8]
print('our 8 worst intervals: us', D[us][w].round(), ' rival', D[rv][w].round())
w = np.argsort(-gap)[:8]
print('8 biggest gap intervals: gap', gap[w].round(), ' activity rank pct', (np.argsort(o)[w] / len(o) * 100).round())
