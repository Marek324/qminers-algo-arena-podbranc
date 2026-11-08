"""Compare teams' PnL curves from a raw leaderboard dump (pnlSeries every 25 s).

usage: .venv/bin/python scripts/r4pb_lb_compare.py logs/r4pb/lb/r3.json [our team] [rival]
"""
import json
import sys
import numpy as np

lb = json.load(open(sys.argv[1]))
us = sys.argv[2] if len(sys.argv) > 2 else 'Podbranč'
rv = sys.argv[3] if len(sys.argv) > 3 else lb[0]['teamName']
S = {}
for r in lb:
    s = r.get('pnlSeries') or []
    if len(s) > 100:
        S[r['teamName']] = np.array([p['pnl'] for p in s], float)
t = np.array([p['elapsedS'] for p in lb[0]['pnlSeries']], float)
n = min(len(v) for v in S.values())
for k in S:
    S[k] = S[k][:n]
t = t[:n]
D = {k: np.diff(v) for k, v in S.items()}
top = [r['teamName'] for r in lb[:8] if r['teamName'] in S]
print(f'{n} points, step {t[1]-t[0]:.0f}s, end {t[-1]:.0f}s')
# volatility proxy: cross-team dispersion of 25 s increments (median abs over top-20 teams)
names = [r['teamName'] for r in lb[:20] if r['teamName'] in D]
disp = np.median(np.abs(np.array([D[k] for k in names])), axis=0)
hi = disp > np.quantile(disp, 0.8)
print(f'busy intervals (top 20% by median |dPnL| across top-20 teams): {hi.sum()} of {len(hi)}')
print(f'{"team":<28}{"PnL":>11}{"corr w us":>10}{"beta":>6}{"busy":>10}{"calm":>10}{"worst25s":>10}{"maxDD":>10}{"1st half":>10}{"2nd half":>10}')
du = D[us]
for k in top + ([us] if us not in top else []):
    d = D[k]; c = np.corrcoef(d, du)[0, 1]; b = np.dot(d, du) / np.dot(du, du)
    cum = S[k]; dd = np.max(np.maximum.accumulate(cum) - cum)
    h = n // 2
    print(f'{k[:27]:<28}{S[k][-1]:>11,.0f}{c:>10.2f}{b:>6.2f}{d[hi].sum():>10,.0f}{d[~hi].sum():>10,.0f}{d.min():>10,.0f}{dd:>10,.0f}{S[k][h]:>10,.0f}{S[k][-1]-S[k][h]:>10,.0f}')
gap = S[rv] - S[us]
print(f'\ngap {rv} - {us} every 1000 s:', ' '.join(f'{gap[int(i)]/1e3:+.0f}k' for i in np.arange(0, n, 1000 / (t[1] - t[0]))), f'end {gap[-1]/1e3:+.0f}k')
g = np.diff(gap)
o = np.argsort(g)
print('gap built in the 10 best 25 s intervals for the rival:', f'{g[o[-10:]].sum():+,.0f}', ' 10 worst:', f'{g[o[:10]].sum():+,.0f}', ' rest:', f'{g[o[10:-10]].sum():+,.0f}')
print('gap per interval when we gain >0 vs we lose:', f'{g[du > 0].sum():+,.0f} / {g[du <= 0].sum():+,.0f}')
for lab, m in (('our dPnL quintiles', du),):
    q = np.quantile(m, [0.2, 0.4, 0.6, 0.8])
    b = np.digitize(m, q)
    print(lab, [f'{g[b == i].sum()/1e3:+.0f}k' for i in range(5)], ' (rival - us gap per quintile of our increment, low->high)')
q = np.quantile(disp, [0.2, 0.4, 0.6, 0.8]); b = np.digitize(disp, q)
print('market activity quintiles', [f'{g[b == i].sum()/1e3:+.0f}k' for i in range(5)], ' (gap, calm->busy)')
