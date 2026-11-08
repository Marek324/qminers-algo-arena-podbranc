"""Fit the 'all-ridge' take model on the first half of a candidate log; print constants to paste into a strategy.

usage: .venv/bin/python scripts/fit_take_ridge.py logs/takecand/round3_public.npz [--frac 0.5] [--y ev5]
"""
import argparse

import numpy as np
from sklearn.linear_model import Ridge

ap = argparse.ArgumentParser()
ap.add_argument('npz')
ap.add_argument('--frac', type=float, default=0.5)
ap.add_argument('--y', default='ev5')
a = ap.parse_args()
z = np.load(a.npz)
X, y, t = z['X'], z[a.y], z['t']
tr = t <= np.quantile(t, a.frac)
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
m = Ridge(alpha=10.0).fit((X[tr] - mu) / sd, y[tr])
w = m.coef_ / sd
b = m.intercept_ - (w * mu).sum()
pred = X @ w + b
cuts = np.quantile(pred[tr], np.linspace(0, 0.99, 100))
vals = [y[tr][pred[tr] >= c].sum() for c in cuts]
c = cuts[int(np.argmax(vals))]
te = ~tr
print(f'cut {c:.4f}  train sel {np.sum(pred[tr] >= c)} sum {max(vals):.0f} | test sel {np.sum(pred[te] >= c)} '
      f'sum {y[te][pred[te] >= c].sum():.0f}  rule test sum {y[te][z["take"][te] == 1].sum():.0f}')
print('ML_W = [' + ', '.join(f'{v:.6g}' for v in w) + ']')
print(f'ML_B = {b:.6g}')
print(f'ML_CUT = {c:.6g}')
for f, v, s in zip(z['feats'], w, sd):
    print(f'  {f:<12}{v:>10.4f}  (per sd {v * s:>7.3f})')
