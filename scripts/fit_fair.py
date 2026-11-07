"""Which online features predict the future mid? Fit on the first half of a tape, test on the second.

Reads the .npz from scripts/record_features.py. Prints out-of-sample R^2 for the current adaptive_mm fair
value, a full linear model, and a gradient-boosting model (how much a non-linear model could add), at
several horizons, plus the linear coefficients.

usage: uv run python scripts/fit_fair.py logs/features/round1.npz [--gbm]
"""
import sys

import numpy as np

z = np.load(sys.argv[1])
D = {c: z['data'][:, i] for i, c in enumerate(z['cols'])}
GRID_MS = 20
n = len(D['t'])
mid = (D['bb'] + D['ba']) / 2
spread = D['ba'] - D['bb']
eps = 1e-9


def lag(x, k):
    out = np.full_like(x, np.nan)
    out[k:] = x[:-k]
    return out


F = {}
for a in ('0.05', '0.1', '0.2', '0.4'):
    sb, sa = D['sb' + a], D['sa' + a]
    F['d' + a] = (sb + sa) / 2 - mid
F['xb0.1'] = D['sb0.1'] - D['bb']
F['xa0.1'] = D['sa0.1'] - D['ba']
F['imb'] = (D['bq'] - D['aq']) / (D['bq'] + D['aq'] + eps)
F['imb2'] = (D['bq2'] - D['aq2']) / (D['bq2'] + D['aq2'] + eps)
F['nimb'] = (D['nb'] - D['na']) / (D['nb'] + D['na'] + eps)
for k in ('f100', 'f300', 'f1s'):
    F[k] = D[k] / D['depth']
    F['sw_' + k] = D['sw_' + k] / D['depth']
F['ofi'] = D['ofi1s'] / D['depth']
F['newimb'] = (D['nbid_new'] - D['nask_new']) / (D['nbid_new'] + D['nask_new'] + 1)
F['r100'] = mid - lag(mid, 5)
F['r500'] = mid - lag(mid, 25)
F['r2s'] = mid - lag(mid, 100)
F['wide'] = (spread >= 2).astype(float)
F['imb_x_tight'] = F['imb'] * (1 - F['wide'])
F['d_x_wide'] = F['d0.1'] * F['wide']
F['logq'] = np.log((D['bq'] + 1) / (D['aq'] + 1))
F['tmid'] = np.log1p(D['tlast_mid'] / 1000)
F['d_x_tmid'] = F['d0.1'] * F['tmid']
# the current adaptive_mm fair-value offset (without the hidden-price check and inventory)
F['adaptive'] = 0.75 * F['d0.1'] + 0.15 * F['imb'] + 0.8 * D['f100'] / D['depth']

names = [k for k in F if k != 'adaptive']
X = np.column_stack([F[k] for k in names])
half = n // 2
start = 200  # skip warm-up rows
for h_ms in (100, 340, 1000):
    k = h_ms // GRID_MS
    y = np.full(n, np.nan)
    y[:-k] = mid[k:] - mid[:-k]
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    ok[:start] = False
    tr = ok.copy(); tr[half:] = False
    te = ok.copy(); te[:half] = False

    def r2(pred, m):
        return 1 - np.mean((y[m] - pred[m]) ** 2) / np.mean((y[m] - y[m].mean()) ** 2)

    a = F['adaptive']
    b = np.dot(a[tr], y[tr]) / np.dot(a[tr], a[tr])
    print(f'\n=== horizon {h_ms} ms  (train {tr.sum()}, test {te.sum()} rows) ===')
    print(f'adaptive fair offset: test R2 {r2(b * a, te):.4f} (best scale {b:.2f})')
    for single in ('d0.05', 'd0.1', 'd0.2', 'd0.4', 'imb', 'ofi', 'sw_f300'):
        x = F[single]
        bb_ = np.dot(x[tr], y[tr]) / np.dot(x[tr], x[tr])
        print(f'  {single:10s} alone: test R2 {r2(bb_ * x, te):.4f}')
    Xa = np.column_stack([X, np.ones(n)])
    mu, sd = X[tr].mean(0), X[tr].std(0) + eps
    Xs = np.column_stack([(X - mu) / sd, np.ones(n)])
    lam = 1e-3 * tr.sum()
    A = Xs[tr].T @ Xs[tr] + lam * np.eye(Xs.shape[1])
    w = np.linalg.solve(A, Xs[tr].T @ y[tr])
    pred = Xs @ w
    print(f'full linear ({len(names)} features): train R2 {r2(pred, tr):.4f}  test R2 {r2(pred, te):.4f}')
    raw = w[:-1] / sd
    order = np.argsort(-np.abs(w[:-1]))
    print('  top standardized coefs: ' + ', '.join(f'{names[i]} {w[i]:+.3f} (raw {raw[i]:+.3f})' for i in order[:12]))
    if '--gbm' in sys.argv:
        from sklearn.ensemble import HistGradientBoostingRegressor
        idx = np.flatnonzero(tr)[::3]
        g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=31).fit(X[idx], y[idx])
        pg = np.full(n, 0.0)
        pg[te] = g.predict(X[te])
        print(f'gradient boosting: test R2 {r2(pg, te):.4f}')
