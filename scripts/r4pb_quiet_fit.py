"""Quiet-regime forecast study: which features add to the strategy's own fair (q3 weights) at 0.5/1/5 s.

Fits on the first half, tests on the second. Prints base R2 and the R2 gain of base + each single feature, then
the full linear coefficients (in fair-price ticks per feature unit).
usage: .venv/bin/python scripts/r4pb_quiet_fit.py logs/features/r4_q3_hmm.npz
"""
import sys
import numpy as np

z = np.load(sys.argv[1]); D = {c: z['data'][:, i] for i, c in enumerate(z['cols'])}
n = len(D['t']); mid = (D['bb'] + D['ba']) / 2; eps = 1e-9
def lag(x, k):
    o = np.full_like(x, np.nan); o[k:] = x[k:] - x[:-k]; return o
def lead(x, k):
    o = np.full_like(x, np.nan); o[:-k] = x[k:] - x[:-k]; return o
def ema_dev(x, k):
    a = 1 / k; e = np.empty_like(x); e[0] = x[0]
    for i in range(1, len(x)): e[i] = e[i-1] + a * (x[i] - e[i-1])
    return x - e
depth = np.maximum(D['depth'], 1.0)
F = {
    'gap': D['s'] - mid, 'imb': (D['bq'] - D['aq']) / (D['bq'] + D['aq'] + eps),
    'imb2': (D['bq2'] - D['aq2']) / (D['bq2'] + D['aq2'] + eps), 'flow_used': D['flow_scale'] * D['flow'] / depth,
    'flow': D['flow'] / depth, 'agg_fast': D['agg_fast'] / depth, 'agg_slow': D['agg_slow'] / depth,
    'spread': D['ba'] - D['bb'], 'touch_skew': (D['s'] - D['bb']) - (D['ba'] - D['s']),
    'dmid_0.1': lag(mid, 5), 'dmid_0.5': lag(mid, 25), 'dmid_2': lag(mid, 100), 'dmid_10': lag(mid, 500),
    'ds_0.5': lag(D['s'], 25), 'ds_2': lag(D['s'], 100), 'mom1s': ema_dev(mid, 50), 'mom10s': ema_dev(mid, 500),
    'inv': D['inv'], 'P': D['P'], 'pv': D['pv'], 'upd_imb': D['n_new_b'] - D['n_new_a'],
}
own = 0.9 * F['gap'] + 1.0 * F['flow_used'] + 0.15 * F['imb']
names = list(F); X = np.column_stack([F[k] for k in names])
half = D['t'] < D['t'][n // 2]; quiet = D['pv'] < 0.3
for h in (0.5, 1.0, 5.0):
    y = lead(mid, int(h / 0.02))
    ok = ~np.isnan(y) & np.all(~np.isnan(X), axis=1) & quiet & (np.abs(y) < 60)
    tr, te = ok & half, ok & ~half
    def fit(cols):
        A = np.column_stack(cols + [np.ones(len(y))])
        b = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
        p = A[te] @ b
        return 1 - np.var(y[te] - p) / np.var(y[te]), b
    r0, _ = fit([own])
    gains = sorted(((fit([own, F[k]])[0] - r0, k) for k in names), reverse=True)
    rl, bl = fit([F[k] for k in names])
    print(f'h {h}s quiet n_test {te.sum()}: base R2 {r0:.4f}  full lin {rl:.4f}')
    print('   +1 feature gains:', ', '.join(f'{k} {g:+.4f}' for g, k in gains[:8]))
    print('   lin coefs:', ', '.join(f'{k} {c:+.3f}' for k, c in zip(names, bl) if abs(c) > 1e-3))

print('\nsingle-feature coefs on top of base (h 0.5 / 1 s):')
for h in (0.5, 1.0):
    y = lead(mid, int(h / 0.02))
    ok = ~np.isnan(y) & np.all(~np.isnan(X), axis=1) & quiet & (np.abs(y) < 60)
    tr, te = ok & half, ok & ~half
    out = []
    for k in ('imb', 'imb2', 'agg_fast', 'flow_used', 'dmid_0.1', 'touch_skew', 'gap', 'agg_slow'):
        A = np.column_stack([own, F[k], np.ones(len(y))])
        b = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
        out.append(f'{k}: own {b[0]:+.2f} feat {b[1]:+.3f}')
    print(f'  h {h}:', ' | '.join(out))
rec = D['agg_fast']; print('agg_fast std', np.nanstd(F['agg_fast'][quiet]), 'flow_used std', np.nanstd(F['flow_used'][quiet]), 'imb std', np.nanstd(F['imb'][quiet]))
