"""Volatile-regime linear forecast from live-computable features; fit on half 1, test on half 2 (and reverse).

usage: .venv/bin/python scripts/r4pb_vol_fit.py logs/features/r4_q3_hmm.npz
"""
import sys
import numpy as np

z = np.load(sys.argv[1]); D = {c: z['data'][:, i] for i, c in enumerate(z['cols'])}
n = len(D['t']); mid = (D['bb'] + D['ba']) / 2; eps = 1e-9; tt = D['t']
def lead(x, k):
    o = np.full_like(x, np.nan); o[:-k] = x[k:] - x[:-k]; return o
def ema_dev(x, tau):
    e = np.empty_like(x); e[0] = x[0]
    for i in range(1, len(x)):
        a = 1 - np.exp(-(tt[i] - tt[i-1]) / tau); e[i] = e[i-1] + a * (x[i] - e[i-1])
    return x - e
depth = np.maximum(D['depth'], 1.0)
F = {
    'gap': D['s'] - mid,
    'flow': D['flow'] / depth,
    'flow_used': D['flow_scale'] * D['flow'] / depth,
    'agg_slow': D['agg_slow'] / depth,
    'imb': (D['bq'] - D['aq']) / (D['bq'] + D['aq'] + eps),
    'imb2': (D['bq2'] - D['aq2']) / (D['bq2'] + D['aq2'] + eps),
    'touch_skew': (D['s'] - D['bb']) - (D['ba'] - D['s']),
    'mom_0.1': ema_dev(mid, 0.1), 'mom_1': ema_dev(mid, 1.0), 'mom_3': ema_dev(mid, 3.0),
    'smom_0.3': ema_dev(D['s'], 0.3), 'smom_1': ema_dev(D['s'], 1.0),
    'gapxP': (D['s'] - mid) * D['P'],
}
names = list(F); X = np.column_stack([F[k] for k in names])
own = 0.8 * F['gap'] + 1.2 * F['flow_used'] + 0.2 * F['mom_1'] - 0.05 * ema_dev(mid, 3.0) * 0 
vol = D['pv'] >= 0.3
h1 = tt < tt[n // 2]
for h in (0.5, 1.0, 2.0):
    y = lead(mid, int(h / 0.02)); ok = ~np.isnan(y) & np.all(np.isfinite(X), axis=1) & vol & (np.abs(y) < 80)
    for lab, tr, te in (('fit h1->h2', ok & h1, ok & ~h1), ('fit h2->h1', ok & ~h1, ok & h1)):
        def r2(cols):
            A = np.column_stack(cols + [np.ones(n)]); b = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
            p = A[te] @ b; return 1 - np.var(y[te] - p) / np.var(y[te]), b
        r0, b0 = r2([own]); rl, bl = r2([F[k] for k in names])
        print(f'h {h}s {lab}: own R2 {r0:.4f} (scale {b0[0]:.2f})  lin R2 {rl:.4f}')
        if lab == 'fit h1->h2':
            print('    coefs:', ', '.join(f'{k} {c:+.3f}' for k, c in zip(names, bl)))

print('\nclean model: gap, EMA(gap) 0.3/1/3 s, flow_used, mom_1, imb2')
gap = F['gap']
def ema(x, tau):
    return x - ema_dev(x, tau)
G = {'gap': gap, 'gap_e0.3': ema(gap, 0.3), 'gap_e1': ema(gap, 1.0), 'gap_e3': ema(gap, 3.0),
     'flow_used': F['flow_used'], 'mom_1': F['mom_1'], 'imb2': F['imb2']}
gn = list(G); GX = np.column_stack([G[k] for k in gn])
for regime, R in (('volatile', vol), ('quiet', ~vol)):
  for h in (0.5, 1.0, 2.0):
    y = lead(mid, int(h / 0.02)); ok = ~np.isnan(y) & np.all(np.isfinite(GX), axis=1) & R & (np.abs(y) < 80)
    out = []
    for lab, tr, te in (('h1->h2', ok & h1, ok & ~h1), ('h2->h1', ok & ~h1, ok & h1)):
        A = np.column_stack([GX, np.ones(n)]); b = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
        p = A[te] @ b; r = 1 - np.var(y[te] - p) / np.var(y[te])
        A0 = np.column_stack([own, np.ones(n)]); b0 = np.linalg.lstsq(A0[tr], y[tr], rcond=None)[0]
        r0 = 1 - np.var(y[te] - A0[te] @ b0) / np.var(y[te])
        out.append(f'{lab} own {r0:.3f} clean {r:.3f} [' + ' '.join(f'{k} {c:+.2f}' for k, c in zip(gn, b)) + ']')
    print(f'{regime} h {h}s: ' + ' || '.join(out))
