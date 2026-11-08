"""Can a non-linear model beat the hmm strategy's own fair price at forecasting the next mid move?

Reads the .npz from scripts/record_hmm_features.py (20 ms grid). Fits on the first half of the tape, tests on the
second, separately for volatile (pv > 0.5) and quiet samples, at several horizons:
- base: the strategy's fair - mid (its own forecast), as a 1-feature linear fit;
- lin: linear model on all features;
- gbm: HistGradientBoosting on all features.
Also prints a trading proxy for the volatile taker: the mean move beyond the touch when the forecast says the
book is at least TH ticks stale (buy at the ask, value at the mid h later).

usage: .venv/bin/python scripts/fit_hmm_gbm.py logs/features/r3_hmm.npz
"""
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

z = np.load(sys.argv[1])
D = {c: z['data'][:, i] for i, c in enumerate(z['cols'])}
n = len(D['t'])
mid = (D['bb'] + D['ba']) / 2
spread = D['ba'] - D['bb']
eps = 1e-9


def lag(x, k):
    o = np.full_like(x, np.nan)
    o[k:] = x[k:] - x[:-k]
    return o


def lead(x, k):
    o = np.full_like(x, np.nan)
    o[:-k] = x[k:] - x[:-k]
    return o


depth = np.maximum(D['depth'], 1.0)
F = {
    'gap': D['s'] - mid,
    'spread': spread,
    'imb': (D['bq'] - D['aq']) / (D['bq'] + D['aq'] + eps),
    'imb2': (D['bq2'] - D['aq2']) / (D['bq2'] + D['aq2'] + eps),
    'flow': D['flow'] / depth,
    'flow_used': D['flow_scale'] * D['flow'] / depth,
    'agg_fast': D['agg_fast'] / depth,
    'agg_slow': D['agg_slow'] / depth,
    'n_trades': D['n_trades'],
    'pv': D['pv'],
    'P': D['P'],
    'cq': D['cq'],
    'cv': D['cv'],
    'upd_b': D['n_new_b'],
    'upd_a': D['n_new_a'],
    'dmid_0.1': lag(mid, 5),
    'dmid_0.5': lag(mid, 25),
    'dmid_2': lag(mid, 100),
    'ds_0.1': lag(D['s'], 5),
    'ds_0.5': lag(D['s'], 25),
    'gap_lag_0.1': lag(D['s'] - mid, 5),
    'touch_skew': (D['s'] - D['bb']) - (D['ba'] - D['s']),
}
names = list(F)
X = np.column_stack([F[k] for k in names])
# the strategy's own forecast (volatile settings of hmm_617_fit: V_K_S 0.8, V_K_FLOW 1.2; quiet K_S .75, K_FLOW 1.0, K_IMB .15)
own_v = 0.8 * F['gap'] + 1.2 * F['flow_used']
own_q = 0.75 * F['gap'] + 1.0 * F['flow_used'] + 0.15 * F['imb']
half = D['t'] < D['t'][n // 2]
vol = D['pv'] > 0.5

for h_s in (0.25, 0.5, 1.0):
    y = lead(mid, int(h_s / 0.02))
    ok = ~np.isnan(y) & np.all(~np.isnan(X), axis=1) & (np.abs(y) < 60)
    for lab, R, own in (('volat', vol, own_v), ('quiet', ~vol, own_q)):
        tr = ok & R & half
        te = ok & R & ~half
        Y = y
        # base
        A = np.column_stack([own[tr], np.ones(tr.sum())])
        b = np.linalg.lstsq(A, Y[tr], rcond=None)[0]
        p_base = b[0] * own[te] + b[1]
        r2 = lambda p: 1 - np.var(Y[te] - p) / np.var(Y[te])  # noqa: E731
        # linear
        A = np.column_stack([X[tr], np.ones(tr.sum())])
        bl = np.linalg.lstsq(A, Y[tr], rcond=None)[0]
        p_lin = np.column_stack([X[te], np.ones(te.sum())]) @ bl
        g = HistGradientBoostingRegressor(max_depth=4, max_iter=200, learning_rate=0.05, min_samples_leaf=200,
                                          l2_regularization=1.0)
        g.fit(X[tr], Y[tr])
        p_gbm = g.predict(X[te])
        print(f'h {h_s}s {lab}: n_test {te.sum()}  R2 base {r2(p_base):.4f}  lin {r2(p_lin):.4f}  gbm {r2(p_gbm):.4f}')
        if lab == 'volat':
            ask = D['ba'][te]
            bid = D['bb'][te]
            fut = (mid + Y)[te]
            m0 = mid[te]
            for name, p in (('base', p_base), ('gbm', p_gbm)):
                row = []
                for th in (0.5, 1.0, 2.0):
                    buy = (m0 + p) - ask >= th
                    sell = bid - (m0 + p) >= th
                    gain = np.r_[fut[buy] - ask[buy], bid[sell] - fut[sell]]
                    row.append(f'th{th}: n {len(gain)} {np.mean(gain) if len(gain) else 0:+.2f}')
                print(f'     take proxy {name}: ' + '  '.join(row))
