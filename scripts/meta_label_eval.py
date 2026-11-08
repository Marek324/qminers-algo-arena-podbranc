"""Compare take policies on logged candidates (scripts/take_candidates.py) by realised executable markout.

Leave-one-quarter-out on the main tape; optional extra tapes are pure out-of-distribution tests (model trained
on all of the main tape). Policies, scored as the summed per-lot markout of the takes they select:
  rule        the strategy's own edge threshold
  filt-*      rule takes kept only when the model predicts positive markout (meta-labelling: filter, never add)
  all-*       any candidate whose predicted markout clears a cut chosen on the training folds
usage: .venv/bin/python scripts/meta_label_eval.py logs/takecand/round3_public.npz [logs/takecand/round2_public.npz]
       [--y ev5]
"""
import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load(path, y):
    z = np.load(path)
    return z['X'], z[y], z['take'].astype(bool), z['t'], list(z['feats'])


def models():
    return {
        'ridge': lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'hgb': lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                      min_samples_leaf=300, l2_regularization=1.0),
    }


def best_cut(pred, y):
    cuts = np.quantile(pred, np.linspace(0.0, 0.99, 100))
    vals = [y[pred >= c].sum() for c in cuts]
    return cuts[int(np.argmax(vals))]


def evaluate(Xtr, ytr, ttr, Xte, yte, tte):
    out = {'rule': (yte[tte].sum(), int(tte.sum()))}
    for name, mk in models().items():
        m = mk().fit(Xtr[ttr], ytr[ttr])
        keep = tte & (m.predict(Xte) > 0)
        out[f'filt-{name}'] = (yte[keep].sum(), int(keep.sum()))
        m = mk().fit(Xtr, ytr)
        c = best_cut(m.predict(Xtr), ytr)
        sel = m.predict(Xte) >= c
        out[f'all-{name}'] = (yte[sel].sum(), int(sel.sum()))
    return out


def show(title, res):
    print(f'\n{title}')
    for k, (v, n) in res.items():
        print(f'  {k:<12}{v:>12,.0f}  n {n:>8,}  per take {v / max(n, 1):>6.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('main')
    ap.add_argument('extra', nargs='*')
    ap.add_argument('--y', default='ev5')
    a = ap.parse_args()
    X, y, take, t, feats = load(a.main, a.y)
    print(f'{a.main}: {len(y):,} candidates, rule takes {take.sum():,}, label {a.y}')
    q = np.searchsorted(np.quantile(t, [0.25, 0.5, 0.75]), t)
    tot = {}
    for k in range(4):
        te = q == k
        res = evaluate(X[~te], y[~te], take[~te], X[te], y[te], take[te])
        show(f'quarter {k + 1} held out', res)
        for name, (v, n) in res.items():
            tot.setdefault(name, [0, 0])
            tot[name][0] += v
            tot[name][1] += n
    show('all quarters (out of fold)', {k: tuple(v) for k, v in tot.items()})
    for path in a.extra:
        X2, y2, take2, _, _ = load(path, a.y)
        show(f'OOD {path} (trained on all of {a.main})', evaluate(X, y, take, X2, y2, take2))
    m = models()['hgb']().fit(X[take], y[take])
    rng = np.random.default_rng(0)
    base = np.mean((m.predict(X[take]) - y[take]) ** 2)
    imp = []
    for j, f in enumerate(feats):
        Xp = X[take].copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        imp.append((np.mean((m.predict(Xp) - y[take]) ** 2) - base, f))
    print('\nin-sample permutation importance (filter model, MSE increase):')
    for v, f in sorted(imp, reverse=True)[:10]:
        print(f'  {f:<12}{v:>10.3f}')


if __name__ == '__main__':
    main()
