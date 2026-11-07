"""Which online features tell the calm/storm regime? Builds causal 1 s features (each also as EWMAs) and scores them
against the HMM's storm state (estimate_model.detect_regimes, 30 s windows; hindsight, so a label only): univariate
AUC for "storm now" and "storm within 30 s", correlation between features, and an out-of-sample logistic regression
(train on the first 60% of the tape, test on the rest).

uv run python scripts/regime_features.py data/round2_public.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from analysis.replay import replay  # noqa: E402
from analysis.tape import load_tape  # noqa: E402
from estimate_model import REGIME_WINDOW_S, detect_jumps, detect_regimes, pd_ffill  # noqa: E402
from tools.regime_filter import RegimeFilter  # noqa: E402

US = 1_000_000


def features(rp, jumps) -> pd.DataFrame:
    t0, t1 = rp.ev_ts[0], rp.ev_ts[-1]
    n = int((t1 - t0) // US)
    sec = lambda ts: ((ts[(ts >= t0) & (ts < t0 + n * US)] - t0) // US).astype(int)  # noqa: E731
    cnt = lambda ts: np.bincount(sec(ts), minlength=n)[:n].astype(float)  # noqa: E731
    lim, ioc = ~rp.od_ioc, rp.od_ioc
    gap = rp.od_cross_gap
    g = t0 + np.arange(n + 1) * US
    idx = np.maximum(np.searchsorted(rp.ev_ts, g, 'right') - 1, 0)
    mid = pd_ffill(rp.mid[idx])
    dm = np.abs(np.diff(mid))
    ev_s = sec(rp.ev_ts)
    m = len(ev_s)
    spread = np.nan_to_num(rp.best_ask - rp.best_bid, nan=5.0)[:m]
    nev = np.maximum(np.bincount(ev_s, minlength=n)[:n], 1)
    f = pd.DataFrame({
        'lims': cnt(rp.od_ts[lim]),
        'far4': cnt(rp.od_ts[lim & np.isfinite(gap) & (gap <= -5)]),
        'far6': cnt(rp.od_ts[lim & np.isfinite(gap) & (gap <= -7)]),
        'absdm': dm,
        'ioc': cnt(rp.od_ts[ioc]),
        'ioc_deep': cnt(rp.od_ts[ioc & np.isfinite(gap) & (gap >= 2)]),
        'spread_gt1': np.bincount(ev_s, weights=(spread > 1).astype(float), minlength=n)[:n] / nev,
        'touch_qty': np.bincount(ev_s, weights=np.nan_to_num((rp.bid_qty + rp.ask_qty)[:m] / 2.0), minlength=n)[:n] / nev,
        'events': nev.astype(float),
    })
    f['wide_at'] = (np.nan_to_num(rp.best_ask - rp.best_bid, nan=5.0)[idx[1:]] > 1).astype(float)
    f['rv5'] = pd.Series(dm**2).rolling(5, min_periods=1).sum().to_numpy()
    f['rv30'] = pd.Series(dm**2).rolling(30, min_periods=1).sum().to_numpy()
    # seconds since the last jump (causal), capped
    jt = np.sort(jumps['t_us'])
    k = np.searchsorted(jt, g[1:], 'right') - 1
    f['since_jump'] = np.where(k >= 0, np.minimum((g[1:] - jt[np.maximum(k, 0)]) / US, 600), 600)
    base = [c for c in f.columns if c not in ('rv5', 'rv30', 'since_jump', 'wide_at', 'lims')]
    for c in base:
        for hl in (2, 10, 30):
            f[f'{c}_ew{hl}'] = f[c].ewm(halflife=hl).mean()
    return f


def main(path: str) -> None:
    rp = replay(load_tape(Path(path)), 10**9)
    jumps = detect_jumps(rp)
    reg = detect_regimes(rp, jumps)
    if not reg:
        print('no regimes found')
        return
    f = features(rp, jumps)
    prob = np.repeat(np.array(reg['storm_prob']), REGIME_WINDOW_S)
    n = min(len(f), len(prob))
    f, storm = f.iloc[:n].reset_index(drop=True), (prob[:n] > 0.5).astype(int)
    ahead = pd.Series(storm[::-1]).rolling(30, min_periods=1).max().to_numpy()[::-1]  # storm within the next 30 s
    ahead = np.r_[ahead[1:], ahead[-1]]
    onset = ((ahead == 1) & (storm == 0)).astype(int)  # calm now, storm within 30 s
    calm_now = storm == 0
    rows = []
    for c in f.columns:
        x = f[c].to_numpy()
        a_now = roc_auc_score(storm, x)
        a_on = roc_auc_score(onset[calm_now], x[calm_now]) if onset[calm_now].any() else np.nan
        rows.append({'feature': c, 'AUC storm now': a_now, 'AUC onset (calm now, storm <30s)': a_on})
    tab = pd.DataFrame(rows)
    tab['|AUC-0.5| now'] = (tab['AUC storm now'] - 0.5).abs()
    print(f'{path}: {n} s, storm share {storm.mean():.2f}, onsets {int(((storm[1:] == 1) & (storm[:-1] == 0)).sum())}')
    print(tab.sort_values('|AUC-0.5| now', ascending=False).round(3).head(30).to_string(index=False))

    top = tab.sort_values('|AUC-0.5| now', ascending=False).feature.head(12).tolist()
    print('\nSpearman correlation among the top features:')
    print(f[top].corr(method='spearman').round(2).to_string())

    cut = int(0.6 * n)
    for name, cols in (('top 12', top), ('ew10 only', [c for c in f.columns if c.endswith('_ew10')]),
                       ('far + spread + ioc (ew10)', ['far4_ew10', 'far6_ew10', 'spread_gt1_ew10', 'ioc_ew10', 'absdm_ew10'])):
        X = np.log1p(np.maximum(f[cols].to_numpy(), 0))
        sc = StandardScaler().fit(X[:cut])
        lr = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(X[:cut]), storm[:cut])
        p = lr.predict_proba(sc.transform(X[cut:]))[:, 1]
        auc = roc_auc_score(storm[cut:], p) if storm[cut:].any() and not storm[cut:].all() else np.nan
        acc = ((p > 0.5) == storm[cut:]).mean()
        print(f'\nlogistic [{name}] out-of-sample: AUC {auc:.3f}, accuracy {acc:.3f} (storm share in test {storm[cut:].mean():.2f})')
        print('  weights:', dict(zip(cols, lr.coef_[0].round(2))))
    js = np.sort(((jumps['t_us'] - rp.ev_ts[0]) // US).astype(int))
    onset_s = [js[i] for i in range(len(js)) if (i == 0 and js[i] > 90) or (i > 0 and js[i] - js[i - 1] >= 90)]
    filter_check(f, storm, reg, onset_s)


def filter_check(f: pd.DataFrame, storm: np.ndarray, reg: dict, onset_s=None) -> None:
    """Fit RegimeFilter's per-state rates on the first 60%, run it over the rest, report AUC, delay and false alarms."""
    n = len(f)
    cut = int(0.6 * n)
    tr = slice(0, cut)
    s_tr = storm[tr].astype(bool)
    rate = lambda c: (float(f[c].to_numpy()[tr][~s_tr].mean()), float(f[c].to_numpy()[tr][s_tr].mean()))  # noqa: E731
    far, ioc, wide, lims = rate('far4'), rate('ioc'), rate('wide_at'), rate('lims')
    print(f'\nRegimeFilter rates fitted on the first 60%: far {far}, ioc {ioc}, wide {wide}; '
          f'durations calm {reg["calm_dur_s"]:.0f}s storm {reg["storm_dur_s"]:.0f}s')
    test = storm[cut:]
    onsets = np.flatnonzero((test[1:] == 1) & (test[:-1] == 0)) + 1
    print(f'  limits/s calm {lims[0]:.1f} storm {lims[1]:.1f}')
    for temper, use_lim in ((0.25, False), (0.25, True), (0.1, True), (0.05, True)):
        flt = RegimeFilter(reg['calm_dur_s'], reg['storm_dur_s'], far, ioc, wide, temper=temper, p0=0.28,
                           lim_rate=lims if use_lim else None)
        ps_all = np.array([flt.update(a, b, int(c), d if use_lim else -1.0)
                           for a, b, c, d in zip(f['far4'], f['ioc'], f['wide_at'], f['lims'], strict=True)])
        ps = ps_all[cut:]
        if onset_s is not None:
            leads = []
            for o in onset_s:
                if o < cut:
                    continue
                w = ps_all[max(0, o - 60):o + 30] > 0.5
                hit = np.flatnonzero(w)
                leads.append((min(o, 60) - hit[0]) if len(hit) else -30)
            lead_txt = f'lead before the first jump (s, + = early) median {np.median(leads) if leads else float("nan"):.0f}, all {leads}'
        else:
            lead_txt = ''
        delays = []
        for o in onsets:
            hit = np.flatnonzero(ps[o:o + 60] > 0.5)
            delays.append(hit[0] if len(hit) else 60)
        calm = test == 0
        print(f'  temper {temper} lim {use_lim}: {lead_txt}')
        print(f'    AUC {roc_auc_score(test, ps):.3f}, accuracy {((ps > 0.5) == test).mean():.3f}, '
              f'onset detection delay median {np.median(delays) if delays else float("nan"):.0f}s (all {delays}), '
              f'false alarms (P>0.5 while calm) {(ps[calm] > 0.5).mean():.1%}, storm seconds caught {(ps[~calm] > 0.5).mean():.1%}')


if __name__ == '__main__':
    main(sys.argv[1])
