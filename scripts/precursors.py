"""Can a regime change or the direction of the next jump be seen coming?

1. Calm stretches: are their lengths memoryless (exponential; then a storm start is unpredictable in principle)?
2. Precursors: per-second causal features averaged over the 60 s (and 10 s) before each storm onset (a jump after
   >= QUIET_S without one) against random calm windows far from any jump; AUC and standardized difference.
3. Direction: does anything before a jump predict its sign (previous jump, net taker flow, new-limit side imbalance,
   drift, the new-order price estimate of S minus the mid)?

uv run python scripts/precursors.py data/round2_public.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from analysis.replay import replay  # noqa: E402
from analysis.tape import load_tape  # noqa: E402
from estimate_model import REGIME_WINDOW_S, detect_jumps, detect_regimes, pd_ffill  # noqa: E402

US = 1_000_000
QUIET_S = 90


def per_second(rp, t) -> pd.DataFrame:
    t0, t1 = rp.ev_ts[0], rp.ev_ts[-1]
    n = int((t1 - t0) // US)
    end = t0 + n * US
    sec = lambda ts: ((ts - t0) // US).astype(int)  # noqa: E731

    def agg(ts, w=None):
        m = (ts >= t0) & (ts < end)
        return np.bincount(sec(ts[m]), weights=None if w is None else w[m], minlength=n)[:n].astype(float)

    lim, ioc = ~rp.od_ioc, rp.od_ioc
    g = t0 + np.arange(n + 1) * US
    mid = pd_ffill(rp.mid[np.maximum(np.searchsorted(rp.ev_ts, g, 'right') - 1, 0)])
    lt, lp, ls, lq = rp.od_ts[lim], rp.od_price[lim].astype(float), rp.od_side[lim].astype(float), rp.od_qty[lim].astype(float)
    it, iside, iq = rp.od_ts[ioc], rp.od_side[ioc].astype(float), rp.od_qty[ioc].astype(float)
    nl = np.maximum(agg(lt), 1)
    lim_mid = mid[np.clip(sec(lt), 0, n - 1)]
    can = t[~t.is_insert].ts_us.to_numpy()
    f = pd.DataFrame({
        'limits': agg(lt),
        'limit_side_imb': agg(lt, ls) / nl,  # +1 all buys
        'limit_size': agg(lt, lq) / nl,
        'limit_px_minus_mid': agg(lt, lp - lim_mid) / nl,  # buys sit below, sells above; the mean leans toward S
        'limit_px_disp': np.sqrt(np.maximum(agg(lt, (lp - lim_mid) ** 2) / nl, 0)),
        'cancels': agg(can),
        'iocs': agg(it),
        'ioc_net_flow': agg(it, iside * iq),
        'ioc_size': agg(it, iq) / np.maximum(agg(it), 1),
        'absdm': np.abs(np.diff(mid)),
        'drift30': pd.Series(np.diff(mid)).rolling(30, min_periods=1).sum().to_numpy(),
        'spread': np.nan_to_num(rp.best_ask - rp.best_bid, nan=5.0)[np.maximum(np.searchsorted(rp.ev_ts, g[1:], 'right') - 1, 0)],
        'touch_qty': np.nan_to_num((rp.bid_qty + rp.ask_qty) / 2.0)[np.maximum(np.searchsorted(rp.ev_ts, g[1:], 'right') - 1, 0)],
    })
    return f


def main(path: str) -> None:
    t = load_tape(Path(path))
    rp = replay(t, 10**9)
    jumps = detect_jumps(rp)
    reg = detect_regimes(rp, jumps)
    f = per_second(rp, t)
    n = len(f)
    t0 = rp.ev_ts[0]
    js = ((jumps['t_us'] - t0) // US).astype(int)
    sz = jumps['size']
    print(f'{path}: {n} s, {len(js)} jumps')

    # 1. calm stretch lengths from the HMM (30 s windows)
    if reg:
        z = (np.array(reg['storm_prob']) > 0.5).astype(int)
        runs, cur, length = {0: [], 1: []}, z[0], 0
        for v in z:
            if v == cur:
                length += 1
            else:
                runs[cur].append(length * REGIME_WINDOW_S)
                cur, length = v, 1
        for k, name in ((0, 'calm'), (1, 'storm')):
            d = np.array(runs[k][1:]) if len(runs[k]) > 2 else np.array(runs[k])
            if len(d) > 3:
                ks = stats.kstest(d / d.mean(), 'expon')
                print(f'  {name} stretches: n {len(d)}, mean {d.mean():.0f}s, CV {d.std() / d.mean():.2f} (exponential = 1), '
                      f'KS vs exponential p {ks.pvalue:.2f}')
    gaps = np.diff(jumps['t_us']) / US
    print(f'  gaps between jumps: CV {gaps.std() / gaps.mean():.2f}; gaps > {QUIET_S}s: {(gaps > QUIET_S).sum()}')

    # 2. precursors: onset = jump after >= QUIET_S quiet
    onsets = [js[i] for i in range(len(js)) if (i == 0 and js[i] > QUIET_S) or (i > 0 and js[i] - js[i - 1] >= QUIET_S)]
    far_from_jump = np.ones(n, bool)
    for j in js:
        far_from_jump[max(0, j - 90):min(n, j + 120)] = False
    rng = np.random.default_rng(0)
    calm_ends = np.flatnonzero(far_from_jump)
    calm_ends = calm_ends[calm_ends > 70]
    print(f'\n  storm onsets (jump after >= {QUIET_S}s quiet): {len(onsets)}; calm reference windows drawn from {len(calm_ends)} s')
    for w, lag in ((60, 1), (10, 1), (3, 0)):
        rows = []
        ref = rng.choice(calm_ends, size=min(400, len(calm_ends)), replace=False)
        for c in f.columns:
            x = f[c].to_numpy()
            on = np.array([x[o - w - lag:o - lag].mean() for o in onsets if o - w - lag >= 0])
            rf = np.array([x[e - w - lag:e - lag].mean() for e in ref])
            if len(on) < 5:
                continue
            y = np.r_[np.ones(len(on)), np.zeros(len(rf))]
            auc = roc_auc_score(y, np.r_[on, rf])
            dz = (on.mean() - rf.mean()) / (rf.std() / np.sqrt(len(on)) + 1e-12)
            rows.append({'feature': c, 'AUC': auc, 'z (onset vs calm)': dz, 'onset mean': on.mean(), 'calm mean': rf.mean()})
        tab = pd.DataFrame(rows).assign(dev=lambda d: (d.AUC - 0.5).abs()).sort_values('dev', ascending=False).drop(columns='dev')
        print(f'\n  window [{-w - lag}s, {-lag}s) before onset vs calm:')
        print(tab.round(3).head(8).to_string(index=False))

    # 3. direction of the next jump
    print('\n  direction of a jump (sign) vs what came before it:')
    sgn = np.sign(sz)
    out = []
    prev = np.r_[0, sgn[:-1]]
    ok = prev != 0
    out.append(('previous jump sign', float(np.mean(sgn[ok] == prev[ok])), int(ok.sum())))
    for c, w in (('ioc_net_flow', 10), ('ioc_net_flow', 2), ('limit_side_imb', 2), ('limit_px_minus_mid', 1), ('drift30', 1)):
        x = f[c].to_numpy()
        pre = np.array([x[max(0, j - w):j].sum() if j > 0 else 0 for j in js])
        m = pre != 0
        out.append((f'sign of {c} over {w}s before', float(np.mean(np.sign(pre[m]) == sgn[m])), int(m.sum())))
    for name, hit, cnt in out:
        se = np.sqrt(0.25 / max(cnt, 1))
        print(f'    {name:<42} hit rate {hit:.2f} (n {cnt}, 0.50 +- {2 * se:.2f} is chance)')


if __name__ == '__main__':
    main(sys.argv[1])
