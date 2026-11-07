"""Does the market switch between regimes? Per window (30 s and 60 s): jump count, calm volatility (jump windows left
out), IOC rate, limit rate, spread>1 share and touch depth; then persistence (autocorrelation, dispersion against a
constant rate) and a 2-state hidden Markov fit per series (Poisson for counts, Gaussian on log for the rest) against a
1-state model by BIC, with the expected state durations.

uv run python scripts/regime_analysis.py data/round2_public.csv [data/round1.csv ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from analysis.replay import replay  # noqa: E402
from analysis.tape import load_tape  # noqa: E402
from estimate_model import detect_jumps, near_jump  # noqa: E402

US = 1_000_000


def windows(rp, t, jumps, w_s: float) -> pd.DataFrame:
    t0, t1 = rp.ev_ts[0], rp.ev_ts[-1]
    W = int(w_s * US)
    n = int((t1 - t0) // W)
    edges = t0 + np.arange(n + 1) * W
    wid = lambda ts: ((ts - t0) // W).astype(int)  # noqa: E731
    cnt = lambda ts: np.bincount(wid(ts[(ts >= t0) & (ts < edges[-1])]), minlength=n)[:n]  # noqa: E731
    # 1 s mid grid for volatility; seconds near a jump are left out of the calm vol
    g = np.arange(t0, edges[-1], US)
    mid = pd.Series(rp.mid[np.maximum(np.searchsorted(rp.ev_ts, g, 'right') - 1, 0)]).ffill().bfill().to_numpy()
    r = np.diff(mid)
    calm = ~near_jump(g[1:], jumps['t_us'], 1 * US, 5 * US)
    gw = wid(g[1:])
    vol = np.array([r[(gw == k) & calm].std() if ((gw == k) & calm).sum() > 5 else np.nan for k in range(n)])
    spread = rp.best_ask - rp.best_bid
    ev_w = wid(rp.ev_ts[rp.ev_ts < edges[-1]])
    out = pd.DataFrame({
        'jumps': cnt(jumps['t_us']),
        'jump_ticks': np.bincount(wid(jumps['t_us'][jumps['t_us'] < edges[-1]]), weights=np.abs(jumps['size'][jumps['t_us'] < edges[-1]]),
                                  minlength=n)[:n] if len(jumps['t_us']) else np.zeros(n),
        'calm_vol': vol,
        'ioc_per_s': cnt(rp.od_ts[rp.od_ioc]) / w_s,
        'limit_per_s': cnt(rp.od_ts[~rp.od_ioc]) / w_s,
        'touch_qty': np.bincount(ev_w, weights=np.nan_to_num((rp.bid_qty + rp.ask_qty)[: len(ev_w)] / 2.0), minlength=n)[:n]
        / np.maximum(np.bincount(ev_w, minlength=n)[:n], 1),
    })
    if spread is not None:
        wide = np.nan_to_num(spread[: len(ev_w)] > 1).astype(float)
        out['spread_gt1'] = np.bincount(ev_w, weights=wide, minlength=n)[:n] / np.maximum(np.bincount(ev_w, minlength=n)[:n], 1)
    return out


def hmm2(x: np.ndarray, kind: str, iters: int = 200):
    """2-state HMM by EM. kind 'poisson' (counts) or 'gauss'. Returns (loglik 2-state, loglik 1-state, params)."""
    x = x[np.isfinite(x)]
    n = len(x)
    if kind == 'poisson':
        lam1 = x.mean()
        ll1 = float((x * np.log(max(lam1, 1e-9)) - lam1 - gammaln(x + 1)).sum())
        lam = np.array([max(np.quantile(x, 0.3), 1e-3), max(np.quantile(x, 0.9), x.mean() * 2, 1e-2)])

        def logp(lam):
            return x[:, None] * np.log(lam[None]) - lam[None] - gammaln(x + 1)[:, None]
    else:
        mu1, sd1 = x.mean(), x.std()
        ll1 = float((-0.5 * ((x - mu1) / sd1) ** 2 - np.log(sd1 * np.sqrt(2 * np.pi))).sum())
        mu = np.quantile(x, [0.25, 0.75])
        sd = np.array([sd1, sd1])

        def logp(par):
            m, s = par
            return -0.5 * ((x[:, None] - m[None]) / s[None]) ** 2 - np.log(s[None] * np.sqrt(2 * np.pi))
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    pi = np.array([0.5, 0.5])
    ll = -np.inf
    for _ in range(iters):
        lp = logp(lam) if kind == 'poisson' else logp((mu, sd))
        la = np.zeros((n, 2))
        la[0] = np.log(pi) + lp[0]
        lA = np.log(A)
        for t in range(1, n):
            la[t] = logsumexp(la[t - 1][:, None] + lA, axis=0) + lp[t]
        lb = np.zeros((n, 2))
        for t in range(n - 2, -1, -1):
            lb[t] = logsumexp(lA + (lp[t + 1] + lb[t + 1])[None], axis=1)
        new_ll = float(logsumexp(la[-1]))
        g = np.exp(la + lb - new_ll)
        xi = np.zeros((2, 2))
        for t in range(n - 1):
            m = la[t][:, None] + lA + (lp[t + 1] + lb[t + 1])[None] - new_ll
            xi += np.exp(m)
        A = xi / xi.sum(1, keepdims=True)
        pi = g[0] / g[0].sum()
        w = g.sum(0)
        if kind == 'poisson':
            lam = np.maximum((g * x[:, None]).sum(0) / w, 1e-4)
        else:
            mu = (g * x[:, None]).sum(0) / w
            sd = np.maximum(np.sqrt((g * (x[:, None] - mu[None]) ** 2).sum(0) / w), 1e-3)
        if new_ll - ll < 1e-6:
            break
        ll = new_ll
    k2, k1 = (2 + 2, 1) if kind == 'poisson' else (4 + 2, 2)
    bic2, bic1 = -2 * new_ll + k2 * np.log(n), -2 * ll1 + k1 * np.log(n)
    state_par = lam if kind == 'poisson' else mu
    order = np.argsort(state_par)
    dur = 1 / np.maximum(1 - np.diag(A), 1e-9)
    occ = g.mean(0)
    return {'bic_gain': float(bic1 - bic2), 'levels': state_par[order].round(3).tolist(),
            'mean_duration_windows': dur[order].round(1).tolist(), 'occupancy': occ[order].round(2).tolist()}


def main() -> None:
    for path in sys.argv[1:]:
        rp = replay(load_tape(Path(path)), 10**9)
        jumps = detect_jumps(rp)
        print(f'\n===== {path}: {len(jumps["t_us"])} jumps (>= 5 ticks / 100 ms)')
        for w in (30, 60):
            df = windows(rp, None, jumps, w)
            print(f'\n-- {w} s windows ({len(df)})')
            rows = []
            for col in df.columns:
                x = df[col].to_numpy(float)
                ok = np.isfinite(x)
                s = pd.Series(x)
                ac = [s.autocorr(L) for L in (1, 2, 5)]
                disp = x[ok].var() / x[ok].mean() if col == 'jumps' and x[ok].mean() > 0 else np.nan
                rows.append({'series': col, 'mean': x[ok].mean(), 'cv': x[ok].std() / max(abs(x[ok].mean()), 1e-9),
                             'ac1': ac[0], 'ac2': ac[1], 'ac5': ac[2], 'var/mean': disp})
            print(pd.DataFrame(rows).round(3).to_string(index=False))
            print('2-state HMM vs 1 state (BIC gain > 10 = strong evidence of regimes):')
            for col, kind in (('jumps', 'poisson'), ('calm_vol', 'gauss'), ('ioc_per_s', 'gauss'), ('spread_gt1', 'gauss')):
                if col not in df or df[col].sum() == 0:
                    continue
                x = df[col].to_numpy(float)
                if kind == 'gauss':
                    x = np.log(np.maximum(x, 1e-4))
                r = hmm2(x, kind)
                lv = r['levels'] if kind == 'poisson' else np.exp(r['levels']).round(3).tolist()
                print(f'  {col:<11} BIC gain {r["bic_gain"]:+8.1f}  levels {lv}  mean duration (windows) {r["mean_duration_windows"]}  occupancy {r["occupancy"]}')
            if w == 60:
                print('cross-correlation of windows (same window):')
                print(df.corr(method='spearman').round(2).to_string())


if __name__ == '__main__':
    main()
