"""Estimate the market-model parameters of a tape: measure, calibrate, fit-check and (optionally) compute the target PnL.

usage:
    uv run python scripts/estimate_model.py data/round2.csv [--target] [--strategies strategies/a.py,strategies/b.py]

Steps (model in analysis/generator.py, background in docs/DATA_ANALYSIS.md):
  1. measure   rates, side split, limit/IOC size distributions, cancel timer, IOC price offsets vs the touch, and whether
               filled orders still get a cancel row; straight from the replayed tape
  2. estimate  X (limit-price distance from the hidden fair price S) from an offline two-sided estimate of S,
               permanent IOC impact by size, and the mid volatility
  3. calibrate grid over X mean/sd, sigma_s and impact_k on short generated tapes, scored against the real tape's
               analysis report (spread mix, volatility, impact, maker markout, touch depth, mid change rate, signal, joins)
  4. fit check a longer generated tape analysed side by side with the real one
  5. target    (--target) mean PnL per hour of an S-aware reference market maker on fresh generated sessions, plus the
               listed strategies on the same sessions

Output: logs/generator/<tape>_params.json (+ size arrays), logs/generator/<tape>_fit.json and a readable
logs/generator/<tape>_fit.md. Needs logs/analysis/<tape>.json (runs the analysis first if it is missing).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import analyze  # noqa: E402
from analysis.generator import PARAM_DIR, GenParams, generate, write_tape  # noqa: E402
from analysis.replay import replay  # noqa: E402
from analysis.report import to_json  # noqa: E402
from analysis.tape import load_tape  # noqa: E402
from analysis.util import US  # noqa: E402

TAPES = PARAM_DIR / 'tapes'


def _g(r, *p):
    for k in p:
        r = r.get(k) if isinstance(r, dict) else None
        if r is None:
            return None
    return r


# metric name -> (getter, scale for the error: relative 'log' or absolute 'abs', tolerance)
METRICS = {
    'spread=1 share': (lambda r: _g(r, 'book', 'spread_share', '1'), 'abs', 0.05),
    'vol 1s (t/√s)': (lambda r: _g(r, 'price', 'vol_ticks_per_sqrt_s', '1s'), 'log', 0.1),
    'impact 60s (t)': (lambda r: _g(r, 'trades', 'impact_ticks', '60s', 'mean_ticks'), 'abs', 0.05),
    'maker markout 1s': (lambda r: _g(r, 'maker', 'markout_ticks', '1s'), 'abs', 0.1),
    'maker markout 0ms': (lambda r: _g(r, 'maker', 'markout_ticks', '0ms'), 'abs', 0.1),
    'touch qty p50': (lambda r: _g(r, 'book', 'touch_qty', 'bid', 'p50'), 'log', 0.2),
    'mid changes/s': (lambda r: _g(r, 'book', 'dynamics', 'mid_changes_per_s'), 'log', 0.15),
    'microprice IC 1s': (lambda r: _g(r, 'signals', 'directional', 'microprice_offset', '1s', 'spearman'), 'abs', 0.05),
    'join share': (lambda r: _g(r, 'orders', 'passive_placement', 'join', 'share'), 'abs', 0.05),
    'trades/s': (lambda r: _g(r, 'trades', 'trades_per_s'), 'log', 0.1),
}


def _series_col(r, col, per_s=False):
    """A column of the report's time series resampled to 1 s: last value per second, or the sum with per_s."""
    s = _g(r, 'series')
    if not s or col not in s.get('columns', []):
        return None
    d = s['data']
    v = d[col] if isinstance(d, dict) else [row[col] if isinstance(row, dict) else row[s['columns'].index(col)] for row in d]
    v = np.asarray(v, dtype=float)
    k = max(1, round(1 / float(s.get('step_s') or 1)))
    if k == 1:
        return v
    n = len(v) // k * k
    return v[:n].reshape(-1, k).sum(1) if per_s else v[k - 1::k]


def _big_moves_per_h(r, thr):
    m = _series_col(r, 'mid')
    if m is None or len(m) < 10:
        return None
    m = pd_ffill(m)
    return float((np.abs(np.diff(m)) >= thr).sum() / (len(m) / 3600)) + 0.5  # +0.5 keeps the log error finite at 0


def _ioc_dispersion(r):
    c = _series_col(r, 'ioc_inserts', per_s=True)
    return None if c is None or c.mean() <= 0 else float(c.var() / c.mean())


# jump metrics (round2): 1 s mid moves of >= 5 and >= 20 ticks per hour, 60 s vol, IOC count dispersion per second
METRICS.update({
    'big 1s moves/h (>=5t)': (lambda r: _big_moves_per_h(r, 5), 'log', 0.25),
    'huge 1s moves/h (>=20t)': (lambda r: _big_moves_per_h(r, 20), 'log', 0.4),
    'vol 60s (t/√s)': (lambda r: _g(r, 'price', 'vol_ticks_per_sqrt_s', '60s'), 'log', 0.1),
    'IOC/s var/mean': (_ioc_dispersion, 'log', 0.15),
    # inserted limit orders per second: keeps the count right when crossing orders are dropped (cross_mode 'drop')
    'limit inserts/s': (lambda r: (_g(r, 'tape', 'limit_inserts') or 0) / max(_g(r, 'tape', 'duration_s') or 1, 1), 'log', 0.03),
})


def metrics(rep: dict) -> dict:
    return {k: f(rep) for k, (f, _, _) in METRICS.items()}


def error(real: dict, gen: dict) -> float:
    err = 0.0
    for k, (_, kind, tol) in METRICS.items():
        a, b = real.get(k), gen.get(k)
        if a is None or b is None:
            continue
        err += (abs(np.log(max(b, 1e-9) / max(a, 1e-9))) if kind == 'log' else abs(b - a)) / tol
    return err


# ---------------------------------------------------------------- 1 + 2: measure and estimate
JUMP_BIN_US = 100_000
JUMP_MIN = 5.0  # a 100 ms mid move of at least this many ticks is a jump (round1's largest 1 s move is 4)


def near_jump(ts, t_jump, before_us, after_us):
    """Mask of times within [t_j - before_us, t_j + after_us] of any jump."""
    if len(t_jump) == 0:
        return np.zeros(len(ts), bool)
    k = np.searchsorted(t_jump, ts + before_us, 'right') - 1
    ok = k >= 0
    out = np.zeros(len(ts), bool)
    out[ok] = ts[ok] - t_jump[k[ok]] <= after_us
    return out


def detect_jumps(rp) -> dict:
    """Jumps in S: 100 ms mid moves of >= JUMP_MIN ticks; consecutive bins merge into one jump. Also estimates the
    calm S volatility (away from jumps), and the excitation after a jump from an event study: extra IOCs and extra
    squared mid moves in the 10 s after it, with one decay time (the mean delay of the extra IOCs)."""
    t0, t1 = rp.ev_ts[0], rp.ev_ts[-1]
    nb = int((t1 - t0) // JUMP_BIN_US)
    edges = t0 + np.arange(nb + 1) * JUMP_BIN_US
    mid = rp.mid[np.maximum(np.searchsorted(rp.ev_ts, edges, 'right') - 1, 0)]
    mid = np.asarray(pd_ffill(mid))
    dm = np.diff(mid)
    big = np.flatnonzero(np.abs(dm) >= JUMP_MIN)
    t_us, size = [], []
    k = 0
    while k < len(big):
        j = k
        while j + 1 < len(big) and big[j + 1] == big[j] + 1 and np.sign(dm[big[j + 1]]) == np.sign(dm[big[k]]):
            j += 1
        t_us.append(edges[big[k]])
        size.append(float(mid[big[j] + 1] - mid[big[k]]))
        k = j + 1
    t_us, size = np.array(t_us, dtype=np.int64), np.array(size)
    dur = (t1 - t0) / US
    out = {'t_us': t_us, 'size': size, 'calm_sigma': None, 'exc_var': 0.0, 'tau': 1.0, 'ioc_boost': 0.0, 'extra_iocs': 0,
           'alpha': float(len(size) / np.log(np.abs(size) / JUMP_MIN).sum()) if len(size) > 5 else 2.5}  # Pareto MLE
    # calm volatility from 1 s mid moves away from jumps
    g = np.arange(0, nb, 10)
    calm_g = ~near_jump(edges[g], t_us, 2 * US, 15 * US)
    r1 = np.diff(mid[g])
    ok = calm_g[1:] & calm_g[:-1]
    if ok.sum() > 100:
        out['calm_sigma'] = float(r1[ok].std())
    if len(t_us) < 5:
        return out
    # event study over the 10 s after each jump (bins 1..100), skipping the jump bins themselves
    ioc_t = rp.od_ts[rp.od_ioc]
    base_ioc = len(ioc_t) / dur * JUMP_BIN_US / US
    nioc = np.bincount(((ioc_t - t0) // JUMP_BIN_US).astype(int), minlength=nb + 1)[:nb]
    jb = ((t_us - t0) // JUMP_BIN_US).astype(int)
    H = np.arange(2, 100)
    ex_ioc = np.array([np.mean([nioc[j + h] for j in jb if j + h < nb]) - base_ioc for h in H])
    total = float(np.clip(ex_ioc, 0, None).sum())
    tau = float((np.clip(ex_ioc, 0, None) * H).sum() / max(total, 1e-9) * JUMP_BIN_US / US) if total > 0 else 1.0
    tau = float(np.clip(tau, 0.3, 10.0))
    base_var = (out['calm_sigma'] or 1.0) ** 2 * JUMP_BIN_US / US
    ex_var = np.array([np.mean([dm[j + h] ** 2 for j in jb if j + h < nb]) - base_var for h in H])
    mean_abs = float(np.mean(np.abs(size)))
    out.update({
        'tau': tau,
        'ioc_boost': total / tau,
        'extra_iocs': int(total * len(t_us)),
        'exc_var': float(max(np.clip(ex_var, 0, None).sum(), 0.0) / (tau * mean_abs)),
    })
    return out


REGIME_WINDOW_S = 30


def detect_regimes(rp, jumps: dict) -> dict | None:
    """Calm/storm regimes: a 2-state HMM (EM) over 30 s windows on three conditionally independent series: jump count
    (Poisson), log calm volatility and log IOC rate (Gaussian). Returns per-state rates measured on the decoded
    windows and the mean stretch durations, or None when one state fits better (BIC) or there are no jumps."""
    from scipy.special import gammaln, logsumexp

    if len(jumps['t_us']) < 10:
        return None
    t0, t1 = rp.ev_ts[0], rp.ev_ts[-1]
    W = REGIME_WINDOW_S * US
    n = int((t1 - t0) // W)
    if n < 20:
        return None
    end = t0 + n * W
    wid = lambda ts: ((ts[(ts >= t0) & (ts < end)] - t0) // W).astype(int)  # noqa: E731
    jc = np.bincount(wid(jumps['t_us']), minlength=n)[:n].astype(float)
    ioc_t = rp.od_ts[rp.od_ioc]
    ic = np.bincount(wid(ioc_t), minlength=n)[:n] / REGIME_WINDOW_S
    g = np.arange(t0, end + 1, US)
    mid = pd_ffill(rp.mid[np.maximum(np.searchsorted(rp.ev_ts, g, 'right') - 1, 0)])
    r = np.diff(mid)
    calm = ~near_jump(g[1:], jumps['t_us'], 1 * US, 5 * US)
    gw = ((g[1:] - t0 - 1) // W).astype(int).clip(0, n - 1)
    vol = np.array([r[(gw == k) & calm].std() if ((gw == k) & calm).sum() > 5 else np.nan for k in range(n)])
    vol = np.where(np.isfinite(vol), vol, np.nanmedian(vol))
    lv, li = np.log(np.maximum(vol, 1e-3)), np.log(np.maximum(ic, 1e-3))

    def emit(lam, mv, sv, mi, si):
        return (jc[:, None] * np.log(lam[None]) - lam[None] - gammaln(jc + 1)[:, None]
                - 0.5 * ((lv[:, None] - mv[None]) / sv[None]) ** 2 - np.log(sv[None])
                - 0.5 * ((li[:, None] - mi[None]) / si[None]) ** 2 - np.log(si[None]))

    lam = np.array([max(np.quantile(jc, 0.3), 0.05), max(np.quantile(jc, 0.9), 1.0)])
    mv, sv = np.quantile(lv, [0.3, 0.9]), np.array([lv.std()] * 2)
    mi, si = np.quantile(li, [0.3, 0.9]), np.array([li.std()] * 2)
    A, pi, ll = np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([0.8, 0.2]), -np.inf
    for _ in range(300):
        lp, lA = emit(lam, mv, sv, mi, si), np.log(A)
        la = np.zeros((n, 2))
        la[0] = np.log(pi) + lp[0]
        for t in range(1, n):
            la[t] = logsumexp(la[t - 1][:, None] + lA, axis=0) + lp[t]
        lb = np.zeros((n, 2))
        for t in range(n - 2, -1, -1):
            lb[t] = logsumexp(lA + (lp[t + 1] + lb[t + 1])[None], axis=1)
        new = float(logsumexp(la[-1]))
        gam = np.exp(la + lb - new)
        xi = sum(np.exp(la[t][:, None] + lA + (lp[t + 1] + lb[t + 1])[None] - new) for t in range(n - 1))
        A, pi, w = xi / xi.sum(1, keepdims=True), gam[0], gam.sum(0)
        lam = np.maximum((gam * jc[:, None]).sum(0) / w, 1e-3)
        mv = (gam * lv[:, None]).sum(0) / w
        sv = np.maximum(np.sqrt((gam * (lv[:, None] - mv[None]) ** 2).sum(0) / w), 0.02)
        mi = (gam * li[:, None]).sum(0) / w
        si = np.maximum(np.sqrt((gam * (li[:, None] - mi[None]) ** 2).sum(0) / w), 0.02)
        if new - ll < 1e-7:
            break
        ll = new
    ll1 = float((jc * np.log(max(jc.mean(), 1e-3)) - jc.mean() - gammaln(jc + 1)).sum()
                - 0.5 * (((lv - lv.mean()) / lv.std()) ** 2).sum() - n * np.log(lv.std())
                - 0.5 * (((li - li.mean()) / li.std()) ** 2).sum() - n * np.log(li.std()))
    bic_gain = (-2 * ll1 + 5 * np.log(n)) - (-2 * new + 14 * np.log(n))
    if bic_gain < 10:
        return None
    storm = int(np.argmax(lam))
    zs = gam[:, storm] > 0.5
    dur = REGIME_WINDOW_S / np.maximum(1 - np.diag(A), 1e-6)
    secs = lambda m: m.sum() * REGIME_WINDOW_S  # noqa: E731
    out = {'bic_gain': float(bic_gain), 'storm_share': float(zs.mean()), 'storm_prob': gam[:, storm].round(4).tolist(),
           'calm_dur_s': float(dur[1 - storm]), 'storm_dur_s': float(dur[storm]),
           'jump_rate_calm': float(jc[~zs].sum() / max(secs(~zs), 1)), 'jump_rate_storm': float(jc[zs].sum() / max(secs(zs), 1)),
           'ioc_rate_calm': float(ic[~zs].mean()), 'ioc_rate_storm': float(ic[zs].mean()) if zs.any() else float(ic.mean()),
           'sigma_calm': float(np.sqrt(np.mean(vol[~zs] ** 2))), 'sigma_storm': float(np.sqrt(np.mean(vol[zs] ** 2))) if zs.any() else 0.0}
    return out


def pd_ffill(a):
    import pandas as pd

    return pd.Series(a).ffill().bfill().to_numpy()


def measure(path: Path, name: str, rep: dict) -> GenParams:
    t = load_tape(path)
    rp = replay(t, 10**9)
    dur = (rp.ev_ts[-1] - rp.ev_ts[0]) / US
    jumps = detect_jumps(rp)
    regimes = detect_regimes(rp, jumps)
    ioc, lim = rp.od_ioc, ~rp.od_ioc
    ins, can = t[t.is_insert], t[~t.is_insert]
    first = dict(zip(ins.OrderId.tolist(), ins.ts_us.tolist(), strict=True))
    life = np.array([ts - first[o] for o, ts in zip(can.OrderId.tolist(), can.ts_us.tolist(), strict=True) if o in first]) / 1e6
    gap = rp.od_cross_gap[ioc]
    off = np.round(gap[np.isfinite(gap)]).astype(int)
    vals, cnt = np.unique(off, return_counts=True)
    ioc_offset = {int(v): float(c / cnt.sum()) for v, c in zip(vals, cnt, strict=True) if c / cnt.sum() > 0.001}
    np.save(PARAM_DIR / f'{name}_limit_sizes.npy', rp.od_qty[lim].astype(np.int32))
    np.save(PARAM_DIR / f'{name}_ioc_sizes.npy', rp.od_qty[ioc].astype(np.int32))
    cancel_rows = len(can) / max(lim.sum(), 1)

    # X: distance of limit prices from an offline (centred, two-sided) estimate of S, minus the ~0.5 tick rounding
    ts, px, sd = rp.od_ts[lim], rp.od_price[lim].astype(float), rp.od_side[lim]
    side_mean = {}
    for side in (1, -1):
        m = sd == side
        cs = np.r_[0, np.cumsum(px[m])]
        a = np.searchsorted(ts[m], ts - 500_000)
        b = np.searchsorted(ts[m], ts + 500_000)
        side_mean[side] = (cs[b] - cs[a]) / np.maximum(b - a, 1)
    x_eff = np.where(sd == 1, (side_mean[1] + side_mean[-1]) / 2 - px, px - (side_mean[1] + side_mean[-1]) / 2)
    x_mean = max(float(x_eff.mean()) - 0.5, 0.05)
    x_std = float(np.sqrt(max(x_eff.var() - 1 / 12, 0.01)))

    # permanent IOC impact: signed mid move from just before to 5 s after, by size bucket -> move = k * size^beta
    # (5 s, not 30 s: the impact is flat by then, and at round2's volatility a 30 s move is mostly noise)
    it, side, q = rp.od_ts[ioc], rp.od_side[ioc].astype(float), rp.od_qty[ioc].astype(float)
    calm = ~near_jump(it, jumps['t_us'], before_us=6 * US, after_us=15 * US)  # jumps would swamp the impact
    it, side, q = it[calm], side[calm], q[calm]
    before = np.maximum(np.searchsorted(rp.ev_ts, it, 'left') - 1, 0)
    j = np.searchsorted(rp.ev_ts, it + 5 * US, 'right') - 1
    move = (rp.mid[j] - rp.mid[before]) * side
    perm = float(np.nanmean(move))
    edges = np.unique(np.quantile(q, [0, .2, .4, .6, .8, .95, 1]))
    bq, bm = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (q >= lo) & (q <= hi) & np.isfinite(move)
        if m.sum() > 50:
            bq.append(float(np.median(q[m])))
            bm.append(float(np.mean(move[m])))
    bq, bm = np.array(bq), np.array(bm)
    pos = bm > 0.01
    if perm > 0.05 and pos.sum() >= 3:
        beta, logk = np.polyfit(np.log(bq[pos]), np.log(bm[pos]), 1)
        beta = float(np.clip(beta, 0.3, 1.0))
        kappa_raw = float(np.exp(logk))
    else:
        beta, kappa_raw = 0.5, 0.0
    if len(jumps['t_us']):
        np.save(PARAM_DIR / f'{name}_jump_sizes.npy', np.abs(jumps['size']).astype(np.float32))
    p = GenParams(
        name=name, duration_s=dur, limit_rate=float(lim.sum() / dur),
        ioc_rate=float((ioc.sum() - jumps['extra_iocs']) / dur),
        sigma_s=jumps['calm_sigma'] or float(_g(rep, 'price', 'vol_ticks_per_sqrt_s', '60s') or 1.0), x_mean=x_mean, x_std=x_std,
        life_mean_s=float(life.mean()) if len(life) else 0.5, cancel_filled=bool(cancel_rows > 0.97), ioc_offset=ioc_offset,
        limit_sizes_file=f'{name}_limit_sizes.npy', ioc_sizes_file=f'{name}_ioc_sizes.npy',
        impact_k=kappa_raw, impact_beta=beta, ioc_mode='S', clamp=True,
        jump_rate=len(jumps['t_us']) / dur, jump_sizes_file=f'{name}_jump_sizes.npy' if len(jumps['t_us']) else '',
        jump_up=float((jumps['size'] > 0).mean()) if len(jumps['t_us']) else 0.5,
        exc_var=jumps['exc_var'], exc_tau=jumps['tau'], ioc_boost=jumps['ioc_boost'], ioc_tau=jumps['tau'],
        jump_alpha=jumps['alpha'], jump_detect=JUMP_MIN,
    )
    if regimes:
        # the per-state rates replace the overall ones; IOCs in a storm are partly the post-jump burst, so the storm
        # rate is kept and the burst is left to the calibration (exc scale)
        p = replace(p, sigma_s=regimes['sigma_calm'], jump_rate=regimes['jump_rate_calm'], ioc_rate=regimes['ioc_rate_calm'],
                    calm_dur_s=regimes['calm_dur_s'], storm_dur_s=regimes['storm_dur_s'], sigma_storm=regimes['sigma_storm'],
                    jump_rate_storm=regimes['jump_rate_storm'], ioc_rate_storm=regimes['ioc_rate_storm'])
    return with_gamma_x(p, x_mean, x_std), {
        'x_eff_mean': float(x_eff.mean()), 'x_eff_std': float(x_eff.std()), 'permanent_impact_5s': perm,
        'impact_by_size': [{'median_size': a, 'move_5s': b} for a, b in zip(bq.tolist(), bm.tolist(), strict=True)],
        'life_cv': float(life.std() / life.mean()) if len(life) else None, 'cancel_rows_per_limit': cancel_rows,
        'regimes': {k: v for k, v in regimes.items() if k != 'storm_prob'} if regimes else None,
        'jumps': {k: v for k, v in jumps.items() if k not in ('t_us', 'size')} | {
            'n': int(len(jumps['t_us'])), 'size_pct_10_50_90_99': np.percentile(np.abs(jumps['size']), [10, 50, 90, 99]).round(1).tolist()
            if len(jumps['t_us']) else []}}


X_LOC = 0.05


def with_gamma_x(p: GenParams, x_mean: float, x_std: float) -> GenParams:
    """X as a shifted gamma with the given mean and sd (the round1 tape analysis found a shifted gamma)."""
    m = max(x_mean - X_LOC, 0.05)
    return replace(p, x_mean=x_mean, x_std=x_std, x_dist='gamma', x_loc=X_LOC, x_a=(m / x_std) ** 2, x_b=x_std**2 / m)


# ---------------------------------------------------------------- 3 + 4: calibrate and check
def _gen_and_measure(args) -> tuple[dict, dict]:
    p, ov, seed, minutes = args
    p = with_gamma_x(replace(p, **ov), ov['x_mean'], ov['x_std'])
    df, _, _ = generate(p, seed=seed, duration_s=minutes * 60)
    path = TAPES / f'calib_{p.name}_{abs(hash(json.dumps(ov, sort_keys=True)))}_{seed}_{os.getpid()}.csv'
    write_tape(df, path)
    try:
        m = metrics(analyze(path))
    finally:
        path.unlink(missing_ok=True)
    return ov, m


def calibrate(p: GenParams, real: dict, workers: int, minutes: float, log) -> tuple[GenParams, list]:
    xs = sorted({round(p.x_mean * f, 3) for f in (0.85, 1.0, 1.15)})
    sds = sorted({round(p.x_std * f, 3) for f in (0.6, 1.0)})
    sig = sorted({round(p.sigma_s * f, 3) for f in (0.85, 1.0)})
    kap = sorted({round(k, 5) for k in ((0.0,) if p.impact_k == 0 else (p.impact_k * 0.6, p.impact_k, p.impact_k * 1.4))})
    if p.storm_dur_s > 0:
        # regimes: sigma scale (both states), impact and the post-jump excitation (0 or half the event-study value,
        # since part of what follows a jump is the storm state itself)
        # and the crossing rule: 'clamp' keeps a crossing limit one tick inside the opposite best, 'drop' skips it (then
        # the true limit rate is higher than the inserted one). round2: drop matches the book's unpredictability.
        grid = [dict(x_mean=a, x_std=p.x_std, sigma_s=round(p.sigma_s * f, 3), sigma_storm=round(p.sigma_storm * f, 3),
                     impact_k=kap[0], exc_var=round(p.exc_var * 0.5, 4), ioc_boost=round(p.ioc_boost * 0.5, 3),
                     cross_mode=cm, limit_rate=round(p.limit_rate * lr, 2))
                for a, f, (cm, lr) in itertools.product(xs, (0.85, 1.0), (('clamp', 1.0), ('drop', 1.2), ('drop', 1.3)))]
    elif p.jump_rate > 0:
        # jumps: the small-jump cutoff (0 = none) and the excitation; X sd and sigma from the measured values
        jmin = (0.0, 2.0, 3.0)
        exc = (round(p.exc_var * 0.5, 4), p.exc_var)
        sig = sorted({round(p.sigma_s * f, 3) for f in (0.6, 0.85)})
        grid = [dict(x_mean=a, x_std=p.x_std, sigma_s=c, impact_k=d, exc_var=e, jump_min=m)
                for a, c, d, e, m in itertools.product(xs, sig, kap, exc, jmin)]
    else:
        grid = [dict(x_mean=a, x_std=b, sigma_s=c, impact_k=d) for a, b, c, d in itertools.product(xs, sds, sig, kap)]
    log(f'calibrating {len(grid)} parameter sets on {minutes:g}-minute generated tapes...')
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_gen_and_measure, [(p, ov, 7, minutes) for ov in grid]))
    scored = sorted(((error(real, m), ov, m) for ov, m in res), key=lambda x: x[0])
    best = scored[0][1]
    return with_gamma_x(replace(p, **best), best['x_mean'], best['x_std']), scored


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('tape', type=Path)
    ap.add_argument('--name', default=None, help='name for the params file (default: tape file name)')
    ap.add_argument('--calib-minutes', type=float, default=15)
    ap.add_argument('--check-minutes', type=float, default=30)
    ap.add_argument('--target', action='store_true', help='also compute the target PnL (slow)')
    ap.add_argument('--strategies', default='', help='comma-separated strategy files to score on the target sessions')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()
    t0 = time.perf_counter()
    log = lambda s: print(f'[{time.perf_counter() - t0:6.0f}s] {s}', flush=True)  # noqa: E731
    name = args.name or args.tape.name.removesuffix('.gz').removesuffix('.csv')
    PARAM_DIR.mkdir(parents=True, exist_ok=True)
    TAPES.mkdir(parents=True, exist_ok=True)

    rep_path = ROOT / 'logs' / 'analysis' / f'{name}.json'
    if not rep_path.exists():
        log(f'no analysis report for {name}; running the analysis first...')
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        rep_path.write_text(to_json(analyze(args.tape)))
    real_rep = json.loads(rep_path.read_text())
    real = metrics(real_rep)

    log('measuring rates, sizes, cancels, IOC pricing, X and impact from the tape...')
    p0, extra = measure(args.tape, name, real_rep)
    log(f'measured: limit {p0.limit_rate:.1f}/s, IOC {p0.ioc_rate:.2f}/s, life {p0.life_mean_s:.3f}s, X ~ ({p0.x_mean:.2f}, {p0.x_std:.2f}), '
        f'sigma {p0.sigma_s:.3f}, impact {p0.impact_k:.4f} x size^{p0.impact_beta:.2f} (permanent impact {extra["permanent_impact_5s"]:+.3f}), '
        f'cancel_filled {p0.cancel_filled}')
    p, scored = calibrate(p0, real, args.workers, args.calib_minutes, log)
    p.save(PARAM_DIR / f'{name}_params.json')
    log(f'best fit: x ({p.x_mean}, {p.x_std}), sigma {p.sigma_s}, impact {p.impact_k} x size^{p.impact_beta:.2f} (error {scored[0][0]:.2f}); saved params')

    log(f'fit check on a {args.check_minutes:g}-minute generated tape...')
    df, _, _ = generate(p, seed=99, duration_s=args.check_minutes * 60)
    check = write_tape(df, TAPES / f'{name}_check.csv')
    gen = metrics(analyze(check))
    rows = [{'metric': k, 'real': real[k], 'generated': gen[k]} for k in METRICS]
    fit = {'name': name, 'params': asdict(p), 'measured': extra, 'fit_error': error(real, gen), 'metrics': rows,
           'calibration_top5': [{'error': e, 'params': ov} for e, ov, _ in scored[:5]]}

    if args.target:
        from analysis.target import compute_target

        strategies = [s for s in args.strategies.split(',') if s]
        log('computing the target PnL (S-aware reference market maker on fresh generated sessions)...')
        fit['target'] = compute_target(name, strategies=strategies, workers=args.workers, log=log)

    (PARAM_DIR / f'{name}_fit.json').write_text(json.dumps(fit, indent=1, default=lambda o: o.item()))
    f = lambda v: '-' if v is None else (f'{v:,.0f}' if abs(v) >= 1000 else f'{v:.3g}')  # noqa: E731
    md = [f'# Generator fit: {name}', '', f'Fit error {fit["fit_error"]:.2f} (sum of |difference| / tolerance; ~1 per metric is a close fit).', '',
          '| metric | real | generated |', '|---|---|---|'] + [f'| {r["metric"]} | {f(r["real"])} | {f(r["generated"])} |' for r in rows]
    md += ['', '## Parameters', '', '```json', json.dumps(asdict(p), indent=1, default=lambda o: o.item()), '```']
    if 'target' in fit:
        tg = fit['target']
        md += ['', '## Target PnL', '', f'S-aware reference market maker: **{tg["target"]["mean_per_hour"]:,.0f} per hour** '
               f'(sd {tg["target"]["sd_per_hour"]:,.0f} across {tg["n_test"]} fresh sessions of {tg["session_minutes"]:g} min), '
               f'setting {tg["oracle_setting"]}.', '']
        for k, v in tg['strategies'].items():
            md.append(f'- {k}: {v["mean_per_hour"]:,.0f} per hour (sd {v["sd_per_hour"]:,.0f})')
    (PARAM_DIR / f'{name}_fit.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    log(f'-> {PARAM_DIR / f"{name}_fit.md"}')


if __name__ == '__main__':
    main()
