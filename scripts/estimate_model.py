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
def measure(path: Path, name: str, rep: dict) -> GenParams:
    t = load_tape(path)
    rp = replay(t, 10**9)
    dur = (rp.ev_ts[-1] - rp.ev_ts[0]) / US
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

    # permanent IOC impact: signed mid move from just before to 30 s after, by size bucket -> move = k * size^beta
    it, side, q = rp.od_ts[ioc], rp.od_side[ioc].astype(float), rp.od_qty[ioc].astype(float)
    before = np.maximum(np.searchsorted(rp.ev_ts, it, 'left') - 1, 0)
    j = np.searchsorted(rp.ev_ts, it + 30 * US, 'right') - 1
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
    p = GenParams(
        name=name, duration_s=dur, limit_rate=float(lim.sum() / dur), ioc_rate=float(ioc.sum() / dur),
        sigma_s=float(_g(rep, 'price', 'vol_ticks_per_sqrt_s', '60s') or 1.0), x_mean=x_mean, x_std=x_std,
        life_mean_s=float(life.mean()) if len(life) else 0.5, cancel_filled=cancel_rows > 0.97, ioc_offset=ioc_offset,
        limit_sizes_file=f'{name}_limit_sizes.npy', ioc_sizes_file=f'{name}_ioc_sizes.npy',
        impact_k=kappa_raw, impact_beta=beta, ioc_mode='S', clamp=True,
    )
    return with_gamma_x(p, x_mean, x_std), {
        'x_eff_mean': float(x_eff.mean()), 'x_eff_std': float(x_eff.std()), 'permanent_impact_30s': perm,
        'impact_by_size': [{'median_size': a, 'move_30s': b} for a, b in zip(bq.tolist(), bm.tolist(), strict=True)],
        'life_cv': float(life.std() / life.mean()) if len(life) else None, 'cancel_rows_per_limit': cancel_rows}


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
        f'sigma {p0.sigma_s:.3f}, impact {p0.impact_k:.4f} x size^{p0.impact_beta:.2f} (permanent impact {extra["permanent_impact_30s"]:+.3f}), '
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

    (PARAM_DIR / f'{name}_fit.json').write_text(json.dumps(fit, indent=1))
    f = lambda v: '-' if v is None else (f'{v:,.0f}' if abs(v) >= 1000 else f'{v:.3g}')  # noqa: E731
    md = [f'# Generator fit: {name}', '', f'Fit error {fit["fit_error"]:.2f} (sum of |difference| / tolerance; ~1 per metric is a close fit).', '',
          '| metric | real | generated |', '|---|---|---|'] + [f'| {r["metric"]} | {f(r["real"])} | {f(r["generated"])} |' for r in rows]
    md += ['', '## Parameters', '', '```json', json.dumps(asdict(p), indent=1), '```']
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
