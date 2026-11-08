"""Is a strategy robust or overfit? Backtests it in the real simulator on variants of every tape and reports how stable it is.

usage:
    uv run python scripts/robustness.py strategies/round1/adaptive_mm.py                 # all tapes in data/
    uv run python scripts/robustness.py strategies/x.py data/a.csv data/b.csv --set QUOTE_SIZE=30 --sensitivity all

Tests, per tape (plain .csv or .csv.gz in data/, deduplicated by name):
  full        the whole tape, plus its theoretical max (share of max = how much of the available money you take)
  chunks      the tape cut into --chunks equal stretches of time, each run from a cold start: is the PnL steady over time?
  mirrored    buys become sells and prices are reflected around the median: an exact mirror image of the market.
              A side-symmetric strategy scores exactly the same; a different score means its buy and sell logic differ
              (a hidden long/short bias that may only fit this tape's direction).
  regime      per chunk, PnL against the chunk's price drift and volatility: does it only make money when the
              market moves (or sits still)?
  bootstrap   --boot sessions of --boot-minutes built from random 60 s blocks of the tape (re-joined in price and time):
              a PnL distribution instead of one number. Weakest test: block joins are artificial.
  regimes     (--regime-boot N, round 2 style tapes) calm/storm labels from estimate_model.detect_regimes:
              regime_split = PnL earned in calm and in storm seconds of the full run; storm_heavy / calm_only = N
              bootstrap sessions built only from storm (or only from calm) 60 s blocks. How does PnL scale if the
              scored tape is stormier or calmer than this one?
  synthetic   (--gen NAME) --gen-sessions sessions of --gen-minutes from logs/generator/NAME_params.json
              (analysis/generator.py): fresh markets from the fitted model. Only as good as the model's transfer.
  sensitivity (opt-in) each tunable constant scaled x0.5 ... x1.5 (bools flipped), run on the first and second half of
              every tape. Sharp drops around the current value mean a lucky peak; if the variant that wins on the first
              half does not win on the second (walk-forward), tuning does not transfer.
Across tapes: PnL, PnL per hour and share of the theoretical max side by side.

Every run uses the scoring config (5 ms latency, |inventory| 100 penalty, 1000-lot limit). The strategy file is never
written: overrides (--set and the sensitivity variants) are applied in memory. Output: logs/robustness/<strategy>.json
(plus a short table on stdout); the dashboard's Robustness tab reads it.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.strategies import discover_params, load_strategy, scoring_config  # noqa: E402

OUT_DIR = ROOT / 'logs' / 'robustness'
TAPE_DIR = OUT_DIR / 'tapes'
SCALES = (0.5, 0.75, 0.9, 1.1, 1.25, 1.5)
SCHEMA_VERSION = 1


# ---------------------------------------------------------------- tapes
def find_tapes() -> list[Path]:
    """Every tape in data/: plain CSVs, plus .csv.gz files that have no plain copy."""
    data = ROOT / 'data'
    plain = {p.name: p for p in data.glob('*.csv')}
    for gz in data.glob('*.csv.gz'):
        plain.setdefault(gz.name.removesuffix('.gz'), gz)
    return [plain[k] for k in sorted(plain)]


def tape_name(path: Path) -> str:
    return path.name.removesuffix('.gz').removesuffix('.csv')


def plain_csv(path: Path) -> Path:
    """The simulator reads plain CSV only: decompress .gz into the cache once."""
    if path.suffix != '.gz':
        return path
    out = TAPE_DIR / path.name.removesuffix('.gz')
    if not out.exists() or out.stat().st_mtime < path.stat().st_mtime:
        out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'rb') as src, out.open('wb') as dst:
            shutil.copyfileobj(src, dst)
    return out


def _digest(path: Path) -> str:
    st = path.stat()
    return hashlib.blake2b(f'{path.resolve()}:{st.st_mtime_ns}:{st.st_size}'.encode(), digest_size=6).hexdigest()


def _write(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out['Time'] = out['Time'].map(lambda v: f'{v:.6f}')
    for c in ('Volume', 'Price'):
        out[c] = out[c].astype('Int64')
    out.to_csv(path, index=False)
    return path


def make_variants(path: Path, n_chunks: int, n_boot: int, boot_minutes: float, halves: bool, seed: int,
                  n_regime: int = 0, gen: tuple | None = None) -> dict:
    """Write (or reuse) the derived tapes for one source tape. Returns {kind: [(label, path, meta), ...]}."""
    src = plain_csv(path)
    tag = f'{tape_name(path)}_{_digest(src)}'
    base = TAPE_DIR / tag
    df = pd.read_csv(src, dtype={'OrderId': 'string', 'Side': 'string', 'Operation': 'string', 'Lifespan': 'string'})
    t = df['Time'].to_numpy(float)
    t0, t1 = float(t[0]), float(t[-1])
    out: dict = {'full': [('full', src, {'t0': t0, 't1': t1})]}

    def cut(k: int, lo: float, hi: float, name: str):
        p = base / f'{name}.csv'
        m = (t >= lo) & ((t < hi) if hi < t1 else (t <= hi))
        if not p.exists():
            _write(df[m], p)
        return p, {'t0': lo, 't1': hi, 'rows': int(m.sum())}

    edges = np.linspace(t0, t1, n_chunks + 1)
    out['chunks'] = [(f'chunk {k + 1}', *cut(k, edges[k], edges[k + 1], f'chunk{n_chunks}_{k}')) for k in range(n_chunks)]
    if halves:
        mid = (t0 + t1) / 2
        out['halves'] = [('first half', *cut(0, t0, mid, 'half_0')), ('second half', *cut(1, mid, t1, 'half_1'))]

    mp = base / 'mirrored.csv'
    if not mp.exists():
        mdf = df.copy()
        ins = mdf['Operation'] == 'Insert'
        p0 = int(np.median(mdf.loc[ins, 'Price']))
        mdf.loc[ins, 'Price'] = 2 * p0 - mdf.loc[ins, 'Price']
        mdf.loc[ins, 'Side'] = mdf.loc[ins, 'Side'].map({'B': 'A', 'A': 'B'})
        _write(mdf, mp)
    out['mirrored'] = [('mirrored', mp, {'t0': t0, 't1': t1})]

    block = 60.0
    ins = (df['Operation'] == 'Insert').to_numpy()
    price = df['Price'].to_numpy(float)
    numeric_ids = df['OrderId'].str.fullmatch(r'\d+').all()
    n_blocks = max(1, int(round(min(boot_minutes * 60, t1 - t0) / block)))

    def session(pool, p, rng):
        """A session of n_blocks random 60 s blocks from `pool` (block start times), re-joined in price and time."""
        if not p.exists():
            parts, cursor, level = [], t0, None
            for j, s in enumerate(rng.choice(pool, size=n_blocks, replace=True)):
                m = (t >= s) & (t < s + block)
                blk = df[m].copy()
                bi = ins[m]
                # join in price: shift the block so its first inserts sit where the previous block ended
                head = price[m][bi][:200]
                if level is not None and len(head):
                    blk.loc[blk['Operation'] == 'Insert', 'Price'] += int(round(level - np.median(head)))
                tail = blk.loc[blk['Operation'] == 'Insert', 'Price'].to_numpy(float)[-200:]
                level = float(np.median(tail)) if len(tail) else level
                blk['Time'] = blk['Time'] - s + cursor
                cursor += block
                blk['OrderId'] = (blk['OrderId'].astype('int64') + (j + 1) * 10**10).astype('string') if numeric_ids else f'{j}_' + blk['OrderId']
                parts.append(blk)
            _write(pd.concat(parts, ignore_index=True), p)
        return p

    out['bootstrap'] = []
    if n_boot:
        starts = np.arange(t0, t1 - block, block)
        rng = np.random.default_rng(seed)
        for b in range(n_boot):
            p = session(starts, base / f'boot{n_blocks}_{seed}_{b}.csv', rng)
            out['bootstrap'].append((f'bootstrap {b + 1}', p, {'blocks': n_blocks}))

    out['storm_heavy'], out['calm_only'], out['regime_labels'] = [], [], None
    if n_regime:
        labels = regime_labels(src)
        if labels is not None:
            out['regime_labels'] = labels
            starts = np.arange(t0, t1 - block, block)
            storm_share = np.array([labels['storm_s'][int(s - t0):int(s - t0 + block)].mean() for s in starts])
            rng = np.random.default_rng(seed + 1)
            for kind, pool in (('storm_heavy', starts[storm_share > 0.5]), ('calm_only', starts[storm_share == 0])):
                if len(pool) < 3:
                    continue
                for b in range(n_regime):
                    p = session(pool, base / f'{kind}{n_blocks}_{seed}_{b}.csv', rng)
                    out[kind].append((f'{kind} {b + 1}', p, {'blocks': n_blocks, 'pool_blocks': int(len(pool))}))

    out['synthetic'] = []
    if gen:
        from analysis.generator import PARAM_DIR, GenParams, generate, write_tape

        gp = GenParams.load(PARAM_DIR / f'{gen[0]}_params.json')
        for k in range(gen[1]):
            p = TAPE_DIR / f'gen_{gen[0]}_{int(gen[2])}m_{seed}_{k}.csv'
            if not p.exists():
                g, _, _ = generate(gp, seed=10_000 + seed * 100 + k, duration_s=gen[2] * 60)
                write_tape(g, p)
            out['synthetic'].append((f'synthetic {k + 1}', p, {'minutes': gen[2]}))
    return out


def regime_labels(src: Path) -> dict | None:
    """Per-second storm flag of a tape (estimate_model.detect_regimes, 30 s windows), or None without regimes."""
    sys.path.insert(0, str(ROOT / 'scripts'))
    from analysis.replay import replay
    from analysis.tape import load_tape
    from estimate_model import REGIME_WINDOW_S, detect_jumps, detect_regimes

    rp = replay(load_tape(src), 10**9)
    reg = detect_regimes(rp, detect_jumps(rp))
    if not reg:
        return None
    prob = np.array(reg['storm_prob'])
    n_s = int((rp.ev_ts[-1] - rp.ev_ts[0]) // 1_000_000) + 1
    storm_s = np.repeat(prob > 0.5, REGIME_WINDOW_S)[:n_s]
    storm_s = np.r_[storm_s, np.zeros(max(0, n_s - len(storm_s)), bool)]
    return {'storm_s': storm_s, 'storm_share': float(storm_s.mean()), 'calm_dur_s': reg['calm_dur_s'], 'storm_dur_s': reg['storm_dur_s']}


# ---------------------------------------------------------------- workers (run in child processes)
def run_backtest(strategy_path: str, csv_path: str, overrides: dict, curve: bool = False) -> dict:
    from simulator.simulator import ExchangeSimulator

    t = time.perf_counter()
    try:
        sim = ExchangeSimulator(strategy=load_strategy(Path(strategy_path), overrides), config=scoring_config())
        res = sim.run(Path(csv_path)).model_dump(mode='json')
        res['error'] = None
        if curve:
            res['curve'] = [[ts / 1e6, v] for ts, v in sim.pnl_curve(res['pnl'])]
    except Exception as exc:  # report a crashing variant instead of losing the whole run
        res = {'pnl': None, 'error': f'{type(exc).__name__}: {exc}'}
    res['runtime_s'] = time.perf_counter() - t
    return res


def run_bounds(csv_path: str) -> dict:
    from analysis.sections.maxpnl import tape_bounds

    return tape_bounds(Path(csv_path))


# ---------------------------------------------------------------- report
def _per_hour(pnl, dur_s):
    return pnl / (dur_s / 3600) if pnl is not None and dur_s else None


def _share(pnl, mx):
    return pnl / mx if pnl is not None and mx else None


def _variants(params, names) -> list[tuple[str, float | None, object]]:
    """(param, scale or None for bool flip, value) for every sensitivity variant."""
    out = []
    for p in params:
        if names != 'all' and p.name not in names:
            continue
        if isinstance(p.default, bool):
            out.append((p.name, None, not p.default))
            continue
        seen = set()
        for s in SCALES:
            v = p.default * s
            if isinstance(p.default, int):
                v = int(round(v))
                if p.default >= 1:
                    v = max(1, v)
            if v != p.default and v not in seen:
                seen.add(v)
                out.append((p.name, s, v))
    return out


def regime_report(name: str, v: dict, results: dict, full: dict) -> dict:
    """regime_split (calm / storm PnL of the full run), storm_heavy / calm_only and synthetic sessions."""
    out: dict = {}
    lab = v.get('regime_labels')
    curve = full.get('curve')
    if lab is not None and curve:
        c = np.array(curve, dtype=float)
        sec = np.arange(0, int(c[-1, 0]) + 1)
        pnl_s = np.interp(sec, c[:, 0] - c[0, 0], c[:, 1])
        dp = np.diff(pnl_s)
        st = lab['storm_s'][:len(dp)]
        st = np.r_[st, np.zeros(len(dp) - len(st), bool)]
        out['regime_split'] = {'storm_share': float(st.mean()), 'calm_pnl': float(dp[~st].sum()), 'storm_pnl': float(dp[st].sum()),
                               'calm_per_hour': float(dp[~st].sum() / max((~st).sum(), 1) * 3600),
                               'storm_per_hour': float(dp[st].sum() / max(st.sum(), 1) * 3600),
                               'calm_dur_s': lab['calm_dur_s'], 'storm_dur_s': lab['storm_dur_s']}
        full.pop('curve', None)
    for kind in ('storm_heavy', 'calm_only', 'synthetic'):
        rows = [{'label': label, 'pnl': results[(name, kind, label, 'default')].get('pnl'),
                 'breach_penalty': results[(name, kind, label, 'default')].get('breach_penalty'),
                 'error': results[(name, kind, label, 'default')].get('error'), **meta} for label, _, meta in v.get(kind, [])]
        if rows:
            pn = [r['pnl'] for r in rows if r['pnl'] is not None]
            minutes = rows[0].get('minutes') or (rows[0].get('blocks', 0) * 60 / 60)
            out[kind] = {'sessions': rows, 'n': len(pn), 'median': float(np.median(pn)) if pn else None,
                         'p05': float(np.quantile(pn, 0.05)) if pn else None, 'share_profitable': float(np.mean(np.array(pn) > 0)) if pn else None,
                         'median_per_hour': float(np.median(pn)) / (minutes / 60) if pn and minutes else None}
    return out


def verdicts(rep: dict) -> list[dict]:
    out = []
    files = [f for f in rep['files'] if f['full'].get('pnl') is not None]
    if files:
        pos = sum(f['full']['pnl'] > 0 for f in files)
        shares = [f['share_of_max'] for f in files if f['share_of_max'] is not None]
        spread = (max(shares) - min(shares)) if len(shares) > 1 else 0.0
        status = 'good' if pos == len(files) and spread < 0.1 else 'warn' if pos >= len(files) / 2 else 'bad'
        out.append({'topic': 'Across tapes', 'status': status,
                    'headline': f'Profitable on {pos} of {len(files)} tapes',
                    'detail': 'share of theoretical max ' + ', '.join(f'{f["name"]} {100 * f["share_of_max"]:.1f}%' for f in files if f['share_of_max'] is not None) + '.'})
    ch = [c for f in rep['files'] for c in f['chunks'] if c.get('pnl') is not None]
    if ch:
        pos = sum(c['pnl'] > 0 for c in ch)
        worst = min(ch, key=lambda c: c['pnl'])
        status = 'good' if pos == len(ch) else 'warn' if pos >= 0.75 * len(ch) else 'bad'
        out.append({'topic': 'Over time', 'status': status, 'headline': f'Profitable in {pos} of {len(ch)} time chunks',
                    'detail': f'worst chunk {worst["pnl"]:,.0f} ({worst["file"]}, {worst["label"]}); chunks run from a cold start.'})
    mir = [(f['name'], f['full']['pnl'], f['mirrored'].get('pnl')) for f in files if f['mirrored'].get('pnl') is not None]
    if mir:
        gaps = [abs(m - p) / max(abs(p), 1.0) for _, p, m in mir]
        g = max(gaps)
        status = 'good' if g < 0.05 else 'warn' if g < 0.3 else 'bad'
        out.append({'topic': 'Side symmetry', 'status': status,
                    'headline': 'Mirrored market scores the same: buy and sell logic are symmetric' if g < 0.05
                    else f'Mirrored market scores {100 * g:.0f}% differently: buy and sell logic differ',
                    'detail': ', '.join(f'{n}: {p:,.0f} normal vs {m:,.0f} mirrored' for n, p, m in mir)
                    + '. A difference is a hidden long/short bias that may only fit this tape\'s direction.'})
    rg = [c for c in ch if c.get('mid_change_ticks') is not None and c.get('vol_ticks_per_sqrt_s') is not None]
    if len(rg) >= 6:
        pnl = np.array([c['pnl'] for c in rg], dtype=float)
        mv = np.abs([c['mid_change_ticks'] for c in rg])
        vol = np.array([c['vol_ticks_per_sqrt_s'] for c in rg], dtype=float)
        c_mv = float(sps.spearmanr(pnl, mv).statistic)
        c_vol = float(sps.spearmanr(pnl, vol).statistic)
        worst = max(abs(c_mv), abs(c_vol))
        status = 'good' if worst < 0.4 else 'warn' if worst < 0.7 else 'bad'
        what = 'how far the price moved' if abs(c_mv) >= abs(c_vol) else 'volatility'
        out.append({'topic': 'Regime dependence', 'status': status,
                    'headline': 'Chunk PnL does not depend much on the market regime' if worst < 0.4 else f'Chunk PnL tracks {what}',
                    'detail': f'Spearman of chunk PnL with |price drift| {c_mv:+.2f}, with volatility {c_vol:+.2f} over {len(rg)} chunks; '
                              'strong = it needs that kind of market.'})
    boots = [b['pnl'] for f in files for b in f['bootstrap'] if b.get('pnl') is not None]
    if boots:
        q = float(np.quantile(boots, 0.05))
        pos = float(np.mean(np.array(boots) > 0))
        status = 'good' if q > 0 else 'warn' if pos >= 0.75 else 'bad'
        out.append({'topic': 'Bootstrap', 'status': status, 'headline': f'{100 * pos:.0f}% of {len(boots)} reshuffled sessions profitable',
                    'detail': f'5th percentile {q:,.0f}, median {float(np.median(boots)):,.0f}.'})
    for f in files:
        rs = f.get('regime_split')
        if rs:
            worse = min(rs['calm_per_hour'], rs['storm_per_hour'])
            status = 'good' if worse > 0 else 'bad'
            out.append({'topic': 'Calm vs storm', 'status': status,
                        'headline': f'{f["name"]}: {rs["calm_per_hour"]:,.0f}/h in calm, {rs["storm_per_hour"]:,.0f}/h in storms',
                        'detail': f'storms {100 * rs["storm_share"]:.0f}% of the time (mean stretch {rs["storm_dur_s"]:.0f} s, calm {rs["calm_dur_s"]:.0f} s); '
                                  f'calm PnL {rs["calm_pnl"]:,.0f}, storm PnL {rs["storm_pnl"]:,.0f}. Losing in one regime means a stormier or calmer scored tape hurts.'})
        for kind, what in (('storm_heavy', 'storm-only'), ('calm_only', 'calm-only'), ('synthetic', 'generated')):
            k = f.get(kind)
            if k and k['n']:
                status = 'good' if k['p05'] > 0 else 'warn' if k['share_profitable'] >= 0.75 else 'bad'
                out.append({'topic': f'{what.capitalize()} sessions', 'status': status,
                            'headline': f'{f["name"]}: {100 * k["share_profitable"]:.0f}% of {k["n"]} {what} sessions profitable',
                            'detail': f'median {k["median"]:,.0f} ({k["median_per_hour"]:,.0f}/h), 5th percentile {k["p05"]:,.0f}.'
                                      + (' Generated markets: only as good as the model transfer.' if kind == 'synthetic' else '')})
    sens = rep.get('sensitivity') or {}
    if sens.get('params'):
        sharp = [p for p, d in sens['params'].items() if d.get('drop_at_25pct') is not None and d['drop_at_25pct'] > 0.3]
        wf = [w for w in sens.get('walk_forward', {}).values() if w.get('rank_corr') is not None]
        rc = float(np.mean([w['rank_corr'] for w in wf])) if wf else None
        status = 'good' if not sharp and (rc is None or rc > 0.5) else 'warn' if len(sharp) <= 2 and (rc is None or rc > 0.2) else 'bad'
        out.append({'topic': 'Parameters', 'status': status,
                    'headline': (f'{len(sharp)} parameter(s) on a sharp peak' if sharp else 'Parameters sit on a plateau'),
                    'detail': (f'sharp: {", ".join(sharp)}; ' if sharp else '') +
                              (f'walk-forward rank correlation first vs second half {rc:+.2f} (low = tuning does not transfer).' if rc is not None else '')})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('strategy', type=Path)
    ap.add_argument('tapes', nargs='*', type=Path, help='tapes to test (default: every tape in data/)')
    ap.add_argument('--set', action='append', default=[], metavar='NAME=VALUE', help='override a strategy constant (repeatable)')
    ap.add_argument('--chunks', type=int, default=6)
    ap.add_argument('--boot', type=int, default=8, help='bootstrap sessions per tape (0 = off)')
    ap.add_argument('--boot-minutes', type=float, default=30)
    ap.add_argument('--sensitivity', default='', help="'all' or comma-separated constant names (default: off)")
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--regime-boot', type=int, default=0, help='storm-only and calm-only bootstrap sessions per tape, plus the calm/storm PnL split (0 = off)')
    ap.add_argument('--gen', default='', help='generator params name (logs/generator/NAME_params.json) for synthetic sessions')
    ap.add_argument('--gen-sessions', type=int, default=6)
    ap.add_argument('--gen-minutes', type=float, default=30)
    args = ap.parse_args()

    t_start = time.perf_counter()
    cls_name, params = discover_params(args.strategy)
    by_name = {p.name: p for p in params}
    overrides = {}
    for kv in args.set:
        k, v = kv.split('=', 1)
        if k not in by_name:
            sys.exit(f'{k} is not a tunable constant of {args.strategy.name}: {", ".join(by_name)}')
        d = by_name[k].default
        overrides[k] = (by_name[k].scope, (v.lower() in ('1', 'true', 'yes')) if isinstance(d, bool) else type(d)(float(v) if isinstance(d, int) else v))
    sens_names = 'all' if args.sensitivity == 'all' else [s for s in args.sensitivity.split(',') if s]
    variants = _variants(params, sens_names) if sens_names else []

    tapes = args.tapes or find_tapes()
    print(f'{args.strategy.name} ({cls_name}) on {len(tapes)} tape(s): {", ".join(p.name for p in tapes)}', flush=True)
    jobs = []  # (key, fn, args)
    files = []
    for tp in tapes:
        name = tape_name(tp)
        v = make_variants(tp, args.chunks, args.boot, args.boot_minutes, bool(variants), args.seed,
                          args.regime_boot, (args.gen, args.gen_sessions, args.gen_minutes) if args.gen else None)
        files.append((name, v))
        for kind in ('full', 'chunks', 'mirrored', 'bootstrap', 'storm_heavy', 'calm_only', 'synthetic'):
            for label, path, _ in v[kind]:
                want_curve = kind == 'full' and v.get('regime_labels') is not None
                jobs.append(((name, kind, label, 'default'), run_backtest, (str(args.strategy), str(path), overrides, want_curve)))
        for label, path, _ in v['full'] + v['chunks']:
            jobs.append(((name, 'bounds', label, ''), run_bounds, (str(path),)))
        for label, path, _ in v.get('halves', []):
            jobs.append(((name, 'halves', label, 'default'), run_backtest, (str(args.strategy), str(path), overrides)))
            for pname, scale, val in variants:
                ov = dict(overrides)
                ov[pname] = (by_name[pname].scope, val)
                jobs.append(((name, 'halves', label, f'{pname}={val}'), run_backtest, (str(args.strategy), str(path), ov)))

    print(f'{len(jobs)} jobs on {args.workers} workers...', flush=True)
    results: dict = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fn, *a): key for key, fn, a in jobs}
        for done, fut in enumerate(as_completed(futs), 1):
            key = futs[fut]
            results[key] = fut.result()
            if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
                print(f'  {done}/{len(jobs)} done ({time.perf_counter() - t_start:.0f} s)', flush=True)

    rep_files = []
    for name, v in files:
        full_meta = v['full'][0][2]
        bounds_full = results[(name, 'bounds', 'full', '')]
        full = results[(name, 'full', 'full', 'default')]
        chunks = []
        for label, _, meta in v['chunks']:
            r = results[(name, 'chunks', label, 'default')]
            b = results[(name, 'bounds', label, '')]
            dur = meta['t1'] - meta['t0']
            chunks.append({'label': label, 'file': name, 't0_s': meta['t0'] - full_meta['t0'], 't1_s': meta['t1'] - full_meta['t0'], 'rows': meta['rows'],
                           'pnl': r.get('pnl'), 'pnl_per_hour': _per_hour(r.get('pnl'), dur), 'fills': r.get('fills'),
                           'breach_penalty': r.get('breach_penalty'), 'theoretical_max': b['theoretical_max'],
                           'mid_change_ticks': b.get('mid_change_ticks'), 'vol_ticks_per_sqrt_s': b.get('vol_ticks_per_sqrt_s'),
                           'share_of_max': _share(r.get('pnl'), b['theoretical_max']), 'error': r.get('error')})
        boot = [{'label': label, 'pnl': results[(name, 'bootstrap', label, 'default')].get('pnl'),
                 'error': results[(name, 'bootstrap', label, 'default')].get('error')} for label, _, _ in v['bootstrap']]
        bp = [b['pnl'] for b in boot if b['pnl'] is not None]
        dur = bounds_full['duration_s']
        rep_files.append({
            'name': name, 'duration_s': dur, 'full': full, 'pnl_per_hour': _per_hour(full.get('pnl'), dur),
            'theoretical_max': bounds_full['theoretical_max'], 'theoretical_min': bounds_full['theoretical_min'],
            'share_of_max': _share(full.get('pnl'), bounds_full['theoretical_max']),
            'mirrored': results[(name, 'mirrored', 'mirrored', 'default')], 'chunks': chunks,
            'chunks_profitable': sum((c['pnl'] or 0) > 0 for c in chunks),
            'bootstrap': boot,
            **regime_report(name, v, results, full),
            'bootstrap_summary': {'n': len(bp), 'p05': float(np.quantile(bp, 0.05)) if bp else None, 'median': float(np.median(bp)) if bp else None,
                                  'share_profitable': float(np.mean(np.array(bp) > 0)) if bp else None},
        })

    sensitivity: dict = {}
    if variants:
        sensitivity = {'scales': list(SCALES), 'params': {}, 'walk_forward': {}}
        names = [name for name, _ in files]
        for pname in dict.fromkeys(p for p, _, _ in variants):
            rows = []
            base_total = sum((results[(n, 'halves', h, 'default')].get('pnl') or 0) for n in names for h in ('first half', 'second half'))
            rows.append({'scale': 1.0, 'value': by_name[pname].default if pname not in overrides else overrides[pname][1], 'total_pnl': base_total,
                         'per_tape': {n: {h: results[(n, 'halves', h, 'default')].get('pnl') for h in ('first half', 'second half')} for n in names}})
            for p2, scale, val in variants:
                if p2 != pname:
                    continue
                key = f'{pname}={val}'
                tot = sum((results[(n, 'halves', h, key)].get('pnl') or 0) for n in names for h in ('first half', 'second half'))
                rows.append({'scale': scale, 'value': val, 'total_pnl': tot,
                             'per_tape': {n: {h: results[(n, 'halves', h, key)].get('pnl') for h in ('first half', 'second half')} for n in names}})
            rows.sort(key=lambda r: (r['scale'] if r['scale'] is not None else 2.0))
            near = [r['total_pnl'] for r in rows if r['scale'] in (0.75, 1.25) or r['scale'] is None]
            drop = (base_total - min(near)) / abs(base_total) if near and base_total else None
            sensitivity['params'][pname] = {'rows': rows, 'drop_at_25pct': drop}
        keys = ['default'] + [f'{p}={v}' for p, _, v in variants]
        for n in names:
            first = np.array([results[(n, 'halves', 'first half', k)].get('pnl') or np.nan for k in keys], dtype=float)
            second = np.array([results[(n, 'halves', 'second half', k)].get('pnl') or np.nan for k in keys], dtype=float)
            ok = np.isfinite(first) & np.isfinite(second)
            best = int(np.nanargmax(first)) if ok.any() else 0
            sensitivity['walk_forward'][n] = {
                'rank_corr': float(sps.spearmanr(first[ok], second[ok]).statistic) if ok.sum() > 3 else None,
                'best_on_first': keys[best], 'its_pnl_second': float(second[best]) if np.isfinite(second[best]) else None,
                'default_pnl_second': float(second[0]) if np.isfinite(second[0]) else None,
            }

    rep = {
        'meta': {'schema_version': SCHEMA_VERSION, 'strategy': args.strategy.name, 'strategy_class': cls_name,
                 'overrides': {k: v for k, (_, v) in overrides.items()}, 'tapes': [p.name for p in tapes], 'chunks': args.chunks,
                 'bootstrap': args.boot, 'boot_minutes': args.boot_minutes, 'seed': args.seed,
                 'sensitivity': sens_names if sens_names else None, 'regime_boot': args.regime_boot,
                 'gen': args.gen or None, 'gen_sessions': args.gen_sessions if args.gen else None, 'gen_minutes': args.gen_minutes if args.gen else None,
                 'generated_at': datetime.now(UTC).isoformat(timespec='seconds'),
                 'runtime_s': time.perf_counter() - t_start, 'jobs': len(jobs)},
        'files': rep_files,
        'sensitivity': sensitivity,
    }
    rep['verdicts'] = verdicts(rep)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.strategy.stem + (('__' + '_'.join(f'{k}-{v}' for k, (_, v) in sorted(overrides.items()))) if overrides else '')
    out = args.out or OUT_DIR / f'{stem}.json'
    out.write_text(json.dumps(rep, indent=1, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else str(o)))

    print(f'\n{"tape":<14}{"pnl":>12}{"/hour":>11}{"of max":>8}{"chunks+":>9}{"mirrored":>11}{"boot p05":>11}')
    for f in rep_files:
        pnl = f['full'].get('pnl')
        print(f'{f["name"]:<14}{pnl if pnl is not None else float("nan"):>12,.0f}{f["pnl_per_hour"] or float("nan"):>11,.0f}'
              f'{100 * (f["share_of_max"] or float("nan")):>7.1f}%{f["chunks_profitable"]:>5}/{len(f["chunks"]):<3}'
              f'{f["mirrored"].get("pnl") if f["mirrored"].get("pnl") is not None else float("nan"):>11,.0f}'
              f'{f["bootstrap_summary"]["p05"] if f["bootstrap_summary"]["p05"] is not None else float("nan"):>11,.0f}')
    for f in rep_files:
        if f.get('regime_split'):
            rs = f['regime_split']
            print(f'{f["name"]}: calm {rs["calm_pnl"]:,.0f} ({rs["calm_per_hour"]:,.0f}/h), storm {rs["storm_pnl"]:,.0f} ({rs["storm_per_hour"]:,.0f}/h)')
        for kind in ('storm_heavy', 'calm_only', 'synthetic'):
            if f.get(kind):
                k = f[kind]
                print(f'{f["name"]} {kind:<12} median {k["median"] or float("nan"):>10,.0f} ({k["median_per_hour"] or float("nan"):>9,.0f}/h)  '
                      f'p05 {k["p05"] or float("nan"):>10,.0f}  profitable {100 * (k["share_profitable"] or 0):.0f}% of {k["n"]}')
    for v_ in rep['verdicts']:
        print(f'[{v_["status"]:>4}] {v_["topic"]}: {v_["headline"]}. {v_["detail"]}')
    print(f'\n-> {out}  ({time.perf_counter() - t_start:.0f} s)')


if __name__ == '__main__':
    main()
