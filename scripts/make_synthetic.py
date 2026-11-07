"""Generate a synthetic dataset from the fitted market model, split into train and validation.

usage:
    uv run python scripts/make_synthetic.py                       # defaults below
    uv run python scripts/make_synthetic.py --exact 20 --random 40 --val-frac 0.33 --minutes 20
    uv run python scripts/make_synthetic.py --base round2_public --out data/synthetic/round2   # round2 twins

Two families of sessions:
  exact    the --base parameters (logs/generator/<base>_params.json, default round1): how a strategy does on a
           market like that tape
  random   each session draws its own parameters from ranges that cover sample_v2, round1 and somewhat beyond, so a
           strategy cannot specialise to one setting (the next round will have different parameters); 30% of them
           have no push per IOC, like sample_v2. With a base that has regimes and jumps (round2), random sessions
           instead scale the base's rates, volatilities, jump rates and regime lengths by the factors in RANGES_R2

Every session gets a seed of its own; train and validation never share seeds. Output in --out (default data/synthetic/,
git-ignored):
  <family>_<split>_<nnn>.csv     the tape, in the simulator's CSV format
  <family>_<split>_<nnn>.npz     the true hidden price S (t_us, s), for analysis only; a strategy never sees it
  manifest.json                  every session's split, family, seed and parameters

The real tapes in data/ are not touched: keep them as the final check, never as training data.
"""

from __future__ import annotations

import argparse
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

from analysis.generator import PARAM_DIR, GenParams, generate, write_tape  # noqa: E402

OUT = ROOT / 'data' / 'synthetic'

# parameter ranges for the randomized family (uniform unless noted)
RANGES = {
    'sigma_s': (0.35, 1.1),
    'x_mean': (0.5, 1.1),
    'x_std': (0.15, 0.5),
    'limit_rate': (160.0, 200.0),
    'ioc_rate': (5.0, 10.0),
    'life_mean_s': (0.4, 0.8),
    'impact_k': (0.0, 0.015),
}
P_NO_IMPACT = 0.3  # share of randomized sessions without the push per IOC (sample_v2 has none)
X_LOC = 0.05

# round2-style bases (regimes and jumps): random sessions scale the base by a factor drawn per session
# (log-uniform). Jump rates go up to 2x: the private evaluation tape had about 1.6x round2_public's jumps
# (docs/EVALUATION_DATASET_ANALYSIS.md).
RANGES_R2 = {
    'sigma_s': (0.7, 1.4),
    'sigma_storm': (0.7, 1.4),
    'jump_rate': (0.5, 2.0),
    'jump_rate_storm': (0.5, 2.0),
    'calm_dur_s': (0.5, 2.0),
    'storm_dur_s': (0.5, 2.0),
    'limit_rate': (0.85, 1.15),
    'ioc_rate': (0.8, 1.25),
    'ioc_rate_storm': (0.8, 1.25),
    'life_mean_s': (0.8, 1.25),
    'x_mean': (0.8, 1.25),
    'impact_k': (0.5, 1.5),
}


def random_params_r2(base: GenParams, rng: np.random.Generator) -> GenParams:
    d = {k: getattr(base, k) * float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) for k, (lo, hi) in RANGES_R2.items()}
    if base.x_dist == 'gamma':
        m = max(d['x_mean'] - base.x_loc, 0.05)
        d['x_a'], d['x_b'] = (m / base.x_std) ** 2, base.x_std ** 2 / m
    return replace(base, name='random', **d)


def random_params(base: GenParams, rng: np.random.Generator) -> GenParams:
    d = {k: float(rng.uniform(*v)) for k, v in RANGES.items()}
    if rng.random() < P_NO_IMPACT:
        d['impact_k'] = 0.0
    m = max(d['x_mean'] - X_LOC, 0.05)
    sizes = str(rng.choice(['round1', 'sample_v2']))
    return replace(
        base, name='random', sigma_s=d['sigma_s'], limit_rate=d['limit_rate'], ioc_rate=d['ioc_rate'],
        life_mean_s=d['life_mean_s'], impact_k=d['impact_k'], impact_beta=0.65,
        x_dist='gamma', x_mean=d['x_mean'], x_std=d['x_std'], x_loc=X_LOC, x_a=(m / d['x_std']) ** 2, x_b=d['x_std'] ** 2 / m,
        cancel_filled=bool(rng.integers(0, 2)), limit_sizes_file=f'{sizes}_limit_sizes.npy', ioc_sizes_file=f'{sizes}_ioc_sizes.npy',
        ioc_mode='S', clamp=True,
    )


def make_one(job: dict) -> dict:
    p = GenParams(**{**job['params'], 'ioc_offset': {int(k): v for k, v in job['params']['ioc_offset'].items()}})
    t = time.perf_counter()
    df, ts, s = generate(p, seed=job['seed'], duration_s=job['minutes'] * 60)
    tape = Path(job['out']) / f'{job["id"]}.csv'
    write_tape(df, tape)
    np.savez_compressed(tape.with_suffix('.npz'), t_us=np.round(ts * 1e6).astype(np.int64), s=s.astype(np.float32))
    return {**{k: job[k] for k in ('id', 'family', 'split', 'seed', 'minutes')}, 'rows': len(df), 'file': tape.name,
            'params': {k: v for k, v in job['params'].items() if k not in ('ioc_offset',)}, 'seconds': time.perf_counter() - t}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--exact', type=int, default=24, help='sessions with the round1 parameters')
    ap.add_argument('--random', type=int, default=48, help='sessions with randomized parameters')
    ap.add_argument('--val-frac', type=float, default=1 / 3, help='share of each family held out for validation')
    ap.add_argument('--minutes', type=float, default=20)
    ap.add_argument('--seed', type=int, default=50_000, help='first seed; sessions use seed, seed+1, ...')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--base', default='round1', help='fitted params to start from: logs/generator/<base>_params.json')
    ap.add_argument('--out', type=Path, default=OUT, help='output directory (default data/synthetic)')
    args = ap.parse_args()

    base = GenParams.load(PARAM_DIR / f'{args.base}_params.json')
    rand = random_params_r2 if base.storm_dur_s > 0 or base.jump_rate > 0 else random_params
    out = args.out.resolve()
    rng = np.random.default_rng(args.seed)
    jobs = []
    seed = args.seed
    for family, n in (('exact', args.exact), ('random', args.random)):
        n_val = int(round(n * args.val_frac))
        for i in range(n):
            split = 'val' if i >= n - n_val else 'train'
            p = base if family == 'exact' else rand(base, rng)
            jobs.append({'id': f'{family}_{split}_{i:03d}', 'family': family, 'split': split, 'seed': seed, 'minutes': args.minutes,
                         'params': asdict(p), 'out': str(out)})
            seed += 1

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f'generating {len(jobs)} sessions of {args.minutes:g} min into {out} on {args.workers} workers...', flush=True)
    done = []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, r in enumerate(ex.map(make_one, jobs), 1):
            done.append(r)
            if k % max(1, len(jobs) // 10) == 0 or k == len(jobs):
                print(f'  {k}/{len(jobs)} ({time.perf_counter() - t0:.0f} s)', flush=True)
    manifest = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'minutes': args.minutes, 'base_params': f'{args.base}_params.json',
                'ranges': RANGES_R2 if rand is random_params_r2 else RANGES, 'sessions': done}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, default=lambda o: o.item()))
    for fam in ('exact', 'random'):
        for sp in ('train', 'val'):
            s = [d for d in done if d['family'] == fam and d['split'] == sp]
            print(f'{fam:7} {sp:5}: {len(s):3d} sessions, {sum(d["rows"] for d in s) / 1e6:.1f}M rows')
    print(f'-> {out / "manifest.json"}  ({time.perf_counter() - t0:.0f} s)')


if __name__ == '__main__':
    main()
