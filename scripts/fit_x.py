"""Likelihood fit of the limit-order offset X (and sigma of S) for a tape, on N slices spread across it.

For each slice, a grid filter tracks the hidden price S from limit-order prices (buy = floor(S - X),
sell = ceil(S + X)) and maximises the exact likelihood over sigma_S and the X distribution (shifted gamma and
lognormal). The slices run in parallel, one per core, and are pooled at the end.

    .venv/bin/python scripts/fit_x.py data/round2_train.csv --slices 20 --workers 12

Writes logs/generator/xfit_<name>/slice_*.json and logs/generator/<name>_xfit.json. About 4-5 min per slice per core.
The sigma it reports has no IOC push, so it comes out a bit above estimate_model.py's sigma_s; use it for X only.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from x_fit_combine import combine_estimates  # noqa: E402
from x_fit_slice import fit_slice  # noqa: E402


def run_one(args: tuple) -> int:
    idx, n_total, tape, out_dir, max_iter = args
    fit_slice(idx, n_total, tape, out_dir, subsample=1, max_iter=max_iter, n_workers=1)
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('tape', type=Path)
    ap.add_argument('--slices', type=int, default=20, help='slices to fit, spread evenly over the tape')
    ap.add_argument('--n-total', type=int, default=96, help='the tape is cut into this many slices; --slices of them are fitted')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--max-iter', type=int, default=80, help='Nelder-Mead iterations per model')
    args = ap.parse_args()

    name = args.tape.name.split('.')[0]
    out_dir = ROOT / 'logs' / 'generator' / f'xfit_{name}'
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, args.n_total // args.slices)
    idxs = list(range(0, args.n_total, step))[: args.slices]
    jobs = [(i, args.n_total, str(args.tape), str(out_dir), args.max_iter) for i in idxs]
    print(f'fitting {len(idxs)} of {args.n_total} slices of {args.tape} on {args.workers} cores', flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for i in ex.map(run_one, jobs):
            print(f'  slice {i} done', flush=True)
    out = ROOT / 'logs' / 'generator' / f'{name}_xfit.json'
    combine_estimates(str(out_dir), str(out))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
