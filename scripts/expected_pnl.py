"""Expected PnL of our strategies on fresh generated sessions of a fitted market model, plus on the real tape.

    .venv/bin/python scripts/expected_pnl.py --name round2 --real data/round2.csv
    .venv/bin/python scripts/expected_pnl.py --name round2 --x-from logs/generator/round2_xfit.json   # variant with the likelihood X

--name is a params file written by estimate_model.py (logs/generator/<name>_params.json). With --x-from, X is replaced
by the pooled shifted-gamma fit of scripts/fit_x.py and saved as <name>_xfit_params.json, which is scored instead.
Writes logs/generator/expected_<name>[_xfit].json. A 20-minute x 8-session run takes about 15-20 min on 12 cores.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.generator import PARAM_DIR, GenParams  # noqa: E402
from analysis.target import compute_target, run_file_strategy  # noqa: E402


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--name', required=True, help='params name in logs/generator/')
    ap.add_argument('--x-from', type=Path, default=None, help='replace X by the pooled gamma fit in this xfit JSON')
    ap.add_argument('--real', type=Path, default=None, help='also run the strategies on this real tape')
    ap.add_argument('--strategies', default='strategies/round1/imbalance_mm.py,strategies/round1/adaptive_mm.py')
    ap.add_argument('--minutes', type=float, default=20)
    ap.add_argument('--n-train', type=int, default=4)
    ap.add_argument('--n-test', type=int, default=8)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    strategies = [s for s in args.strategies.split(',') if s]

    name = args.name
    if args.x_from:
        g = json.loads(args.x_from.read_text())['shifted_gamma']
        k, th, loc = g['k']['inv_var_mean'], g['theta']['inv_var_mean'], g['loc']['inv_var_mean']
        base = GenParams.load(PARAM_DIR / f'{args.name}_params.json')
        name = f'{args.name}_xfit'
        replace(base, name=name, x_dist='gamma', x_a=k, x_b=th, x_loc=loc,
                x_mean=loc + k * th, x_std=(k ** 0.5) * th).save(PARAM_DIR / f'{name}_params.json')
        log(f'X replaced by the likelihood fit: {loc:.3f} + Gamma({k:.3f}, {th:.3f}), mean {loc + k * th:.3f}')

    out: dict = {'params': json.loads((PARAM_DIR / f'{name}_params.json').read_text())}
    if args.real:
        log(f'strategies on the real tape {args.real}')
        import pandas as pd
        t = pd.read_csv(args.real, usecols=['Time'])['Time']
        hours = (t.iloc[-1] - t.iloc[0]) / 3600
        with ProcessPoolExecutor(len(strategies)) as ex:
            real = list(ex.map(run_file_strategy, [str(args.real)] * len(strategies), strategies))
        out['real'] = {Path(s).name: {'pnl': r, 'per_hour': r / hours} for s, r in zip(strategies, real)}
        log(f'  {out["real"]}')

    log(f'expected PnL for {name}')
    out['expected'] = compute_target(name, minutes=args.minutes, n_train=args.n_train, n_test=args.n_test,
                                     strategies=strategies, workers=args.workers, log=log)
    path = PARAM_DIR / f'expected_{name}.json'
    path.write_text(json.dumps(out, indent=1))
    e = out['expected']
    log(f'reference market maker {e["target"]["mean_per_hour"]:,.0f}/h; '
        + ', '.join(f'{k} {v["mean_per_hour"]:,.0f}/h' for k, v in e['strategies'].items()))
    log(f'-> {path}')


if __name__ == '__main__':
    main()
