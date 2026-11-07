"""Monte Carlo estimate of a strategy's expected PnL given generator parameters.

For each parameter set (gen_tape.py flags) it generates N fresh sessions (different seeds), backtests every
strategy on each, and reports mean / sd / standard error, scaled to a full round (--round-s, default 12000 s).
Run several parameter sets to see how much the EV depends on the fit (usually more than on session noise).

  .venv/bin/python scripts/ev_estimate.py --n 16 --duration 1200 \
      --strategies strategies/imbalance_mm.py,strategies/adaptive_mm.py \
      --set "mine=--kappa 0.0075 --alpha 0.7 --sigma 0.5" --set "theirs=--kappa 0.009 --alpha 0.65 --sigma 0.75"
"""
import argparse
import json
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TMP = Path('data/synthetic/ev_tmp')


def session(job):
    name, flags, seed, dur, strategies = job
    base = TMP / f'{name}_{seed}'
    subprocess.run(['.venv/bin/python', 'scripts/gen_tape.py', '--out', str(base), '--duration', str(dur),
                    '--seed', str(seed)] + shlex.split(flags), check=True, capture_output=True)
    out = {}
    for s in strategies:
        d = f'logs/ev/{name}_{seed}_{Path(s).stem}'
        subprocess.run(['.venv/bin/exchange-simulator', '--strategy', s, '--backtest', f'{base}.csv',
                        '--log-dir', d, '--no-logs'], capture_output=True)
        r = json.load(open(d + '/result.json'))
        out[Path(s).stem] = r['pnl']  # already net of breach penalty
    for ext in ('.csv', '_S.npz'):
        Path(f'{base}{ext}').unlink()
    return name, seed, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--duration', type=float, default=1200)
    ap.add_argument('--round-s', type=float, default=12000)
    ap.add_argument('--strategies', required=True)
    ap.add_argument('--set', action='append', required=True, help='name=gen_tape flags')
    ap.add_argument('--jobs', type=int, default=10)
    ap.add_argument('--seed0', type=int, default=1000)
    a = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    strategies = a.strategies.split(',')
    sets = [s.split('=', 1) for s in a.set]
    jobs = [(n, f, a.seed0 + i, a.duration, strategies) for n, f in sets for i in range(a.n)]
    res: dict = {}
    with ThreadPoolExecutor(a.jobs) as ex:
        for name, seed, out in ex.map(session, jobs):
            for st, pnl in out.items():
                res.setdefault((name, st), []).append(pnl)
    scale = a.round_s / a.duration
    print(f'EV per {a.round_s:.0f}s round, from {a.n} sessions of {a.duration:.0f}s (scaled x{scale:g})')
    print(f'{"params":10s} {"strategy":16s} {"mean":>10s} {"se":>8s} {"sd(1 round)":>12s} {"p5..p95 of a round":>22s}')
    summary = {}
    for (name, st), v in res.items():
        v = np.array(v) * scale
        # sessions are short; a full round averages scale of them, so its sd is sd_session*scale/sqrt(scale)
        sd_round = v.std(ddof=1) / np.sqrt(scale)
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
        print(f'{name:10s} {st:16s} {m:>10,.0f} {se:>8,.0f} {sd_round:>12,.0f} '
              f'{m - 1.645 * sd_round:>10,.0f}..{m + 1.645 * sd_round:<10,.0f}')
        summary[f'{name}/{st}'] = dict(mean=m, se=se, sd_round=sd_round, n=len(v))
    Path('logs/ev').mkdir(parents=True, exist_ok=True)
    json.dump(summary, open('logs/ev/summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
