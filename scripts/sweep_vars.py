"""Backtest constant overrides of one strategy on one or more tapes in parallel and print PnL vs the base.

usage: .venv/bin/python scripts/sweep_vars.py strategies/round3/hmm_deep_clear_r3.py TAG 'A=1,B=2' 'A=3' ...
env:   TAPES=data/round3_train.csv[,more.csv]   JOBS=8   OUT=logs/sweep
Each variant is written to $OUT/TAG/v_<i>.py (the base is v_0); results go to $OUT/TAG/<i>_<tape>/.
"""
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

base, tag, variants = sys.argv[1], sys.argv[2], sys.argv[3:]
tapes = os.environ.get('TAPES', 'data/round3_train.csv').split(',')
jobs_n = int(os.environ.get('JOBS', '8'))
out = Path(os.environ.get('OUT', 'logs/sweep')) / tag
out.mkdir(parents=True, exist_ok=True)
text = Path(base).read_text()
jobs = []
for i, v in enumerate(['BASE'] + variants):
    t = text
    if v != 'BASE':
        for kv in v.split(','):
            k, val = kv.split('=', 1)
            t, n = re.subn(rf'^(\s*){k} = [^#\n]*', rf'\g<1>{k} = {val}  ', t, count=1, flags=re.M)
            if not n:
                raise SystemExit(f'constant {k} not found')
    p = out / f'v_{i}.py'
    p.write_text(t)
    for j, tp in enumerate(tapes):
        jobs.append((v, j, p, tp, out / f'{i}_{Path(tp).stem}'))


def run(jb):
    v, j, p, tp, d = jb
    subprocess.run(['.venv/bin/exchange-simulator', '--strategy', str(p), '--backtest', tp, '--log-dir', str(d),
                    '--no-logs'], capture_output=True)
    try:
        r = json.loads((d / 'result.json').read_text())
        return v, j, r['pnl'], r['breach_penalty']
    except (OSError, ValueError, KeyError):
        return v, j, float('nan'), 0.0


res: dict[str, list] = {}
with ThreadPoolExecutor(jobs_n) as ex:
    for v, j, pnl, pen in ex.map(run, jobs):
        res.setdefault(v, [None] * len(tapes))[j] = (pnl, pen)
b = res['BASE']
for v, r in res.items():
    parts = '  '.join(f'{x[0]:>9,.0f} ({x[0] - y[0]:+8,.0f}){" pen " + f"{x[1]:,.0f}" if x[1] else ""}'
                      for x, y in zip(r, b))
    tot, btot = sum(x[0] for x in r), sum(y[0] for y in b)
    tail = f'   total {tot:>9,.0f} ({tot - btot:+8,.0f})' if len(tapes) > 1 else ''
    print(f'{v:60s} {parts}{tail}', flush=True)
