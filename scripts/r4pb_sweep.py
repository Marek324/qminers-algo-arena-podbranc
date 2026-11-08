"""Backtest variants of one strategy on round4_public and print PnL, both halves and penalty.

usage: .venv/bin/python scripts/r4pb_sweep.py strategies/round4/pb_chase_r4.py TAG "CH_K=1;CH_X=5" "CH_K=3" ...
Each spec sets top-level constants (`;`-separated). Runs -j backtests at a time (default 4) into logs/r4pb/TAG/.
"""
import argparse
import csv
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def set_const(text: str, name: str, value: str) -> str:
    out, n = re.subn(rf'^(\s*){name} = [^#\n]*', rf'\g<1>{name} = {value}  ', text, count=1, flags=re.M)
    if n == 0:
        raise SystemExit(f'constant {name} not found')
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('strategy', type=Path)
    ap.add_argument('tag')
    ap.add_argument('specs', nargs='+')
    ap.add_argument('-j', '--jobs', type=int, default=4)
    ap.add_argument('--tape', default='data/round4_public.csv')
    ap.add_argument('--split', type=float, default=3000.0)
    args = ap.parse_args()
    base = args.strategy.read_text()
    root = Path('logs/r4pb') / args.tag
    root.mkdir(parents=True, exist_ok=True)

    def run(i_spec):
        i, spec = i_spec
        src = base
        for kv in filter(None, spec.split(';')):
            k, v = kv.split('=', 1)
            src = set_const(src, k.strip(), v.strip())
        path = root / f'v_{i}.py'
        path.write_text(src)
        out = root / f'run_{i}'
        subprocess.run(['.venv/bin/exchange-simulator', '--strategy', str(path), '--backtest', args.tape,
                        '--log-dir', str(out), '--no-logs'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            r = json.loads((out / 'result.json').read_text())
            rows = list(csv.reader(open(out / 'pnl.csv')))[1:]
            h1 = [float(p) for t, p in rows if float(t) <= args.split][-1]
            line = f'{i:>3} {r["pnl"]:>11,.0f}  h1 {h1:>10,.0f}  h2 {r["pnl"] - h1:>10,.0f}  pen {r["breach_penalty"]:>7,.0f}  {spec}'
        except Exception as e:  # noqa: BLE001
            line = f'{i:>3} FAILED {e}  {spec}'
        print(line, flush=True)
        return line

    with ThreadPoolExecutor(args.jobs) as ex:
        lines = list(ex.map(run, enumerate(args.specs)))
    (root / 'summary.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
