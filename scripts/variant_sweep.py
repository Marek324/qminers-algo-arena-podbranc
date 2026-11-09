"""Run named constant-override variants of one strategy on several tapes in parallel; print a PnL table.

usage: .venv/bin/python scripts/variant_sweep.py strategies/round4/jteam_r4.py grid.txt tape1.csv tape2.csv \
           --out logs/sweep_fire -j 16
grid file: one variant per line, `name: NAME=value, NAME=value` (empty override list = the file as is).
Writes <out>/<variant>__<tape>/result.json and <out>/results.csv (variant, tape, pnl, penalty, fills).
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
    ap.add_argument('grid', type=Path)
    ap.add_argument('tapes', type=Path, nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('-j', '--jobs', type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    src = a.strategy.read_text()
    variants = []
    for line in a.grid.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, _, rest = line.partition(':')
        text = src
        for kv in filter(None, (x.strip() for x in rest.split(','))):
            k, v = (s.strip() for s in kv.split('=', 1))
            text = set_const(text, k, v)
        path = a.out / f'{name.strip()}.py'
        path.write_text(text)
        variants.append((name.strip(), path))

    def run(job):
        (name, path), tape = job
        d = a.out / f'{name}__{tape.stem}'
        res = d / 'result.json'
        if not res.exists():
            subprocess.run(['.venv/bin/exchange-simulator', '--strategy', str(path), '--backtest', str(tape),
                            '--log-dir', str(d), '--no-logs'], capture_output=True)
        try:
            r = json.loads(res.read_text())
        except (OSError, ValueError):
            return name, tape.stem, None, None, None
        print(f'{name:<28}{tape.stem:<22}{r["pnl"]:>12,.0f}  pen {r["breach_penalty"]:>9,.0f}', flush=True)
        return name, tape.stem, r['pnl'], r['breach_penalty'], r['fills']

    jobs = [(v, t) for v in variants for t in a.tapes]
    with ThreadPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(run, jobs))
    with open(a.out / 'results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant', 'tape', 'pnl', 'penalty', 'fills'])
        w.writerows(rows)
    tapes = [t.stem for t in a.tapes]
    print('\n' + f'{"variant":<28}' + ''.join(f'{t:>22}' for t in tapes) + f'{"total":>14}')
    for name, _ in variants:
        vals = [next((r[2] for r in rows if r[0] == name and r[1] == t), None) for t in tapes]
        tot = sum(v for v in vals if v is not None)
        print(f'{name:<28}' + ''.join(f'{(v or 0):>22,.0f}' for v in vals) + f'{tot:>14,.0f}')


if __name__ == '__main__':
    main()
