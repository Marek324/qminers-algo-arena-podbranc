"""Run named strategy variants on several tapes in parallel and print/save a PnL table.

usage: .venv/bin/python scripts/r3_sweep.py strategies/round3/x.py data/round3_public.csv data/r3_eval_twin.csv \
           -v base= -v lev2=LEV_POS=300,LEV_EDGE=3 [-g NAME=v1,v2,v3] [-j 10] [--out logs/sweeps/name.csv]

-v NAME=K1=V1,K2=V2  one named variant (empty override list = file as is); repeatable.
-g K=v1,v2,...       one-parameter grid: adds a variant per value; repeatable (each grid is separate, not crossed).
Constants may be indented (class attributes). Runs .venv/bin/exchange-simulator --no-logs. A run's result.json lands
in logs/sweeps/<out stem>/<variant>_<tape>/.
"""
import argparse
import csv
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / '.venv' / 'bin' / 'exchange-simulator'


def set_const(text: str, name: str, value: str) -> str:
    out, n = re.subn(rf'^(\s*){name} = [^#\n]*', rf'\g<1>{name} = {value}  ', text, count=1, flags=re.M)
    if n == 0:
        raise SystemExit(f'constant {name} not found')
    return out


def run(strategy: Path, tape: Path, out: Path) -> dict:
    t0 = time.time()
    p = subprocess.run([str(SIM), '--strategy', str(strategy), '--backtest', str(tape), '--log-dir', str(out),
                        '--no-logs'], capture_output=True, text=True, cwd=ROOT)
    if p.returncode != 0:
        return {'pnl': float('nan'), 'error': p.stderr[-400:], 'sec': time.time() - t0}
    r = json.loads((out / 'result.json').read_text())
    r['sec'] = time.time() - t0
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('strategy', type=Path)
    ap.add_argument('tapes', type=Path, nargs='+')
    ap.add_argument('-v', '--variant', action='append', default=[])
    ap.add_argument('-g', '--grid', action='append', default=[])
    ap.add_argument('-j', '--jobs', type=int, default=10)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    variants: list[tuple[str, list[tuple[str, str]]]] = []
    for v in args.variant:
        name, _, rest = v.partition('=')
        kv = [tuple(x.split('=', 1)) for x in rest.split(',') if x]
        variants.append((name, kv))
    for g in args.grid:
        k, vals = g.split('=', 1)
        for val in vals.split(','):
            variants.append((f'{k}={val}', [(k, val)]))
    if not variants:
        variants = [('base', [])]

    stem = args.out.stem if args.out else f'{args.strategy.stem}_{int(time.time())}'
    work = ROOT / 'logs' / 'sweeps' / stem
    work.mkdir(parents=True, exist_ok=True)
    src = args.strategy.read_text()
    jobs = []
    for i, (name, kv) in enumerate(variants):
        text = src
        for k, val in kv:
            text = set_const(text, k, val)
        path = work / f'v{i}.py'
        path.write_text(text)
        for tape in args.tapes:
            jobs.append((i, tape, path, work / f'v{i}_{tape.stem}'))

    with ThreadPoolExecutor(args.jobs) as pool:
        futs = {(i, tape.stem): pool.submit(run, path, tape, out) for i, tape, path, out in jobs}
        res = {k: f.result() for k, f in futs.items()}

    rows = []
    for i, (name, _) in enumerate(variants):
        row = {'variant': name}
        tot = 0.0
        for tape in args.tapes:
            r = res[(i, tape.stem)]
            row[tape.stem] = round(r['pnl'])  if r['pnl'] == r['pnl'] else 'ERR'
            row[f'{tape.stem}_pen'] = round(r.get('breach_penalty', 0))
            row[f'{tape.stem}_fills'] = r.get('fills', 0)
            if 'error' in r:
                row['error'] = r['error'].replace('\n', ' | ')
            tot += r['pnl'] if r['pnl'] == r['pnl'] else 0
        row['total'] = round(tot)
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    outp = args.out or (work / 'results.csv')
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'{"variant":<40}' + ''.join(f'{t.stem[:16]:>18}{"pen":>9}' for t in args.tapes) + f'{"total":>12}')
    for r in rows:
        print(f'{r["variant"][:40]:<40}' + ''.join(f'{str(r[t.stem]):>18}{r[t.stem + "_pen"]:>9}' for t in args.tapes)
              + f'{r["total"]:>12}' + (f'  ERROR {r["error"][:200]}' if 'error' in r else ''))
    print(f'saved {outp}')


if __name__ == '__main__':
    main()
