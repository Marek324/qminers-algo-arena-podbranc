"""Backtest a strategy over a parameter grid and several tapes, in parallel (--no-logs runs).

Usage:
  python3 -I scripts/sweep.py strategies/round1/adaptive_mm.py data/round1_train.csv \
      -p TAKE_THRESHOLD=0.5,1,2 -p TAKE_POS=1,5

Each -p overrides a `NAME = value` constant in the strategy file, at module level or in a class body (found with
ast, so text in docstrings is never touched); comma-separated values form a grid. Prints one row per combination: PnL per tape, total, worst tape, fills, max q95 |inventory|, penalty.
Temporary strategy copies and outputs go to the system temp dir.
"""
import argparse
import ast
import itertools
import json
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run(strategy: Path, tape: Path, out: Path) -> dict:
    subprocess.run(
        ['uv', 'run', 'exchange-simulator', '--strategy', str(strategy), '--backtest', str(tape),
         '--log-dir', str(out), '--no-logs'],
        check=True, capture_output=True,
    )
    return json.loads((out / 'result.json').read_text())


def set_param(text: str, name: str, value: str) -> str:
    """Return text with the first module-level or class-body `name = ...` assignment set to value (comment kept)."""
    tree = ast.parse(text)
    bodies = [tree.body] + [n.body for n in tree.body if isinstance(n, ast.ClassDef)]
    for body in bodies:
        for node in body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                continue
            if node.end_lineno != node.lineno:
                raise SystemExit(f'parameter {name} spans several lines; edit it by hand')
            lines = text.split('\n')
            i = node.lineno - 1
            lines[i], n = re.subn(rf'^(\s*){name}\s*=\s*[^#]*', rf'\g<1>{name} = {value}  ', lines[i], count=1)
            if n == 0:
                raise SystemExit(f'could not rewrite line {node.lineno} for {name}')
            return '\n'.join(lines)
    raise SystemExit(f'parameter {name} not found')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('strategy', type=Path)
    ap.add_argument('tapes', type=Path, nargs='+')
    ap.add_argument('-p', '--param', action='append', default=[])
    ap.add_argument('-j', '--jobs', type=int, default=8)
    args = ap.parse_args()

    names, grids = [], []
    for p in args.param:
        name, values = p.split('=', 1)
        names.append(name)
        grids.append(values.split(','))
    source = args.strategy.read_text()
    tmp = Path(tempfile.mkdtemp(prefix='sweep_'))

    combos = list(itertools.product(*grids)) or [()]
    jobs = []
    for i, combo in enumerate(combos):
        text = source
        for name, value in zip(names, combo):
            text = set_param(text, name, value)
        path = tmp / f'combo{i}.py'
        path.write_text(text)
        for j, tape in enumerate(args.tapes):
            jobs.append((i, j, path, tape, tmp / f'out{i}_{j}'))

    results: dict = {}
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(run, path, tape, out): (i, j) for i, j, path, tape, out in jobs}
        for fut, (i, j) in futures.items():
            results[(i, j)] = fut.result()

    head = ' '.join(f'{n:>14}' for n in names)
    tapes = ' '.join(f'{t.stem[-10:]:>10}' for t in args.tapes)
    print(f'{head} {tapes} {"total":>9} {"worst":>9} {"fills":>6} {"q95":>4} {"penalty":>8}')
    for i, combo in enumerate(combos):
        rs = [results[(i, j)] for j in range(len(args.tapes))]
        pnls = [r['pnl'] for r in rs]
        row = ' '.join(f'{v:>14}' for v in combo)
        per = ' '.join(f'{p:>10.0f}' for p in pnls)
        print(f'{row} {per} {sum(pnls):>9.0f} {min(pnls):>9.0f} {sum(r["fills"] for r in rs):>6}'
              f' {max(r["abs_inventory_q95"] for r in rs):>4} {sum(r["breach_penalty"] for r in rs):>8.0f}')


if __name__ == '__main__':
    main()
