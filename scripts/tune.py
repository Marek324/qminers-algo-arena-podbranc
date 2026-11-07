"""Coordinate-descent tuning of a strategy's top-level constants with an overfitting guard.

Every candidate value is backtested on each tape chunk (e.g. the four quarters of a round). A change is accepted
only if the total PnL rises by at least --min-gain (fraction of the current total) AND the new value beats the
current one on at least --min-wins chunks. Passes repeat until nothing changes. Prints a log and the final
settings; writes the tuned copy to --out.

usage: .venv/bin/python scripts/tune.py strategies/regime_mm.py data/q/r2_q1.csv data/q/r2_q2.csv \
           data/q/r2_q3.csv data/q/r2_q4.csv --grid scripts/tune_grid_regime.txt --out strategies/regime_mm_tuned.py
grid file: one line per constant, `NAME = v1, v2, v3` (the current value is added automatically).
"""
import argparse
import json
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def set_const(text: str, name: str, value: str) -> str:
    out, n = re.subn(rf'^(\s*){name} = [^#\n]*', rf'\g<1>{name} = {value}  ', text, count=1, flags=re.M)
    if n == 0:
        raise SystemExit(f'constant {name} not found')
    return out


def get_const(text: str, name: str) -> str:
    m = re.search(rf'^\s*{name} = ([^#\n]*)', text, flags=re.M)
    return m.group(1).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('strategy', type=Path)
    ap.add_argument('tapes', type=Path, nargs='+')
    ap.add_argument('--grid', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('-j', '--jobs', type=int, default=6)
    ap.add_argument('--min-gain', type=float, default=0.01)
    ap.add_argument('--min-wins', type=int, default=3)
    ap.add_argument('--passes', type=int, default=3)
    args = ap.parse_args()

    grid = []
    for line in args.grid.read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            name, vals = line.split('=', 1)
            grid.append((name.strip(), [v.strip() for v in vals.split(',')]))
    text = args.strategy.read_text()
    tmp = Path(tempfile.mkdtemp(prefix='tune_'))
    cache: dict = {}
    counter = [0]

    def run_one(src: str, tape: Path) -> float:
        counter[0] += 1
        k = counter[0]
        path = tmp / f's{k}.py'
        path.write_text(src)
        out = tmp / f'o{k}'
        subprocess.run(['.venv/bin/exchange-simulator', '--strategy', str(path), '--backtest', str(tape),
                        '--log-dir', str(out), '--no-logs'], check=True, capture_output=True)
        r = json.loads((out / 'result.json').read_text())
        return r['pnl'] - r['breach_penalty'] * 0  # pnl already includes the penalty

    def evaluate(srcs: list[str]) -> list[list[float]]:
        todo = [(s, t) for s in srcs if s not in cache for t in args.tapes]
        with ThreadPoolExecutor(args.jobs) as pool:
            res = list(pool.map(lambda st: run_one(*st), todo))
        it = iter(res)
        for s in srcs:
            if s not in cache:
                cache[s] = [next(it) for _ in args.tapes]
        return [cache[s] for s in srcs]

    base = evaluate([text])[0]
    print(f'base {sum(base):,.0f}  chunks {[round(x) for x in base]}', flush=True)
    for p in range(args.passes):
        changed = False
        for name, vals in grid:
            cur = get_const(text, name)
            cands = [v for v in vals if v != cur]
            srcs = [set_const(text, name, v) for v in cands]
            res = evaluate(srcs)
            best = None
            for v, s, r in zip(cands, srcs, res):
                wins = sum(a > b for a, b in zip(r, base))
                ok = sum(r) >= sum(base) * (1 + args.min_gain) and wins >= args.min_wins
                print(f'  {name}={v:<12} total {sum(r):>10,.0f} ({sum(r) - sum(base):+,.0f})  wins {wins}/{len(r)}'
                      f'{"  ok" if ok else ""}', flush=True)
                if ok and (best is None or sum(r) > sum(best[2])):
                    best = (v, s, r)
            if best:
                text, base = best[1], best[2]
                changed = True
                print(f'ACCEPT {name} {cur} -> {best[0]}: total {sum(base):,.0f} chunks {[round(x) for x in base]}',
                      flush=True)
            else:
                print(f'keep {name} = {cur}', flush=True)
        args.out.write_text(text)
        print(f'--- pass {p + 1} done: total {sum(base):,.0f}', flush=True)
        if not changed:
            break
    print('final', {n: get_const(text, n) for n, _ in grid})


if __name__ == '__main__':
    main()
