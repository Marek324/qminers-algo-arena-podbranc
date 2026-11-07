"""Parameter sweep for class-level constants (`    NAME = value` inside the strategy class): rewrite them, backtest each variant, print PnL.

usage: uv run python scripts/sweep_class.py strategies/x.py data/sample.csv out_dir 'NAME=v1,v2' 'NAME2=v1,v2' ...
"""
import itertools, json, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

src, data, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
grid = [(k, v.split(',')) for k, v in (a.split('=') for a in sys.argv[4:])]
out.mkdir(parents=True, exist_ok=True)
text = Path(src).read_text()

def run(combo):
    t = text
    for (k, _), v in zip(grid, combo):
        t, n = re.subn(rf'^(\s+{k} = ).*$', rf'\g<1>{v}', t, count=1, flags=re.M)
        assert n == 1, k
    tag = '_'.join(f'{k}{v}' for (k, _), v in zip(grid, combo))
    f = out / f'{tag}.py'; f.write_text(t)
    subprocess.run(['uv', 'run', 'exchange-simulator', '--strategy', str(f), '--backtest', data,
                    '--log-dir', str(out / tag), '--no-logs'], capture_output=True, check=True)
    r = json.loads((out / tag / 'result.json').read_text())
    return combo, r['pnl'], r['breach_penalty'], r['abs_inventory_q997']

with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(run, itertools.product(*[v for _, v in grid])))
for combo, pnl, pen, q in sorted(res, key=lambda x: -x[1])[:25]:
    print(dict(zip([k for k, _ in grid], combo)), round(pnl), pen, q)
