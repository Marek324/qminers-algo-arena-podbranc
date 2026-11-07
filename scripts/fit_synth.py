"""Compare synthetic tapes (scripts/gen_tape.py) with round1 on the stats that matter for PnL.

Each arg is a set of gen_tape flags; every set is generated for --duration seconds, analysed with
analysis.py, and backtested with strategies/imbalance_mm.py. Prints impact by IOC size (1s/5s mean),
maker markout, realized vol, touch depth and PnL per 12000s, next to round1.

  .venv/bin/python scripts/fit_synth.py "--kappa 0.02 --alpha 0.5" "--kappa 0.01 --alpha 0.6 --sigma 0.7"
"""
import json
import shlex
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DUR = 1200
STRAT = 'strategies/imbalance_mm.py'


def stats(rep):
    imp = [round((r['1s'] + r['5s']) / 2, 3) for r in rep['trades']['impact_by_size']]
    return dict(
        imp=imp,
        mk1s=round(rep['maker']['markout_ticks']['1s'], 3),
        vol=round(rep['indicators']['evaluation']['realized_vol_60s']['feature']['mean'], 3),
        touch=round(rep['book']['touch_qty']['bid']['mean']),
        sp1=round(rep['book']['spread_share']['1'], 3),
        hind=round(rep['max_pnl']['theoretical_max']['pnl_per_hour']),
    )


def run(i_flags):
    i, flags = i_flags
    sys.path.insert(0, '.')
    from analysis import analyze
    name = f'data/synthetic/fit_{i}'
    subprocess.run(['.venv/bin/python', 'scripts/gen_tape.py', '--out', name, '--duration', str(DUR),
                    '--seed', '7'] + shlex.split(flags), check=True, capture_output=True)
    st = stats(analyze(Path(name + '.csv')))
    d = f'logs/fit/{i}'
    subprocess.run(['.venv/bin/exchange-simulator', '--strategy', STRAT, '--backtest', name + '.csv',
                    '--log-dir', d, '--no-logs'], capture_output=True)
    st['pnl_12k'] = round(json.load(open(d + '/result.json'))['pnl'] * 12000 / DUR)
    for ext in ('.csv', '_S.npz'):
        Path(name + ext).unlink()
    return flags, st


if __name__ == '__main__':
    r1 = stats(json.load(open('logs/analysis/round1.json')))
    r1['pnl_12k'] = 615583
    print(f'{"round1":40s} {r1}', flush=True)
    with ProcessPoolExecutor(6) as ex:
        for flags, st in ex.map(run, enumerate(sys.argv[1:])):
            print(f'{flags:40s} {st}', flush=True)
