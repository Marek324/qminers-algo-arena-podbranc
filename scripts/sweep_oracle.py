"""Run scripts/oracle_s_mm.py (true S) over a few parameter sets in parallel; print pnl.

  .venv/bin/python scripts/sweep_oracle.py data/synthetic/synth_short.csv "K_S=1,K_IMB=0,K_FLOW=0" "K_S=1,EDGE=0.3" ...
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

tape = sys.argv[1]
s_path = tape[:-4] + '_S.npz'
sets = sys.argv[2:]
strat = os.environ.get('SWEEP_STRAT', 'scripts/oracle_s_mm.py')


def run(i_ps):
    i, ps = i_ps
    d = f'logs/sweep_oracle/{os.path.basename(tape)[:-4]}_{i}'
    env = dict(os.environ, ORACLE_S=s_path, ORACLE_PARAMS=ps)
    subprocess.run(['.venv/bin/exchange-simulator', '--strategy', strat, '--backtest', tape,
                    '--log-dir', d, '--no-logs'], env=env, capture_output=True)
    r = json.load(open(d + '/result.json'))
    return ps, r['pnl'], r['breach_penalty'], r['traded_volume'], r['abs_inventory_q50']


with ThreadPoolExecutor(int(os.environ.get('JOBS', '8'))) as ex:
    for ps, pnl, br, vol, q50 in ex.map(run, enumerate(sets)):
        print(f'{pnl:>12,.0f}  breach={br:<8,.0f} vol={vol:<8} invq50={q50:<4} {ps}', flush=True)
