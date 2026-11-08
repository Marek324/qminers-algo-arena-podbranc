"""Run hmm_qm_diag on a tape and save its per-fill log (with order metadata) to npz."""
import sys
from pathlib import Path
import numpy as np
from simulator.main import load_strategy_from_file
from simulator.simulator import ExchangeSimulator, SimulationConfig
strat = load_strategy_from_file(sys.argv[1])
res = ExchangeSimulator(strat, SimulationConfig()).run(Path(sys.argv[2]))
print('pnl', getattr(res, 'pnl', res))
cols = 't sign price qty t_sent vol take qahead inside imb fx sdev inv pv_fill own_lvl'.split()
np.savez(sys.argv[3], f=np.array(strat.flog, dtype=float), cols=np.array(cols))
