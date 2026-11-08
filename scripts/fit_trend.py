"""Fit a cubic in elapsed fraction u = t / T to a tape's 1 s mid: S(u) = c0 + c1 u + c2 u^2 + c3 u^3.

usage: .venv/bin/python scripts/fit_trend.py data/<tape>.csv
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.replay import replay  # noqa: E402
from analysis.tape import load_tape  # noqa: E402

US = 1_000_000
rp = replay(load_tape(Path(sys.argv[1])), grid_step_us=US)
ts = rp.ev_ts
mid = (rp.best_bid + rp.best_ask) / 2
ok = np.isfinite(mid)
ts, mid = ts[ok], mid[ok]
sec = np.arange(ts[0], ts[-1], US)
m = mid[np.clip(np.searchsorted(ts, sec, 'right') - 1, 0, len(ts) - 1)]
u = np.arange(len(m)) / len(m)
for deg in (2, 3, 4):
    c = np.polyfit(u, m, deg)
    fit = np.polyval(c, u)
    print(f'deg {deg}: coefs high->low {np.round(c, 2).tolist()}  resid sd {np.std(m - fit):.1f}')
    print('   fit every 600 s:', ' '.join(f'{x:.0f}' for x in fit[::600]))
