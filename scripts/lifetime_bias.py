"""Order lifetimes (cancel time - insert time) on a tape: distribution, drift over the session, side / size effects,
and whether per-second bid-vs-ask cancel imbalance predicts the next mid move.

usage: .venv/bin/python scripts/lifetime_bias.py data/<tape>.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.replay import replay  # noqa: E402
from analysis.tape import load_tape  # noqa: E402

path = Path(sys.argv[1])
df = pd.read_csv(path)
print('columns:', list(df.columns))
for c in ('Operation', 'Side', 'Lifespan'):
    print(c, df[c].value_counts().to_dict())
ins = df[df.Operation == 'Insert']
lim = ins[ins.Lifespan == 'G'].set_index('OrderId')
can = df[df.Operation == 'Cancel'].drop_duplicates('OrderId').set_index('OrderId')
j = lim[['Time', 'Side', 'Volume', 'Price']].join(can[['Time']], rsuffix='_c', how='inner')
j['life'] = j.Time_c - j.Time
j = j[j.life >= 0]
print(f'\nlimit inserts {len(lim)}, with a cancel {len(j)} ({len(j) / len(lim):.1%})')
q = j.life.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(3).to_dict()
print(f'life mean {j.life.mean():.3f} s, sd {j.life.std():.3f} (exponential: sd = mean), quantiles {q}')
print('memoryless check, mean remaining life given age > a:', {a: round((j.life[j.life > a] - a).mean(), 3) for a in (0, 0.25, 0.5, 1, 2)})
print('mean life by side:', j.groupby('Side').life.mean().round(3).to_dict())
j['vb'] = pd.qcut(j.Volume, 4, duplicates='drop')
print('mean life by size quartile:', j.groupby('vb', observed=True).life.mean().round(3).to_dict())
j['blk'] = (j.Time // 600 * 600).astype(int)
print('mean life per 600 s block:', j.groupby('blk').life.mean().round(3).to_dict())

# observable signal: per-second count of cancelled lots on each side, vs next mid move
rp = replay(load_tape(path), grid_step_us=1_000_000)
ts = rp.ev_ts / 1e6
mid = (rp.best_bid + rp.best_ask) / 2
ok = np.isfinite(mid)
ts, mid = ts[ok], mid[ok]
sec = np.arange(np.ceil(ts[0]), np.floor(ts[-1]))
m = mid[np.clip(np.searchsorted(ts, sec, 'right') - 1, 0, len(ts) - 1)]
cs = can.join(lim[['Side', 'Volume']], rsuffix='_i', how='inner')
cs['s'] = np.floor(cs.Time).astype(int)
sd = cs.Side_i if 'Side_i' in cs else cs.Side
bidc = cs[sd == 'B'].groupby('s').Volume_i.sum() if 'Volume_i' in cs else cs[sd == 'B'].groupby('s').Volume.sum()
askc = cs[sd == 'A'].groupby('s').Volume_i.sum() if 'Volume_i' in cs else cs[sd == 'A'].groupby('s').Volume.sum()
idx = sec.astype(int)
b = bidc.reindex(idx, fill_value=0).to_numpy(float)
a = askc.reindex(idx, fill_value=0).to_numpy(float)
imb = (b - a) / np.maximum(b + a, 1)
for h in (1, 5, 30):
    fut = np.r_[m[h:] - m[:-h], np.full(h, np.nan)]
    k = np.isfinite(fut)
    r = np.corrcoef(imb[k], fut[k])[0, 1]
    print(f'corr(bid-ask cancelled-lot imbalance this second, mid move next {h:2d} s) = {r:+.3f} (n {k.sum()}, se {1 / np.sqrt(k.sum()):.3f})')
