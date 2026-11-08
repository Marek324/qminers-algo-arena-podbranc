"""Markout curve of own fills by fill type and regime, out to 120 s: where does each leg's edge peak?

usage: .venv/bin/python scripts/markout_curve.py logs/<full-logs run> data/<tape>.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.replay import replay  # noqa: E402
from analysis.tape import load_tape  # noqa: E402

US = 1_000_000
H = (0.1, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 60, 120)

run, tape = Path(sys.argv[1]), Path(sys.argv[2])
rp = replay(load_tape(tape), grid_step_us=US)
ts = rp.ev_ts
mid = (rp.best_bid + rp.best_ask) / 2
ok = np.isfinite(mid)
ts, mid = ts[ok], mid[ok]


def mid_at(t):
    return mid[np.clip(np.searchsorted(ts, t, 'right') - 1, 0, len(ts) - 1)]


t0 = ts[0]
sec = np.arange(t0, ts[-1], US)
m1 = mid_at(sec)
vol30 = pd.Series(np.abs(np.diff(m1, prepend=m1[0]))).rolling(30, center=True, min_periods=1).mean().to_numpy()
storm = vol30 > 2.0 * np.median(vol30)
tr = pd.read_csv(run / 'trades.csv')
own = tr[(tr.taker_owner == 'user') | (tr.maker_owner == 'user')].copy()
tb = own.taker_side == 'B'
own['taking'] = own.taker_owner == 'user'
own['sign'] = np.where(own.taking, np.where(tb, 1, -1), np.where(tb, -1, 1))
t = own.timestamp_us.to_numpy()
own['storm'] = storm[np.clip(((t - t0) // US).astype(int), 0, len(storm) - 1)]
inv = np.cumsum((own.sign * own.quantity).to_numpy())
inv_before = inv - (own.sign * own.quantity).to_numpy()
own['opening'] = np.abs(inv) > np.abs(inv_before)
for h in H:
    own[f'{h}'] = own.sign * (mid_at(t + int(h * US)) - own.price)
pd.set_option('display.width', 250)
for keys in (['storm', 'taking'], ['storm', 'taking', 'opening']):
    g = own.groupby(keys)
    out = pd.DataFrame({f'{h}': g.apply(lambda d, h=h: (d[f'{h}'] * d.quantity).sum() / d.quantity.sum()) for h in H})
    out.insert(0, 'lots', g.quantity.sum())
    print(f'\nticks/lot markout vs horizon (s), by {keys}:')
    print(out.round(2).to_string())
# inventory carry: PnL of holding inventory over the next h seconds, sampled each second
pnl_inv = pd.Series(inv, index=t)
inv_s = np.zeros(len(sec))
idx = np.searchsorted(t, sec, 'right') - 1
inv_s[idx >= 0] = inv[idx[idx >= 0]]
print('\ninventory carry per second by |inv| bucket and regime: mean inv*(mid[t+h]-mid[t]) (ticks)')
for h in (1, 5, 10, 30, 60):
    fut = np.r_[m1[h:], np.full(h, m1[-1])] - m1
    c = inv_s * fut
    b = np.where(np.abs(inv_s) >= 95, 'cap', np.where(np.abs(inv_s) >= 50, 'mid', 'low'))
    df = pd.DataFrame({'c': c, 'b': b, 's': storm[:len(c)]})
    print(h, df.groupby(['s', 'b']).c.agg(['mean', 'count']).round(1).to_dict())
# carry of the next 1 s / 5 s by age of the position (time since the last fill that grew |inv|), per regime
op_t = t[own.opening.to_numpy()]
k = np.searchsorted(op_t, sec, 'right') - 1
age = np.where(k >= 0, (sec - op_t[np.maximum(k, 0)]) / US, 1e9)
ab = pd.cut(age, [-1, 1, 2, 3, 5, 8, 12, 20, 40, 1e10], labels=['<1', '1-2', '2-3', '3-5', '5-8', '8-12', '12-20', '20-40', '40+'])
for h in (1, 3, 5):
    fut = np.r_[m1[h:], np.full(h, m1[-1])] - m1
    df = pd.DataFrame({'c': inv_s * fut / np.maximum(np.abs(inv_s), 1), 'tot': inv_s * fut, 'age': ab, 's': storm[:len(sec)], 'big': np.abs(inv_s) >= 50})
    df = df[df.big]
    print(f'\n|inv|>=50: next {h}s carry by position age: per lot (ticks), total (ticks*lots), seconds')
    print(df.groupby(['s', 'age'], observed=True).agg(per_lot=('c', 'mean'), total=('tot', 'sum'), n=('c', 'size')).round(2).to_string())
