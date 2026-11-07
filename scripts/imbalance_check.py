"""How well do touch imbalance and decayed OFI predict price moves on a tape? (market orders only)

Usage: uv run python -I scripts/imbalance_check.py data/<file>.csv [ofi_tau_ms]

For each signal, events are bucketed into deciles. Per decile it prints:
  P(up)      probability the next mid change is up
  dmid@1s    mean mid change (ticks) 1 s later
  dmid@5s    mean mid change (ticks) 5 s later
A taker signal is only worth trading if |dmid| in its extreme deciles clearly exceeds the half
spread paid on entry plus the half spread paid on exit (that is, the mean spread).
The OFI here is the same as in strategies/adaptive_mm.py: forecast = 0.25 * ofi / avg touch depth.
"""
import collections
import sys

import numpy as np

path = sys.argv[1]
tau_us = float(sys.argv[2]) * 1000 if len(sys.argv) > 2 else 1_000_000.0

book = {'B': {}, 'A': {}}  # side -> price -> OrderedDict(order_id -> qty)
where = {}


def best(side):
    lv = book[side]
    if not lv:
        return None
    return max(lv) if side == 'B' else min(lv)


def remove_qty(side, price, oid, amount):
    lvl = book[side][price]
    lvl[oid] -= amount
    if lvl[oid] <= 0:
        del lvl[oid]
        del where[oid]
    if not lvl:
        del book[side][price]


ts_l, bid_l, ask_l, imb_l, ofi_l = [], [], [], [], []
prev = None
ofi = 0.0
avg_depth = 0.0
last_ts = 0
with open(path) as f:
    next(f)
    for line in f:
        t, op, oid, side, vol, price, life = line.rstrip('\n').split(',')
        ts = int(round(float(t) * 1e6))
        if op == 'Insert':
            price, qty = int(price), int(vol)
            opp = 'A' if side == 'B' else 'B'
            while qty > 0:
                b = best(opp)
                if b is None or (side == 'B' and price < b) or (side == 'A' and price > b):
                    break
                for moid, mq in list(book[opp][b].items()):
                    take = min(qty, mq)
                    qty -= take
                    remove_qty(opp, b, moid, take)
                    if qty == 0:
                        break
            if qty > 0 and life == 'G':
                book[side].setdefault(price, collections.OrderedDict())[oid] = qty
                where[oid] = (side, price)
        else:
            loc = where.get(oid)
            if loc:
                s, p = loc
                remove_qty(s, p, oid, book[s][p][oid])
        bb, ba = best('B'), best('A')
        if bb is None or ba is None:
            continue
        qb, qa = sum(book['B'][bb].values()), sum(book['A'][ba].values())
        if ts > last_ts:
            ofi *= np.exp(-(ts - last_ts) / tau_us)
            last_ts = ts
        if prev is not None:
            pb, pqb, pa, pqa = prev
            ofi += (qb if bb >= pb else 0) - (pqb if bb <= pb else 0) - (qa if ba <= pa else 0) + (pqa if ba >= pa else 0)
        prev = (bb, qb, ba, qa)
        depth = (qb + qa) / 2
        avg_depth = depth if avg_depth == 0 else avg_depth + 0.01 * (depth - avg_depth)
        ts_l.append(ts)
        bid_l.append(bb)
        ask_l.append(ba)
        imb_l.append((qb - qa) / (qb + qa))
        ofi_l.append(0.25 * ofi / avg_depth)

ts = np.array(ts_l)
mid = (np.array(bid_l) + np.array(ask_l)) / 2
spread = np.array(ask_l) - np.array(bid_l)
print(f'{path}: {len(mid)} events, {ts[-1] / 1e6:.0f} s, mid {mid[0]:.1f} -> {mid[-1]:.1f} '
      f'(min {mid.min():.1f}, max {mid.max():.1f}), mean spread {spread.mean():.2f} ticks')

# next mid change direction
chg = np.flatnonzero(np.diff(mid) != 0) + 1  # indices where mid changed
nxt = np.searchsorted(chg, np.arange(len(mid)), side='right')  # first change strictly after i
valid = nxt < len(chg)
up = np.zeros(len(mid), dtype=bool)
up[valid] = mid[chg[nxt[valid]]] > mid[np.arange(len(mid))[valid]]
# mid change after horizons
fut = {}
for h in (1_000_000, 5_000_000):
    j = np.searchsorted(ts, ts + h, side='left')
    ok = j < len(ts)
    d = np.full(len(ts), np.nan)
    d[ok] = mid[j[ok]] - mid[ok]
    fut[h] = d

for name, sig in (('imbalance', np.array(imb_l)), ('ofi forecast (ticks)', np.array(ofi_l))):
    edges = np.quantile(sig, np.linspace(0.1, 0.9, 9))
    k = np.searchsorted(edges, sig, side='right')
    print(f'\n{name}: decile -> mean value, n, P(up), dmid@1s, dmid@5s')
    for d in range(10):
        m = k == d
        if not m.any():
            continue
        mv = m & valid
        print(f'  {d}  mean={sig[m].mean():+8.3f}  n={m.sum():8d}  P(up)={up[mv].mean():.3f}'
              f'  dmid@1s={np.nanmean(fut[1_000_000][m]):+7.3f}  dmid@5s={np.nanmean(fut[5_000_000][m]):+7.3f}')
    for thr in (1.0, 2.0):
        if name.startswith('ofi'):
            sel = np.abs(sig) > thr
            if sel.any():
                s = np.sign(sig[sel])
                print(f'  |forecast|>{thr}: {sel.mean() * 100:.1f}% of events, signed dmid@1s='
                      f'{np.nanmean(s * fut[1_000_000][sel]):+.3f}  @5s={np.nanmean(s * fut[5_000_000][sel]):+.3f}'
                      f'  (round-trip cost ~ mean spread {spread.mean():.2f})')
