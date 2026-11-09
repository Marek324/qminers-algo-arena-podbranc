"""Big jumps only: how much stale liquidity can a 5 ms-latency sniper take, and what is it worth?

usage: .venv/bin/python scripts/big_jump_stale.py data/round4_train.csv [THR ticks] [FAR ticks]
1. Replay the tape with a price-level book (IOCs sweep it, cancels remove what is left).
2. A trigger is a new limit order placed FAR or more ticks beyond the opposite best price (an ask far above the best ask
   means S jumped up). The jump is "big" if the mid moves THR or more ticks in that direction within 2 s.
3. 6 ms after the first trigger of each big jump (detect + 5 ms latency + 1 ms), count the stale lots on the run-over
   side priced at least MARGIN ticks better than the new S (trigger price -+ X_V), up to 900 lots, and their edge.
   Also when the first market trade against those stale levels happens.
"""
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1]
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 25
FAR = int(sys.argv[3]) if len(sys.argv) > 3 else 10
X_V = 2.0
MARGIN = 2.0
CAP = 900

bids = defaultdict(int)   # price -> qty
asks = defaultdict(int)
orders = {}                # id -> [side, price, qty]
best = {'b': None, 'a': None}


def bb():
    return max(bids) if bids else None


def ba():
    return min(asks) if asks else None


rows = []
with open(path) as f:
    next(f)
    for line in f:
        p = line.rstrip('\n').split(',')
        rows.append(p)

T = np.array([float(r[0]) for r in rows])
# pass 1: touch path (cheap) via the full replay, collecting triggers and mids
trig = []        # (row index, time, direction, price)
mids_t, mids = [], []
snap_req = {}    # row index -> list of trigger ids to snapshot after this row
pending = []     # (time to snapshot, trigger id)
snaps = {}
first_stale_trade = {}


def remove(oid, q=None):
    o = orders.get(oid)
    if o is None:
        return
    s, px, qty = o
    take = qty if q is None else min(q, qty)
    book = bids if s == 'B' else asks
    book[px] -= take
    if book[px] <= 0:
        del book[px]
    o[2] -= take
    if o[2] <= 0:
        del orders[oid]


level_q = {'B': defaultdict(list), 'A': defaultdict(list)}   # price -> FIFO of order ids

ins = []   # (t, cls)
pi = 0
for i, r in enumerate(rows):
    t = T[i]
    while pending and pending[0][0] <= t:
        _, k = pending.pop(0)
        d, s_new = trig[k][2], trig[k][4]
        if d > 0:   # up jump: stale asks below s_new - MARGIN
            lv = sorted(px for px in asks if px <= s_new - MARGIN)
            lots = edge = 0
            for px in lv:
                q = min(asks[px], CAP - lots)
                lots += q
                edge += q * (s_new - px)
                if lots >= CAP:
                    break
        else:
            lv = sorted((px for px in bids if px >= s_new + MARGIN), reverse=True)
            lots = edge = 0
            for px in lv:
                q = min(bids[px], CAP - lots)
                lots += q
                edge += q * (px - s_new)
                if lots >= CAP:
                    break
        snaps[k] = (lots, edge)
    op = r[1]
    if op == 'Insert':
        oid, side, qty, px, life = r[2], r[3], int(r[4]), int(r[5]), r[6]
        b0, a0 = bb(), ba()
        if life == 'F':
            crosses = (side == 'B' and a0 is not None and px >= a0) or (side == 'A' and b0 is not None and px <= b0)
            ins.append((t, (1 if side == 'B' else -1) if crosses else 0))
        else:
            vis = True
            if side == 'B' and b0 is not None and px < b0 - 19:
                vis = False
            if side == 'A' and a0 is not None and px > a0 + 19:
                vis = False
            ins.append((t, (1 if side == 'B' else -1) if vis else 0))
        if life == 'F':   # IOC: sweep
            left = qty
            if side == 'B':
                while left and asks and min(asks) <= px:
                    lvl = min(asks)
                    fifo = level_q['A'][lvl]
                    while left and fifo:
                        m = fifo[0]
                        if m not in orders:
                            fifo.pop(0)
                            continue
                        q = min(left, orders[m][2])
                        remove(m, q)
                        left -= q
                        if m not in orders:
                            fifo.pop(0)
                    for k in list(first_stale_trade.keys()):
                        pass
                    if not fifo and lvl in asks:
                        del asks[lvl]
            else:
                while left and bids and max(bids) >= px:
                    lvl = max(bids)
                    fifo = level_q['B'][lvl]
                    while left and fifo:
                        m = fifo[0]
                        if m not in orders:
                            fifo.pop(0)
                            continue
                        q = min(left, orders[m][2])
                        remove(m, q)
                        left -= q
                        if m not in orders:
                            fifo.pop(0)
                    if not fifo and lvl in bids:
                        del bids[lvl]
        else:
            orders[oid] = [side, px, qty]
            (bids if side == 'B' else asks)[px] += qty
            level_q[side][px].append(oid)
            if b0 is not None and a0 is not None:
                if side == 'A' and px >= a0 + FAR:
                    k = len(trig)
                    trig.append((i, t, 1, px, px - X_V, px - a0))
                    pending.append((t + 0.006, k))
                elif side == 'B' and px <= b0 - FAR:
                    k = len(trig)
                    trig.append((i, t, -1, px, px + X_V, b0 - px))
                    pending.append((t + 0.006, k))
    elif op == 'Cancel':
        remove(r[2])
    b1, a1 = bb(), ba()
    if b1 is not None and a1 is not None and (not mids or (b1 + a1) / 2 != mids[-1]):
        mids_t.append(t)
        mids.append((b1 + a1) / 2)

mids_t = np.array(mids_t)
mids = np.array(mids)
# big jumps: first trigger whose direction the mid confirms by THR within 2 s; merge triggers within 5 s
# every trigger: big (mid moves THR within 2 s) or false; depth and same-direction triggers within 30 ms
stats = []
tt = np.array([x[1] for x in trig]); dd = np.array([x[2] for x in trig])
for k, (i, t, d, px, s_new, dep) in enumerate(trig):
    j0 = np.searchsorted(mids_t, t) - 1
    j1 = np.searchsorted(mids_t, t + 2.0)
    big = False
    if j0 >= 0 and j1 > j0:
        m0 = mids[j0]
        mv = (mids[j0:j1].max() - m0) if d > 0 else (m0 - mids[j0:j1].min())
        big = mv >= THR
    a, b = np.searchsorted(tt, t - 0.030), np.searchsorted(tt, t, side='right')
    n30 = int(np.sum(dd[a:b] == d))
    stats.append((big, dep, n30, snaps.get(k, (0, 0))[1]))
st = np.array(stats, float)
print('triggers by depth / confirmations (n same-direction far orders in the last 30 ms incl. this one):')
for lo, hi in ((10, 13), (13, 16), (16, 21), (21, 999)):
    for nmin in (1, 2, 3):
        m = (st[:, 1] >= lo) & (st[:, 1] < hi) & (st[:, 2] >= nmin)
        if m.any():
            print(f'  depth {lo:>3}-{hi:<3} n>={nmin}: triggers {int(m.sum()):5d}  big {int(st[m, 0].sum()):4d}  edge on big {st[m & (st[:, 0] > 0), 3].sum():>9,.0f}')
out = []
last_t = -1e9
for k, (i, t, d, px, s_new, dep) in enumerate(trig):
    if t - last_t < 5:
        continue
    j0 = np.searchsorted(mids_t, t) - 1
    j1 = np.searchsorted(mids_t, t + 2.0)
    if j0 < 0 or j1 <= j0:
        continue
    m0 = mids[j0]
    mv = (mids[j0:j1].max() - m0) if d > 0 else (m0 - mids[j0:j1].min())
    if mv >= THR:
        last_t = t
        lots, edge = snaps.get(k, (0, 0))
        out.append((t, d, mv, lots, edge))
print(f'{path}: {len(trig)} far (>= {FAR}) triggers, {len(out)} big jumps (>= {THR} ticks in 2 s)')
tot = 0
for t, d, mv, lots, edge in out:
    tot += edge
    print(f'  t {t:8.2f}  dir {d:+d}  move {mv:6.1f}  stale lots after 6 ms {lots:4d}  edge vs new S {edge:>9,.0f}')
print(f'total takeable edge (lot-ticks = $) {tot:,.0f}')

# rule scan: fire when, within W, >= G missing inserts and visible new orders only on one side (>= 1)
it = np.array([x[0] for x in ins]); ic = np.array([x[1] for x in ins])
jt = np.array([o[0] for o in out]); jd = np.array([o[1] for o in out]); je = np.array([o[4] for o in out])
for W in (0.010, 0.020, 0.030):
    for G in (2, 3, 4, 6):
        fires = []
        last = -1.0
        miss = np.cumsum(ic == 0); vb = np.cumsum(ic == 1); va = np.cumsum(ic == -1)
        lo = np.searchsorted(it, it - W)
        for k in range(len(it)):
            if it[k] - last < 0.5:
                continue
            a = lo[k]
            m = miss[k] - (miss[a - 1] if a > 0 else 0)
            if m < G:
                continue
            b_ = vb[k] - (vb[a - 1] if a > 0 else 0)
            s_ = va[k] - (va[a - 1] if a > 0 else 0)
            d = 1 if (b_ >= 1 and s_ == 0) else (-1 if (s_ >= 1 and b_ == 0) else 0)
            if d:
                fires.append((it[k], d)); last = it[k]
        hit = set(); good = bad = 0
        for ft, d in fires:
            j = np.where((jt >= ft - 0.05) & (jt <= ft + 0.05) & (jd == d))[0]
            if len(j):
                good += 1; hit.add(int(j[0]))
            else:
                bad += 1
        print(f'W {W*1000:3.0f} ms G {G}: fires {len(fires):5d}  on a big jump {good:4d}  false {bad:5d}  jumps caught {len(hit):3d}/{len(out)}  their edge {je[list(hit)].sum() if hit else 0:>9,.0f}')
