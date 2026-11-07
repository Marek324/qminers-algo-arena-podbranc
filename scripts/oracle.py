"""Perfect-foresight PnL ceilings for a tape: how much could *any* strategy have made?

Usage: uv run python -I scripts/oracle.py data/sample.csv [max_pos ...]     (default max_pos: 50 100)

Replays the tape (market orders only) and reports, for each position limit P:

1. path bound   = P * sum |mid change|
   Always holding P lots in the direction of the next move, at the mid, with no spread and
   unlimited liquidity. A loose upper bound on any strategy.

2. taker oracle = best PnL of a perfect-foresight trader who only takes liquidity: after every
   event they may buy up to the ask queue at the best ask, or sell up to the bid queue at the best
   bid, keeping |inventory| <= P. The end position is marked at the touch. Solved exactly by
   dynamic programming over inventory.
   It is optimistic: no latency, and taking never removes liquidity from later events.
   It is pessimistic in one way: no passive (maker) fills and no levels beyond the touch.

Compare a strategy's pnl with the taker oracle at the same position limit to see what share of
the available edge it captures.
"""
import collections
import sys

import numpy as np
from scipy.ndimage import maximum_filter1d

path = sys.argv[1]
limits = [int(x) for x in sys.argv[2:]] or [50, 100]

# ---- replay: price -> deque of [order_id, qty] per side, FIFO
book = {'B': {}, 'A': {}}
where = {}  # order_id -> (side, price)


def best(side):
    lv = book[side]
    if not lv:
        return None
    return max(lv) if side == 'B' else min(lv)


def level_qty(side, price):
    return sum(q for _, q in book[side][price].values())


def remove_qty(side, price, oid, amount):
    lvl = book[side][price]
    lvl[oid][1] -= amount
    if lvl[oid][1] <= 0:
        del lvl[oid]
        del where[oid]
    if not lvl:
        del book[side][price]


bids, asks, bqs, aqs = [], [], [], []
with open(path) as f:
    next(f)
    for line in f:
        t, op, oid, side, vol, price, life = line.rstrip('\n').split(',')
        if op == 'Insert':
            price, qty = int(price), int(vol)
            opp = 'A' if side == 'B' else 'B'
            while qty > 0:
                b = best(opp)
                if b is None or (side == 'B' and price < b) or (side == 'A' and price > b):
                    break
                for moid, (_, mq) in list(book[opp][b].items()):
                    take = min(qty, mq)
                    qty -= take
                    remove_qty(opp, b, moid, take)
                    if qty == 0:
                        break
            if qty > 0 and life == 'G':
                book[side].setdefault(price, collections.OrderedDict())[oid] = [oid, qty]
                where[oid] = (side, price)
        else:
            loc = where.get(oid)
            if loc:
                s, p = loc
                remove_qty(s, p, oid, book[s][p][oid][1])
        bb, ba = best('B'), best('A')
        if bb is None or ba is None:
            continue
        bids.append(bb)
        asks.append(ba)
        bqs.append(level_qty('B', bb))
        aqs.append(level_qty('A', ba))

bids = np.array(bids, dtype=np.float64)
asks = np.array(asks, dtype=np.float64)
mids = (bids + asks) / 2
print(f'{path}: {len(mids)} events with a two-sided book')
print(f'  mid: start {mids[0]:.1f}  end {mids[-1]:.1f}  min {mids.min():.1f}  max {mids.max():.1f}')
print(f'  mean spread {np.mean(asks - bids):.2f} ticks, mean touch size bid {np.mean(bqs):.1f} / ask {np.mean(aqs):.1f}')
path_len = float(np.abs(np.diff(mids)).sum())
print(f'  sum |mid change| = {path_len:.1f} ticks')

NEG = -1e18
for P in limits:
    n = 2 * P + 1
    pos = np.arange(-P, P + 1, dtype=np.float64)
    V = np.full(n, NEG)
    V[P] = 0.0  # start flat with zero cash
    pad = np.full(2 * P, NEG)
    for bid, ask, bq, aq in zip(bids, asks, bqs, aqs):
        a = min(aq, 2 * P)
        b = min(bq, 2 * P)
        # buy: V[p] = max_{p-a <= p' <= p} V[p'] + (p' - p) * ask   (trailing window of a+1)
        W = np.concatenate([pad[:a], V + pos * ask])
        vb = maximum_filter1d(W, a + 1, mode='constant', cval=NEG)[(a + 1) // 2:(a + 1) // 2 + n] - pos * ask
        # sell: V[p] = max_{p <= p' <= p+b} V[p'] + (p' - p) * bid  (leading window of b+1)
        W = np.concatenate([V + pos * bid, pad[:b]])
        vs = maximum_filter1d(W, b + 1, mode='constant', cval=NEG)[(b + 1) // 2:(b + 1) // 2 + n] - pos * bid
        V = np.maximum(V, np.maximum(vb, vs))
    final = V + np.where(pos >= 0, pos * bids[-1], pos * asks[-1])
    best_i = int(np.argmax(final))
    print(f'  P={P:4d}: path bound {P * path_len:12,.0f}   taker oracle {final[best_i]:12,.0f}'
          f'   (flat at end: {final[P]:12,.0f})')
