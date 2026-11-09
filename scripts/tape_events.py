"""Replay a tape with a minimal price-time book and save per-event touch, IOCs and new limit prices.

Output .npz arrays (one row per tape row unless noted):
  t (s), bb, ba, bq, aq  touch after the row (market only; 0 when a side is empty)
  ev_kind  0 = limit insert, 1 = IOC, 2 = cancel
  ev_side  +1 buy, -1 sell, 0 cancel
  ev_px, ev_qty  order price / size
  ev_exec  lots the IOC executed (0 for others)
  ev_lvls  price levels the IOC traded at
  ev_deep  deepest price the IOC traded at (0 if none)

The IOC price is round(S +- 1), so ioc price - side gives an S reading. Used by scripts/jump_study.py.

usage: uv run python scripts/tape_events.py data/round2_train.csv logs/features/round2_events.npz
"""
import sys
from collections import deque

import numpy as np


def main(src: str, dst: str) -> None:
    bids: dict[int, deque] = {}
    asks: dict[int, deque] = {}
    orders: dict[int, list] = {}  # id -> [side, price, qty]
    best_b = None
    best_a = None
    rows = []
    with open(src) as f:
        next(f)
        for line in f:
            t, op, oid, side, vol, px, life = line.rstrip('\n').split(',')
            oid = int(oid)
            kind, sgn, p, q, ex, lv, deep = 2, 0, 0, 0, 0, 0, 0
            if op == 'Insert':
                p = int(px)
                q = int(vol)
                sgn = 1 if side == 'B' else -1
                ioc = life == 'F'
                kind = 1 if ioc else 0
                rem = q
                book = asks if sgn > 0 else bids
                levels = set()
                while rem > 0 and book:
                    bp = min(book) if sgn > 0 else max(book)
                    if (sgn > 0 and bp > p) or (sgn < 0 and bp < p):
                        break
                    dq = book[bp]
                    while rem > 0 and dq:
                        o = dq[0]
                        fq = min(rem, o[2])
                        o[2] -= fq
                        rem -= fq
                        levels.add(bp)
                        if o[2] == 0:
                            dq.popleft()
                            orders.pop(o[3], None)
                    if not dq:
                        del book[bp]
                ex = q - rem
                lv = len(levels)
                if levels:
                    deep = max(levels) if sgn > 0 else min(levels)
                if not ioc and rem > 0:
                    o = [sgn, p, rem, oid]
                    orders[oid] = o
                    (bids if sgn > 0 else asks).setdefault(p, deque()).append(o)
            else:
                o = orders.pop(oid, None)
                if o is not None:
                    book = bids if o[0] > 0 else asks
                    dq = book.get(o[1])
                    if dq is not None:
                        try:
                            dq.remove(o)
                        except ValueError:
                            pass
                        if not dq:
                            del book[o[1]]
            bb = max(bids) if bids else 0
            ba = min(asks) if asks else 0
            bq = sum(o[2] for o in bids[bb]) if bids else 0
            aq = sum(o[2] for o in asks[ba]) if asks else 0
            rows.append((float(t), bb, ba, bq, aq, kind, sgn, p, q, ex, lv, deep))
    a = np.array(rows, dtype=np.float64)
    names = ['t', 'bb', 'ba', 'bq', 'aq', 'ev_kind', 'ev_side', 'ev_px', 'ev_qty', 'ev_exec', 'ev_lvls', 'ev_deep']
    np.savez_compressed(dst, **{n: a[:, i] for i, n in enumerate(names)})
    print('saved', dst, a.shape)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
