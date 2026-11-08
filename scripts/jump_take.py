"""Is there money in taking stale orders after a jump? For every new market limit order placed >= FAR ticks behind the
touch (S has jumped away from the book on that side), look at the book LATENCY later and count the stale liquidity
on the run-over side priced better than the new S estimate (the far order's price +- X_MEAN), and the edge per lot.
Also checks how much of that edge the mid realizes 2 s later (does the book really catch up).

uv run python scripts/jump_take.py data/round2_public.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.tape import load_tape  # noqa: E402

US = 1_000_000
LATENCY = 5_000
X_MEAN = 0.8


def main(path: str, far: int = 6, cap: int = 200) -> None:
    t = load_tape(Path(path))
    ts, op, oid = t.ts_us.to_numpy(), t.is_insert.to_numpy(), t.OrderId.to_numpy()
    side = t.Side.astype(str).to_numpy()
    vol = t.Volume.fillna(0).to_numpy()
    px = t.Price.fillna(0).to_numpy()
    ioc = (t.Lifespan.astype(str).to_numpy() == 'F')
    bids: dict[int, int] = {}
    asks: dict[int, int] = {}
    orders: dict[int, tuple[str, int, int]] = {}
    pending = []  # (due_ts, run-over side 'B'/'A', S estimate, trigger ts)
    events = []
    mids_t, mids_v = [], []

    def best(book, hi):
        return (max(book) if hi else min(book)) if book else None

    for i in range(len(t)):
        now = ts[i]
        while pending and pending[0][0] <= now:
            due, s, S, t0 = pending.pop(0)
            book = bids if s == 'B' else asks
            if s == 'B':
                lv = sorted(((p, q) for p, q in book.items() if p > S), reverse=True)
            else:
                lv = sorted((p, q) for p, q in book.items() if p < S)
            take, edge = 0, 0.0
            for p, q in lv:
                k = min(q, cap - take)
                if k <= 0:
                    break
                take += k
                edge += k * abs(p - S)
            events.append((t0, s, S, take, edge))
        if op[i]:
            s, q, p = side[i], int(vol[i]), int(px[i])
            if ioc[i]:
                book = asks if s == 'B' else bids
                rem = q
                while rem > 0 and book:
                    lvl = min(book) if s == 'B' else max(book)
                    if (s == 'B' and lvl > p) or (s == 'A' and lvl < p):
                        break
                    k = min(rem, book[lvl])
                    rem -= k
                    book[lvl] -= k
                    if book[lvl] == 0:
                        del book[lvl]
                    need = k
                    for o in [o for o, v in orders.items() if v[1] == lvl and v[0] != s][:50]:
                        if need <= 0:
                            break
                        d = min(need, orders[o][2])
                        orders[o] = (orders[o][0], lvl, orders[o][2] - d)
                        need -= d
                        if orders[o][2] == 0:
                            del orders[o]
            else:
                bb, ba = best(bids, True), best(asks, False)
                if s == 'B' and ba is not None and (ba - 1 - p) >= far:
                    pending.append((now + LATENCY, 'B', p + X_MEAN, now))  # buy far below: S fell, bids are stale
                elif s == 'A' and bb is not None and (p - bb - 1) >= far:
                    pending.append((now + LATENCY, 'A', p - X_MEAN, now))
                book = bids if s == 'B' else asks
                book[p] = book.get(p, 0) + q
                orders[oid[i]] = (s, p, q)
        else:
            o = orders.pop(oid[i], None)
            if o is not None and o[2] > 0:
                book = bids if o[0] == 'B' else asks
                if o[1] in book:  # the simplified replay can already have emptied this level
                    book[o[1]] -= o[2]
                    if book[o[1]] <= 0:
                        del book[o[1]]
        bb, ba = best(bids, True), best(asks, False)
        if bb is not None and ba is not None:
            mids_t.append(now)
            mids_v.append((bb + ba) / 2)

    mt, mv = np.array(mids_t), np.array(mids_v)
    E = pd.DataFrame(events, columns=['t', 'side', 'S', 'take', 'edge'])
    dur = (ts[-1] - ts[0]) / US
    # dedupe: one opportunity per side per 1 s burst
    E = E.sort_values('t')
    E['burst'] = ((E.t.diff() > US) | (E.side != E.side.shift())).cumsum()
    first = E.groupby('burst').first()
    best_ = E.groupby('burst').agg(take=('take', 'max'), edge=('edge', 'max'))
    j0 = np.searchsorted(mt, first.t.to_numpy(), 'right') - 1
    j2 = np.searchsorted(mt, first.t.to_numpy() + 2 * US, 'right') - 1
    sgn = np.where(first.side == 'B', -1.0, 1.0)
    realized = (mv[j2] - mv[j0]) * sgn
    implied = np.abs(mv[j0] - first.S.to_numpy())
    print(f'{path}: far >= {far} ticks, cap {cap} lots, latency {LATENCY / 1000:.0f} ms')
    print(f'  trigger orders {len(E)}, bursts {len(first)} ({len(first) / dur * 3600:.0f}/h)')
    print(f'  first trigger of a burst: stale lots takeable after latency: mean {first['take'].mean():.0f}, '
          f'share with any {(first['take'] > 0).mean():.0%}; edge (lot-ticks vs new S) mean {first['edge'].mean():,.0f}, total {first['edge'].sum():,.0f} '
          f'({first['edge'].sum() / dur * 3600:,.0f}/h)')
    print(f'  best trigger within each burst: edge total {best_['edge'].sum():,.0f} ({best_['edge'].sum() / dur * 3600:,.0f}/h)')
    print(f'  mid move toward the jump within 2 s: mean {realized.mean():.2f} ticks vs implied gap mid-S {implied.mean():.2f}; '
          f'share realized {np.mean(realized) / max(np.mean(implied), 1e-9):.0%}')
    big = first['edge'] > 1000
    print(f'  bursts with edge > 1000: {big.sum()}, their edge {first['edge'][big].sum():,.0f}')


if __name__ == '__main__':
    main(sys.argv[1], *(int(a) for a in sys.argv[2:]))
