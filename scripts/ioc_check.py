"""Are IOCs informed, momentum-driven or bursty? Replays the book and measures, per IOC:
side vs delta (recent new-limit prices minus mid), side vs the past mid move, signed mid move after the IOC by size,
and clustering of IOC arrivals and signs.

uv run python scripts/ioc_check.py data/round2_train.csv [data/round1_train.csv ...]
"""

import sys
from bisect import bisect_left, bisect_right
from collections import deque

import numpy as np
import pandas as pd


def run(path):
    df = pd.read_csv(path)
    t = df.Time.to_numpy()
    op = df.Operation.to_numpy()
    oid = df.OrderId.to_numpy()
    side = df.Side.to_numpy()
    vol = df.Volume.to_numpy()
    px = df.Price.to_numpy()
    life = df.Lifespan.to_numpy()

    bids, asks = {}, {}  # price -> qty
    orders = {}  # id -> (side, price, qty)
    recent = deque()  # (t, price) of new limit orders
    mid_t, mid_v = [], []
    iocs = []  # (t, sign, size, mid_before, delta)
    best_b = best_a = None

    def best():
        return (max(bids) if bids else None), (min(asks) if asks else None)

    for i in range(len(df)):
        ti = t[i]
        if op[i] == 'Insert':
            s, p, q = side[i], px[i], vol[i]
            if life[i] == 'G':
                book = bids if s == 'B' else asks
                book[p] = book.get(p, 0) + q
                orders[oid[i]] = [s, p, q]
                recent.append((ti, p))
            else:
                b, a = best()
                mid = (b + a) / 2 if b is not None and a is not None else None
                while recent and recent[0][0] < ti - 0.3:
                    recent.popleft()
                delta = (np.mean([r[1] for r in recent]) - mid) if (recent and mid is not None) else np.nan
                sign = 1 if s == 'B' else -1
                iocs.append((ti, sign, q, mid if mid is not None else np.nan, delta))
                # execute against the book
                rem = q
                book = asks if sign > 0 else bids
                while rem > 0 and book:
                    lvl = min(book) if sign > 0 else max(book)
                    if (sign > 0 and lvl > p) or (sign < 0 and lvl < p):
                        break
                    take = min(rem, book[lvl])
                    rem -= take
                    book[lvl] -= take
                    if book[lvl] == 0:
                        del book[lvl]
                    # reduce resting orders FIFO-agnostic: drop qty lazily by price (fine for mid tracking)
                    need = take
                    for k in [k for k, o in orders.items() if o[1] == lvl and o[0] != s][: 50]:
                        if need <= 0:
                            break
                        d = min(need, orders[k][2])
                        orders[k][2] -= d
                        need -= d
                        if orders[k][2] == 0:
                            del orders[k]
        else:
            o = orders.pop(oid[i], None)
            if o is not None and o[2] > 0:
                book = bids if o[0] == 'B' else asks
                book[o[1]] -= o[2]
                if book[o[1]] <= 0:
                    del book[o[1]]
        b, a = best()
        if b is not None and a is not None and (b, a) != (best_b, best_a):
            best_b, best_a = b, a
            mid_t.append(ti)
            mid_v.append((b + a) / 2)

    mid_t = np.array(mid_t)
    mid_v = np.array(mid_v)

    def mid_at(x):
        j = np.searchsorted(mid_t, x, side='right') - 1
        return mid_v[np.clip(j, 0, len(mid_v) - 1)]

    I = pd.DataFrame(iocs, columns=['t', 'sign', 'size', 'mid', 'delta']).dropna()
    for h in (1, 5, 10, 30):
        I[f'm{h}'] = I.sign * (mid_at(I.t + h) - I.mid)
    for h in (1, 5):
        I[f'past{h}'] = np.sign(I.mid - mid_at(I.t - h))

    print(f'\n=== {path}: {len(I):,} IOCs over {t[-1]:.0f}s ({len(I) / t[-1]:.2f}/s)')
    print('P(buy) by delta bucket (new limits last 300ms minus mid):')
    I['db'] = pd.cut(I.delta, [-9, -0.75, -0.25, 0.25, 0.75, 9])
    print((I.groupby('db', observed=True).sign.agg(['count', lambda s: (s > 0).mean()])).to_string())
    for h in (1, 5):
        g = I.groupby(f'past{h}').sign.agg(['count', lambda s: (s > 0).mean()])
        print(f'P(buy) by sign of the past {h}s mid move:\n{g.to_string()}')
    print('signed mid move after IOC (ticks) by size:')
    I['sb'] = pd.qcut(I['size'], 6, duplicates='drop')
    print(I.groupby('sb', observed=True)[['m1', 'm5', 'm10', 'm30']].mean().round(3).to_string())
    k, b = np.polyfit(np.log(I.groupby('sb', observed=True)['size'].mean()),
                      np.log(I.groupby('sb', observed=True)['m10'].mean().clip(1e-4)), 1)
    print(f'power fit m10 ~ {np.exp(b):.4f} * size^{k:.2f}')
    s = I.sign.to_numpy()
    print(f'sign autocorr lag1..5: {[round(float(np.corrcoef(s[:-L], s[L:])[0, 1]), 3) for L in range(1, 6)]}')
    gaps = np.diff(I.t.to_numpy())
    print(f'inter-IOC gaps: mean {gaps.mean():.3f}s, CV {gaps.std() / gaps.mean():.2f} (Poisson = 1)')
    cnt = np.bincount(I.t.astype(int))
    print(f'IOCs per second: mean {cnt.mean():.2f}, var/mean {cnt.var() / cnt.mean():.2f} (Poisson = 1)')
    net = pd.Series(I.sign.to_numpy() * I['size'].to_numpy()).groupby((I.t // 10).astype(int).to_numpy()).sum()
    dm = pd.Series(mid_at(np.arange(len(net) + 1) * 10.0)).diff().shift(-1).iloc[: len(net)].to_numpy()
    ok = ~np.isnan(dm)
    print(f'corr(net IOC volume in 10s bin, mid change same bin) {np.corrcoef(net.to_numpy()[ok], dm[ok])[0, 1]:.3f}; '
          f'next bin {np.corrcoef(net.to_numpy()[ok][:-1], dm[ok][1:])[0, 1]:.3f}; '
          f'net-volume autocorr {pd.Series(net.to_numpy()).autocorr():.3f}')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        run(p)
