"""Economics of a leveraged jump trade, straight from the tape (no strategy run).

At each jump signal (CONFIRM far orders on one side within 50 ms, FAR+ ticks behind the opposite best, agreeing to 3
ticks), 5 ms later take every stale level better than the new S estimate (least extreme far order -+ X_MEAN, then
-+ MARGIN) up to CAP lots. After a hold of T, sell those same lots back into the book as it is then (walking the
levels; the IOC exit) and charge the penalty for T. Reports the average and total net per (trigger, margin, T), and
the gross before the penalty, so a profitable leverage design shows up as a positive row. Our own orders are not in
the book and do not move it; exits only see market orders.

uv run python scripts/lever_econ.py data/round3_public.csv
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
HOLDS_MS = (50, 100, 200, 500, 1000, 2000)
PENALTY_PER_US = 10_000 / US


def walk(levels, lots):
    """Proceeds of trading `lots` through (price, qty) levels in order; None if the book is too thin."""
    got, left = 0.0, lots
    for p, q in levels:
        k = min(q, left)
        got += k * p
        left -= k
        if left <= 0:
            return got
    return None


def main(path: str) -> None:
    t = load_tape(Path(path))
    ts, op, oid = t.ts_us.to_numpy(), t.is_insert.to_numpy(), t.OrderId.to_numpy()
    side = t.Side.astype(str).to_numpy()
    vol = t.Volume.fillna(0).to_numpy()
    px = t.Price.fillna(0).to_numpy()
    ioc = t.Lifespan.astype(str).to_numpy() == 'F'
    bids: dict[int, int] = {}
    asks: dict[int, int] = {}
    orders: dict[int, list] = {}
    far_recent: list = []           # (t, side 'B'/'A', price)
    last_sig = {'B': -10**12, 'A': -10**12}
    pend_entry: list = []           # (due, side, s_new, sig_t, far_px)
    pend_exit: list = []            # (due, idx, hold_ms)
    trades: list = []               # entries: dict(side, levels taken per (far, margin, cap))
    results: list = []
    CONFIGS = [(far, conf, margin) for far in (8, 10, 15) for conf in (2, 3) for margin in (3.0, 6.0, 10.0, 15.0, 20.0)]
    CAP = 900  # extra lots beyond a flat-to-100 position

    for i in range(len(t)):
        now = ts[i]
        # entries due: snapshot the stale side
        while pend_entry and pend_entry[0][0] <= now:
            _, s, s_new, sig_t, cfg = pend_entry.pop(0)
            far, conf, margin = cfg
            if s == 'B':  # buys far below: S fell; sell into bids above s_new + margin
                lv = sorted(((p, q) for p, q in bids.items() if p >= s_new + margin), reverse=True)
            else:
                lv = sorted(((p, q) for p, q in asks.items() if p <= s_new - margin))
            lots, cost = 0, 0.0
            for p, q in lv:
                k = min(q, CAP - lots)
                if k <= 0:
                    break
                lots += k
                cost += k * p
            if lots >= 10:
                idx = len(trades)
                trades.append({'cfg': cfg, 's': s, 'lots': lots, 'entry_px': cost / lots, 't': now})
                for h in HOLDS_MS:
                    pend_exit.append((now + h * 1000, idx, h))
                pend_exit.sort()
        while pend_exit and pend_exit[0][0] <= now:
            _, idx, h = pend_exit.pop(0)
            tr = trades[idx]
            if tr['s'] == 'B':  # we sold stale bids (short); buy back from the asks now
                proceeds = walk(sorted(asks.items()), tr['lots'])
                gross = None if proceeds is None else tr['lots'] * tr['entry_px'] - proceeds
            else:               # we bought stale asks (long); sell into the bids now
                proceeds = walk(sorted(bids.items(), reverse=True), tr['lots'])
                gross = None if proceeds is None else proceeds - tr['lots'] * tr['entry_px']
            if gross is not None:
                results.append((*tr['cfg'], h, tr['lots'], gross, gross - (h * 1000 + LATENCY) * PENALTY_PER_US))
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
                        orders[o][2] -= d
                        need -= d
                        if orders[o][2] == 0:
                            del orders[o]
            else:
                bb = max(bids) if bids else None
                ba = min(asks) if asks else None
                behind = (ba - 1 - p) if (s == 'B' and ba is not None) else ((p - bb - 1) if (s == 'A' and bb is not None) else -1)
                if behind >= 8:
                    far_recent = [f for f in far_recent if now - f[0] <= 50_000]
                    for far, conf, margin in CONFIGS:
                        if behind < far or now - last_sig.get((s, far, conf, margin), -10**12) < US:
                            continue
                        agree = [f for f in far_recent if f[1] == s and abs(f[2] - p) <= 3 and f[3] >= far]
                        if len(agree) + 1 >= conf:
                            pp = max([f[2] for f in agree] + [p]) if s == 'B' else min([f[2] for f in agree] + [p])
                            s_new = pp + X_MEAN if s == 'B' else pp - X_MEAN
                            pend_entry.append((now + LATENCY, s, s_new, now, (far, conf, margin)))
                            last_sig[(s, far, conf, margin)] = now
                    far_recent.append((now, s, p, behind))
                    pend_entry.sort(key=lambda x: x[0])
                book = bids if s == 'B' else asks
                book[p] = book.get(p, 0) + q
                orders[oid[i]] = [s, p, q]
        else:
            o = orders.pop(oid[i], None)
            if o is not None and o[2] > 0:
                book = bids if o[0] == 'B' else asks
                if o[1] in book:
                    book[o[1]] -= o[2]
                    if book[o[1]] <= 0:
                        del book[o[1]]

    R = pd.DataFrame(results, columns=['far', 'conf', 'margin', 'hold_ms', 'lots', 'gross', 'net'])
    dur_h = (ts[-1] - ts[0]) / US / 3600
    g = R.groupby(['far', 'conf', 'margin', 'hold_ms']).agg(n=('net', 'size'), lots=('lots', 'mean'),
                                                          gross_avg=('gross', 'mean'), net_avg=('net', 'mean'),
                                                          net_total=('net', 'sum'))
    g['net_per_h'] = g.net_total / dur_h
    print(f'{path}: leveraged jump trades, extra lots up to {CAP}, IOC entry 5 ms after the signal and IOC exit after the hold')
    print(g.round(0).sort_values('net_total', ascending=False).head(25).to_string())
    print('\nbest hold per (far, conf, margin):')
    best = g.reset_index().sort_values('net_total', ascending=False).groupby(['far', 'conf', 'margin']).head(1)
    print(best.round(0).head(15).to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1])
