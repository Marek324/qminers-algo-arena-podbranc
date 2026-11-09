"""Hindsight and reactive upper bounds on what leverage over 100 can earn on a tape.

Leverage is a rental problem: the penalty is a flat 10,000 $/s while |I| > 100, whatever the size of the excess.
An excursion: take Q extra lots at time t (IOC walking the visible book), give them back at t + h (IOC walking the
book then), pay 10,000 * h. We choose the set of non-overlapping excursions with the largest total (weighted interval
scheduling, a backward DP over a 2 ms grid).

- hindsight: any t, best side and hold. No strategy can beat it (it knows the future, its fills do not move the book).
- reactive(X): entry only where the mid already moved >= X ticks in the last 20 ms as seen 5 ms before the entry
  (the latency), on that side only; the exit is still chosen with hindsight. This bounds "detect the jump, then ride".

.venv/bin/python scripts/lever_oracle.py data/round3_train.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.replay import replay  # noqa: E402
from analysis.tape import load_tape  # noqa: E402

STEP_US = 2_000
HOLDS = (1, 2, 3, 5, 10, 25, 50, 100, 250, 500)  # grid steps (2 ms .. 1 s)
QS = (100, 300, 900)
PEN_PER_STEP = 10_000 * STEP_US / 1e6
LAT = 3       # grid steps of latency (6 ms >= 5 ms)
LOOK = 10     # grid steps of look-back (20 ms)
REACT_X = (1, 2, 3, 5, 8)


def walk_cost(best, depth, q, sign):
    """Cash for q lots walking the visible levels (price = best + sign*k); a thin book rests at the last level."""
    cum = np.cumsum(depth, axis=1)
    prev = np.concatenate([np.zeros((len(depth), 1), dtype=cum.dtype), cum[:, :-1]], axis=1)
    take = np.clip(q - prev, 0, depth)
    k = np.arange(depth.shape[1])
    cash = (take * (best[:, None] + sign * k)).sum(axis=1).astype(float)
    cash += (q - take.sum(axis=1)) * (best + sign * (depth.shape[1] - 1))
    return cash


def schedule(vals):
    """Max total of non-overlapping excursions; vals[h][g] = net of entering at g for h steps. Returns (total, n)."""
    G = len(next(iter(vals.values())))
    best = np.max(np.stack(list(vals.values())), axis=0)
    pos = np.flatnonzero(best > 0)
    if not len(pos):
        return 0.0, 0
    cand = [h for h in vals if (vals[h] > 0).any()]
    opt = np.zeros(G + max(HOLDS) + 2)
    hit = {}
    pset = set(pos.tolist())
    for g in range(pos.max(), pos.min() - 1, -1):
        o = opt[g + 1]
        if g in pset:
            for h in cand:
                x = vals[h][g]
                if x > 0 and x + opt[g + h + 1] > o:
                    o = x + opt[g + h + 1]
                    hit[g] = h
        opt[g] = o
    chosen, g = [], pos.min()
    while g <= pos.max():
        if opt[g] > opt[g + 1] and g in hit:
            chosen.append((g, hit[g]))
            g += hit[g] + 1
        else:
            g += 1
    schedule.last = chosen
    return float(opt[pos.min()]), len(chosen)


def main(path):
    r = replay(load_tape(Path(path)), STEP_US)
    idx = r.state_index(r.grid_ts)
    bb, ba = r.best_bid[idx], r.best_ask[idx]
    ok = ~(np.isnan(bb) | np.isnan(ba))
    bb, ba = np.nan_to_num(bb), np.nan_to_num(ba)
    mid = (bb + ba) / 2
    hours = (r.grid_ts[-1] - r.grid_ts[0]) / 3.6e9
    G = len(bb)
    # signed mid move seen at g - LAT over the previous LOOK steps
    pre = np.zeros(G)
    pre[LAT + LOOK:] = mid[LOOK:G - LAT] - mid[:G - LAT - LOOK]
    print(f'{path}: {G} grid points of {STEP_US / 1000:.0f} ms, {hours:.2f} h')
    for q in QS:
        buy = walk_cost(ba, r.grid_ask_depth, q, +1)
        sell = walk_cost(bb, r.grid_bid_depth, q, -1)
        lng, sht = {}, {}
        for h in HOLDS:
            a, b = slice(0, G - h), slice(h, G)
            both = ok[a] & ok[b]
            L = np.full(G, -np.inf)
            S = np.full(G, -np.inf)
            L[a] = np.where(both, sell[b] - buy[a], -np.inf) - PEN_PER_STEP * h
            S[a] = np.where(both, sell[a] - buy[b], -np.inf) - PEN_PER_STEP * h
            lng[h], sht[h] = L, S
        tot, n = schedule({h: np.maximum(lng[h], sht[h]) for h in HOLDS})
        print(f'\nQ = {q}: hindsight {tot:,.0f} $ ({tot / hours:,.0f} $/h), {n} excursions')
        if DUMP:
            rows = [(int(r.grid_ts[g]), 1 if lng[h][g] >= sht[h][g] else -1, h * STEP_US,
                     round(float(max(lng[h][g], sht[h][g])))) for g, h in schedule.last]
            out = ROOT / 'logs' / f'lever_schedule_{Path(path).stem}_q{q}.json'
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(rows))
            print(f'  schedule (entry_us, side, hold_us, net) -> {out}')
            continue
        for x in REACT_X:
            up, dn = pre >= x, pre <= -x
            v = {h: np.where(up, lng[h], np.where(dn, sht[h], -np.inf)) for h in HOLDS}
            tot, n = schedule(v)
            print(f'  reactive, mid moved >= {x} ticks in 20 ms: {tot:,.0f} $ ({tot / hours:,.0f} $/h), {n} excursions,'
                  f' {(up | dn).sum()} trigger points')
        # expected value: no hindsight at all. Enter at the first grid point of each trigger episode, fixed hold h.
        print('  expected net per excursion (no hindsight), first point of each trigger episode:')
        print('    X  episodes  ' + '  '.join(f'{h * STEP_US // 1000:>7d}ms' for h in HOLDS))
        for x in REACT_X:
            up, dn = pre >= x, pre <= -x
            first_up = up & ~np.r_[False, up[:-1]]
            first_dn = dn & ~np.r_[False, dn[:-1]]
            row = []
            for h in HOLDS:
                net = np.r_[lng[h][first_up], sht[h][first_dn]]
                net = net[np.isfinite(net)]
                row.append(net.mean() if len(net) else np.nan)
            print(f'  {x:3d}  {first_up.sum() + first_dn.sum():8d}  ' + '  '.join(f'{v:9,.0f}' for v in row))


DUMP = '--dump' in sys.argv

if __name__ == '__main__':
    main(sys.argv[1])
