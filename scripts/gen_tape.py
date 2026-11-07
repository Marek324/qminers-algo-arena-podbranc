"""Generate a synthetic tape from the round1 generator model (docs/DATA_ANALYSIS.md).

Model ("engine A"):
- Hidden fair price S: Brownian motion, dS = sigma dW, S0 = 10000 (ticks), plus a permanent jump
  side * kappa * size**alpha at every IOC (round1's big IOCs move the price for good).
- Limit orders: Poisson LIMIT_RATE/s, side a fair coin. Distance X ~ loc + Gamma(k, theta).
  buy = floor(S - X), sell = ceil(S + X), clamped so they never cross the market touch.
  Size = max(1, round(exp(N(size_mu, size_sigma)))).
  Each limit gets a cancel timer Exp(CANCEL_RATE) at insert; the cancel row is written
  even if the order was filled meanwhile (round1 does this; the exchange ignores it).
- IOCs: Poisson IOC_RATE/s, side a fair coin, price round(S + IOC_OFF) / round(S - IOC_OFF),
  size = max(1, round(exp(N(ioc_mu, ioc_sigma)))).
- Timestamps in whole microseconds, strictly increasing (min gap 1us). Ids from 1000001, +1 per insert.

Writes <out>.csv (simulator format) and <out>_S.npz with the true S at every row:
  t_us (int64, row timestamp), S (float64, S right after that row), mid (market-only mid after that row).

Usage:
  .venv/bin/python scripts/gen_tape.py --seed 1                 # -> data/synthetic/synth_r1_s1.csv
  .venv/bin/python scripts/gen_tape.py --seed 1 2 3             # the three twins used in docs/DATA_ANALYSIS.md
  .venv/bin/python scripts/gen_tape.py --out data/synthetic/synth_short --duration 1200 --seed 2
"""

from __future__ import annotations

import argparse
import heapq
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

# round1 fits (docs/DATA_ANALYSIS.md, findings/barbora/direct_params.json)
R1 = dict(
    sigma=0.5,                 # diffusive part only; IOC impact below adds the rest (mid vol ~0.95)
    limit_rate=179.80,
    x_k=1.72, x_loc=0.054, x_theta=0.32,
    size_mu=2.658, size_sigma=1.799,
    cancel_rate=2.10,          # mean timer 0.476s; measured hazard ~2.0-2.14/s on every level
    ioc_rate=8.007,
    ioc_off=1.0,
    ioc_mu=4.106, ioc_sigma=1.396,
    # permanent impact of an IOC on S: S += side * kappa * q**alpha, q = order size
    # (impact_on=0) or executed size (impact_on=1). Not in DATA_ANALYSIS.md; fitted to
    # round1's impact_by_size table (logs/analysis/round1.json).
    # Fit (scripts/fit_synth.py, 1200s tapes): kappa 0.0075, alpha 0.7, sigma 0.5 reproduces round1's
    # impact curve, maker markout (0.07), touch depth, hindsight max and imbalance_mm PnL (~610k/12000s).
    # PnL is sensitive to kappa: 0.0065 gives ~660-700k.
    kappa=0.0075, alpha=0.7, impact_on=0.0,
)


def generate(out: str, duration: float, seed: int, p: dict) -> None:
    rng = np.random.default_rng(seed)
    lam = p['limit_rate'] + p['ioc_rate']
    n = int(lam * duration * 1.02) + 1000
    gaps = rng.exponential(1.0 / lam, n)
    t_arr = np.cumsum(gaps)
    t_arr = t_arr[t_arr < duration]
    n = len(t_arr)
    is_ioc = rng.random(n) < p['ioc_rate'] / lam
    is_buy = rng.random(n) < 0.5
    xdist = p['x_loc'] + rng.gamma(p['x_k'], p['x_theta'], n)
    lsize = np.maximum(1, np.rint(np.exp(rng.normal(p['size_mu'], p['size_sigma'], n)))).astype(np.int64)
    isize = np.maximum(1, np.rint(np.exp(rng.normal(p['ioc_mu'], p['ioc_sigma'], n)))).astype(np.int64)
    life = rng.exponential(1.0 / p['cancel_rate'], n)
    # S noise: one normal per arrival and a pool for cancel events
    z_arr = rng.standard_normal(n)
    sigma = p['sigma']

    # book state (market orders only)
    orders: dict[int, list] = {}         # id -> [side(+1 buy/-1 sell), price, qty]
    levels: dict[tuple, deque] = {}      # (side, price) -> deque of ids
    lvl_qty: dict[tuple, int] = {}
    bid_heap: list[int] = []             # -price
    ask_heap: list[int] = []
    cancels: list[tuple] = []            # (time, id)

    def best_bid():
        while bid_heap:
            px = -bid_heap[0]
            if lvl_qty.get((1, px), 0) > 0:
                return px
            heapq.heappop(bid_heap)
        return None

    def best_ask():
        while ask_heap:
            px = ask_heap[0]
            if lvl_qty.get((-1, px), 0) > 0:
                return px
            heapq.heappop(ask_heap)
        return None

    def add(oid, side, px, q):
        orders[oid] = [side, px, q]
        key = (side, px)
        dq = levels.get(key)
        if dq is None:
            dq = deque()
            levels[key] = dq
        dq.append(oid)
        old = lvl_qty.get(key, 0)
        lvl_qty[key] = old + q
        if old == 0:
            heapq.heappush(bid_heap if side > 0 else ask_heap, -px if side > 0 else px)

    def remove(oid):
        o = orders.pop(oid, None)
        if o is None:
            return
        key = (o[0], o[1])
        lvl_qty[key] -= o[2]
        # id stays in deque; skipped lazily during matching

    def ioc(side, px, q):
        opp = -side
        q0 = q
        while q > 0:
            b = best_ask() if side > 0 else best_bid()
            if b is None or (side > 0 and b > px) or (side < 0 and b < px):
                return q0 - q
            key = (opp, b)
            dq = levels[key]
            while q > 0 and dq:
                oid = dq[0]
                o = orders.get(oid)
                if o is None or o[1] != b:
                    dq.popleft()
                    continue
                f = min(q, o[2])
                o[2] -= f
                lvl_qty[key] -= f
                q -= f
                if o[2] == 0:
                    del orders[oid]
                    dq.popleft()
        return q0

    rows: list[str] = []
    s_t = np.empty(2 * n + 10, dtype=np.int64)
    s_v = np.empty_like(s_t, dtype=np.float64)
    s_mid = np.empty_like(s_v)
    nrow = 0
    S = 10000.0
    t_prev = 0.0
    last_us = 0
    next_id = 1000001
    ai = 0
    zc = rng.standard_normal(len(s_t))   # S noise for cancel rows
    zi = 0
    t0 = time.time()

    while ai < n or cancels:
        ta = t_arr[ai] if ai < n else math.inf
        if cancels and cancels[0][0] <= ta:
            tc, oid = heapq.heappop(cancels)
            if tc >= duration:
                cancels.clear()
                if ai >= n:
                    break
                continue
            dt = tc - t_prev
            S += sigma * math.sqrt(dt) * zc[zi]
            zi += 1
            t_prev = tc
            us = max(last_us + 1, int(round(tc * 1e6)))
            last_us = us
            remove(oid)
            rows.append(f'{us / 1e6:.6f},Cancel,{oid},,,,')
        else:
            dt = ta - t_prev
            S += sigma * math.sqrt(dt) * z_arr[ai]
            t_prev = ta
            us = max(last_us + 1, int(round(ta * 1e6)))
            last_us = us
            oid = next_id
            next_id += 1
            side = 1 if is_buy[ai] else -1
            sc = 'B' if side > 0 else 'A'
            if is_ioc[ai]:
                px = math.floor(S + side * p['ioc_off'] + 0.5)
                q = int(isize[ai])
                filled = ioc(side, px, q)
                qi = filled if p['impact_on'] else q
                if p['kappa'] and qi > 0:
                    S += side * p['kappa'] * qi ** p['alpha']
                rows.append(f'{us / 1e6:.6f},Insert,{oid},{sc},{q},{px},F')
            else:
                x = xdist[ai]
                q = int(lsize[ai])
                if side > 0:
                    px = math.floor(S - x)
                    a = best_ask()
                    if a is not None and px >= a:
                        px = a - 1
                else:
                    px = math.ceil(S + x)
                    b = best_bid()
                    if b is not None and px <= b:
                        px = b + 1
                add(oid, side, px, q)
                heapq.heappush(cancels, (ta + life[ai], oid))
                rows.append(f'{us / 1e6:.6f},Insert,{oid},{sc},{q},{px},G')
            ai += 1
        s_t[nrow] = last_us
        s_v[nrow] = S
        bb_, ba_ = best_bid(), best_ask()
        s_mid[nrow] = (bb_ + ba_) / 2 if bb_ is not None and ba_ is not None else np.nan
        nrow += 1
        if nrow % 1_000_000 == 0:
            print(f'  {nrow:,} rows, t={t_prev:.0f}s, {time.time() - t0:.0f}s elapsed', flush=True)

    with open(out + '.csv', 'w') as f:
        f.write('Time,Operation,OrderId,Side,Volume,Price,Lifespan\n')
        f.write('\n'.join(rows))
        f.write('\n')
    np.savez_compressed(out + '_S.npz', t_us=s_t[:nrow], S=s_v[:nrow], mid=s_mid[:nrow])
    print(f'wrote {out}.csv ({nrow:,} rows, {time.time() - t0:.0f}s) and {out}_S.npz')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', help='output path without extension (default data/synthetic/synth_r1_s<seed>)')
    ap.add_argument('--duration', type=float, default=12000.0)
    ap.add_argument('--seed', type=int, nargs='+', default=[1], help='one tape per seed; same seed, same tape')
    for k, v in R1.items():
        ap.add_argument('--' + k.replace('_', '-'), type=float, default=v)
    a = ap.parse_args()
    p = {k: getattr(a, k) for k in R1}
    if a.out and len(a.seed) > 1:
        ap.error('--out takes a single --seed')
    for seed in a.seed:
        out = a.out or f'data/synthetic/synth_r1_s{seed}'
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        generate(out, a.duration, seed, p)


if __name__ == '__main__':
    main()
