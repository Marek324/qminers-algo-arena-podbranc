"""Diagnose a tape: print the numbers that decide which strategy to run (see docs/PLAYBOOK.md).

usage: uv run python scripts/diagnose.py data/<round>.csv

Replays the tape through the real engine with a passive recorder (no orders),
then measures:
  1. structure: spread, depth at the touch, IOC rate and size, order lifetimes
  2. informed flow: mid move after aggressive trades, in the aggressor's direction
  3. maker edge: what a resting order filled by those trades earns vs the later mid
  4. imbalance signal: does touch size imbalance predict the next mid move
  5. price process: variance ratio (>1 trending, <1 mean reverting)
and a suggested strategy family.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.strategy import BaseStrategy


class Recorder(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.b = []
        self.tr = []

    def on_order_book_change(self, s, trades):
        if s.bids and s.asks:
            bb, ba = s.bids[0].price, s.asks[0].price
            bq = sum(o.quantity for o in s.bids if o.price == bb)
            aq = sum(o.quantity for o in s.asks if o.price == ba)
            self.b.append((s.timestamp_us, bb, ba, bq, aq))
        for t in trades:
            self.tr.append((t.timestamp_us, t.price, t.quantity, 1 if t.book_side == 'ask' else -1))


def main(path: str) -> None:
    tape = pd.read_csv(path)
    rec = Recorder()
    ExchangeSimulator(rec, SimulationConfig()).run(Path(path))
    b = pd.DataFrame(rec.b, columns=['t', 'bb', 'ba', 'bq', 'aq'])
    tr = pd.DataFrame(rec.tr, columns=['t', 'p', 'q', 's'])
    T = b.t.values
    mid = ((b.bb + b.ba) / 2).values
    dur = tape.Time.max()

    def idx(t):
        return np.clip(np.searchsorted(T, t, side='left') - 1, 0, len(T) - 1)

    print(f'== {path}: {len(tape):,} rows, {dur / 60:.1f} min')

    # 1. structure
    dt = np.diff(T, append=T[-1])
    sp = (b.ba - b.bb).values
    spd = pd.Series(dt).groupby(sp).sum() / dt.sum()
    ins = tape[tape.Operation == 'Insert']
    ioc = ins[ins.Lifespan == 'F']
    g = ins[ins.Lifespan == 'G'].set_index('OrderId').Time
    canc = tape[tape.Operation == 'Cancel'].set_index('OrderId').Time
    life = (canc - g).dropna()
    print('\n[1] structure')
    print('  spread (time share):', {int(k): round(v, 3) for k, v in spd.items() if v > 0.001})
    print(f'  touch qty median: bid {b.bq.median():.0f}  ask {b.aq.median():.0f}')
    print(f'  IOC: {len(ioc) / dur:.2f}/s, median {ioc.Volume.median():.0f} lots, '
          f'{ioc.Volume.sum() / dur:.0f} lots/s, buy share {(ioc.Side == "B").mean():.2f}')
    print(f'  resting order lifetime median: {life.median():.2f}s')

    # 2 + 3. aggressor events: mid move after, and maker edge
    ev = tr.groupby(['t', 's']).agg(q=('q', 'sum'), vw=('p', 'mean')).reset_index()
    i0 = idx(ev.t.values)
    m0 = mid[i0]
    print('\n[2] mid move after aggressive trades, in aggressor direction (ticks)')
    move = {}
    for h in [0.1, 1, 3, 5, 30]:
        j = idx(ev.t.values + int(h * 1e6)) + 1
        j = np.clip(j, 0, len(mid) - 1)
        move[h] = float((ev.s * (mid[j] - m0)).mean())
    print('  ', {f'{h}s': round(v, 2) for h, v in move.items()})
    half_spread = float((sp * dt).sum() / dt.sum()) / 2
    print(f'  avg half-spread: {half_spread:.2f}')

    print('\n[3] maker edge per lot vs later mid (what a resting order earns)')
    edge = {}
    for h in [1, 5, 30]:
        j = idx(ev.t.values + int(h * 1e6))
        edge[h] = float((ev.s * (ev.vw - mid[j]) * ev.q).sum() / ev.q.sum())
    print('  ', {f'{h}s': round(v, 2) for h, v in edge.items()})
    # split by aggressor size: big orders that empty a level are often informed
    j5 = idx(ev.t.values + 5_000_000)
    ev['e5'] = ev.s * (ev.vw - mid[j5])
    ev['qb'] = pd.qcut(ev.q, 5, duplicates='drop')
    by = ev.groupby('qb', observed=True).apply(lambda g: (g.e5 * g.q).sum() / g.q.sum(), include_groups=False)
    print('   5s edge by aggressor size:', {str(k): round(v, 2) for k, v in by.items()})
    small_edge = float(by.iloc[0])

    # 4. imbalance
    sub = b.iloc[:: max(1, len(b) // 200_000)]
    Ts = sub.t.values
    ms = ((sub.bb + sub.ba) / 2).values
    imb = ((sub.bq - sub.aq) / (sub.bq + sub.aq)).values
    print('\n[4] touch imbalance -> future mid move (correlation)')
    corr = {}
    for h in [0.5, 1, 5]:
        j = np.clip(np.searchsorted(Ts, Ts + int(h * 1e6)), 0, len(Ts) - 1)
        corr[h] = float(np.corrcoef(imb, ms[j] - ms)[0, 1])
    print('  ', {f'{h}s': round(v, 3) for h, v in corr.items()})

    # 5. price process
    sec = pd.Series(mid, index=T // 1_000_000).groupby(level=0).last()
    r1 = sec.diff().dropna()
    vr = {k: float(sec.diff(k).var() / (k * r1.var())) for k in [5, 30, 120] if len(sec) > 3 * k}
    print('\n[5] variance ratio (1 = random walk, >1 trending, <1 mean reverting)')
    print('  ', {f'{k}s': round(v, 2) for k, v in vr.items()},
          f' price range {tape.Price.min():.0f}-{tape.Price.max():.0f}')

    # verdict
    print('\n== suggestion')
    follow = move[3] - 2 * half_spread
    if follow > 1:
        print(f'  FLOW FOLLOWING: mid runs {move[3]:.1f} ticks in 3s after aggression, '
              f'more than the round-trip spread. Start from a momentum taker')
    elif edge[5] > 0.2:
        print(f'  MARKET MAKING: resting fills earn {edge[5]:.2f} ticks/lot after 5s and flow is '
              f'uninformed. Start from strategies/round1/imbalance_mm.py')
    elif small_edge > 0.2:
        print(f'  MARKET MAKING, with care: average fill edge is small ({edge[5]:.2f}) but fills against '
              f'small aggressors earn {small_edge:.2f}; big aggressors are informed. Start from '
              f'strategies/round1/imbalance_mm.py and use the flow term (K_FLOW) so quotes step away after big orders')
    else:
        print('  UNCLEAR: neither edge is large. Check adverse selection by trade size and time of day.')
    if corr[0.5] > 0.2:
        print(f'  imbalance is predictive (corr {corr[0.5]:.2f}): use it to skew quotes / fair value')


if __name__ == '__main__':
    main(sys.argv[1])
