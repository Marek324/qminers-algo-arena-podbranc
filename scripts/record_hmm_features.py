"""Replay a tape with an hmm-family strategy and record its internal state on a fixed time grid.

The strategy trades normally; every GRID_US the recorder saves the market touch and the strategy's estimates
(Kalman S and variance, regime probability, flow, depth, learned gaps) plus a few cheap tape features. Used to fit
non-linear forecasts of the next mid move offline (scripts/fit_hmm_gbm.py).

usage: .venv/bin/python scripts/record_hmm_features.py strategies/round3/hmm_617_fit.py data/round3_train.csv \
           logs/features/r3_hmm.npz [grid_us]
"""
import math
import sys
from pathlib import Path

import numpy as np

from simulator.main import load_strategy_from_file
from simulator.simulator import ExchangeSimulator, SimulationConfig

COLS = ['t', 'bb', 'ba', 'bq', 'aq', 'bq2', 'aq2', 's', 'P', 'pv', 'cq', 'cv', 'flow', 'depth', 'flow_scale',
        'agg_fast', 'agg_slow', 'n_trades', 'n_new_b', 'n_new_a', 'inv']


def main() -> None:
    strat = load_strategy_from_file(sys.argv[1])
    grid = int(sys.argv[4]) if len(sys.argv) > 4 else 20_000
    rows: list[tuple] = []
    st = {'next': 0, 'af': 0.0, 'as': 0.0, 'tl': 0, 'nt': 0.0, 'nb': 0, 'na': 0, 'lb': None, 'la': None}
    orig = strat.on_order_book_change

    def hook(snapshot, trades):
        orig(snapshot, trades)
        now = snapshot.timestamp_us
        dt = now - st['tl']
        st['tl'] = now
        st['af'] *= math.exp(-dt / 100_000)
        st['as'] *= math.exp(-dt / 1_000_000)
        st['nt'] *= math.exp(-dt / 1_000_000)
        for tr in trades:
            if tr.taker_owner == 'market':
                q = tr.quantity if tr.book_side == 'ask' else -tr.quantity
                st['af'] += q
                st['as'] += q
                st['nt'] += 1
        if snapshot.bids is not st['lb']:
            st['lb'] = snapshot.bids
            st['nb'] += 1
        if snapshot.asks is not st['la']:
            st['la'] = snapshot.asks
            st['na'] += 1
        if now < st['next'] or not snapshot.bids or not snapshot.asks:
            return
        st['next'] = now + grid
        bb = ba = None
        bq = aq = bq2 = aq2 = 0
        for o in snapshot.bids:
            if o.owner != 'market':
                continue
            if bb is None:
                bb = o.price
            if o.price == bb:
                bq += o.quantity
            elif o.price >= bb - 2:
                bq2 += o.quantity
            else:
                break
        for o in snapshot.asks:
            if o.owner != 'market':
                continue
            if ba is None:
                ba = o.price
            if o.price == ba:
                aq += o.quantity
            elif o.price <= ba + 2:
                aq2 += o.quantity
            else:
                break
        if bb is None or ba is None:
            return
        rows.append((now / 1e6, bb, ba, bq, aq, bq2, aq2, strat.s, strat.P, strat.pv, strat.cq, strat.cv, strat.flow,
                     strat.depth, strat.flow_scale, st['af'], st['as'], st['nt'], st['nb'], st['na'], strat.inv))
        st['nb'] = st['na'] = 0

    strat.on_order_book_change = hook
    res = ExchangeSimulator(strat, SimulationConfig()).run(Path(sys.argv[2]))
    np.savez_compressed(sys.argv[3], data=np.array(rows, dtype=np.float64), cols=np.array(COLS))
    print('pnl', res.pnl, 'rows', len(rows))


if __name__ == '__main__':
    main()
