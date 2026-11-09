"""Record a strategy's S estimate (attribute `s`), regime (`pv`) and the visible new market orders and trades.

usage: .venv/bin/python scripts/s_probe.py strategies/<x>.py data/round3_train.csv logs/features/<out>.npz
Saves per callback: t (s), mid, s_hat, pv; per new market limit order seen: nt, nside, npx, nbb, nba (touch before);
per market-taker trade: tt, tpx, tq, tside (+1 buyer-initiated).
Compare s_hat with the true S from IOC prices in logs/features/<tape>_events.npz (scripts/s_eval.py).
"""
import sys
from pathlib import Path

import numpy as np

from simulator.main import load_strategy_from_file
from simulator.simulator import ExchangeSimulator, SimulationConfig

strat = load_strategy_from_file(sys.argv[1])
cb = strat.on_order_book_change
T, M, S, PV = [], [], [], []
NT, NS, NP, NBB, NBA = [], [], [], [], []
TT, TP, TQ, TS = [], [], [], []
st = {'mx': 0, 'bb': 0, 'ba': 0}


def wrapped(snap, trades):
    now = snap.timestamp_us / 1e6
    bb0, ba0 = st['bb'], st['ba']
    mx = st['mx']
    for views, sgn in ((snap.bids, 1), (snap.asks, -1)):
        for o in views:
            if o.owner == 'market':
                i = int(o.order_id)
                if i > mx:
                    NT.append(now); NS.append(sgn); NP.append(o.price); NBB.append(bb0); NBA.append(ba0)
                    st['mx2'] = max(st.get('mx2', 0), i)
    st['mx'] = max(mx, st.get('mx2', 0))
    for tr in trades:
        if tr.taker_owner == 'market':
            TT.append(now); TP.append(tr.price); TQ.append(tr.quantity); TS.append(1 if tr.book_side == 'ask' else -1)
    cb(snap, trades)
    bm = [o.price for o in snap.bids if o.owner == 'market'][:1]
    am = [o.price for o in snap.asks if o.owner == 'market'][:1]
    if bm and am:
        st['bb'], st['ba'] = bm[0], am[0]
        T.append(now); M.append((bm[0] + am[0]) / 2); S.append(strat.s); PV.append(strat.pv)


strat.on_order_book_change = wrapped
res = ExchangeSimulator(strat, SimulationConfig()).run(Path(sys.argv[2]))
print('pnl', res.pnl)
np.savez_compressed(sys.argv[3], t=np.array(T), mid=np.array(M), s=np.array(S), pv=np.array(PV),
                    nt=np.array(NT), nside=np.array(NS), npx=np.array(NP), nbb=np.array(NBB), nba=np.array(NBA),
                    tt=np.array(TT), tpx=np.array(TP), tq=np.array(TQ), tside=np.array(TS))
