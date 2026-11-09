"""Log every take candidate of a strategy with features and its executable markout (meta-labelling data).

Runs the strategy in the simulator (in-process), and on each book event where fair is at least EDGE_LO
through the opposite best price, records one row per side: features at decision time, whether the strategy's
own rule takes, and the label. Label = what one lot taken at that price earns: the order lands 5 ms later,
fills only if the opposite best is still at that price or better, and is marked against the mid h seconds on.

usage: .venv/bin/python scripts/take_candidates.py strategies/round3/hmm_deep_clear_ks.py data/round3_train.csv \
           --out logs/takecand/r3.npz
Works for strategies with the hmm_deep_clear_ks attribute names (s, P, pv, mid, inv, flow_scale, flow, depth).
"""
import argparse
import importlib.util
import math
from collections import deque
from pathlib import Path

import numpy as np

from simulator.main import (MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS,
                            POSITION_BREACH_PENALTY_PER_SECOND)
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.strategy import BaseStrategy

EDGE_LO = 0.0
DEDUP_US = 50_000
HORIZONS = (1.0, 5.0, 30.0)
FEATS = ['edge', 'ds', 'dimb', 'dfx', 'pv', 'P', 'spread', 'tq', 'oq', 'depth', 'dinv', 'r01', 'r1', 'r5',
         'n_new1', 'far1', 'tvol1', 'dtflow1', 'since_vol', 'djv', 'vol_regime']


def load_class(path: Path):
    spec = importlib.util.spec_from_file_location('strat_mod', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    classes = sorted((v for v in vars(mod).values()
                      if isinstance(v, type) and issubclass(v, BaseStrategy) and v is not BaseStrategy),
                     key=lambda c: c.__name__)
    return classes[0]


def make_logger(base):
    class Logger(base):
        def __init__(self):
            super().__init__()
            self.rows = []        # (t, side, price, take, *feats)
            self.ts = []          # every event: t, bb, ba
            self.bbs = []
            self.bas = []
            self.last_rec = {1: (None, 0), -1: (None, 0)}
            self.new_t = deque()      # (t, far) of new market orders
            self.tk = deque()         # (t, signed qty) of market taker trades
            self.mids = deque()       # (t, mid), 5 s
            self.vol_since = 0
            self.prev_vol = False
            self._seen_max = ''

        def _observe(self, sign, price, depth, now):
            far = 1 if sign * (price - self.mid) > 1.5 else 0
            self.new_t.append((now, far))
            super()._observe(sign, price, depth, now)

        def on_order_book_change(self, snapshot, trades):
            super().on_order_book_change(snapshot, trades)
            bids, asks = snapshot.bids, snapshot.asks
            if not bids or not asks:
                return
            bb = ba = None
            bq = aq = 0
            for o in bids:
                if o.owner == 'market':
                    if bb is None:
                        bb = o.price
                    elif o.price != bb:
                        break
                    bq += o.quantity
            for o in asks:
                if o.owner == 'market':
                    if ba is None:
                        ba = o.price
                    elif o.price != ba:
                        break
                    aq += o.quantity
            if bb is None or ba is None:
                return
            now = snapshot.timestamp_us
            mid = (bb + ba) / 2
            self.ts.append(now)
            self.bbs.append(bb)
            self.bas.append(ba)
            for tr in trades:
                if tr.taker_owner == 'market':
                    self.tk.append((now, tr.quantity if tr.book_side == 'ask' else -tr.quantity))
            self.mids.append((now, mid))
            while self.mids and now - self.mids[0][0] > 5_000_000:
                self.mids.popleft()
            while self.new_t and now - self.new_t[0][0] > 1_000_000:
                self.new_t.popleft()
            while self.tk and now - self.tk[0][0] > 1_000_000:
                self.tk.popleft()
            vol = self.pv >= self.P_ON
            if vol and not self.prev_vol:
                self.vol_since = now
            self.prev_vol = vol
            fx = self.flow_scale * (self.flow / self.depth if self.depth else 0.0)
            imb = (bq - aq) / (bq + aq)
            if vol:
                fair = mid + self.V_K_S * (self.s - mid) + self.V_K_FLOW * fx - self.V_K_INV * self.inv
                thr = self.V_EDGE_TAKE
            else:
                fair = mid + self.K_S * (self.s - mid) + self.K_IMB * imb + self.K_FLOW * fx - self.K_INV * self.inv
                thr = self.EDGE_TAKE
            for d, price, tq, oq in ((1, ba, aq, bq), (-1, bb, bq, aq)):
                edge = d * (fair - price)
                if edge < EDGE_LO:
                    continue
                key = (price, vol)
                lk, lt = self.last_rec[d]
                if lk == key and now - lt < DEDUP_US:
                    continue
                self.last_rec[d] = (key, now)

                def ret(dt):
                    for t0, m0 in self.mids:
                        if now - t0 <= dt:
                            return d * (mid - m0)
                    return 0.0
                n_new = len(self.new_t)
                far = sum(f for _, f in self.new_t)
                tvol = sum(abs(q) for _, q in self.tk)
                tfl = sum(q for _, q in self.tk)
                self.rows.append((now, d, price, 1 if edge >= thr else 0,
                                  edge, d * (self.s - mid), d * imb, d * fx, self.pv, self.P, ba - bb, tq, oq,
                                  self.depth, d * self.inv / 100.0, ret(100_000), ret(1_000_000), ret(5_000_000),
                                  n_new, far / n_new if n_new else 0.0, tvol, d * tfl,
                                  min(30.0, (now - self.vol_since) / 1e6) if vol else -1.0,
                                  d * self.jv, 1.0 if vol else 0.0))
    return Logger


def labels(rows, ts, bbs, bas):
    ts = np.asarray(ts)
    bbs = np.asarray(bbs, float)
    bas = np.asarray(bas, float)
    mids = (bbs + bas) / 2
    t = rows[:, 0]
    d = rows[:, 1]
    p = rows[:, 2]
    i5 = np.minimum(np.searchsorted(ts, t + ORDER_ACCEPT_DELAY_MS * 1000), len(ts) - 1)
    surv = np.where(d > 0, bas[i5] <= p, bbs[i5] >= p).astype(float)
    out = {'surv': surv}
    for h in HORIZONS:
        ih = np.minimum(np.searchsorted(ts, t + h * 1e6), len(ts) - 1)
        out[f'ev{h:g}'] = surv * d * (mids[ih] - p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('strategy', type=Path)
    ap.add_argument('tape', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    strat = make_logger(load_class(a.strategy))()
    cfg = SimulationConfig(public_depth=20, run_log_path=None, book_delta_log_path=None, event_log_path=None,
                           trade_log_path=None, min_log_level='INFO', max_position=MAX_POSITION,
                           max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
                           order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS, show_progress=False, progress_fd=2)
    res = ExchangeSimulator(strategy=strat, config=cfg).run(a.tape)
    rows = np.asarray(strat.rows, float)
    lab = labels(rows, strat.ts, strat.bbs, strat.bas)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, t=rows[:, 0], side=rows[:, 1], price=rows[:, 2], take=rows[:, 3],
                        X=rows[:, 4:], feats=np.array(FEATS), **lab)
    print(f'pnl {res.pnl:,.0f}  rows {len(rows):,}  taken-by-rule {int(rows[:, 3].sum()):,}')
    for k, v in lab.items():
        print(k, f'mean {v.mean():.3f}  rule-taken mean {v[rows[:, 3] == 1].mean():.3f}')


if __name__ == '__main__':
    main()
