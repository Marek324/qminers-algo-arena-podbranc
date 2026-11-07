"""Target PnL from the fitted market model: what a market maker that knows the hidden fair price S can expect to earn.

The reference strategy sees the true S at the current time (never the future) and otherwise trades through the real
simulator with the scoring settings (5 ms latency, queue position, |inventory| 100 limit). It quotes around S, skews
for inventory and optionally takes when the touch is far enough on the wrong side of S. Its few settings are tuned on
one batch of generated sessions; the target is its mean PnL per hour on a fresh batch, with the spread across sessions.

No submitted strategy can see S, so this is a target to aim at (an estimate of what perfect decoding of the hidden
state is worth), not a guarantee. Our own strategies run on the same fresh sessions give their expected PnL under the
model, which is also an out-of-sample check that does not depend on the one real tape.
"""

from __future__ import annotations

import itertools
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from simulator.models import OrderRequest, OrderStatus, OrderType, Side
from simulator.strategy import BaseStrategy

from .generator import PARAM_DIR, GenParams, generate, write_tape
from .strategies import load_strategy, scoring_config

TAPE_DIR = PARAM_DIR / 'tapes'
MAX_POS = 100


class OracleMarketMaker(BaseStrategy):
    """Quotes floor(S - half) / ceil(S + half), shifted by -skew * inventory; takes when |S - touch| >= take_edge."""

    def __init__(self, s_times_us: np.ndarray, s_values: np.ndarray, half: float, size: int, skew: float, take_edge: float | None) -> None:
        super().__init__()
        self.st, self.sv = s_times_us, s_values
        self.half, self.size, self.skew, self.take_edge = half, size, skew, take_edge
        self.inv = 0
        self.orders: dict[str, list] = {}  # id -> [side, price, remaining, cancelling]

    def _s(self) -> float:
        i = int(np.searchsorted(self.st, self.now_us(), side='right')) - 1
        return float(self.sv[max(i, 0)])

    def _resting(self, side: Side) -> int:
        return sum(o[2] for o in self.orders.values() if o[0] == side)

    def on_order_book_change(self, snapshot, trades) -> None:
        if not snapshot.bids or not snapshot.asks:
            return
        bb, ba = snapshot.bids[0].price, snapshot.asks[0].price
        fair = self._s() - self.skew * self.inv
        if self.take_edge is not None:
            if fair - ba >= self.take_edge and self.inv < MAX_POS:
                q = min(self.size, MAX_POS - self.inv - self._resting(Side.BUY))
                if q > 0:
                    oid = self.send_order(OrderRequest(side=Side.BUY, price=ba, quantity=q, order_type=OrderType.IOC))
                    self.orders[oid] = [Side.BUY, ba, q, True]
            elif bb - fair >= self.take_edge and self.inv > -MAX_POS:
                q = min(self.size, MAX_POS + self.inv - self._resting(Side.SELL))
                if q > 0:
                    oid = self.send_order(OrderRequest(side=Side.SELL, price=bb, quantity=q, order_type=OrderType.IOC))
                    self.orders[oid] = [Side.SELL, bb, q, True]
        want = {Side.BUY: min(math.floor(fair - self.half), ba - 1), Side.SELL: max(math.ceil(fair + self.half), bb + 1)}
        for side, px in want.items():
            live = [(oid, o) for oid, o in self.orders.items() if o[0] == side and not o[3]]
            if live and live[0][1][1] == px:
                continue
            for oid, o in live:
                self.cancel_order(oid)
                o[3] = True
            room = (MAX_POS - self.inv if side == Side.BUY else MAX_POS + self.inv) - self._resting(side)
            q = min(self.size, room)
            if q > 0:
                oid = self.send_order(OrderRequest(side=side, price=px, quantity=q, order_type=OrderType.LIMIT))
                self.orders[oid] = [side, px, q, False]

    def on_private_order_update(self, event) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        if event.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            self.inv += event.filled_quantity if o[0] == Side.BUY else -event.filled_quantity
            o[2] = event.remaining_quantity
            if event.status == OrderStatus.FILLED:
                del self.orders[event.order_id]
        elif event.status in (OrderStatus.CANCELED, OrderStatus.REJECTED):
            del self.orders[event.order_id]


# ---------------------------------------------------------------- sessions and runs (child processes)
def session(name: str, seed: int, minutes: float) -> tuple[str, str]:
    """Generate (or reuse) one session; returns (tape path, S path .npz)."""
    p = GenParams.load(PARAM_DIR / f'{name}_params.json')
    tape = TAPE_DIR / f'{name}_{int(minutes)}m_seed{seed}.csv'
    spath = tape.with_suffix('.npz')
    if not tape.exists() or not spath.exists():
        df, ts, S = generate(p, seed=seed, duration_s=minutes * 60)
        write_tape(df, tape)
        np.savez(spath, t_us=np.round(ts * 1e6).astype(np.int64), s=S)
    return str(tape), str(spath)


def run_oracle(tape: str, spath: str, cfg: tuple) -> float:
    from simulator.simulator import ExchangeSimulator

    d = np.load(spath)
    strat = OracleMarketMaker(d['t_us'], d['s'], *cfg)
    return ExchangeSimulator(strategy=strat, config=scoring_config()).run(Path(tape)).pnl


def run_file_strategy(tape: str, strategy: str) -> float:
    from simulator.simulator import ExchangeSimulator

    return ExchangeSimulator(strategy=load_strategy(Path(strategy)), config=scoring_config()).run(Path(tape)).pnl


GRID = list(itertools.product((-0.5, 0.0, 0.5, 1.0), (100, 200), (0.002, 0.005, 0.01), (None, 0.75, 1.0, 1.5)))


def compute_target(name: str, minutes: float = 20, n_train: int = 4, n_test: int = 8, strategies: list[str] | None = None,
                   workers: int = 8, log=print) -> dict:
    """Tune the oracle on n_train sessions, report it (and `strategies`) on n_test fresh sessions, per hour."""
    with ProcessPoolExecutor(workers) as ex:
        train = list(ex.map(session, [name] * n_train, range(1000, 1000 + n_train), [minutes] * n_train))
        test = list(ex.map(session, [name] * n_test, range(2000, 2000 + n_test), [minutes] * n_test))
        log(f'{name}: {n_train} training + {n_test} test sessions of {minutes:g} min generated; tuning {len(GRID)} oracle settings')
        futs = {(cfg, k): ex.submit(run_oracle, t, s, cfg) for cfg in GRID for k, (t, s) in enumerate(train)}
        train_pnl = {cfg: np.mean([futs[(cfg, k)].result() for k in range(n_train)]) for cfg in GRID}
        best = max(GRID, key=lambda c: train_pnl[c])
        log(f'  best oracle setting {best}: {train_pnl[best] * 60 / minutes:,.0f}/hour on training sessions')
        oracle = list(ex.map(run_oracle, [t for t, _ in test], [s for _, s in test], [best] * n_test))
        strat = {Path(sp).name: list(ex.map(run_file_strategy, [t for t, _ in test], [sp] * n_test)) for sp in (strategies or [])}
    scale = 60 / minutes

    def summ(x):
        x = np.asarray(x, dtype=float) * scale
        return {'mean_per_hour': float(x.mean()), 'sd_per_hour': float(x.std(ddof=1)) if len(x) > 1 else None,
                'se_per_hour': float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None, 'sessions_per_hour': x.tolist()}

    return {
        'params': name, 'session_minutes': minutes, 'n_train': n_train, 'n_test': n_test,
        'oracle_setting': {'half': best[0], 'size': best[1], 'skew': best[2], 'take_edge': best[3]},
        'target': summ(oracle),
        'strategies': {k: summ(v) for k, v in strat.items()},
    }


# ---------------------------------------------------------------- floor / target / ceiling for one strategy
def run_strategy_true_s(tape: str, spath: str, strategy: str, overrides: dict) -> float:
    """Run a strategy file with its hidden-price estimate replaced by the true S (strategies keeping s_bid/s_ask)."""
    from simulator.simulator import ExchangeSimulator

    d = np.load(spath)
    t_us, s = d['t_us'], d['s']
    strat = load_strategy(Path(strategy), overrides)
    if not (hasattr(strat, 's_bid') and hasattr(strat, 's_ask') and hasattr(strat, '_scan_new')):
        raise ValueError(f'{strategy} has no s_bid/s_ask hidden-price estimate to replace')

    def true_scan(views, sign):
        i = int(np.searchsorted(t_us, strat.now_us(), side='right')) - 1
        strat.s_bid = strat.s_ask = float(s[max(i, 0)])

    strat._scan_new = true_scan
    return ExchangeSimulator(strategy=strat, config=scoring_config()).run(Path(tape)).pnl


def run_bounds_on_session(tape: str) -> float:
    from .sections.maxpnl import tape_bounds

    return tape_bounds(Path(tape))['theoretical_max']


def compute_bracket(name: str, strategy: str, grid: dict[str, list], minutes: float = 20, n_train: int = 4, n_test: int = 8,
                    workers: int = 8, log=print) -> dict:
    """Floor (strategy as is), target (strategy fed the true S, re-tuned over `grid`) and ceiling (expected
    theoretical max) per hour, on fresh sessions of the fitted market model `name`."""
    scope = {p.name: p.scope for p in __import__('analysis.strategies', fromlist=['discover_params']).discover_params(Path(strategy))[1]}
    combos = [dict(zip(grid, vals, strict=True)) for vals in itertools.product(*grid.values())]
    with ProcessPoolExecutor(workers) as ex:
        train = list(ex.map(session, [name] * n_train, range(1000, 1000 + n_train), [minutes] * n_train))
        test = list(ex.map(session, [name] * n_test, range(2000, 2000 + n_test), [minutes] * n_test))
        log(f'{name}: tuning {len(combos)} true-S settings of {Path(strategy).name} on {n_train} sessions')
        futs = {(k, i): ex.submit(run_strategy_true_s, t, s, strategy, {n: (scope[n], v) for n, v in c.items()})
                for k, c in enumerate(combos) for i, (t, s) in enumerate(train)}
        score = [np.mean([futs[(k, i)].result() for i in range(n_train)]) for k in range(len(combos))]
        best = combos[int(np.argmax(score))]
        ov = {n: (scope[n], v) for n, v in best.items()}
        log(f'  best true-S setting {best}; scoring floor / target / ceiling on {n_test} fresh sessions')
        target = list(ex.map(run_strategy_true_s, [t for t, _ in test], [s for _, s in test], [strategy] * n_test, [ov] * n_test))
        floor = list(ex.map(run_file_strategy, [t for t, _ in test], [strategy] * n_test))
        ceiling = list(ex.map(run_bounds_on_session, [t for t, _ in test]))
    scale = 60 / minutes

    def summ(x):
        x = np.asarray(x, dtype=float) * scale
        return {'mean_per_hour': float(x.mean()), 'sd_per_hour': float(x.std(ddof=1)), 'sessions_per_hour': x.tolist()}

    return {'params': name, 'strategy': Path(strategy).name, 'session_minutes': minutes, 'n_test': n_test,
            'true_s_setting': best, 'floor': summ(floor), 'target': summ(target), 'ceiling': summ(ceiling)}
