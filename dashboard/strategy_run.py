"""Run a strategy through the real simulator and record everything the dashboard needs to replay it.

The simulator is used exactly as `exchange-simulator` uses it (same loader, same scoring config); nothing in
`simulator/` is changed. Recording happens from outside, by wrapping methods on the objects:

- the simulator's order book (`insert`, `remove`, `reduce_quantity`): every order-level change, market and user,
  so the true book (with your orders in their queues) can be rebuilt at any tape row;
- the strategy's callbacks: the trades and best bid/ask after every tape row, and the exchange's replies to you;
- the strategy's order methods: what it sent, cancelled or shrank, and when;
- the strategy's state ("what it is thinking"): its scalar attributes plus an optional `self.debug` dict, captured
  whenever it acts and otherwise every STATE_EVERY_US of market time.

Tape row i is the i-th call of `on_order_book_change`; everything that happens while row i is processed (your
orders landing, the row itself, fills) belongs to row i.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import time
from array import array
from bisect import bisect_left, insort
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import engine  # noqa: E402
from simulator.main import (  # noqa: E402
    MAX_POSITION,
    MAX_POTENTIAL_POSITION,
    ORDER_ACCEPT_DELAY_MS,
    POSITION_BREACH_PENALTY_PER_SECOND,
)
from simulator.strategy import BaseStrategy  # noqa: E402
from simulator.models import Side  # noqa: E402
from simulator.simulator import ExchangeSimulator, SimulationConfig  # noqa: E402

_BUY, _SELL = Side.BUY, Side.SELL

STATE_EVERY_US = 50_000
OP_INSERT, OP_REMOVE, OP_REDUCE = 0, 1, 2
ACT_SEND, ACT_CANCEL, ACT_MODIFY = 0, 1, 2
SCALARS = (bool, int, float, str, type(None))


def user_num(order_id: str) -> int:
    """'u000000012' -> -12 (user ids are negative in the numeric arrays, market ids positive)."""
    return -int(order_id[1:])


def _oid(order_id: str) -> int:
    return -int(order_id[1:]) if order_id[:1] == 'u' else int(order_id)


def _value(v):
    """A state value worth showing, or a sentinel to skip it."""
    if isinstance(v, SCALARS):
        return v
    if hasattr(v, 'value') and isinstance(v.value, str):  # enums such as Side
        return v.value
    if isinstance(v, (tuple, list, set, frozenset)) and len(v) <= 8:
        return repr(v if not isinstance(v, (set, frozenset)) else sorted(v, key=repr))
    if isinstance(v, dict) and len(v) <= 8:
        return repr(v)
    return _SKIP


_SKIP = object()


# ---------------------------------------------------------------- order-level book rebuilt from recorded ops
class OpBook:
    """Order book rebuilt from recorded changes (no matching: the simulator already did it).

    Orders are `[oid, side, price, qty]` with side 0 bid / 1 ask; user orders have negative ids.
    """

    __slots__ = ('orders', 'levels_', 'prices')

    def __init__(self) -> None:
        self.orders: dict[int, list] = {}
        self.levels_: tuple[dict[int, dict[int, None]], dict[int, dict[int, None]]] = ({}, {})
        self.prices: tuple[list[int], list[int]] = ([], [])

    def insert(self, oid: int, side: int, price: int, qty: int) -> None:
        lv = self.levels_[side]
        q = lv.get(price)
        if q is None:
            q = lv[price] = {}
            insort(self.prices[side], price)
        q[oid] = None
        self.orders[oid] = [oid, side, price, qty]

    def remove(self, oid: int) -> None:
        o = self.orders.pop(oid, None)
        if o is None:
            return
        lv = self.levels_[o[1]]
        q = lv[o[2]]
        del q[oid]
        if not q:
            del lv[o[2]]
            ps = self.prices[o[1]]
            del ps[bisect_left(ps, o[2])]

    def reduce(self, oid: int, qty: int) -> None:
        o = self.orders.get(oid)
        if o is not None:
            o[3] -= qty

    def snapshot(self) -> list[tuple[int, int, int, int]]:
        out = []
        for side in (0, 1):
            for p in self.prices[side]:
                out.extend(tuple(self.orders[o]) for o in self.levels_[side][p])
        return out

    def restore(self, snap) -> None:
        self.__init__()
        for oid, side, price, qty in snap:
            self.insert(oid, side, price, qty)

    def levels(self) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
        """(bids best-first, asks best-first) as (price, lots, orders); only levels with live quantity."""
        out = []
        for side in (0, 1):
            rows = []
            prices = reversed(self.prices[0]) if side == 0 else self.prices[1]
            for p in prices:
                live = [self.orders[o][3] for o in self.levels_[side][p] if self.orders[o][3] > 0]
                if live:
                    rows.append((p, sum(live), len(live)))
            out.append(rows)
        return out[0], out[1]

    def level_orders(self, side: int, price: int) -> list[tuple[int | str, int]]:
        """Orders at one level in queue order as (id, lots); user ids come back as 'you #n'."""
        q = self.levels_[side].get(price, {})
        return [(o if o > 0 else f'you #{-o}', self.orders[o][3]) for o in q if self.orders[o][3] > 0]

    def user_qty(self, side: int, price: int) -> int:
        return sum(self.orders[o][3] for o in self.levels_[side].get(price, {}) if o < 0)

    def user_orders(self) -> list[list]:
        return [o for o in self.orders.values() if o[0] < 0 and o[3] > 0]


# ---------------------------------------------------------------- recorded run
@dataclass
class StrategyRun:
    strategy_name: str
    strategy_class: str
    overrides: dict
    result: dict
    runtime_s: float
    error: str | None
    replay: engine.Replay  # replay-like view of the true market (with your orders), for the existing charts
    # book changes
    op_ev: np.ndarray
    op_kind: np.ndarray
    op_oid: np.ndarray
    op_side: np.ndarray
    op_price: np.ndarray
    op_qty: np.ndarray
    chunk: int
    checkpoints: list
    # trades: owner flags line up with replay.t_* arrays
    t_taker_user: np.ndarray
    t_maker_user: np.ndarray
    # actions
    act_ev: np.ndarray
    act_kind: np.ndarray
    act_oid: np.ndarray
    act_side: np.ndarray  # 0 buy, 1 sell, -1 n/a
    act_price: np.ndarray
    act_qty: np.ndarray
    act_ioc: np.ndarray
    # exchange replies to you
    pv_ev: np.ndarray
    pv_oid: np.ndarray
    pv_status: list
    pv_fill_qty: np.ndarray
    pv_fill_price: np.ndarray
    pv_remaining: np.ndarray
    pv_message: list
    # state captures
    st_ev: np.ndarray
    st_keys: list
    st_values: dict  # key -> list aligned with st_ev (None where the key did not exist yet)
    # pnl curve (1 s samples from the simulator)
    pnl_t: np.ndarray
    pnl_v: np.ndarray

    def book_at(self, pos: int) -> OpBook:
        """The true book after tape row `pos`."""
        c = pos // self.chunk
        book = OpBook()
        book.restore(self.checkpoints[c])
        a = int(np.searchsorted(self.op_ev, c * self.chunk, side='left'))
        b = int(np.searchsorted(self.op_ev, pos, side='right'))
        for k, oid, side, price, qty in zip(self.op_kind[a:b].tolist(), self.op_oid[a:b].tolist(), self.op_side[a:b].tolist(),
                                            self.op_price[a:b].tolist(), self.op_qty[a:b].tolist(), strict=True):
            if k == OP_INSERT:
                book.insert(oid, side, price, qty)
            elif k == OP_REMOVE:
                book.remove(oid)
            else:
                book.reduce(oid, qty)
        return book

    def state_at(self, pos: int) -> tuple[int, dict, dict]:
        """(capture index, values, previous values) of the last state capture at or before row `pos`."""
        i = int(np.searchsorted(self.st_ev, pos, side='right')) - 1
        if i < 0:
            return -1, {}, {}
        cur = {k: self.st_values[k][i] for k in self.st_keys}
        prev = {k: self.st_values[k][i - 1] for k in self.st_keys} if i > 0 else {}
        return i, cur, prev

    def series(self, key: str) -> np.ndarray:
        vals = self.st_values.get(key, [])
        return np.array([v if isinstance(v, (int, float)) and not isinstance(v, bool) else (float(v) if isinstance(v, bool) else np.nan)
                         for v in vals], dtype=float)


@dataclass
class Param:
    name: str
    scope: str  # 'module' (NAME = v at top level) or 'class' (NAME = v inside the strategy class)
    default: bool | int | float
    comment: str


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f'dash_strategy_{abs(hash((str(path), time.time_ns())))}', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strategy_class(module, path: Path) -> type:
    """The class the simulator would use: first BaseStrategy subclass in sorted dir() order."""
    for name in dir(module):
        cand = getattr(module, name)
        if isinstance(cand, type) and issubclass(cand, BaseStrategy) and cand is not BaseStrategy:
            return cand
    raise ImportError(f'No BaseStrategy subclass found in {path}')


def _is_param(name: str, value) -> bool:
    return name.isupper() and not name.startswith('_') and isinstance(value, (bool, int, float))


def discover_params(path: Path) -> tuple[str, list[Param]]:
    """(strategy class name, tunable constants): UPPERCASE numbers/booleans at module level or in the class body."""
    module = _load_module(path)
    cls = _strategy_class(module, path)
    source = path.read_text()
    comments = {}
    for m in re.finditer(r'^\s*([A-Z_][A-Z0-9_]*)\s*(?::[^=]+)?=[^#\n]*#\s*(.+)$', source, flags=re.M):
        comments.setdefault(m.group(1), m.group(2).strip())
    defined = set(re.findall(r'^\s*([A-Z_][A-Z0-9_]*)\s*(?::[^=]+)?=', source, flags=re.M))
    params = []
    for name, value in vars(cls).items():
        if _is_param(name, value) and name in defined:
            params.append(Param(name, 'class', value, comments.get(name, '')))
    class_names = {p.name for p in params}
    for name, value in vars(module).items():
        if _is_param(name, value) and name in defined and name not in class_names:
            params.append(Param(name, 'module', value, comments.get(name, '')))
    return cls.__name__, params


def load_strategy(path: Path, overrides: dict[str, tuple[str, object]] | None = None) -> BaseStrategy:
    """A fresh instance of the strategy in `path`, with constants overridden in memory ({name: (scope, value)}).

    The file is never written. A fresh module is loaded every time, so overrides never leak between runs.
    """
    module = _load_module(path)
    cls = _strategy_class(module, path)
    for name, (scope, value) in (overrides or {}).items():
        if scope == 'class':
            setattr(cls, name, value)
        else:
            setattr(module, name, value)
    return cls()


def strategy_files() -> list[Path]:
    return sorted(p for p in (ROOT / 'strategies').glob('*.py') if not p.name.startswith('_'))


def run_strategy(strategy_path: Path, tape_path: Path, tape: engine.Tape, overrides: dict | None = None) -> StrategyRun:
    """Run `strategy_path` on `tape_path` in the simulator and record it. `tape` is the parsed CSV (same rows).

    `overrides` ({name: (scope, value)}) replaces strategy constants in memory for this run only.
    """
    t_start = time.perf_counter()
    strategy = load_strategy(strategy_path, overrides)
    config = SimulationConfig(
        max_position=MAX_POSITION,
        max_potential_position=MAX_POTENTIAL_POSITION,
        position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
        order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS,
    )
    sim = ExchangeSimulator(strategy=strategy, config=config)
    n = len(tape.time)
    chunk = max(2000, n // 400)

    ev = [0]  # current tape row
    op_ev, op_kind, op_oid, op_side, op_price, op_qty = (array('q') for _ in range(6))
    checkpoints: list = []
    shadow = OpBook()

    book = sim.book
    orig_insert, orig_remove, orig_reduce = book.insert, book.remove, book.reduce_quantity

    def rec(kind, oid, side, price, qty):
        op_ev.append(ev[0])
        op_kind.append(kind)
        op_oid.append(oid)
        op_side.append(side)
        op_price.append(price)
        op_qty.append(qty)

    def w_insert(order):
        orig_insert(order)
        oid = _oid(order.order_id)
        side = 0 if order.side.value == 'B' else 1
        rec(OP_INSERT, oid, side, order.price, order.remaining_quantity)
        shadow.insert(oid, side, order.price, order.remaining_quantity)

    def w_remove(order_id):
        out = orig_remove(order_id)
        if out is not None:
            oid = _oid(order_id)
            rec(OP_REMOVE, oid, 0, 0, out.remaining_quantity)
            shadow.remove(oid)
        return out

    def w_reduce(order, amount):
        orig_reduce(order, amount)
        oid = _oid(order.order_id)
        rec(OP_REDUCE, oid, 0, 0, amount)
        shadow.reduce(oid, amount)

    book.insert, book.remove, book.reduce_quantity = w_insert, w_remove, w_reduce

    # cancel_qty per market cancel row: lots that row removed (0 = ignored)
    cancel_qty = np.zeros(n, dtype=np.int32)
    bb = np.full(n, engine.NONE, dtype=np.int32)
    ba = np.full(n, engine.NONE, dtype=np.int32)
    t_ev, t_us, t_price, t_qty, t_side, t_taker, t_maker, t_tu, t_mu = (array('q') for _ in range(9))
    user_ioc: dict[int, int] = {}

    act_ev, act_kind, act_oid, act_side, act_price, act_qty, act_ioc = (array('q') for _ in range(7))
    pv_ev, pv_oid, pv_fq, pv_fp, pv_rem = (array('q') for _ in range(5))
    pv_status: list = []
    pv_msg: list = []

    st_ev: list[int] = []
    st_values: dict[str, list] = {}
    last_capture_us = [-(10**18)]
    acted = [False]

    orig_send, orig_cancel, orig_modify = strategy.send_order, strategy.cancel_order, strategy.modify_order

    def w_send(order):
        oid = orig_send(order)
        n_ = user_num(oid)
        ioc = 1 if order.order_type.value.upper() in ('IOC', 'F') else 0
        user_ioc[n_] = ioc
        act_ev.append(ev[0]); act_kind.append(ACT_SEND); act_oid.append(n_)  # noqa: E702
        act_side.append(0 if order.side.value == 'B' else 1); act_price.append(order.price); act_qty.append(order.quantity)  # noqa: E702
        act_ioc.append(ioc)
        acted[0] = True
        return oid

    def w_cancel(order_id):
        orig_cancel(order_id)
        act_ev.append(ev[0]); act_kind.append(ACT_CANCEL); act_oid.append(user_num(order_id) if order_id[:1] == 'u' else 0)  # noqa: E702
        act_side.append(-1); act_price.append(0); act_qty.append(0); act_ioc.append(0)  # noqa: E702
        acted[0] = True

    def w_modify(order_id, new_quantity):
        orig_modify(order_id, new_quantity)
        act_ev.append(ev[0]); act_kind.append(ACT_MODIFY); act_oid.append(user_num(order_id) if order_id[:1] == 'u' else 0)  # noqa: E702
        act_side.append(-1); act_price.append(0); act_qty.append(new_quantity); act_ioc.append(0)  # noqa: E702
        acted[0] = True

    strategy.send_order, strategy.cancel_order, strategy.modify_order = w_send, w_cancel, w_modify

    orig_obc, orig_pou = strategy.on_order_book_change, strategy.on_private_order_update
    is_cancel = tape.is_cancel.tolist()
    tape_ioc = tape.ioc.tolist()
    tape_oid = tape.oid.tolist()

    def capture(i):
        values = {}
        for k, v in vars(strategy).items():
            if k in ('_ctx', 'debug', 'send_order', 'cancel_order', 'modify_order', 'on_order_book_change', 'on_private_order_update'):
                continue
            v = _value(v)
            if v is not _SKIP:
                values[k] = v
        dbg = getattr(strategy, 'debug', None)
        if isinstance(dbg, dict):
            for k, v in dbg.items():
                v = _value(v)
                values[f'debug.{k}'] = v if v is not _SKIP else repr(v)[:80]
        values['sim.inventory'] = sim._inventory
        values['sim.cash'] = sim._cash
        values['sim.pnl'] = sim._mark_to_market_pnl()
        values['sim.breach_penalty'] = sim._breach_penalty
        values['sim.resting_buy'] = sim.book.resting_user_quantity(_BUY)
        values['sim.resting_sell'] = sim.book.resting_user_quantity(_SELL)
        k_ = len(st_ev)
        for key in values:
            if key not in st_values:
                st_values[key] = [None] * k_
        for key, col in st_values.items():
            col.append(values.get(key))
        st_ev.append(i)

    def w_obc(snapshot, trades):
        i = ev[0]
        for tr in trades:
            t_ev.append(i)
            t_us.append(tr.timestamp_us)
            t_price.append(tr.price)
            t_qty.append(tr.quantity)
            t_side.append(0 if tr.book_side == 'ask' else 1)
            tu = tr.taker_owner == 'user'
            mu = tr.maker_owner == 'user'
            t_taker.append(user_num(tr.taker_order_id) if tu else int(tr.taker_order_id))
            t_maker.append(user_num(tr.maker_order_id) if mu else int(tr.maker_order_id))
            t_tu.append(1 if tu else 0)
            t_mu.append(1 if mu else 0)
        if snapshot.bids:
            bb[i] = snapshot.bids[0].price
        if snapshot.asks:
            ba[i] = snapshot.asks[0].price
        acted[0] = False
        orig_obc(snapshot, trades)
        if acted[0] or snapshot.timestamp_us - last_capture_us[0] >= STATE_EVERY_US:
            capture(i)
            last_capture_us[0] = snapshot.timestamp_us
        ev[0] = i + 1
        if (i + 1) % chunk == 0:
            checkpoints.append(shadow.snapshot())

    def w_pou(event):
        pv_ev.append(ev[0])
        pv_oid.append(user_num(event.order_id))
        pv_status.append(event.status.value)
        pv_fq.append(event.filled_quantity or 0)
        pv_fp.append(event.fill_price or 0)
        pv_rem.append(event.remaining_quantity or 0)
        pv_msg.append(event.message or '')
        orig_pou(event)

    strategy.on_order_book_change, strategy.on_private_order_update = w_obc, w_pou

    checkpoints.append(shadow.snapshot())  # row 0 starts from an empty book
    error = None
    result: dict = {}
    try:
        res = sim.run(tape_path)
        result = res.model_dump(mode='json')
        curve = sim.pnl_curve(res.pnl)
    except Exception as exc:  # a crashing strategy still shows what it did up to the crash
        error = f'{type(exc).__name__}: {exc}'
        curve = list(zip(sim._pnl_sample_times_us, sim._pnl_samples, strict=False))

    a = lambda x, dt=np.int64: np.frombuffer(x, dtype=np.int64).astype(dt) if len(x) else np.zeros(0, dtype=dt)  # noqa: E731
    op_ev_a, op_kind_a, op_oid_a = a(op_ev), a(op_kind, np.int8), a(op_oid)
    op_side_a, op_price_a, op_qty_a = a(op_side, np.int8), a(op_price, np.int32), a(op_qty, np.int32)

    # cancel rows: lots removed by the row's own order id at that row
    rm = op_kind_a == OP_REMOVE
    for e, oid, q in zip(op_ev_a[rm].tolist(), op_oid_a[rm].tolist(), op_qty_a[rm].tolist(), strict=True):
        if e < n and is_cancel[e] and tape_oid[e] == oid:
            cancel_qty[e] = q

    t_ev_a = a(t_ev)
    t_taker_a, t_tu_a = a(t_taker), a(t_tu, bool)
    t_ioc_a = np.array([user_ioc.get(int(tk), 0) if tu else (tape_ioc[e] if e < n else 0)
                        for e, tk, tu in zip(t_ev_a.tolist(), t_taker_a.tolist(), t_tu_a.tolist(), strict=True)], dtype=np.int8)
    rp = _replay_like(tape, chunk, bb, ba, cancel_qty, t_ev_a, a(t_us), a(t_price, np.int32), a(t_qty, np.int32), a(t_side, np.int8), t_ioc_a,
                      t_taker_a, a(t_maker))

    pnl_t = np.array([p[0] for p in curve], dtype=np.int64) / 1e6 if curve else np.zeros(0)
    pnl_v = np.array([p[1] for p in curve], dtype=float) if curve else np.zeros(0)
    return StrategyRun(
        strategy_name=strategy_path.name, strategy_class=type(strategy).__name__, overrides=dict(overrides or {}), result=result,
        runtime_s=time.perf_counter() - t_start, error=error, replay=rp,
        op_ev=op_ev_a, op_kind=op_kind_a, op_oid=op_oid_a, op_side=op_side_a, op_price=op_price_a, op_qty=op_qty_a,
        chunk=chunk, checkpoints=checkpoints, t_taker_user=t_tu_a, t_maker_user=a(t_mu, bool),
        act_ev=a(act_ev), act_kind=a(act_kind, np.int8), act_oid=a(act_oid), act_side=a(act_side, np.int8), act_price=a(act_price),
        act_qty=a(act_qty), act_ioc=a(act_ioc, np.int8),
        pv_ev=a(pv_ev), pv_oid=a(pv_oid), pv_status=pv_status, pv_fill_qty=a(pv_fq), pv_fill_price=a(pv_fp), pv_remaining=a(pv_rem),
        pv_message=pv_msg, st_ev=np.array(st_ev, dtype=np.int64), st_keys=list(st_values), st_values=st_values,
        pnl_t=pnl_t, pnl_v=pnl_v,
    )


def _replay_like(tape, chunk, bb, ba, cancel_qty, t_ev, t_us, t_price, t_qty, t_side, t_ioc, t_taker, t_maker) -> engine.Replay:
    """An engine.Replay built from the simulator run, so the dashboard's charts and tables work unchanged."""
    tcat = t_side.astype(np.int64) + 2 * t_ioc.astype(np.int64)
    T = len(t_ev)
    cum_qty = np.zeros((4, T + 1))
    cum_cnt = np.zeros((4, T + 1), dtype=np.int64)
    for k in range(4):
        m = tcat == k
        cum_qty[k, 1:] = np.cumsum(np.where(m, t_qty, 0))
        cum_cnt[k, 1:] = np.cumsum(m)
    cum_notional = np.zeros(T + 1)
    cum_notional[1:] = np.cumsum(t_qty.astype(np.int64) * t_price.astype(np.int64))
    n = len(tape.time)
    cat = np.where(tape.is_cancel, -1, tape.side + 2 * tape.ioc).astype(np.int8)
    starts = list(range(0, n, chunk)) + [n]
    chunk_counters = np.zeros((len(starts), 11))
    acc = np.zeros(11)
    for k in range(1, len(starts)):
        acc = acc + engine.count_range(cat, tape.vol, cancel_qty, starts[k - 1], starts[k])
        chunk_counters[k] = acc
    return engine.Replay(
        tape=tape, chunk=chunk, checkpoints=[], best_bid=bb, best_ask=ba, cancel_qty=cancel_qty,
        t_ev=t_ev, t_us=t_us, t_price=t_price, t_qty=t_qty, t_side=t_side, t_ioc=t_ioc, t_taker=t_taker, t_maker=t_maker,
        cum_qty=cum_qty, cum_cnt=cum_cnt, cum_notional=cum_notional, chunk_counters=chunk_counters, cat=cat, dups=0,
    )

