"""Replay a raw market-data CSV through a price-time-priority book and derive the trades.

Mirrors the matching rules of `simulator/simulator.py` (market-only: no strategy orders):
an incoming order matches the opposite side while prices cross, every fill at the
resting (maker) order's price, best price and front of queue first; a G residual rests,
an F (IOC) residual is dropped; a cancel of an unknown or already-filled id is ignored.

The tape is replayed once into compact numpy arrays (`Replay`). The book at an arbitrary
event is rebuilt from the nearest full-book checkpoint (taken every `chunk` events).
"""

from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import dataclass
from typing import IO

import numpy as np
import pandas as pd

NONE = -1  # "no bid" / "no ask" in the best-price arrays
CAT_NAMES = ('Buy aggr. G', 'Sell aggr. G', 'Buy aggr. F (IOC)', 'Sell aggr. F (IOC)')  # cat = side + 2 * is_ioc
_NOFILLS: tuple = ()


class Book:
    """Price-time priority book. Orders are `[id, side, price, remaining]`; levels `[queue, head, total, count]`.

    Cancelled orders stay in their queue with remaining == 0 and are skipped lazily.
    """

    __slots__ = ('orders', 'bids', 'asks', 'bp', 'ap', 'dups')

    def __init__(self) -> None:
        self.orders: dict[int, list] = {}
        self.bids: dict[int, list] = {}
        self.asks: dict[int, list] = {}
        self.bp: list[int] = []  # bid prices ascending (best = last)
        self.ap: list[int] = []  # ask prices ascending (best = first)
        self.dups = 0

    def _rest(self, order: list) -> None:
        if order[1] == 0:
            levels, prices = self.bids, self.bp
        else:
            levels, prices = self.asks, self.ap
        price = order[2]
        level = levels.get(price)
        if level is None:
            level = levels[price] = [[], 0, 0, 0]
            insort(prices, price)
        level[0].append(order)
        level[2] += order[3]
        level[3] += 1
        self.orders[order[0]] = order

    def insert(self, oid: int, side: int, price: int, vol: int, ioc: int):
        """Match an incoming order, rest the G residual. Returns a list of fills `(price, qty, maker_id)` (or an empty tuple)."""
        rem = vol
        fills = _NOFILLS
        buy = side == 0
        levels = self.asks if buy else self.bids
        prices = self.ap if buy else self.bp
        orders = self.orders
        while rem > 0 and prices:
            p = prices[0] if buy else prices[-1]
            if (price < p) if buy else (price > p):
                break
            level = levels[p]
            queue = level[0]
            h = level[1]
            maker = queue[h]
            while maker[3] == 0:
                h += 1
                maker = queue[h]
            t = rem if rem < maker[3] else maker[3]
            rem -= t
            maker[3] -= t
            level[2] -= t
            if fills is _NOFILLS:
                fills = []
            fills.append((p, t, maker[0]))
            if maker[3] == 0:
                del orders[maker[0]]
                level[3] -= 1
                h += 1
            if level[2] == 0:
                del levels[p]
                if buy:
                    del prices[0]
                else:
                    prices.pop()
            else:
                if h > 64 and h * 2 > len(queue):
                    level[0] = queue[h:]
                    h = 0
                level[1] = h
        if rem > 0 and not ioc:
            if oid in orders:
                self.dups += 1  # the simulator would raise here
            else:
                self._rest([oid, side, price, rem])
        return fills

    def cancel(self, oid: int) -> int:
        """Remove a resting order; returns the removed quantity (0 if unknown or already filled)."""
        order = self.orders.pop(oid, None)
        if order is None:
            return 0
        levels, prices = (self.bids, self.bp) if order[1] == 0 else (self.asks, self.ap)
        price = order[2]
        level = levels[price]
        qty = order[3]
        order[3] = 0
        level[2] -= qty
        level[3] -= 1
        if level[2] == 0:
            del levels[price]
            del prices[bisect_left(prices, price)]
        elif len(level[0]) - level[1] > 2 * level[3] + 32:
            level[0] = [o for o in level[0][level[1]:] if o[3] > 0]
            level[1] = 0
        return qty

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Live orders in per-level queue order as (ids, sides, prices, remaining)."""
        rows = []
        for levels, prices in ((self.bids, self.bp), (self.asks, self.ap)):
            for p in prices:
                level = levels[p]
                rows.extend(o for o in level[0][level[1]:] if o[3] > 0)
        arr = np.array(rows, dtype=np.int64).reshape(-1, 4)
        return arr[:, 0].copy(), arr[:, 1].astype(np.int8), arr[:, 2].copy(), arr[:, 3].astype(np.int32)

    def restore(self, snap) -> None:
        self.__init__()
        for oid, side, price, rem in zip(*(a.tolist() for a in snap)):
            self._rest([oid, side, price, rem])

    def levels(self) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
        """(bids best-first, asks best-first) as lists of (price, volume, order_count)."""
        bids = [(p, self.bids[p][2], self.bids[p][3]) for p in reversed(self.bp)]
        asks = [(p, self.asks[p][2], self.asks[p][3]) for p in self.ap]
        return bids, asks


@dataclass
class Tape:
    """Parsed CSV as columnar arrays (one entry per row)."""

    ts_us: np.ndarray  # int64 microseconds, made non-decreasing
    time: np.ndarray  # float64 seconds (= ts_us / 1e6)
    is_cancel: np.ndarray  # bool
    oid: np.ndarray  # int64 order id
    side: np.ndarray  # int8: 0 B, 1 A, 2 none (cancel)
    vol: np.ndarray  # int32
    price: np.ndarray  # int32
    ioc: np.ndarray  # int8: 1 if Lifespan == F
    time_fixed: int = 0  # rows whose time went backwards


def _codes(series: pd.Series, mapping: dict[str, int], default: int) -> np.ndarray:
    cat = series.astype('category')
    lut = np.array([mapping.get(str(c), -99) for c in cat.cat.categories] + [default], dtype=np.int16)
    codes = cat.cat.codes.to_numpy()
    out = lut[np.where(codes < 0, len(lut) - 1, codes)]
    return out


def parse_csv(source: str | IO[bytes]) -> Tape:
    """Parse a market CSV (path or binary file object) the way `Simulator.iter_backtest` does."""
    df = pd.read_csv(source, dtype={'Operation': 'category', 'Side': 'category', 'Lifespan': 'category'})
    for col in ('Time', 'Operation', 'OrderId', 'Side', 'Volume', 'Price'):
        if col not in df.columns:
            raise ValueError(f'missing column {col!r}')
    ops = _codes(df['Operation'], {'Insert': 0, 'Cancel': 1}, -99)
    if (ops < 0).any():
        bad = df['Operation'][ops < 0].iloc[0]
        raise ValueError(f'Unsupported operation: {bad!r}')
    is_cancel = ops == 1
    side = _codes(df['Side'], {'B': 0, 'A': 1}, 2)
    if ((side < 0) | ((side == 2) & ~is_cancel)).any():
        raise ValueError('Insert row with a Side other than B/A')
    side[is_cancel] = 2
    oid_col = df['OrderId']
    if oid_col.dtype.kind != 'i':
        oid_col = pd.Series(pd.factorize(oid_col.astype(str))[0], dtype=np.int64)
    ioc = _codes(df['Lifespan'], {'F': 1}, 0) if 'Lifespan' in df.columns else np.zeros(len(df), np.int16)
    ioc = np.where(ioc == 1, 1, 0).astype(np.int8)
    ts = np.rint(df['Time'].to_numpy(np.float64) * 1_000_000).astype(np.int64)
    running = np.maximum.accumulate(ts) if len(ts) else ts
    fixed = int((running != ts).sum())
    return Tape(
        ts_us=running,
        time=running / 1e6,
        is_cancel=is_cancel,
        oid=oid_col.to_numpy(np.int64),
        side=side.astype(np.int8),
        vol=np.nan_to_num(df['Volume'].to_numpy(np.float64)).astype(np.int32),
        price=np.nan_to_num(df['Price'].to_numpy(np.float64)).astype(np.int32),
        ioc=ioc,
        time_fixed=fixed,
    )


@dataclass
class Replay:
    tape: Tape
    chunk: int
    checkpoints: list
    best_bid: np.ndarray  # int32, NONE if empty; state AFTER each event
    best_ask: np.ndarray
    cancel_qty: np.ndarray  # int32: quantity removed by a cancel row (0 = ignored / not a cancel)
    # trades (sorted by event index)
    t_ev: np.ndarray
    t_us: np.ndarray
    t_price: np.ndarray
    t_qty: np.ndarray
    t_side: np.ndarray  # aggressor side: 0 buy, 1 sell
    t_ioc: np.ndarray  # aggressor was F
    t_taker: np.ndarray
    t_maker: np.ndarray
    cum_qty: np.ndarray  # (4, T+1) by aggressor category
    cum_cnt: np.ndarray  # (4, T+1)
    cum_notional: np.ndarray  # (T+1,)
    chunk_counters: np.ndarray  # (n_chunks+1, 11) counters for events [0, c*chunk)
    cat: np.ndarray  # int8 insert category (side + 2*ioc), -1 for cancels
    dups: int

    @property
    def n(self) -> int:
        return len(self.tape.time)

    @property
    def t_time(self) -> np.ndarray:
        return self.t_us / 1e6


def replay(tape: Tape, chunk: int | None = None) -> Replay:
    n = len(tape.time)
    chunk = chunk or max(2000, n // 400)
    ids = tape.oid.tolist()
    cancels = tape.is_cancel.tolist()
    sides = tape.side.tolist()
    vols = tape.vol.tolist()
    prices = tape.price.tolist()
    iocs = tape.ioc.tolist()
    us = tape.ts_us.tolist()

    book = Book()
    bp, ap = book.bp, book.ap
    insert, cancel = book.insert, book.cancel
    bb = [NONE] * n
    ba = [NONE] * n
    cq = [0] * n
    t_ev: list[int] = []
    t_us: list[int] = []
    t_price: list[int] = []
    t_qty: list[int] = []
    t_side: list[int] = []
    t_ioc: list[int] = []
    t_taker: list[int] = []
    t_maker: list[int] = []
    checkpoints = []
    for c0 in range(0, n, chunk):
        checkpoints.append(book.snapshot())
        for i in range(c0, min(n, c0 + chunk)):
            if cancels[i]:
                cq[i] = cancel(ids[i])
            else:
                fills = insert(ids[i], sides[i], prices[i], vols[i], iocs[i])
                if fills:
                    for p, q, m in fills:
                        t_ev.append(i)
                        t_us.append(us[i])
                        t_price.append(p)
                        t_qty.append(q)
                        t_side.append(sides[i])
                        t_ioc.append(iocs[i])
                        t_taker.append(ids[i])
                        t_maker.append(m)
            if bp:
                bb[i] = bp[-1]
            if ap:
                ba[i] = ap[0]

    a = lambda x, dt: np.array(x, dtype=dt)  # noqa: E731
    t_side_a, t_ioc_a = a(t_side, np.int8), a(t_ioc, np.int8)
    t_qty_a, t_price_a = a(t_qty, np.int64), a(t_price, np.int64)
    tcat = t_side_a + 2 * t_ioc_a
    T = len(t_ev)
    cum_qty = np.zeros((4, T + 1))
    cum_cnt = np.zeros((4, T + 1), dtype=np.int64)
    for k in range(4):
        m = tcat == k
        cum_qty[k, 1:] = np.cumsum(np.where(m, t_qty_a, 0))
        cum_cnt[k, 1:] = np.cumsum(m)
    cum_notional = np.zeros(T + 1)
    cum_notional[1:] = np.cumsum(t_qty_a * t_price_a)

    cq_a = a(cq, np.int32)
    cat = np.where(tape.is_cancel, -1, tape.side + 2 * tape.ioc).astype(np.int8)
    starts = list(range(0, n, chunk)) + [n]
    chunk_counters = np.zeros((len(starts), 11))
    acc = np.zeros(11)
    for k in range(1, len(starts)):
        acc = acc + count_range(cat, tape.vol, cq_a, starts[k - 1], starts[k])
        chunk_counters[k] = acc

    return Replay(
        tape=tape, chunk=chunk, checkpoints=checkpoints,
        best_bid=a(bb, np.int32), best_ask=a(ba, np.int32), cancel_qty=cq_a,
        t_ev=a(t_ev, np.int64), t_us=a(t_us, np.int64), t_price=t_price_a.astype(np.int32),
        t_qty=t_qty_a.astype(np.int32), t_side=t_side_a, t_ioc=t_ioc_a,
        t_taker=a(t_taker, np.int64), t_maker=a(t_maker, np.int64),
        cum_qty=cum_qty, cum_cnt=cum_cnt, cum_notional=cum_notional,
        chunk_counters=chunk_counters, cat=cat, dups=book.dups,
    )


def count_range(cat: np.ndarray, vol: np.ndarray, cq: np.ndarray, a: int, b: int) -> np.ndarray:
    """Counters over events [a, b): 4 insert counts, 4 insert volumes (by cat), cancels effective / ignored, cancelled volume."""
    c = cat[a:b]
    ins = c >= 0
    out = np.zeros(11)
    out[0:4] = np.bincount(c[ins], minlength=4)
    out[4:8] = np.bincount(c[ins], weights=vol[a:b][ins], minlength=4)
    q = cq[a:b]
    can = ~ins
    out[8] = np.count_nonzero(can & (q > 0))
    out[9] = np.count_nonzero(can & (q == 0))
    out[10] = q.sum()
    return out


def counters_at(r: Replay, pos: int) -> np.ndarray:
    """Counters over events [0, pos]."""
    m = pos + 1
    c = m // r.chunk
    return r.chunk_counters[c] + count_range(r.cat, r.tape.vol, r.cancel_qty, c * r.chunk, m)


def book_at(r: Replay, pos: int) -> Book:
    """Book state after event `pos` (rebuilt from the nearest checkpoint)."""
    t = r.tape
    c = pos // r.chunk
    book = Book()
    book.restore(r.checkpoints[c])
    ids, cancels, sides, vols, prices, iocs = (
        x[c * r.chunk: pos + 1].tolist() for x in (t.oid, t.is_cancel, t.side, t.vol, t.price, t.ioc)
    )
    for i, o in enumerate(ids):
        if cancels[i]:
            book.cancel(o)
        else:
            book.insert(o, sides[i], prices[i], vols[i], iocs[i])
    return book


def trade_count_through(r: Replay, pos: int) -> int:
    """Number of trades caused by events 0..pos."""
    return int(np.searchsorted(r.t_ev, pos, side='right'))
