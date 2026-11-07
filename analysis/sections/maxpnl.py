"""Theoretical maximum PnL: the best any strategy could score on this tape under the simulator's rules, knowing the future.

One dynamic program over inventory -MAX_POS..MAX_POS walks the tape in time order and combines both ways to trade:

- take: at any book state an order can land on (sent in the callback after event j, it meets the book after the last
  event before ts_j + 5 ms), buy at the best ask / sell at the best bid, up to the quantity there.
- make: for every market insert that crosses or improves the book, a resting user order on the other side could have
  been filled by it, with price-time priority (recorded by the replay, see `Replay.pm_*`):
    * at a new price level (inside the spread, or a gap in the book the insert sweeps through): the user is alone
      there, so first; the insert reaches it after any better-priced market liquidity.
    * at an existing level: behind the market orders that were already there when it could have joined, i.e. right
      after that level last traded (joining earlier would have been filled by that trade).
    * beyond the book, at the insert's own limit price, for whatever the book could not fill.
  The options for one insert are alternatives (the user's lower-priced orders are consumed first and shrink what
  reaches higher ones), so they are combined with a max, not added.

Ends flat by selling/buying the leftover into the final touch (the simulator's markout). |inventory| stays <= 100:
going outside costs 10,000 per second, which no edge here pays for. Not modelled: the user's own fills changing what
the market does next (the tape is fixed, so later market trades still happen as recorded), and the 1000-lot cap on
resting size (never binding with orders placed just in time). PnL is in the simulator's units (price x lots).
"""

import numpy as np

from ..context import Context
from ..replay import LATENCY_US
from ..util import US, safe_div

MAX_POS = 100
NEG = float('-inf')
STATES = np.arange(-MAX_POS, MAX_POS + 1, dtype=float)


def _window_max(w: np.ndarray, length: int, forward: bool) -> np.ndarray:
    """out[i] = max(w[i-length+1 .. i]) (trailing) or max(w[i .. i+length-1]) (forward), -inf outside."""
    n = len(w)
    length = min(length, n)
    m = w.copy()
    k = 1
    while 2 * k <= length:
        sh = np.full(n, NEG)
        if forward:
            sh[:-k] = m[k:]
        else:
            sh[k:] = m[:-k]
        np.maximum(m, sh, out=m)
        k *= 2
    rest = length - k
    if rest > 0:
        sh = np.full(n, NEG)
        if forward:
            sh[:-rest] = m[rest:]
        else:
            sh[rest:] = m[:-rest]
        np.maximum(m, sh, out=m)
    return m


def _trade(v: np.ndarray, side: int, price: float, qty: int) -> np.ndarray:
    """Value after optionally trading up to `qty` lots at `price` (side +1 buy, -1 sell).

    `price` is signed by the objective: +price maximizes PnL, -price minimizes it.
    """
    w = v + STATES * price
    # buy: s' = s + k, from s in [s'-qty, s']; sell: s' = s - k, from s in [s', s'+qty]
    return _window_max(w, int(qty) + 1, forward=side < 0) - STATES * price


def _ops(ctx: Context, use_take: bool, use_make: bool) -> list[tuple]:
    """Time-ordered operations: ('take', bid, bid_qty, ask, ask_qty) or ('make', [(side, price, qty), ...])."""
    rp = ctx.rp
    ops: list[tuple[int, int, tuple]] = []
    if use_take:
        ts = rp.ev_ts
        n = len(ts)
        land = np.unique(np.minimum(np.searchsorted(ts, ts + LATENCY_US, side='left'), n) - 1)
        land = land[land >= 0]
        ok = np.isfinite(rp.best_bid[land]) & np.isfinite(rp.best_ask[land])
        land = land[ok]
        # A state is applied before the next event. Consecutive states with the same touch prices and no make
        # opportunity between them are one decision; the size available is the largest seen in the run.
        b, a = rp.best_bid[land], rp.best_ask[land]
        bq, aq = rp.bid_qty[land], rp.ask_qty[land]
        seg = np.searchsorted(np.unique(rp.pm_event), land + 1, side='left') if use_make else np.zeros(len(land), dtype=np.int64)
        new = np.r_[True, (b[1:] != b[:-1]) | (a[1:] != a[:-1]) | (seg[1:] != seg[:-1])] if len(land) else land.astype(bool)
        starts = np.flatnonzero(new)
        if len(starts):
            bq_max = np.maximum.reduceat(bq, starts)
            aq_max = np.maximum.reduceat(aq, starts)
            for k, bi, bqi, ai, aqi in zip(land[starts].tolist(), b[starts].tolist(), bq_max.tolist(), a[starts].tolist(), aq_max.tolist(), strict=True):
                ops.append((k + 1, 0, ('take', bi, int(bqi), ai, int(aqi))))
    if use_make and len(rp.pm_event):
        ev = rp.pm_event
        starts = np.flatnonzero(np.r_[True, ev[1:] != ev[:-1]])
        ends = np.r_[starts[1:], len(ev)]
        side, price, qty = rp.pm_side.tolist(), rp.pm_price.tolist(), rp.pm_qty.tolist()
        for s0, e0 in zip(starts.tolist(), ends.tolist(), strict=True):
            ops.append((int(ev[s0]), 1, ('make', [(side[j], float(price[j]), qty[j]) for j in range(s0, e0)])))
    ops.sort(key=lambda o: (o[0], o[1]))
    return ops


def _solve(ctx: Context, use_take: bool, use_make: bool, curve_t: np.ndarray, sign: int = 1) -> tuple[dict, np.ndarray]:
    """Best (sign=+1) or worst (sign=-1) PnL; values are kept as sign * PnL so the DP always maximizes."""
    rp = ctx.rp
    ops = _ops(ctx, use_take, use_make)
    v = np.full(len(STATES), NEG)
    v[MAX_POS] = 0.0
    # Curve: best PnL achievable by time t ending flat at t.
    op_ts = np.array([rp.ev_ts[min(o[0], len(rp.ev_ts) - 1)] for o in ops], dtype=np.int64)
    curve_ix = np.searchsorted(op_ts, curve_t, side='right')  # ops applied before curve point j
    curve = np.zeros(len(curve_t))
    c = 0
    for i, (_, _, op) in enumerate(ops):
        while c < len(curve_t) and curve_ix[c] <= i:
            curve[c] = v[MAX_POS]
            c += 1
        if op[0] == 'take':
            _, bid, bq, ask, aq = op
            v = _trade(v, 1, sign * ask, aq)
            v = _trade(v, -1, sign * bid, bq)
        else:
            best = v
            for side, price, qty in op[1]:
                best = np.maximum(best, _trade(v, side, sign * price, qty))
            v = best
    while c < len(curve_t):
        curve[c] = v[MAX_POS]
        c += 1

    bb, ba = rp.best_bid[-1], rp.best_ask[-1]
    last = float(rp.tr_price[-1]) if len(rp.tr_price) else np.nan
    bb = bb if np.isfinite(bb) else last
    ba = ba if np.isfinite(ba) else last
    final = v + sign * np.where(STATES > 0, STATES * bb, STATES * ba)
    j = int(np.nanargmax(final)) if np.isfinite(final).any() else MAX_POS
    pnl = sign * float(final[j]) if np.isfinite(final[j]) else 0.0
    curve *= sign
    if len(curve):
        curve[-1] = pnl
    hours = ctx.duration_s / 3600 if ctx.duration_s else None
    return {
        'pnl': pnl,
        'pnl_per_hour': safe_div(pnl, hours),
        'inventory_before_markout': int(STATES[j]),
        'operations': len(ops),
    }, curve


def max_pnl_section(ctx: Context, curve_step_us: int) -> dict:
    rp = ctx.rp
    k = max(int(np.ceil((ctx.t1 - ctx.t0) / curve_step_us)), 1)
    curve_t = ctx.t0 + curve_step_us * np.arange(1, k + 1, dtype=np.int64)
    curve_t[-1] = max(curve_t[-1], ctx.t1)

    both, c_both = _solve(ctx, True, True, curve_t)
    make, c_make = _solve(ctx, False, True, curve_t)
    take, c_take = _solve(ctx, True, False, curve_t)
    worst, c_worst = _solve(ctx, True, True, curve_t, sign=-1)
    kinds = rp.pm_kind
    return {
        'note': 'best (theoretical_max) and worst (theoretical_min) PnL any strategy could score knowing the future, under the '
        'simulator rules: 5 ms latency, price-time priority, |inventory| <= 100, flat at the end; see analysis/sections/maxpnl.py',
        'max_position': MAX_POS,
        'theoretical_max': both,
        'making_only': make,
        'taking_only': take,
        'theoretical_min': worst,
        'opportunities': {
            'make_events': int(len(np.unique(rp.pm_event))),
            'make_options': int(len(kinds)),
            'new_level_options': int((kinds == 0).sum()),
            'queue_options': int((kinds == 1).sum()),
            'beyond_book_options': int((kinds == 2).sum()),
            'make_volume_new_level': int(rp.pm_qty[kinds == 0].sum()),
        },
        'curve': {
            'step_s': curve_step_us / US,
            'rows': int(len(curve_t)),
            'note': 'best PnL achievable by time t if flat at t',
            'columns': ['t_s', 'theoretical_max', 'making_only', 'taking_only', 'theoretical_min'],
            'data': {'t_s': (curve_t - ctx.t0) / US, 'theoretical_max': c_both, 'making_only': c_make, 'taking_only': c_take,
                     'theoretical_min': c_worst},
        },
    }


def tape_bounds(path) -> dict:
    """Theoretical max/min (and making/taking only) PnL for one tape file, without the rest of the report."""
    from ..context import GRID_STEP_US
    from ..context import Context as _Ctx
    from ..replay import replay
    from ..tape import load_tape

    tape = load_tape(path)
    rp = replay(tape, GRID_STEP_US)
    ctx = _Ctx(tape=tape, rp=rp, series_step_us=US, bin_step_us=60 * US)
    sec = max_pnl_section(ctx, 60 * US)
    out = {k: sec[k]['pnl'] for k in ('theoretical_max', 'theoretical_min', 'making_only', 'taking_only')}
    out['duration_s'] = ctx.duration_s
    mid = ctx.mid_t[np.isfinite(ctx.mid_t)]
    out['mid_change_ticks'] = float(mid[-1] - mid[0]) if len(mid) else None
    m1 = ctx.mid_at(np.arange(ctx.t0, ctx.t1 + 1, US, dtype=np.int64))
    r1 = np.diff(m1)
    r1 = r1[np.isfinite(r1)]
    out['vol_ticks_per_sqrt_s'] = float(r1.std()) if len(r1) > 2 else None
    return out
