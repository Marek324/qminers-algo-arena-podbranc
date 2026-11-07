"""Replaying a tape through a price-time priority book, recording everything the stats need.

Matching follows `simulator/simulator.py` for market-only flow: an insert crosses
the opposite side best price first, front of queue first, fills at the maker's
price, then a LIMIT remainder rests and an IOC remainder dies. A cancel removes
the order if it is resting and is a no-op otherwise.

State "at time t" means the book after every event with timestamp <= t.
"""

from bisect import insort
from dataclasses import dataclass

import numpy as np
import pandas as pd

LATENCY_US = 5_000  # send/cancel/modify take effect 5 ms later
DEPTH_LEVELS = 20  # ticks per side kept in the depth grid, like the default --public-depth

# Order outcomes (od_outcome)
RESTING_AT_END = 0
FILLED = 1
CANCELED = 2
IOC_DONE = 3  # IOC that did not fully fill (remainder killed, possibly zero fill)


@dataclass
class Replay:
    """Arrays describing the replayed tape. Prices are raw ticks; nan means that side was empty."""

    # Per event, state after the event.
    ev_ts: np.ndarray  # int64 us
    best_bid: np.ndarray  # float64
    best_ask: np.ndarray
    bid_qty: np.ndarray  # int64, quantity at best bid
    ask_qty: np.ndarray
    bid_orders: np.ndarray  # int32, orders queued at best bid
    ask_orders: np.ndarray

    # Per trade.
    tr_ts: np.ndarray
    tr_event: np.ndarray  # event index that produced the trade
    tr_price: np.ndarray
    tr_qty: np.ndarray
    tr_sign: np.ndarray  # +1 buyer-initiated (ask lifted), -1 seller-initiated
    tr_taker: np.ndarray  # order index of taker
    tr_maker: np.ndarray  # order index of maker

    # Per inserted order (index = insert order).
    od_ts: np.ndarray
    od_side: np.ndarray  # +1 buy, -1 sell
    od_price: np.ndarray
    od_qty: np.ndarray
    od_ioc: np.ndarray  # bool
    od_rel_touch: np.ndarray  # float: side*(price - same-side best) at arrival; >0 improves, 0 joins, <0 behind
    od_cross_gap: np.ndarray  # float: side*(price - opposite best) at arrival; >=0 means marketable
    od_queue_ahead: np.ndarray  # int64: quantity already at its price when it rested (-1 if never rested)
    od_taker_fill: np.ndarray  # quantity filled on arrival as taker
    od_maker_fill: np.ndarray  # quantity filled while resting
    od_first_maker_fill_ts: np.ndarray  # -1 if never filled as maker
    od_end_ts: np.ndarray  # cancel/fill/IOC time, -1 if resting at end
    od_outcome: np.ndarray  # RESTING_AT_END / FILLED / CANCELED / IOC_DONE
    od_cancel_rel_touch: np.ndarray  # float: side*(price - same-side best) just before the cancel

    # Depth grid: state at grid times.
    grid_ts: np.ndarray
    grid_bid_depth: np.ndarray  # (G, DEPTH_LEVELS) quantity at best_bid - k
    grid_ask_depth: np.ndarray  # (G, DEPTH_LEVELS) quantity at best_ask + k

    # Tape anomalies.
    n_cancel_not_resting: int  # cancels of filled/cancelled/unknown orders (no-ops)
    n_cancel_unknown: int  # cancels of ids never inserted
    n_cancel_after_fill: int  # cancels of orders already fully filled
    n_cancel_twice: int  # cancels of orders already cancelled
    n_duplicate_insert: int  # inserts reusing an id that is still resting (simulator would raise)

    # Passive fill options: for each market insert that crosses or improves the book, where a resting user order
    # could have been and how much of this order would reach it (price-time priority). Rows of one event are
    # alternatives, not additive. pm_side is the user's side (+1 buy, -1 sell).
    pm_event: np.ndarray
    pm_side: np.ndarray
    pm_price: np.ndarray
    pm_qty: np.ndarray
    pm_kind: np.ndarray  # 0 new price level (inside the spread or in a gap: user is first), 1 behind the queue at an existing level, 2 beyond the book at the order's limit

    @property
    def mid(self) -> np.ndarray:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> np.ndarray:
        return self.best_ask - self.best_bid

    def state_index(self, t_us: np.ndarray) -> np.ndarray:
        """Index of the event whose post-state holds at each time (-1 before the first event)."""
        return np.searchsorted(self.ev_ts, t_us, side='right') - 1

    def mid_at(self, t_us: np.ndarray) -> np.ndarray:
        idx = self.state_index(np.asarray(t_us))
        mid = self.mid
        out = np.full(idx.shape, np.nan)
        ok = idx >= 0
        out[ok] = mid[idx[ok]]
        return out


def replay(tape: pd.DataFrame, grid_step_us: int) -> Replay:
    """Replay the tape and return the recorded arrays."""
    n = len(tape)
    ts_list = tape['ts_us'].tolist()
    ins_list = tape['is_insert'].tolist()
    id_list = tape['OrderId'].tolist()
    side_list = tape['side_sign'].tolist()
    qty_list = tape['Volume'].fillna(0).astype('int64').tolist()
    px_list = tape['Price'].fillna(0).astype('int64').tolist()
    ioc_list = tape['ioc'].tolist()

    nan = float('nan')
    ev_bb = [nan] * n
    ev_ba = [nan] * n
    ev_bq = [0] * n
    ev_aq = [0] * n
    ev_bn = [0] * n
    ev_an = [0] * n

    tr_ts: list[int] = []
    tr_ev: list[int] = []
    tr_px: list[int] = []
    tr_q: list[int] = []
    tr_sg: list[int] = []
    tr_tk: list[int] = []
    tr_mk: list[int] = []

    od_ts: list[int] = []
    od_side: list[int] = []
    od_px: list[int] = []
    od_q: list[int] = []
    od_ioc: list[bool] = []
    od_rel: list[float] = []
    od_gap: list[float] = []
    od_ahead: list[int] = []
    od_tf: list[int] = []
    od_mf: list[int] = []
    od_ff: list[int] = []
    od_end: list[int] = []
    od_out: list[int] = []
    od_crel: list[float] = []

    # Book: per side, price -> {order_idx: remaining}, price -> total, sorted price list.
    bids: dict[int, dict[int, int]] = {}
    asks: dict[int, dict[int, int]] = {}
    btot: dict[int, int] = {}
    atot: dict[int, int] = {}
    bpx: list[int] = []  # ascending; best bid = bpx[-1]
    apx: list[int] = []  # ascending; best ask = apx[0]
    live: dict[str, int] = {}  # resting order id -> order idx
    id_last: dict[str, int] = {}  # any id ever inserted -> last order idx
    id_list_of: list[str] = []  # order idx -> id
    od_ev: list[int] = []  # order idx -> event index of its insert

    # For passive options: when an order sent after event j lands (first event >= ts_j + latency), and the
    # last event at which each price level traded, per side.
    ts_np = np.asarray(ts_list, dtype=np.int64)
    land_after = np.searchsorted(ts_np, ts_np + LATENCY_US, side='left').tolist() if n else []
    first_land = land_after[0] if n else 0
    last_tr_ask: dict[int, int] = {}
    last_tr_bid: dict[int, int] = {}
    pm_ev: list[int] = []
    pm_sd: list[int] = []
    pm_px: list[int] = []
    pm_q: list[int] = []
    pm_k: list[int] = []

    def ahead_of_user(lvl: dict[int, int], price_last_trade: int | None, i: int) -> int | None:
        """Market quantity ahead of a user order that joined this level right after its last trade; None if it cannot have."""
        ell = first_land if price_last_trade is None else land_after[price_last_trade]
        if ell > i:
            return None
        ahead = 0
        for oi, oq in lvl.items():
            if od_ev[oi] >= ell:
                break
            ahead += oq
        return ahead

    grid_ts: list[int] = []
    grid_b: list[list[int]] = []
    grid_a: list[list[int]] = []
    next_grid = ts_list[0] if n else 0
    levels = range(DEPTH_LEVELS)

    n_cancel_not_resting = n_cancel_unknown = n_cancel_after_fill = n_cancel_twice = n_dup = 0

    for i in range(n):
        ts = ts_list[i]

        if next_grid < ts:
            # Snapshot the state that holds for all grid points before this event.
            if bpx:
                bb = bpx[-1]
                snap_b = [btot.get(bb - k, 0) for k in levels]
            else:
                snap_b = [0] * DEPTH_LEVELS
            if apx:
                ba = apx[0]
                snap_a = [atot.get(ba + k, 0) for k in levels]
            else:
                snap_a = [0] * DEPTH_LEVELS
            while next_grid < ts:
                grid_ts.append(next_grid)
                grid_b.append(snap_b)
                grid_a.append(snap_a)
                next_grid += grid_step_us

        if ins_list[i]:
            oid = id_list[i]
            s = side_list[i]
            p = px_list[i]
            q = qty_list[i]
            ioc = ioc_list[i]
            idx = len(od_ts)
            if oid in live:
                n_dup += 1
            id_last[oid] = idx

            same_best = (bpx[-1] if bpx else None) if s == 1 else (apx[0] if apx else None)
            if i >= first_land:
                # Passive options for a user order on the other side of this insert.
                bb = bpx[-1] if bpx else None
                ba = apx[0] if apx else None
                if s == 1:  # market buy: user sells at x
                    x = p if ba is None else min(p, ba - 1)
                    if bb is None or x > bb:
                        pm_ev.append(i); pm_sd.append(-1); pm_px.append(x); pm_q.append(q); pm_k.append(0)  # noqa: E702
                    if ba is not None and p >= ba:
                        cum = 0
                        prev = None
                        for lp in apx:
                            if lp > p or cum >= q:
                                break
                            if prev is not None and lp - 1 > prev:
                                pm_ev.append(i); pm_sd.append(-1); pm_px.append(lp - 1); pm_q.append(q - cum); pm_k.append(0)  # noqa: E702
                            ahead = ahead_of_user(asks[lp], last_tr_ask.get(lp), i)
                            if ahead is not None and q - cum - ahead > 0:
                                pm_ev.append(i); pm_sd.append(-1); pm_px.append(lp); pm_q.append(q - cum - ahead); pm_k.append(1)  # noqa: E702
                            cum += atot[lp]
                            prev = lp
                        if cum < q and prev is not None and p > prev:
                            pm_ev.append(i); pm_sd.append(-1); pm_px.append(p); pm_q.append(q - cum); pm_k.append(2)  # noqa: E702
                else:  # market sell: user buys at x
                    x = p if bb is None else max(p, bb + 1)
                    if ba is None or x < ba:
                        pm_ev.append(i); pm_sd.append(1); pm_px.append(x); pm_q.append(q); pm_k.append(0)  # noqa: E702
                    if bb is not None and p <= bb:
                        cum = 0
                        prev = None
                        for lp in reversed(bpx):
                            if lp < p or cum >= q:
                                break
                            if prev is not None and lp + 1 < prev:
                                pm_ev.append(i); pm_sd.append(1); pm_px.append(lp + 1); pm_q.append(q - cum); pm_k.append(0)  # noqa: E702
                            ahead = ahead_of_user(bids[lp], last_tr_bid.get(lp), i)
                            if ahead is not None and q - cum - ahead > 0:
                                pm_ev.append(i); pm_sd.append(1); pm_px.append(lp); pm_q.append(q - cum - ahead); pm_k.append(1)  # noqa: E702
                            cum += btot[lp]
                            prev = lp
                        if cum < q and prev is not None and p < prev:
                            pm_ev.append(i); pm_sd.append(1); pm_px.append(p); pm_q.append(q - cum); pm_k.append(2)  # noqa: E702
            opp_best = (apx[0] if apx else None) if s == 1 else (bpx[-1] if bpx else None)
            od_ts.append(ts)
            od_ev.append(i)
            od_side.append(s)
            od_px.append(p)
            od_q.append(q)
            od_ioc.append(ioc)
            od_rel.append(nan if same_best is None else float(s * (p - same_best)))
            od_gap.append(nan if opp_best is None else float(s * (p - opp_best)))
            od_mf.append(0)
            od_ff.append(-1)
            od_crel.append(nan)

            remaining = q
            if s == 1:
                opx, olv, otot = apx, asks, atot
                while remaining and opx:
                    lp = opx[0]
                    if lp > p:
                        break
                    lvl = olv[lp]
                    while remaining and lvl:
                        mk = next(iter(lvl))
                        mq = lvl[mk]
                        t = remaining if remaining < mq else mq
                        remaining -= t
                        tr_ts.append(ts)
                        tr_ev.append(i)
                        tr_px.append(lp)
                        tr_q.append(t)
                        tr_sg.append(1)
                        last_tr_ask[lp] = i
                        tr_tk.append(idx)
                        tr_mk.append(mk)
                        od_mf[mk] += t
                        if od_ff[mk] < 0:
                            od_ff[mk] = ts
                        if t == mq:
                            del lvl[mk]
                            od_end[mk] = ts
                            od_out[mk] = FILLED
                            if live.get(id_list_of[mk]) == mk:
                                del live[id_list_of[mk]]
                        else:
                            lvl[mk] = mq - t
                        otot[lp] -= t
                    if not lvl:
                        del olv[lp]
                        del otot[lp]
                        opx.pop(0)
            else:
                opx, olv, otot = bpx, bids, btot
                while remaining and opx:
                    lp = opx[-1]
                    if lp < p:
                        break
                    lvl = olv[lp]
                    while remaining and lvl:
                        mk = next(iter(lvl))
                        mq = lvl[mk]
                        t = remaining if remaining < mq else mq
                        remaining -= t
                        tr_ts.append(ts)
                        tr_ev.append(i)
                        tr_px.append(lp)
                        tr_q.append(t)
                        tr_sg.append(-1)
                        last_tr_bid[lp] = i
                        tr_tk.append(idx)
                        tr_mk.append(mk)
                        od_mf[mk] += t
                        if od_ff[mk] < 0:
                            od_ff[mk] = ts
                        if t == mq:
                            del lvl[mk]
                            od_end[mk] = ts
                            od_out[mk] = FILLED
                            if live.get(id_list_of[mk]) == mk:
                                del live[id_list_of[mk]]
                        else:
                            lvl[mk] = mq - t
                        otot[lp] -= t
                    if not lvl:
                        del olv[lp]
                        del otot[lp]
                        opx.pop()

            od_tf.append(q - remaining)
            id_list_of.append(oid)
            if remaining == 0:
                od_ahead.append(-1)
                od_end.append(ts)
                od_out.append(FILLED)
            elif ioc:
                od_ahead.append(-1)
                od_end.append(ts)
                od_out.append(IOC_DONE)
            else:
                if s == 1:
                    lv, tot, pxs = bids, btot, bpx
                else:
                    lv, tot, pxs = asks, atot, apx
                lvl = lv.get(p)
                if lvl is None:
                    lvl = lv[p] = {}
                    tot[p] = 0
                    insort(pxs, p)
                od_ahead.append(tot[p])
                lvl[idx] = remaining
                tot[p] += remaining
                live[oid] = idx
                od_end.append(-1)
                od_out.append(RESTING_AT_END)
        else:
            oid = id_list[i]
            idx = live.pop(oid, None)
            if idx is None:
                n_cancel_not_resting += 1
                prev = id_last.get(oid)
                if prev is None:
                    n_cancel_unknown += 1
                elif od_out[prev] == CANCELED:
                    n_cancel_twice += 1
                elif od_out[prev] == FILLED:
                    n_cancel_after_fill += 1
            else:
                s = od_side[idx]
                p = od_px[idx]
                if s == 1:
                    lv, tot, pxs = bids, btot, bpx
                    best = bpx[-1]
                else:
                    lv, tot, pxs = asks, atot, apx
                    best = apx[0]
                od_crel[idx] = float(s * (p - best))
                lvl = lv[p]
                rq = lvl.pop(idx)
                tot[p] -= rq
                if not lvl:
                    del lv[p]
                    del tot[p]
                    pxs.remove(p)
                od_end[idx] = ts
                od_out[idx] = CANCELED

        if bpx:
            bb = bpx[-1]
            ev_bb[i] = bb
            ev_bq[i] = btot[bb]
            ev_bn[i] = len(bids[bb])
        if apx:
            ba = apx[0]
            ev_ba[i] = ba
            ev_aq[i] = atot[ba]
            ev_an[i] = len(asks[ba])

    if n and next_grid <= ts_list[-1]:
        if bpx:
            snap_b = [btot.get(bpx[-1] - k, 0) for k in levels]
        else:
            snap_b = [0] * DEPTH_LEVELS
        if apx:
            snap_a = [atot.get(apx[0] + k, 0) for k in levels]
        else:
            snap_a = [0] * DEPTH_LEVELS
        while next_grid <= ts_list[-1]:
            grid_ts.append(next_grid)
            grid_b.append(snap_b)
            grid_a.append(snap_a)
            next_grid += grid_step_us

    i64 = np.int64
    return Replay(
        ev_ts=np.asarray(ts_list, dtype=i64),
        best_bid=np.asarray(ev_bb, dtype=float),
        best_ask=np.asarray(ev_ba, dtype=float),
        bid_qty=np.asarray(ev_bq, dtype=i64),
        ask_qty=np.asarray(ev_aq, dtype=i64),
        bid_orders=np.asarray(ev_bn, dtype=np.int32),
        ask_orders=np.asarray(ev_an, dtype=np.int32),
        tr_ts=np.asarray(tr_ts, dtype=i64),
        tr_event=np.asarray(tr_ev, dtype=i64),
        tr_price=np.asarray(tr_px, dtype=i64),
        tr_qty=np.asarray(tr_q, dtype=i64),
        tr_sign=np.asarray(tr_sg, dtype=np.int8),
        tr_taker=np.asarray(tr_tk, dtype=i64),
        tr_maker=np.asarray(tr_mk, dtype=i64),
        od_ts=np.asarray(od_ts, dtype=i64),
        od_side=np.asarray(od_side, dtype=np.int8),
        od_price=np.asarray(od_px, dtype=i64),
        od_qty=np.asarray(od_q, dtype=i64),
        od_ioc=np.asarray(od_ioc, dtype=bool),
        od_rel_touch=np.asarray(od_rel, dtype=float),
        od_cross_gap=np.asarray(od_gap, dtype=float),
        od_queue_ahead=np.asarray(od_ahead, dtype=i64),
        od_taker_fill=np.asarray(od_tf, dtype=i64),
        od_maker_fill=np.asarray(od_mf, dtype=i64),
        od_first_maker_fill_ts=np.asarray(od_ff, dtype=i64),
        od_end_ts=np.asarray(od_end, dtype=i64),
        od_outcome=np.asarray(od_out, dtype=np.int8),
        od_cancel_rel_touch=np.asarray(od_crel, dtype=float),
        grid_ts=np.asarray(grid_ts, dtype=i64),
        grid_bid_depth=np.asarray(grid_b, dtype=i64).reshape(-1, DEPTH_LEVELS),
        grid_ask_depth=np.asarray(grid_a, dtype=i64).reshape(-1, DEPTH_LEVELS),
        n_cancel_not_resting=n_cancel_not_resting,
        n_cancel_unknown=n_cancel_unknown,
        n_cancel_after_fill=n_cancel_after_fill,
        n_cancel_twice=n_cancel_twice,
        n_duplicate_insert=n_dup,
        pm_event=np.asarray(pm_ev, dtype=i64),
        pm_side=np.asarray(pm_sd, dtype=np.int8),
        pm_price=np.asarray(pm_px, dtype=i64),
        pm_qty=np.asarray(pm_q, dtype=i64),
        pm_kind=np.asarray(pm_k, dtype=np.int8),
    )
