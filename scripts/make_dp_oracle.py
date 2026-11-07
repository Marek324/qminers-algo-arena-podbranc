"""Build a deliberately overfit strategy that replays the hindsight-optimal trade plan of one tape.

usage: uv run python scripts/make_dp_oracle.py data/round2_public.csv strategies/dumb.py [JOIN_WINDOW_MS]

JOIN_WINDOW_MS: a queue-joining order is assumed sent at most this long before its fill (default 300).

Runs the same dynamic program as analysis/sections/maxpnl.py (best PnL knowing the future, |inventory| <= 100),
traces the best path back and writes every trade on it into the strategy file as a scheduled order:
- take: IOC at the touch price, sent after the event from which it lands on the planned book state;
- make: limit order at the planned price, sent just in time (or right after the level's last trade, for a place
  in an existing queue) and cancelled after the market order that should fill it.
The strategy counts callbacks (one per tape row) to know the event index. Only meaningful on that exact tape.

Result on round2_public (2026-11-07): plan worth 1.44M (300 ms join window), backtest 1.04-1.05M, below the
simpler future-mid oracle (scripts/make_oracle.py, 1.19M). Known flaw: the DP counts the same lots twice. A market
order that lands inside the spread both fills our planned resting order and stays in the book as a stale quote the
plan later takes, so about 60% of the planned IOCs find nothing. Fixing it needs per-order tracking in the replay.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from analysis.context import GRID_STEP_US  # noqa: E402
from analysis.replay import LATENCY_US, replay  # noqa: E402
from analysis.sections.maxpnl import _window_max  # noqa: E402
from analysis.tape import load_tape  # noqa: E402

MAX_POS = 100
NEG = float('-inf')
STATES = np.arange(-MAX_POS, MAX_POS + 1, dtype=float)


def trade(v, side, price, qty):
    w = v + STATES * price
    return _window_max(w, int(qty) + 1, forward=side < 0) - STATES * price


def back(v_in, v_out_s, s, side, price, qty):
    """State before a trade of up to qty at price that ends in state s with value v_out_s; None if not reachable."""
    w = v_in + STATES * price
    i = s + MAX_POS
    rng = range(i, max(-1, i - qty - 1), -1) if side > 0 else range(i, min(len(w), i + qty + 1))
    for j in rng:  # nearest first: the smallest trade that reaches the value
        if abs(w[j] - STATES[i] * price - v_out_s) < 1e-6:
            return j - MAX_POS
    return None


tape_path, out = Path(sys.argv[1]), Path(sys.argv[2])
join_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
rp = replay(load_tape(tape_path), GRID_STEP_US, join_window_us=int(join_ms * 1000))
ts = rp.ev_ts
n = len(ts)
land_all = np.minimum(np.searchsorted(ts, ts + LATENCY_US, side='left'), n) - 1  # state an order sent after j meets

# ---- operations, as in maxpnl._ops, keeping kind and join for makes
ops = []
land = np.unique(land_all)
land = land[land >= 0]
ok = np.isfinite(rp.best_bid[land]) & np.isfinite(rp.best_ask[land])
land = land[ok]
b, a = rp.best_bid[land], rp.best_ask[land]
bq, aq = rp.bid_qty[land], rp.ask_qty[land]
# One take decision per run of states with the same touch prices: the lots there are mostly the same orders, so
# taking them at two states of the run would count them twice (the DP does not track what we consumed).
seg = np.zeros(len(land), dtype=np.int64)
new = np.r_[True, (b[1:] != b[:-1]) | (a[1:] != a[:-1]) | (seg[1:] != seg[:-1])]
starts = np.flatnonzero(new)
# take the size of the state the order really lands on (the run's first state), not the run's max
for k, bi, bqi, ai, aqi in zip(land[starts].tolist(), b[starts].tolist(), bq[starts].tolist(), a[starts].tolist(), aq[starts].tolist()):
    ops.append((k + 1, 0, ('take', k, int(bi), int(bqi), int(ai), int(aqi))))
ev = rp.pm_event
st = np.flatnonzero(np.r_[True, ev[1:] != ev[:-1]])
en = np.r_[st[1:], len(ev)]
sd, px, qt, kd, jn = rp.pm_side.tolist(), rp.pm_price.tolist(), rp.pm_qty.tolist(), rp.pm_kind.tolist(), rp.pm_join.tolist()
for s0, e0 in zip(st.tolist(), en.tolist()):
    ops.append((int(ev[s0]), 1, ('make', int(ev[s0]), [(sd[j], int(px[j]), int(qt[j]), kd[j], jn[j]) for j in range(s0, e0)])))
ops.sort(key=lambda o: (o[0], o[1]))
print(f'{len(ops)} operations')

# ---- forward DP, keeping the value vector before every op
v = np.full(len(STATES), NEG)
v[MAX_POS] = 0.0
vs = np.empty((len(ops), len(STATES)))
for i, (_, _, op) in enumerate(ops):
    vs[i] = v
    if op[0] == 'take':
        _, _, bid, bqty, ask, aqty = op
        v = trade(v, 1, ask, aqty)
        v = trade(v, -1, bid, bqty)
    else:
        best = v
        for side, price, qty, _, _ in op[2]:
            best = np.maximum(best, trade(v, side, price, qty))
        v = best
bb, ba = rp.best_bid[-1], rp.best_ask[-1]
final = v + np.where(STATES > 0, STATES * bb, STATES * ba)
s = int(np.nanargmax(final)) - MAX_POS
print(f'DP max {final[s + MAX_POS]:,.0f}, final inventory {s}')

# ---- trace back: actions (send_event, fill_event, end_event, side, price, qty, ioc)
acts = []
for i in range(len(ops) - 1, -1, -1):
    op = ops[i][2]
    v_in = vs[i]
    v_out = vs[i + 1] if i + 1 < len(ops) else v
    target = v_out[s + MAX_POS]
    if op[0] == 'take':
        _, k, bid, bqty, ask, aqty = op
        v_mid = trade(v_in, 1, ask, aqty)
        s_mid = back(v_mid, target, s, -1, bid, bqty)
        s_in = back(v_in, v_mid[s_mid + MAX_POS], s_mid, 1, ask, aqty)
        j = int(np.searchsorted(land_all, k, side='right') - 1)  # latest event whose order lands on state k
        assert land_all[j] == k
        if s_mid != s:
            acts.append((j, k + 1, -1, -1, bid, s_mid - s, 1))
        if s_in != s_mid:
            acts.append((j, k + 1, -1, 1, ask, s_mid - s_in, 1))
        s = s_in
    else:
        _, e, alts = op
        if abs(v_in[s + MAX_POS] - target) < 1e-6:
            continue
        for side, price, qty, kind, join in alts:
            s_in = back(v_in, target, s, side, price, qty)
            if s_in is not None:
                break
        else:
            raise RuntimeError(f'no alternative reaches the value at op {i}')
        if kind == 1:
            j = max(join, 0)
        else:
            j = int(np.searchsorted(ts, ts[e] - LATENCY_US, side='right') - 1)
        acts.append((max(j, 0), e, e, side, price, abs(s - s_in), 0))
        s = s_in
assert s == 0, s
acts.sort()
print(f'{len(acts)} actions: {sum(a[6] for a in acts)} takes, {sum(1 - a[6] for a in acts)} makes')


def lst(xs):
    return '[' + ','.join(map(str, xs)) + ']'


send = [a[0] for a in acts]
d_send = [send[0]] + [send[i] - send[i - 1] for i in range(1, len(send))]
src = f'''"""DUMMY ORACLE: replays the hindsight-optimal plan for {tape_path.name}. Only meaningful on {tape_path}. Never submit.

Generated by scripts/make_dp_oracle.py. Counts callbacks (one per tape row) to know the event index and sends
each planned order on its scheduled event: IOCs for takes, limit orders for makes (cancelled after the event that
should fill them). Order sizes are corrected for any drift between the real and the planned inventory.
"""
from __future__ import annotations

import heapq

from simulator.models import OrderBookSnapshot, OrderRequest, OrderStatus, OrderType, PrivateOrderEvent, Side, Trade
from simulator.strategy import BaseStrategy

MAX_POS = 100
CORRECT = 2   # adjust order sizes by (real - planned) inventory: 0 never, 1 takes only, 2 all orders

# planned actions, sorted by send event
D_SEND = {lst(d_send)}
FILL = {lst([a[1] for a in acts])}
END = {lst([a[2] for a in acts])}
SIDE = {lst([a[3] for a in acts])}
PX = {lst([a[4] for a in acts])}
QTY = {lst([a[5] for a in acts])}
IOC = {lst([a[6] for a in acts])}


class DumbOracle(BaseStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.send_ev = []
        t = 0
        for d in D_SEND:
            t += d
            self.send_ev.append(t)
        # planned inventory after each fill event
        order = sorted(range(len(FILL)), key=lambda i: FILL[i])
        self.plan_ev = []
        self.plan_inv = []
        inv = 0
        for i in order:
            inv += SIDE[i] * QTY[i]
            self.plan_ev.append(FILL[i])
            self.plan_inv.append(inv)
        self.pp = 0
        self.planned = 0
        self.n = -1
        self.p = 0
        self.inv = 0
        self.orders: dict[str, list] = {{}}   # id -> [sign, remaining]
        self.ends: list = []                 # heap of (end event, order id)

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        self.n += 1
        n = self.n
        while self.pp < len(self.plan_ev) and self.plan_ev[self.pp] <= n:
            self.planned = self.plan_inv[self.pp]
            self.pp += 1
        ends = self.ends
        while ends and ends[0][0] <= n:
            _, oid = heapq.heappop(ends)
            if oid in self.orders:
                self.cancel_order(oid)
        send_ev = self.send_ev
        while self.p < len(send_ev) and send_ev[self.p] <= n:
            self._act(self.p)
            self.p += 1

    def _act(self, i: int) -> None:
        sign = SIDE[i]
        q = QTY[i]
        if CORRECT == 2 or (IOC[i] and CORRECT == 1):
            q -= sign * (self.inv - self.planned)
        resting = 0
        for o in self.orders.values():
            if o[0] == sign:
                resting += o[1]
        q = min(q, MAX_POS - sign * self.inv - resting)
        if q <= 0:
            return
        side = Side.BUY if sign > 0 else Side.SELL
        ot = OrderType.IOC if IOC[i] else OrderType.LIMIT
        oid = self.send_order(OrderRequest(side=side, price=PX[i], quantity=q, order_type=ot))
        self.orders[oid] = [sign, q]
        if not IOC[i]:
            heapq.heappush(self.ends, (END[i], oid))

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        o = self.orders.get(event.order_id)
        if o is None:
            return
        st = event.status
        if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
            self.inv += o[0] * event.filled_quantity
            o[1] = event.remaining_quantity
            if st == OrderStatus.FILLED:
                self.orders.pop(event.order_id, None)
        elif st == OrderStatus.CANCELED or st == OrderStatus.REJECTED:
            self.orders.pop(event.order_id, None)
'''
out.write_text(src)
print(f'{out}: {out.stat().st_size / 1e6:.2f} MB')
