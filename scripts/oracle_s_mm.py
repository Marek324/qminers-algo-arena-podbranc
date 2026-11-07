"""ORACLE (local analysis only, NOT submittable): imbalance_mm with the TRUE hidden price S.

Reads S from the generator's npz (scripts/gen_tape.py) instead of estimating it from new
limit orders. Use it on a synthetic tape to measure how much PnL a perfect S would add,
i.e. the ceiling of this strategy family.

  ORACLE_S=data/synthetic/synth_r1_s1_S.npz  path to the S file (required)
  ORACLE_PARAMS="K_S=1,K_IMB=0"    optional class-attribute overrides (for sweeps)
"""


from __future__ import annotations

import math

from simulator.models import (
    OrderBookSnapshot,
    OrderRequest,
    OrderStatus,
    OrderType,
    PrivateOrderEvent,
    Side,
    Trade,
)
from simulator.strategy import BaseStrategy

import os

import numpy as np


class OracleSMMStrategy(BaseStrategy):
    QTY = 200
    MAX_POS = 100
    EDGE = 0.4           # min ticks between the target price and fair
    K_IMB = 0.15         # ticks of fair shift at full imbalance
    K_INV = 0.002        # ticks of fair shift per lot of inventory
    KEEP_LEVELS = 1      # how many ticks behind the target an order may stay
    MIN_NEW = 10         # don't send orders smaller than this
    EDGE_IN = 0.5        # min edge to quote inside a 2+ tick spread
    K_FLOW = 0.5         # ticks of fair shift per touch-depth of recent aggressive volume
    FLOW_TAU_US = 200_000
    DEPTH_EMA = 0.001
    K_S = 0.75           # weight of the hidden-price estimate: fair moves K_S of the way from mid to S_hat
    S_ALPHA = 0.1        # weight of each new limit order in the per-side price average
    S_LEVELS = 2         # look for new orders this many ticks behind the best price
    EDGE_TAKE = 0.25     # cross the spread (bid at the best ask) when fair - best_ask >= this

    def __init__(self) -> None:
        super().__init__()
        self.inv = 0
        # order_id -> [sign, price, remaining]
        self.orders: dict[str, list] = {}
        self.cancel_sent: set[str] = set()
        self.flow = 0.0          # signed aggressive volume, fading over FLOW_TAU_US
        self.flow_t = 0
        self.depth = 0.0         # running average touch size
        # hidden fair price: average price of newly arrived market limit orders, per side
        self.s_bid = 0.0
        self.s_ask = 0.0
        self.max_id = ''         # highest market order id seen so far (ids only grow)
        self.last_bids = None
        self.last_asks = None
        z = np.load(os.environ['ORACLE_S'])
        self.st = z['t_us'].tolist()
        self.sv = z['S'].tolist()
        self.sp = 0
        for kv in os.environ.get('ORACLE_PARAMS', '').split(','):
            if kv.strip():
                k, v = kv.split('=')
                setattr(self, k.strip(), float(v) if '.' in v or 'e' in v else int(v))

    def _cancel(self, oid: str) -> None:
        if oid not in self.cancel_sent:
            self.cancel_order(oid)
            self.cancel_sent.add(oid)

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bids, asks = snapshot.bids, snapshot.asks
        if not bids or not asks:
            return
        # touch of market orders only (ours excluded)
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

        mid = (bb + ba) / 2
        fair = mid + self.K_IMB * (bq - aq) / (bq + aq) - self.K_INV * self.inv
        if self.K_S:
            now = snapshot.timestamp_us
            st, sp, ns = self.st, self.sp, len(self.st) - 1
            while sp < ns and st[sp + 1] <= now:
                sp += 1
            self.sp = sp
            fair += self.K_S * (self.sv[sp] - mid)
        if self.K_FLOW:
            now = snapshot.timestamp_us
            self.flow *= math.exp((self.flow_t - now) / self.FLOW_TAU_US)
            self.flow_t = now
            for tr in trades:
                if tr.taker_owner == 'market':
                    self.flow += tr.quantity if tr.book_side == 'ask' else -tr.quantity
            half = (bq + aq) / 2
            self.depth = half if self.depth == 0 else self.depth + self.DEPTH_EMA * (half - self.depth)
            fair += self.K_FLOW * self.flow / self.depth

        if fair - ba >= self.EDGE_TAKE:
            pb = ba
        elif ba - bb >= 2 and fair - (bb + 1) >= self.EDGE_IN:
            pb = bb + 1
        else:
            pb = bb
            while fair - pb < self.EDGE:
                pb -= 1
        if bb - fair >= self.EDGE_TAKE:
            pa = bb
        elif ba - bb >= 2 and (ba - 1) - fair >= self.EDGE_IN:
            pa = ba - 1
        else:
            pa = ba
            while pa - fair < self.EDGE:
                pa += 1
        self._side(1, pb)
        self._side(-1, pa)

    def _scan_new(self, views, sign: int) -> None:
        """Fold market orders newer than max_id, within S_LEVELS of the best price, into the side's average."""
        first = views[0].price
        mx = self.max_id
        lm = len(mx)
        a = self.S_ALPHA
        for o in views:
            if (o.price - first) * sign > self.S_LEVELS:
                break
            if o.owner != 'market':
                continue
            oid = o.order_id
            if len(oid) > lm or (len(oid) == lm and oid > mx):
                if sign < 0:
                    self.s_bid = o.price if not self.s_bid else self.s_bid + a * (o.price - self.s_bid)
                else:
                    self.s_ask = o.price if not self.s_ask else self.s_ask + a * (o.price - self.s_ask)
                if len(oid) > len(self.max_id) or (len(oid) == len(self.max_id) and oid > self.max_id):
                    self.max_id = oid

    def _side(self, sign: int, target: int) -> None:
        resting = 0
        at_target = 0
        keep = []
        for oid, o in self.orders.items():
            if o[0] != sign or oid in self.cancel_sent:
                if o[0] == sign:
                    resting += o[2]
                continue
            behind = (target - o[1]) * sign      # >0: less aggressive than target
            if behind < 0 or behind > self.KEEP_LEVELS:
                self._cancel(oid)
                resting += o[2]
                continue
            resting += o[2]
            if behind == 0:
                at_target += o[2]
            else:
                keep.append((behind, oid, o[2]))
        if at_target >= self.QTY:
            return
        room = self.MAX_POS - sign * self.inv - resting
        want = self.QTY - at_target
        if room < min(want, self.MIN_NEW) and keep:
            # free room: cancel the deepest kept order
            keep.sort(reverse=True)
            self._cancel(keep[0][1])
            return
        q = min(want, room)
        if q < self.MIN_NEW:
            return
        side = Side.BUY if sign > 0 else Side.SELL
        oid = self.send_order(OrderRequest(side=side, price=target, quantity=q, order_type=OrderType.LIMIT))
        self.orders[oid] = [sign, target, q]

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        oid = event.order_id
        o = self.orders.get(oid)
        if o is None:
            return
        st = event.status
        if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
            self.inv += o[0] * event.filled_quantity
            o[2] -= event.filled_quantity
            if o[2] <= 0:
                self._drop(oid)
        elif st == OrderStatus.CANCELED:
            self._drop(oid)
        elif st == OrderStatus.REJECTED and oid not in self.cancel_sent:
            self._drop(oid)

    def _drop(self, oid: str) -> None:
        self.orders.pop(oid, None)
        self.cancel_sent.discard(oid)
