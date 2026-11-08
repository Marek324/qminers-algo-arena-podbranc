"""Jump-event specialist: snipe the stale side at a jump, work off the inventory into the
post-jump IOC surge, then go idle until the next jump. No baseline market making (lab_kimi).

Round 2 (docs/DATA_ANALYSIS.md, round2_public): the hidden fair price S jumps 20-115
ticks instantly ~9 times an hour (plus many smaller 6-20 tick moves), coin-flip
direction, no warning, no reversion. What the strategy sees (measured on the tape,
findings/lab_kimi/sniper/):

- A jump is revealed ~6-40 ms in by NEW resting limit orders landing at the new level:
  their implied level (ask price - 1 / bid price + 1) sits >= N_TICKS from the current
  mid. In calm markets a resting order essentially never lands that far out (99.99th
  percentile of displacement is 5.9 ticks), so the trigger is nearly false-alarm-free.
- The market's stale orders on the run-over side are cancelled within 5-35 ms and swept
  by the post-jump IOC surge (market IOCs at round(S +- 1): ~47/s for ~1 s, ~19/s for
  ~4 s). New-level "wall" orders (500-3300 lots) re-anchor the book in 50-250 ms.
- Passive maker fills mark out at -0.4 ticks/lot @1 s on this tape (jumps dominate the
  loss), so the strategy never rests quotes without an inventory to work off.

Machine per jump event (all parameters are top-level constants):

1. TRIGGER: a new resting market order (id above every id seen) whose implied level is
   N_TICKS from the mid of the PREVIOUS event (the far order itself drags the post-event
   mid halfway toward itself). A second new order within CONFIRM_DIST of that level
   within CONFIRM_US, or CONFIRM_VOL market lots swept on the run-over side, confirms it
   (kills fat-order false alarms; a lone deep order confirms nothing).
2. SNIPE: cancel everything; if the run-over side still holds stale quotes (its best
   price is STALE_GAP beyond the estimated level), IOC in the jump direction with limit
   est -/+ SNIPE_EDGE, size bounded by the inventory cap minus everything already live
   on that side (resting, pending-cancel and in flight can all still fill).
   A late trigger (jump already completed) fails this gate, so it never chases.
3. REDUCE: once the snipe resolves, rest ONE quote on the side that reduces inventory
   (after an up-jump snipe we are long: an ask at round(est) + HARVEST_OFF), re-centred
   as new orders refine the running level estimate (EMA of new-order implied levels).
   The surge IOCs, priced round(S +- 1), lift the quote with about HARVEST_OFF ticks of
   edge. After HARVEST_US the quote steps out to UNWIND_EDGE and stays until UNWIND_US,
   working off what is left. No quote is ever sent without inventory behind it, and no
   opposite-side quote is ever sent: that is the adverse-selection side of the surge.
4. IDLE: flat or UNWIND_US after the event, residual flattened with an IOC at the touch;
   no orders at all until the next trigger. Opposite-direction triggers within
   LOCK_DIR_US of the last event are ignored (post-jump re-anchoring noise); snipes are
   at most one per SEND_COOLDOWN_US.

Inventory is counted from fills only. Every send is sized so that even if ALL live
orders on one side fill, |inventory| stays within MAX_POS (100): the room calculation
counts resting, pending-cancel and in-flight orders alike. A trigger while quoting
cancels the side it runs over at once (5 ms latency beats the 20-40 ms stale-fill
window about half the time).

Detector + snipe measured offline (findings/lab_kimi/sniper/sniper_sim3.py, marked at
the mean new-order level 2 s after each fill): +33k ticks over the tape with zero
false-alarm cost on 8 calm minutes.
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

# Detector
N_TICKS = 7.0          # trigger: new order implied level this far from the pre-event mid
ASK_OFF = 1.0          # implied level of a new ask: price - ASK_OFF
BID_OFF = 1.0          # implied level of a new bid: price + BID_OFF
CONFIRM_US = 30_000    # confirmation window after the trigger
CONFIRM_DIST = 5.0     # a confirming order lands within this of the trigger level
CONFIRM_VOL = 200      # ... or this many market lots swept on the run-over side ...
CONFIRM_VOL_GAP = 4.0  # ... at prices this far beyond the trigger level
EST_ALPHA = 0.4        # EMA weight of each new near-level order in the level estimate
EST_MAX_DEV = 6.0      # new orders further than this from est don't refine it
LOCK_DIR_US = 2_000_000     # after an event, ignore opposite-direction triggers
SEND_COOLDOWN_US = 1_500_000  # min time between snipe sends
REARM_US = 500_000      # after an event ends, wait before re-triggering while idle

# Snipe
SNIPE_EDGE = 2.0        # IOC limit this many ticks inside the estimated level
SNIPE_SIZE = 100        # max lots per snipe
STALE_GAP = 3.0         # run-over best price must be this far beyond est (0: no gate)

# Reducing quote (harvest the surge, then work off the rest)
HARVEST_US = 3_000_000      # quote tightly for this long after the event
HARVEST_OFF = 1         # quote at round(est) +- this during the harvest
UNWIND_US = 15_000_000     # keep quoting (wider) until this long after the event
UNWIND_EDGE = 1          # quote at round(est) +- this during the unwind
QUOTE_SIZE = 100         # max lots per reducing quote (capped by |inventory| too)
MIN_QTY = 5              # smallest order worth sending
REPLACE_TICKS = 1        # re-centre a resting quote when the target moves this much

# Risk
MAX_POS = 100           # hard cap on |inventory| even if every live order fills

BUY = Side.BUY
SELL = Side.SELL
LIMIT = OrderType.LIMIT
IOC = OrderType.IOC

IDLE = 0
SNIPING = 1
HARVEST = 2
UNWIND = 3

_TERMINAL = (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)
_FILL = (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED)


class LabKimiSniper(BaseStrategy):
    """See the module docstring. State: phase, trigger, level estimate, our orders."""

    def __init__(self) -> None:
        super().__init__()
        self._inventory = 0
        self._state = IDLE
        self._event_ts = 0
        self._event_dir = 0
        self._est = 0.0
        # order_id -> [side, price, remaining, is_ioc, cancelling]
        self._orders: dict[str, list] = {}
        self._side_of: dict[str, Side] = {}
        self._trig: dict | None = None
        self._max_id = ''
        self._last_bids: list | None = None
        self._last_asks: list | None = None
        self._last_mid = 0.0
        self._last_dir = 0
        self._last_event_ts = -10 ** 12
        self._last_send = -10 ** 12
        self._flat_id: str | None = None
        self._snipe_id: str | None = None

    # ---------- order plumbing ----------

    def _send(self, side: Side, price: int, qty: int, order_type: OrderType) -> str:
        order_id = self.send_order(OrderRequest(side=side, price=price, quantity=qty,
                                                order_type=order_type))
        self._orders[order_id] = [side, price, qty, order_type is IOC, False]
        self._side_of[order_id] = side
        return order_id

    def _live_qty(self, side: Side) -> int:
        """Live quantity on `side`: resting, pending-cancel and in flight can all still fill."""
        total = 0
        for order in self._orders.values():
            if order[0] is side and order[2] > 0:
                total += order[2]
        return total

    def _room(self, side: Side) -> int:
        """Lots that can still be added on `side` keeping the worst case within MAX_POS."""
        if side is BUY:
            return MAX_POS - self._inventory - self._live_qty(side)
        return MAX_POS + self._inventory - self._live_qty(side)

    def _cancel_side(self, side: Side) -> None:
        for order_id, order in self._orders.items():
            if order[0] is side and not order[3] and not order[4]:
                self.cancel_order(order_id)
                order[4] = True

    def _cancel_all(self) -> None:
        for order_id, order in self._orders.items():
            if not order[3] and not order[4]:
                self.cancel_order(order_id)
                order[4] = True

    def _manage_quote(self, side: Side, price: int, want: int) -> None:
        """Hold `want` resting lots at `price` on `side`; cancel anything else there."""
        if price <= 0:
            self._cancel_side(side)
            return
        at_price = 0
        stale: list[str] = []
        for order_id, order in self._orders.items():
            if order[0] is not side or order[3] or order[4]:
                continue
            if order[1] == price:
                at_price += order[2]
            elif abs(order[1] - price) >= REPLACE_TICKS:
                stale.append(order_id)
        for order_id in stale:
            self.cancel_order(order_id)
            self._orders[order_id][4] = True
        if want <= 0:
            return
        gap = want - at_price
        if gap >= MIN_QTY:
            qty = min(gap, self._room(side))
            if qty >= MIN_QTY:
                self._send(side, price, qty, LIMIT)

    # ---------- level estimate and detector ----------

    def _scan_new(self, levels: list) -> list[tuple[str, int]]:
        """Return (side, price) of market orders never seen before, best first.

        Order ids only grow, so an id above the highest seen so far is a new insert.
        Only orders visible in the public depth are scanned, like everything the
        strategy knows.
        """
        out = []
        top_id = self._max_id
        for o in levels:
            if o.owner != 'market':
                continue
            oid = o.order_id
            if len(oid) > len(top_id) or (len(oid) == len(top_id) and oid > top_id):
                out.append(('B' if o.side is Side.BUY else 'A', o.price))
                if len(oid) > len(self._max_id) or (len(oid) == len(self._max_id) and oid > self._max_id):
                    self._max_id = oid
        return out

    def _touch(self, levels: list) -> tuple[int | None, int]:
        """Best market price on one side and its quantity, ignoring our own orders."""
        best = None
        qty = 0
        for order in levels:
            if order.owner != 'market':
                continue
            if best is None:
                best = order.price
            elif order.price != best:
                break
            qty += order.quantity
        return best, qty

    def _step_detector(self, new_orders: list[tuple[str, int]], trades: list[Trade],
                       mid: float, ts: int) -> None:
        """Trigger on a far new order; confirm on a second order or a run-over sweep."""
        for side, price in new_orders:
            implied = price - ASK_OFF if side == 'A' else price + BID_OFF
            if self._trig is None:
                if ts - self._last_send < SEND_COOLDOWN_US:
                    continue
                if ts - self._last_event_ts < REARM_US and self._state == IDLE:
                    continue
                if abs(implied - mid) < N_TICKS:
                    continue
                dirn = 1 if implied > mid else -1
                if dirn == -self._last_dir and ts - self._last_event_ts < LOCK_DIR_US:
                    continue
                self._trig = dict(ts=ts, dirn=dirn, est=implied, n_conf=0)
                # a trigger while quoting runs over one side: pull it at once
                if self._state in (HARVEST, UNWIND):
                    self._cancel_side(SELL if dirn > 0 else BUY)
                continue
            dev = self._trig['dirn'] * (implied - self._trig['est'])
            if -CONFIRM_DIST <= dev <= CONFIRM_DIST + EST_MAX_DEV:
                self._trig['est'] += EST_ALPHA * (implied - self._trig['est'])
                self._trig['n_conf'] += 1
        if self._trig is not None and trades:
            run_over = 'ask' if self._trig['dirn'] > 0 else 'bid'
            vol = 0
            for t in trades:
                if t.taker_owner != 'market' or t.maker_owner != 'market':
                    continue
                if t.book_side == run_over and \
                        self._trig['dirn'] * (self._trig['est'] - t.price) >= CONFIRM_VOL_GAP:
                    vol += t.quantity
            if vol >= CONFIRM_VOL:
                self._trig['n_conf'] += 5

    # ---------- event core ----------

    def _on_event(self, ts: int) -> None:
        """A confirmed level move: cancel everything, snipe if there is prey, then reduce."""
        trig = self._trig
        self._trig = None
        self._last_dir = trig['dirn']
        self._last_event_ts = ts
        self._event_ts = ts
        self._event_dir = trig['dirn']
        self._est = trig['est']
        side = BUY if trig['dirn'] > 0 else SELL
        live = self._live_qty(side)
        self._cancel_all()
        self._flat_id = None
        size = min(SNIPE_SIZE, MAX_POS - self._inventory * trig['dirn'] - live)
        sniped = False
        if size > 0:
            if trig['dirn'] > 0:
                ba, _ = self._touch(self._last_asks)
                if ba is not None and ba <= self._est - STALE_GAP:
                    limit = math.floor(self._est - SNIPE_EDGE)
                    self._snipe_id = self._send(BUY, limit, size, IOC)
                    sniped = True
            else:
                bb, _ = self._touch(self._last_bids)
                if bb is not None and bb >= self._est + STALE_GAP:
                    limit = math.ceil(self._est + SNIPE_EDGE)
                    self._snipe_id = self._send(SELL, limit, size, IOC)
                    sniped = True
        if sniped:
            self._state = SNIPING
            self._last_send = ts
        elif self._inventory != 0:
            self._state = HARVEST
        else:
            self._state = IDLE

    def _reduce_quote(self, side: Side, bb: int, ba: int) -> int:
        """Reducing-side quote price: beyond round(est), never crossing the band."""
        base = int(round(self._est))
        off = HARVEST_OFF if self._state == HARVEST else UNWIND_EDGE
        if side is SELL:
            price = base + off
            if bb is not None:
                price = max(price, bb + 1)
            return price
        price = base - off
        if ba is not None:
            price = min(price, ba - 1)
        return price

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        ts = snapshot.timestamp_us
        bb, _ = self._touch(snapshot.bids)
        ba, _ = self._touch(snapshot.asks)
        self._last_bids = snapshot.bids
        self._last_asks = snapshot.asks
        if bb is None or ba is None:
            return
        mid = (bb + ba) / 2
        if self._last_mid <= 0:
            self._last_mid = mid

        new_orders = self._scan_new(snapshot.bids) + self._scan_new(snapshot.asks)

        # refine the level estimate from near-level orders while a phase is active
        if self._state != IDLE:
            for side, price in new_orders:
                implied = price - ASK_OFF if side == 'A' else price + BID_OFF
                if abs(implied - self._est) <= EST_MAX_DEV:
                    self._est += EST_ALPHA * (implied - self._est)

        # the trigger compares a new order's implied level with the mid BEFORE this
        # event: the far order itself drags the post-event mid halfway toward itself
        self._step_detector(new_orders, trades, self._last_mid, ts)
        self._last_mid = mid
        if self._trig is not None:
            if self._trig['n_conf'] >= 1:
                self._on_event(ts)
            elif ts - self._trig['ts'] > CONFIRM_US:
                self._trig = None

        if self._state == HARVEST and ts - self._event_ts > HARVEST_US:
            self._state = UNWIND
            self._cancel_all()
        if self._state == UNWIND and ts - self._event_ts > UNWIND_US:
            self._state = IDLE
            self._cancel_all()

        if self._state in (HARVEST, UNWIND):
            inv = self._inventory
            if inv == 0:
                self._cancel_all()
                self._state = IDLE
            else:
                side = SELL if inv > 0 else BUY
                price = self._reduce_quote(side, bb, ba)
                self._manage_quote(side, price, min(QUOTE_SIZE, abs(inv), self._room(side)))
        elif self._state == IDLE:
            if self._inventory != 0 and self._flat_id is None:
                self._flatten(bb, ba)

    def _flatten(self, bb: int, ba: int) -> None:
        """End of an event: clear any residual inventory with an IOC at the touch."""
        inv = self._inventory
        if inv > 0:
            qty = min(inv, self._room(SELL))
            if qty > 0:
                self._flat_id = self._send(SELL, bb, qty, IOC)
        elif inv < 0:
            qty = min(-inv, self._room(BUY))
            if qty > 0:
                self._flat_id = self._send(BUY, ba, qty, IOC)

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        """Count inventory from fills; drop finished orders; advance the phase machine."""
        status = event.status
        order = self._orders.get(event.order_id)
        if order is None:
            side = self._side_of.get(event.order_id)
            if side is not None and status in _FILL:
                self._inventory += event.filled_quantity if side is BUY else -event.filled_quantity
        else:
            if status in _FILL:
                self._inventory += event.filled_quantity if order[0] is BUY else -event.filled_quantity
                order[2] = event.remaining_quantity
                if status is OrderStatus.FILLED:
                    self._forget(event.order_id)
            elif status is OrderStatus.CANCELED or status is OrderStatus.REJECTED:
                # REJECTED covers both a rejected order and a cancel that lost the race
                self._forget(event.order_id)
        # phase transition: the snipe IOC has resolved (filled, cancelled or rejected)
        if self._snipe_id is not None and event.order_id == self._snipe_id and status in _TERMINAL:
            self._snipe_id = None
            if self._state == SNIPING:
                self._state = HARVEST if self._inventory != 0 else IDLE

    def _forget(self, order_id: str) -> None:
        self._orders.pop(order_id, None)
        if self._flat_id == order_id:
            self._flat_id = None
