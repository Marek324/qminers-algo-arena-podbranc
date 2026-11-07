"""Round 2 take-and-unwind: strategies/adaptive_mm_vol.py with passive quotes that mostly close inventory.

Why: a full-logs run of adaptive_mm_vol.py on data/round2_public.csv (pnl 131,725) split by fill type, marked out
against later market trade prices (ticks per lot):

| our fills | lots | 1 s | 5 s | 30 s |
|---|---|---|---|---|
| crossing LIMIT (fair EDGE_TAKE through the opposite best) | 257k | +0.75 | +0.87 | +0.57 |
| passive, closing inventory | 223k | -0.22 | -0.35 | +0.21 |
| passive, opening inventory (incl. from flat) | 76k | -0.51 | -0.13 | -0.37 |
| taker-leg IOC | 8k | -0.72 | -2.90 | +1.33 |

The crossing orders earn the money; passive fills that open a position and the IOC taker leg lose it. Passive quotes
of 200 lots also flipped inventory between +100 and -100 in one fill (|inventory| >= 90 for 56% of the time), which
left no room for the profitable crosses.

Changes from adaptive_mm_vol.py:
- TAKE_ENABLED = False: the OFI/IOC taker leg is off.
- Passive size per side = lots that close inventory + PASSIVE_OPEN (opening lots, scaled by the maker safety
  switch), capped by QUOTE_SIZE. PASSIVE_OPEN = 0 quotes passively only to unwind; PASSIVE_OPEN = QUOTE_SIZE is the
  old behaviour. Crossing orders still use the full QUOTE_SIZE, capped by the risk room.
- TRIM: resting passive size above that target (e.g. after a cross reduced inventory) is shrunk or cancelled.
- K_INV_QUOTE: inventory skew for passive prices only, so a stronger skew unwinds faster without making the
  crossing decision trade on inventory. K_INV still skews the fair used for crossing (default = old behaviour).
- An order smaller than MIN_NEW is still sent when it is the whole closing size (no tiny leftover positions).

- Round 2 tuning (single tape, so differences under ~15k are noise; pnl_se is about 16k):
  | change | pnl |
  |---|---|
  | adaptive_mm_vol.py | 131,725 |
  | these changes, old settings | 174,377 |
  | EDGE_TAKE 0.25 -> 0.12 | 209,944 |
  | S_ALPHA 0.1 -> 0.25 (faster hidden-price estimate), EDGE_TAKE 0.13 | 231,426 |
  | KF: Kalman filter for the hidden price (tools/kalman_price.py), EDGE_TAKE 0.2, K_S 0.65 | 235,976 |
  The Kalman estimate predicts the mid clearly better (first 1,100 s: R^2 of the 0.1 s move 0.107 vs 0.071 for the
  running averages, 0.5 s: 0.090 vs 0.062), but PnL rose only about 5k, within noise. Its jump inflation is what
  matters (KF_JUMP 0: 206k). |inventory| median is now 80.
  Freeing that capacity did not pay: crosses sized by signal strength (CROSS_FULL) and cheaper exit crosses
  (EXIT_MULT < 1) both lost 3k-43k, as did a stronger crossing skew (K_INV). Positions are worth holding: the
  crosses still mark out well at 30 s.
  Tried and left off: K_INV_QUOTE 0.004+ (0.008: 55k, unwinding fast is costly), K_FLOW 0.5/1.2 (worse),
  EDGE 0.6/1.0 (same), CROSS_DEEP (same), GUARD (same or worse).

Original docstring of adaptive_mm_vol.py:

Volatility-adaptive market maker: strategies/adaptive_mm_plus.py with its margins scaled by live volatility,
plus a maker-side safety switch. Built for round 2, which moved about 4x faster than round 1.

Why: round 2 (data/round2_public.csv) has mid volatility 3.8 ticks/sqrt(s) against 0.92 in round 1, and large IOCs push
the price about twice as hard (+1.27 ticks 1 s after a 183+ lot order, against +0.60). adaptive_mm_plus kept round-1
margins (0.4 ticks) and lost 59k there; with round-2-sized margins and a stronger flow term (EDGE 2.2, K_FLOW 2.0,
FLOW_TAU 300 ms, hand-set) it made +90k.

Changes from adaptive_mm_plus.py:
- Volatility scaling (VOL_SCALE): RealizedVol measures the mid's volatility online (1 s samples, about 5 min memory,
  fast warm-up). scale = clamp(vol / VOL_REF, VOL_SCALE_MIN, VOL_SCALE_MAX) with VOL_REF = round 1's 0.92, so round-1-like
  markets get exactly the old settings. Then EDGE, EDGE_IN, KEEP_EDGE grow as scale**EDGE_VOL_POW, EDGE_TAKE as
  scale**TAKE_VOL_POW, K_FLOW as scale**FLOW_VOL_POW, FLOW_TAU_US as scale**FLOW_TAU_VOL_POW, K_INV as scale**INV_VOL_POW
  and S_LEVELS as scale**S_LEVELS_VOL_POW. The edge, flow and flow-memory powers come from the round 2 sweep; the
  others are untested first guesses.
- Maker safety switch (MAKER_CHECK): FillMarkout scores every limit-order fill against the mid MK_HORIZON_US later.
  When that running markout falls from MK_FULL to MK_ZERO ticks/lot (both times the volatility scale), the quote size
  shrinks linearly to MK_FLOOR of QUOTE_SIZE. On round 2 the market's passive fills lose -0.40 ticks/lot after 1 s.

Untested as a whole: run the commands at the end of this docstring before trusting it.

Original docstring of adaptive_mm_plus.py:

Adaptive market maker, upgraded: strategies/adaptive_mm.py (David's) plus room-freeing by shrinking.

Changes from adaptive_mm.py (everything else is identical; see its docstring below):
- SHRINK: when the target order needs room, the deepest kept order is shrunk with modify_order by the lots needed
  (it keeps its queue place) instead of being cancelled. A shrink that loses the race against a fill comes back
  REJECTED; the order is then still live and stays tracked.
- EDGE 0.4 instead of 0.3 (more robust across sessions; see the EDGE comment).
- K_INV 0.0015 instead of 0.002: the only setting change that helped both on the real round 1 and on fresh
  synthetic round-1 sessions (QUOTE_SIZE 150 and EDGE_TAKE 0.31 helped only on synthetic data).

Results (validation = 30 synthetic round-1 sessions never used for choosing; per hour):
| | real round 1 | synthetic validation | better than adaptive_mm in |
|---|---|---|---|
| adaptive_mm.py | 662,359 | 192.1k/h | |
| SHRINK only | 660,741 | 196.0k/h | 97% of sessions |
| SHRINK + K_INV 0.0015 (this file) | 678,057 | 206.2k/h | 100% of sessions |

Original docstring of adaptive_mm.py:

Adaptive market maker.

Maker leg (the money on round 1): quotes passively around a fair price.
- Fair = mid, moved K_S of the way toward a hidden-price estimate, plus small terms for touch
  imbalance, recent aggressive volume and inventory. The tape's limit orders are placed around a
  hidden price that random-walks, and the visible mid lags it by about 0.5 s; the estimate is the
  average of running means of newly arrived limit-order prices on each side. (Fair-value design
  and hidden-price estimate from strategies/imbalance_mm.py.)
- Quote prices: the best price that keeps at least EDGE ticks from fair; cross the spread when fair
  is EDGE_TAKE beyond the opposite best price; inside a 2+ tick spread only with EDGE_IN of edge.
- Orders are as large as the position cap allows. An order the target moves a tick away from is
  kept rather than re-sent, and the deepest kept order is cancelled when room is needed. An order
  the target moves *behind* is also kept while it has KEEP_EDGE (0.25) of edge left, so a small
  dip in fair does not throw away its queue place (joining still needs EDGE, 0.3).

Taker leg (safety, idle on round 1): holds a directional position when order flow imbalance (OFI,
Cont, Kukanov & Stoikov) predicts a move, but only while a paper-traded 1-lot shadow copy of it is
making money, so on tapes without momentum it stays off.

Risk: every order is sized so that even if all our open orders on one side fill, |inventory| stays
within MAX_POS (the exchange penalty starts above 100). Inventory is counted from fills only, so a
cancel that loses the race against a fill cannot put it out of sync. Our own orders are ignored
when reading the book, so no signal reacts to us.

Hidden-price check: on tapes where orders are not placed around a hidden price, the estimate
misleads the quotes (control tapes: about -84k per hour). So its weight is scaled by how well it
has predicted the mid 0.5 s ahead over roughly the last 5 minutes, and drops to 0 when it hasn't.
That cut the control-tape losses to -6k ... -24k with no cost on round 1.

Run:  uv run exchange-simulator --strategy strategies/adaptive_mm_r2.py --backtest data/round2_public.csv --log-dir logs/r2 --no-logs
      (and the same with data/round1.csv: round 1 must stay at about 673k)
Tune: python3 -I scripts/sweep.py strategies/adaptive_mm_r2.py data/round2_public.csv -p PASSIVE_OPEN=0,25,50,200
Experiment history: docs/RESEARCH.md.
"""
from __future__ import annotations

import math
from collections import deque

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

# Forecast
OFI_WEIGHT = 0.25  # forecast ticks per average touch depth of OFI
OFI_TAU_US = 1_000_000  # OFI decay time constant
DEPTH_EMA = 0.01  # per-event weight of the average touch size

# Taker
TAKE_ENABLED = False  # its IOCs lost -2.9 ticks/lot at 5 s on round 2
TAKE_THRESHOLD = 1.0  # |forecast| in ticks that sets a direction
EXIT_THRESHOLD = 0.5  # |forecast| in ticks below which the direction is flat
TAKE_POS = 50  # lots held at full size
TAKE_SIZE = 50  # lots per IOC

# Momentum check
EDGE_CHECK = True
EDGE_TAU_US = 300_000_000  # decay of the shadow's PnL (about the last 5 minutes)
EDGE_FULL = 60.0  # shadow PnL (ticks per lot) needed for full TAKE_POS
LATENCY_US = 5_000  # exchange latency, applied to the shadow

# Fair value (maker)
K_S = 0.65  # round 2 with KF: 0.6 234k, 0.75 232k, 1.0 220k; fair moves this share of the way from the mid to the hidden-price estimate
S_ALPHA = 0.25  # round 2 sweep: 0.05 208k, 0.1 210k, 0.2 226k, 0.3 227k, 0.45 215k; weight of each newly arrived limit order in its side's running price average
S_LEVELS = 2  # new orders count up to this many ticks behind the best price
K_IMB = 0.15  # ticks of fair shift at full touch imbalance
K_FLOW = 0.8  # ticks of fair shift per average touch size of recent aggressive volume
FLOW_TAU_US = 100_000  # decay of the aggressive volume
FLOW_DEPTH_EMA = 0.001  # per-event weight of the touch size that normalises the flow
K_INV = 0.0015  # round 2 (crossing skew): 0 229k, 0.004 164k; ticks of fair shift per lot of inventory away from target; adaptive_mm.py uses 0.002

# Hidden-price check: K_S is scaled by how well the estimate has recently predicted the mid
# Kalman filter for the hidden price (replaces the two running averages of S_ALPHA when KF is on)
KF = True
KF_Q = 0.15  # round 2: 0.05 230k, 0.15 230k, 0.4 231k; diffusion of S, ticks^2 per second, x vol scale**2 (scale about 4 on round 2 -> about 2.5)
KF_JUMP = 1e-3  # round 2: 0 206k, 2e-4 230k, 1e-3 232k, 3e-3 226k, 1e-2 219k; variance added per aggressive market order: KF_JUMP * lots**1.3 (about (0.014 * lots**0.65)^2)
KF_PUSH = 0.0  # round 2: 0.005 232k, 0.015 226k; mean push of S per aggressive order: KF_PUSH * lots**0.65 (round 1: double-counted, so off)
KF_R0 = 0.3  # starting observation noise, ticks^2 (learned online)
KF_ALPHA = 0.01  # learning rate of the observation noise and of the order offset from S
KF_CENSOR = True  # round 2: off 228k; a new order one tick from the opposite best is only a bound on S
KF_BOUND = True  # a market buy that traded at P proves S >= P - 1.5 (IOCs are priced at round(S +- 1))

HP_CHECK = True
HP_HORIZON_US = 500_000  # the mid move the estimate is checked against
HP_SAMPLE_US = 50_000  # one check sample per this much time
HP_TAU_US = 300_000_000  # decay of the check's statistics (about the last 5 minutes)
HP_CORR_FULL = 0.3  # correlation at which the full K_S is used; 0 or less turns the term off
HP_MIN_WEIGHT = 100.0  # samples needed before the term is used at all

# Maker
MAKE_ENABLED = True
QUOTE_SIZE = 200  # lots wanted at the target price (capped by the risk room); 150 won only on synthetic data
EDGE = 0.4  # min ticks between a passive quote and fair (0.3 was 5k better on public round 1 but 0.4 wins on every
# synthetic twin and stress session: +19k to +20k per twin, +3.7% across 8 harder markets)
EDGE_IN = 0.5  # min ticks of edge to quote inside a 2+ tick spread
EDGE_TAKE = 0.2  # with KF on: 0.17 229k, 0.22 230k, 0.28 219k; with KF off (EMA) 0.13 is best: cross (bid at the best ask) when fair is this far through it (x vol scale); round 2 sweep at S_ALPHA 0.25: 0.09 218k, 0.12 229k, 0.15 229k (0.25 was 174k at S_ALPHA 0.1)
LADDER_LEVELS = 2  # an order the target moves away from is kept up to LADDER_LEVELS - 1 ticks behind it
MIN_NEW = 10  # smallest new order worth sending
KEEP_EDGE = 0.25  # an existing order priced better than the target is kept while it has this much edge

# Upgrade
SHRINK = True  # free room for the target order by shrinking the deepest kept order instead of cancelling it

# Round 2: passive quotes mostly unwind, crosses take the position
PASSIVE_OPEN = 0  # passive lots per side beyond those that close inventory; QUOTE_SIZE = adaptive_mm_vol behaviour
TRIM = True  # shrink or cancel resting passive size above the side's target size
# Position capacity: spend the +-MAX_POS on the strongest signals and give it back once they are gone
CROSS_FULL = 1.0  # round 2: 1.5 233k, 2 224k, 3 208k (1.0: 236k); a cross buys up to a position of MAX_POS * min(1, edge / (EDGE_TAKE * CROSS_FULL)); 1.0 = always full
EXIT_MULT = 1.0  # round 2: 0 193k, 0.4 218k, 0.7 224k, 1.5 233k, 2.5 204k (1.0: 236k); crossing that only reduces |inventory| needs EDGE_TAKE * EXIT_MULT (< 1 exits sooner); 1.0 = off
GUARD = False  # (round 2: off 231k, 300 lots/300 ms 229k, 100 lots/1 s 216k, 1000 lots/1 s 232k) after a big market sweep, pull passive quotes on the side it is heading to (crossing still allowed)
GUARD_LOTS = 300  # market aggressive lots on one side within one event that trigger the guard
GUARD_US = 300_000  # how long the guard lasts
CROSS_DEEP = False  # (round 2: 208k on vs 210k off) cross up to the deepest price still EDGE_TAKE inside fair, not only the best opposite price
K_INV_QUOTE = 0.0015  # ticks of passive-price shift per lot of inventory (x vol scale**INV_VOL_POW); K_INV for crossing

# Volatility scaling: margins and the flow response follow the live volatility of the mid
VOL_SCALE = True
VOL_REF = 0.92  # round 1's mid volatility, ticks per sqrt(second); scale 1 keeps the round-1 settings exactly
VOL_SCALE_MIN = 1.0  # never smaller margins than round 1's
VOL_SCALE_MAX = 8.0
EDGE_VOL_POW = 1.2  # EDGE, EDGE_IN, KEEP_EDGE x scale**1.2 (round 2 sweep: edge 0.4 -> about 2.2 at scale 4.1)
TAKE_VOL_POW = 1.0  # EDGE_TAKE x scale**1.0 (guess: same selectivity relative to the size of moves)
FLOW_VOL_POW = 0.65  # K_FLOW x scale**0.65 (round 2 sweep: 0.8 -> 2.0)
FLOW_TAU_VOL_POW = 0.78  # FLOW_TAU_US x scale**0.78 (round 2 sweep: 100 -> 300 ms)
INV_VOL_POW = 1.0  # K_INV x scale**1.0 (guess: inventory keeps its weight against the larger margins)
S_LEVELS_VOL_POW = 1.0  # S_LEVELS x scale (guess: in faster markets new orders land further from the touch)

# Maker safety switch: shrink quote size while our own limit-order fills have been losing
MAKER_CHECK = True
MK_HORIZON_US = 1_000_000  # each fill is scored against the mid this long afterwards
MK_FULL = 0.0  # running markout (ticks/lot, x volatility scale) from which the full QUOTE_SIZE is used
MK_ZERO = -0.3  # running markout at which the size is down to MK_FLOOR
MK_FLOOR = 0.25  # smallest share of QUOTE_SIZE, so the switch keeps measuring

# Risk
MAX_POS = 100  # worst case |inventory| if every open order on one side fills (penalty only above 100)

BUY = Side.BUY
SELL = Side.SELL
LIMIT = OrderType.LIMIT
IOC = OrderType.IOC


def _touch(levels: list) -> tuple[int | None, int]:
    """Best market price on one side and the market quantity at it, ignoring our own orders."""
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


class RealizedVol:
    """Online volatility in ticks per sqrt(second): exponentially weighted variance of price changes sampled about
    once per `sample_us` (copy of tools/volatility.py)."""

    def __init__(self, sample_us: int = 1_000_000, n_eff: float = 300.0, min_samples: int = 5) -> None:
        self.sample_us = sample_us
        self.n_eff = n_eff
        self.min_samples = min_samples
        self.var = 0.0
        self.n = 0
        self.last_ts = -1
        self.last_price = 0.0

    def update(self, ts_us: int, price: float) -> bool:
        """Feed one observation; True when a new sample was taken."""
        if self.last_ts < 0:
            self.last_ts = ts_us
            self.last_price = price
            return False
        if ts_us - self.last_ts < self.sample_us:
            return False
        dt = (ts_us - self.last_ts) / 1_000_000
        r2 = (price - self.last_price) ** 2 / dt
        self.n += 1
        w = max(1.0 / self.n_eff, 1.0 / self.n)
        self.var += w * (r2 - self.var)
        self.last_ts = ts_us
        self.last_price = price
        return True

    def vol(self) -> float | None:
        if self.n < self.min_samples:
            return None
        return math.sqrt(self.var)


class HiddenPriceKalman:
    """1-D Kalman filter for the hidden price S from new limit-order prices (copy of tools/kalman_price.py; see
    its module docstring for the model)."""

    def __init__(self, q: float = 2.0, jump: float = 2e-4, jump_pow: float = 1.3, r0: float = 0.3,
                 r_alpha: float = 0.01, r_min: float = 0.05, m0: float = 1.0, m_alpha: float = 0.01,
                 push: float = 0.0, push_pow: float = 0.65) -> None:
        self.q = q  # ticks^2 per second
        self.jump = jump
        self.jump_pow = jump_pow
        self.r = r0
        self.r_alpha = r_alpha
        self.r_min = r_min
        self.m = m0
        self.m_alpha = m_alpha
        self.push = push
        self.push_pow = push_pow
        self.s = None
        self.p = 1.0
        self.ts = 0

    def _predict(self, ts_us: int) -> None:
        if ts_us > self.ts:
            self.p += self.q * (ts_us - self.ts) * 1e-6
            self.ts = ts_us

    def trade(self, ts_us: int, sign: int, qty: float) -> None:
        """Aggressive market volume `qty` in one event; sign +1 buyers lifted the ask, -1 sellers hit the bid."""
        if qty <= 0:
            return
        self._predict(ts_us)
        self.p += self.jump * qty ** self.jump_pow
        if self.push and self.s is not None:
            self.s += sign * self.push * qty ** self.push_pow

    def observe(self, ts_us: int, price: float, sign: int, capped: bool = False) -> None:
        """A new limit order at `price`; sign +1 buy, -1 sell. `capped`: it sits at the most aggressive price
        the generator allows (one tick from the opposite best), so it is only a bound."""
        if self.s is None:
            self.s = price + sign * self.m
            self.ts = ts_us
            return
        self._predict(ts_us)
        innov = price + sign * self.m - self.s
        if capped and innov * sign < 0:
            return  # consistent with the bound, nothing learned
        p = self.p
        self.m += self.m_alpha * (sign * (self.s - price) - self.m)
        self.r += self.r_alpha * (innov * innov - p - self.r)
        if self.r < self.r_min:
            self.r = self.r_min
        k = p / (p + self.r)
        self.s += k * innov
        self.p = (1.0 - k) * p

    def bound_low(self, lo: float) -> None:
        if self.s is not None and self.s < lo:
            self.s = lo

    def bound_high(self, hi: float) -> None:
        if self.s is not None and self.s > hi:
            self.s = hi


class FillMarkout:
    """Running markout per lot, in ticks, of our fills `horizon_us` after each fill (copy of tools/markout.py)."""

    def __init__(self, horizon_us: int = 1_000_000, n_eff_lots: float = 20_000.0, min_lots: float = 500.0) -> None:
        self.horizon_us = horizon_us
        self.n_eff_lots = n_eff_lots
        self.min_lots = min_lots
        self.pending: deque = deque()
        self.avg = 0.0
        self.lots = 0.0

    def add_fill(self, ts_us: int, sign: int, price: float, qty: int) -> None:
        self.pending.append((ts_us, sign, price, qty))

    def update(self, ts_us: int, mid: float) -> None:
        pending = self.pending
        while pending and pending[0][0] + self.horizon_us <= ts_us:
            _, sign, price, qty = pending.popleft()
            self.lots += qty
            w = min(1.0, qty / min(self.lots, self.n_eff_lots))
            self.avg += w * (sign * (mid - price) - self.avg)

    def value(self) -> float | None:
        if self.lots < self.min_lots:
            return None
        return self.avg


class AdaptiveMarketMakerR2(BaseStrategy):
    """See the module docstring. State: inventory and target position, our open orders, the
    decayed signals (OFI, aggressive flow, average touch size) and the shadow taker."""

    def __init__(self) -> None:
        super().__init__()
        self._inventory = 0
        self._dir = 0
        self._target = 0
        # order_id -> [side, price, remaining_quantity, is_ioc, cancelling, shrinking]
        self._orders: dict[str, list] = {}
        self._taker_id: str | None = None

        self._prev: tuple[int, int, int, int] | None = None
        self._ofi = 0.0
        self._avg_depth = 0.0
        self._last_ts = 0

        self._flow = 0.0
        self._flow_depth = 0.0
        self._flow_ts = 0
        self._s_bid = 0.0
        self._s_ask = 0.0
        self._max_id = ''
        self._last_bids: list | None = None
        self._last_asks: list | None = None
        self._hp_pending: deque = deque()  # (ts, delta, mid) samples waiting for their horizon
        self._hp_next = 0
        self._hp_ts = 0
        self._hp_sxy = 0.0
        self._hp_sxx = 0.0
        self._hp_syy = 0.0
        self._hp_w = 0.0

        self._vol = RealizedVol()
        self._mk = FillMarkout(MK_HORIZON_US)
        self._kf = HiddenPriceKalman(q=KF_Q, jump=KF_JUMP, r0=KF_R0, r_alpha=KF_ALPHA, m_alpha=KF_ALPHA, push=KF_PUSH)
        self._scale = 1.0
        self._edge_eff = EDGE
        self._edge_in_eff = EDGE_IN
        self._keep_eff = KEEP_EDGE
        self._take_eff = EDGE_TAKE
        self._k_flow_eff = K_FLOW
        self._flow_tau_eff = float(FLOW_TAU_US)
        self._k_inv_eff = K_INV
        self._s_levels_eff = S_LEVELS
        self._open_size = PASSIVE_OPEN
        self._k_inv_q_eff = K_INV_QUOTE
        self._guard_bid_until = 0  # no passive bids before this time (a big sell sweep)
        self._guard_ask_until = 0  # no passive asks before this time (a big buy sweep)

        self._sh_pos = 0
        self._sh_cash = 0.0
        self._sh_equity = 0.0
        self._sh_pending: tuple[int, int, int] | None = None  # (ready_ts, +1 buy / -1 sell, limit price)
        self._edge = 0.0

    def _update_signals(self, ts: int, bid: int, bid_qty: int, ask: int, ask_qty: int) -> None:
        """Decay the signals to `ts`, then add this event.

        OFI scores the change at the touch since the previous event: a bid queue that grows (or a best
        bid that moves up) is buying pressure, a bid queue that shrinks (or a best bid that moves down)
        is selling pressure, and mirror-wise for the ask.
        """
        dt = ts - self._last_ts
        self._last_ts = ts
        if dt > 0:
            self._ofi *= math.exp(-dt / OFI_TAU_US)
            self._edge *= math.exp(-dt / EDGE_TAU_US)

        if self._prev is not None:
            pb, pbq, pa, paq = self._prev
            e = 0
            if bid >= pb:
                e += bid_qty
            if bid <= pb:
                e -= pbq
            if ask <= pa:
                e -= ask_qty
            if ask >= pa:
                e += paq
            self._ofi += e
        self._prev = (bid, bid_qty, ask, ask_qty)

        depth = (bid_qty + ask_qty) / 2
        self._avg_depth = depth if self._avg_depth == 0.0 else self._avg_depth + DEPTH_EMA * (depth - self._avg_depth)

    def _forecast(self) -> float:
        """Expected mid move in ticks: OFI divided by the average touch size, so it is scale-free."""
        depth = self._avg_depth if self._avg_depth > 0 else 1.0
        return OFI_WEIGHT * self._ofi / depth

    def _room(self, side: Side) -> int:
        """Lots that can still be added on `side` while keeping the worst case within MAX_POS."""
        open_qty = 0
        for order in self._orders.values():
            if order[0] is side:
                open_qty += order[2]
        if side is BUY:
            return MAX_POS - (self._inventory + open_qty)
        return MAX_POS + (self._inventory - open_qty)

    def _send(self, side: Side, price: int, qty: int, order_type: OrderType) -> str:
        order_id = self.send_order(OrderRequest(side=side, price=price, quantity=qty, order_type=order_type))
        self._orders[order_id] = [side, price, qty, order_type is IOC, False, False]
        return order_id

    def _shadow_fill(self, ts: int) -> None:
        """Fill the shadow's pending IOC against the book it would have met on arrival (the previous touch)."""
        pending = self._sh_pending
        if pending is None or ts < pending[0] or self._prev is None:
            return
        self._sh_pending = None
        pb, _, pa, _ = self._prev
        if pending[1] > 0 and pa <= pending[2]:
            self._sh_pos += 1
            self._sh_cash -= pa
        elif pending[1] < 0 and pb >= pending[2]:
            self._sh_pos -= 1
            self._sh_cash += pb

    def _shadow_step(self, ts: int, bid: int, ask: int) -> None:
        """Paper-trade the 1-lot shadow toward the taker's direction and add its PnL change to `edge`.

        The shadow follows the real rules: its IOC lands LATENCY_US later and misses if the book moved
        away. It buys at the ask, sells at the bid and also unwinds by taking, so `edge` (its PnL marked
        at the touch, decayed over EDGE_TAU_US) is a conservative measure of the taker's momentum edge.
        """
        if self._sh_pending is None and self._sh_pos != self._dir:
            if self._dir > self._sh_pos:
                self._sh_pending = (ts + LATENCY_US, 1, ask)
            else:
                self._sh_pending = (ts + LATENCY_US, -1, bid)
        pos = self._sh_pos
        equity = self._sh_cash + (pos * bid if pos > 0 else pos * ask)
        self._edge += equity - self._sh_equity
        self._sh_equity = equity

    def _size(self) -> int:
        """Taker size: TAKE_POS * clamp(edge / EDGE_FULL, 0, 1). Zero while the shadow is losing."""
        if not EDGE_CHECK:
            return TAKE_POS
        scale = self._edge / EDGE_FULL
        if scale <= 0:
            return 0
        return round(TAKE_POS * min(1.0, scale))

    def _run_taker(self, forecast: float, bid: int, ask: int) -> None:
        """Set the direction and target position, and send an IOC toward the target if needed.

        The direction turns long or short beyond +-TAKE_THRESHOLD, goes flat inside +-EXIT_THRESHOLD and is
        otherwise kept. With size 0 nothing is sent, not even to flatten: that would only pay the spread
        to unwind maker inventory. With one IOC in flight at most, it buys at the ask or sells at the
        bid, and only in the direction the forecast points; unwinding against it is left to the quotes.
        """
        if forecast > TAKE_THRESHOLD:
            self._dir = 1
        elif forecast < -TAKE_THRESHOLD:
            self._dir = -1
        elif abs(forecast) < EXIT_THRESHOLD:
            self._dir = 0
        size = self._size()
        self._target = self._dir * size
        if size == 0 or self._taker_id is not None:
            return

        gap = self._target - self._inventory
        if gap > 0 and forecast > 0:
            side, price = BUY, ask
        elif gap < 0 and forecast < 0:
            side, price = SELL, bid
        else:
            return
        qty = min(abs(gap), TAKE_SIZE, self._room(side))
        if qty > 0:
            self._taker_id = self._send(side, price, qty, IOC)

    def _scan_new(self, views: list, sign: int, ts: int = 0, cap: int = 0) -> None:
        """Fold market orders that arrived since the last scan into their side's running price average
        (and into the Kalman filter when KF is on).

        Order ids only grow, so a market order id above the highest one seen so far is new. Only orders
        within S_LEVELS ticks of the best price count (sign: -1 bids, +1 asks). `cap` is the most aggressive
        price a new order may have (one tick from the opposite best); an order there is only a bound on S.
        """
        first = views[0].price
        top_id = self._max_id
        for o in views:
            if (o.price - first) * sign > self._s_levels_eff:
                break
            if o.owner != 'market':
                continue
            oid = o.order_id
            if len(oid) > len(top_id) or (len(oid) == len(top_id) and oid > top_id):
                if KF:
                    self._kf.observe(ts, o.price, -sign, KF_CENSOR and o.price == cap)
                if sign < 0:
                    self._s_bid = o.price if not self._s_bid else self._s_bid + S_ALPHA * (o.price - self._s_bid)
                else:
                    self._s_ask = o.price if not self._s_ask else self._s_ask + S_ALPHA * (o.price - self._s_ask)
                if len(oid) > len(self._max_id) or (len(oid) == len(self._max_id) and oid > self._max_id):
                    self._max_id = oid

    def _fair(self, snapshot: OrderBookSnapshot, trades: list[Trade], bid: int, bid_qty: int, ask: int, ask_qty: int) -> float:
        """Fair price: the mid, moved toward the hidden price and by imbalance, aggressive flow and inventory.

        The tape's limit orders are placed around a hidden price S that random-walks; the visible mid
        lags it by about 0.5 s. S is estimated as the average of the bid-side and ask-side running means
        of newly arrived limit-order prices. Aggressive volume (buys +, sells -), fading over FLOW_TAU_US,
        keeps moving the price for a fraction of a second after big IOCs.
        """
        mid = (bid + ask) / 2
        fair = mid + K_IMB * (bid_qty - ask_qty) / (bid_qty + ask_qty) - self._k_inv_eff * (self._inventory - self._target)
        if K_S:
            ts = snapshot.timestamp_us
            hi_buy = 0
            lo_sell = 0
            if KF and trades:
                buy_vol = 0
                sell_vol = 0
                for t in trades:
                    if t.taker_owner == 'market':
                        if t.book_side == 'ask':
                            buy_vol += t.quantity
                            if t.price > hi_buy:
                                hi_buy = t.price
                        else:
                            sell_vol += t.quantity
                            if not lo_sell or t.price < lo_sell:
                                lo_sell = t.price
                self._kf.trade(ts, 1, buy_vol)
                self._kf.trade(ts, -1, sell_vol)
            if snapshot.bids is not self._last_bids:
                self._last_bids = snapshot.bids
                self._scan_new(snapshot.bids, -1, ts, ask - 1)
            if snapshot.asks is not self._last_asks:
                self._last_asks = snapshot.asks
                self._scan_new(snapshot.asks, 1, ts, bid + 1)
            if KF:
                if KF_BOUND and hi_buy:
                    self._kf.bound_low(hi_buy - 1.5)
                if KF_BOUND and lo_sell:
                    self._kf.bound_high(lo_sell + 1.5)
                est = self._kf.s
            else:
                est = (self._s_bid + self._s_ask) / 2 if self._s_bid and self._s_ask else None
            if est is not None:
                delta = est - mid
                fair += K_S * self._hp_scale(ts, delta, mid) * delta
        if K_FLOW:
            ts = snapshot.timestamp_us
            self._flow *= math.exp((self._flow_ts - ts) / self._flow_tau_eff)
            self._flow_ts = ts
            for t in trades:
                if t.taker_owner == 'market':
                    self._flow += t.quantity if t.book_side == 'ask' else -t.quantity
            half = (bid_qty + ask_qty) / 2
            self._flow_depth = half if self._flow_depth == 0 else self._flow_depth + FLOW_DEPTH_EMA * (half - self._flow_depth)
            fair += self._k_flow_eff * self._flow / self._flow_depth
        return fair

    def _hp_scale(self, ts: int, delta: float, mid: float) -> float:
        """Share of K_S to use, from how well delta = estimate - mid has predicted the mid recently.

        Every HP_SAMPLE_US a (delta, mid) sample is stored; once HP_HORIZON_US has passed, it is paired
        with the mid move since then. The correlation of delta with that move, over decayed sums
        (about the last HP_TAU_US), sets the scale: 0 while it is <= 0 or there are too few samples,
        1 from HP_CORR_FULL up. On tapes where orders are not placed around a hidden price, the
        estimate does not predict the mid and the term switches itself off.
        """
        if not HP_CHECK:
            return 1.0
        if ts > self._hp_ts:
            f = math.exp((self._hp_ts - ts) / HP_TAU_US)
            self._hp_sxy *= f
            self._hp_sxx *= f
            self._hp_syy *= f
            self._hp_w *= f
            self._hp_ts = ts
        pending = self._hp_pending
        while pending and pending[0][0] + HP_HORIZON_US <= ts:
            _, d0, m0 = pending.popleft()
            y = mid - m0
            self._hp_sxy += d0 * y
            self._hp_sxx += d0 * d0
            self._hp_syy += y * y
            self._hp_w += 1.0
        if ts >= self._hp_next:
            pending.append((ts, delta, mid))
            self._hp_next = ts + HP_SAMPLE_US
        if self._hp_w < HP_MIN_WEIGHT or self._hp_sxy <= 0 or self._hp_syy <= 0:
            return 0.0
        corr = self._hp_sxy / math.sqrt(self._hp_sxx * self._hp_syy)
        return min(1.0, corr / HP_CORR_FULL)

    def _targets(self, fair: float, fair_q: float, bid: int, ask: int, take_b: float, take_a: float) -> tuple[int, int]:
        """Target price per side: `fair` decides crossing, `fair_q` (fair with the passive inventory skew)
        places passive quotes.

        Bid: the best ask if fair is EDGE_TAKE or more above it (cross; the quotes unwind it later);
        else one tick inside a 2+ tick spread if that still leaves EDGE_IN from fair_q; otherwise the
        highest price at or below the best bid that is at least EDGE below fair_q. The ask mirrors it.
        """
        edge, edge_in = self._edge_eff, self._edge_in_eff
        if fair - ask >= take_b:
            pb = max(ask, math.floor(fair - take_b)) if CROSS_DEEP else ask
        elif ask - bid >= 2 and fair_q - (bid + 1) >= edge_in:
            pb = bid + 1
        else:
            pb = bid
            while fair_q - pb < edge:
                pb -= 1
        if bid - fair >= take_a:
            pa = min(bid, math.ceil(fair + take_a)) if CROSS_DEEP else bid
        elif ask - bid >= 2 and (ask - 1) - fair_q >= edge_in:
            pa = ask - 1
        else:
            pa = ask
            while pa - fair_q < edge:
                pa += 1
        return pb, pa

    def _cross_size(self, x: float, take: float, pos: int) -> int:
        """Lots for a cross whose fair is `x` ticks through the opposite best, with `pos` = inventory signed in the
        cross direction. Below the full threshold (an EXIT_MULT exit) it only flattens; otherwise it goes up to a
        position of MAX_POS * min(1, x / (take * CROSS_FULL))."""
        if x < take:
            return max(0, -pos)
        if CROSS_FULL <= 1.0:
            return QUOTE_SIZE
        frac = min(1.0, x / (take * CROSS_FULL))
        return max(0, round(MAX_POS * frac) - pos)

    def _side_size(self, side: Side, crossing: bool) -> int:
        """Lots wanted on `side`: QUOTE_SIZE when crossing (capped later by the risk room); otherwise the lots
        that close inventory plus the opening size, capped by QUOTE_SIZE."""
        if crossing:
            return QUOTE_SIZE
        inv = self._inventory
        closing = max(0, -inv) if side is BUY else max(0, inv)
        return min(QUOTE_SIZE, closing + self._open_size)

    def _manage_ladder(self, side: Side, top: int | None, fair: float, size: int) -> None:
        """Hold `size` lots at `top`, and keep (never move) older orders up to LADDER_LEVELS - 1 ticks behind it.

        When the target moves a tick away from a resting order, the order stays where it is and keeps
        its queue place. An order priced better than `top` is also kept, and counts as the order at
        `top`, while it still has KEEP_EDGE of edge (joining needs EDGE); otherwise it is cancelled, as
        are orders deeper than the band. Cancelled orders count towards risk until confirmed. New size
        goes to `top`, capped by the risk room and (with TRIM) by `size` minus what already rests; when
        there is no room, the deepest kept order is shrunk or cancelled to make some. With TRIM, resting
        size above `size` is shrunk away, deepest order first.
        """
        step = -1 if side is BUY else 1
        at_top = 0
        resting = 0
        shrinking = False
        kept = []
        top_ids = []
        for order_id, order in self._orders.items():
            if order[0] is not side or order[3] or order[4]:
                continue
            if order[5]:
                shrinking = True
            depth = (order[1] - top) * step if top is not None else -1
            if depth < 0 and top is not None and (fair - order[1] if side is BUY else order[1] - fair) >= self._keep_eff:
                at_top += order[2]  # queue hysteresis: a placed order with KEEP_EDGE left stands in for the target order
                top_ids.append((0, order_id))
            elif depth < 0 or depth >= LADDER_LEVELS:
                self.cancel_order(order_id)
                order[4] = True
                continue
            elif depth == 0:
                at_top += order[2]
                top_ids.append((0, order_id))
            elif not order[5]:
                kept.append((depth, order_id))
            resting += order[2]
        if top is None or top <= 0:
            return
        excess = resting - size
        if TRIM and not shrinking and excess > 0 and (excess >= MIN_NEW or size == 0):
            kept.sort(reverse=True)
            order_id = (kept + top_ids)[0][1]
            order = self._orders[order_id]
            if SHRINK and order[2] > excess:
                self.modify_order(order_id, order[2] - excess)  # keeps its queue place
                order[5] = True
            else:
                self.cancel_order(order_id)
                order[4] = True
            return
        if at_top >= size:
            return
        room = self._room(side)
        if TRIM:
            room = min(room, size - resting)
        want = size - at_top
        if room < min(want, MIN_NEW) and kept:
            kept.sort(reverse=True)
            order_id = kept[0][1]
            order = self._orders[order_id]
            left = order[2] - max(min(want, MAX_POS) - max(room, 0), MIN_NEW)
            if SHRINK and left >= MIN_NEW:
                self.modify_order(order_id, left)  # keeps its queue place
                order[5] = True
            else:
                self.cancel_order(order_id)
                order[4] = True
            return
        qty = min(want, room)
        if qty >= MIN_NEW or (qty == want and qty > 0):
            self._send(side, top, qty, LIMIT)

    def _rescale(self) -> None:
        """Recompute the volatility-scaled settings after a new volatility sample."""
        v = self._vol.vol() if VOL_SCALE else None
        s = 1.0 if v is None else min(VOL_SCALE_MAX, max(VOL_SCALE_MIN, v / VOL_REF))
        self._scale = s
        e = s ** EDGE_VOL_POW
        self._edge_eff = EDGE * e
        self._edge_in_eff = EDGE_IN * e
        self._keep_eff = KEEP_EDGE * e
        self._take_eff = EDGE_TAKE * s ** TAKE_VOL_POW
        self._k_flow_eff = K_FLOW * s ** FLOW_VOL_POW
        self._flow_tau_eff = FLOW_TAU_US * s ** FLOW_TAU_VOL_POW
        self._kf.q = KF_Q * s * s
        self._k_inv_eff = K_INV * s ** INV_VOL_POW
        self._k_inv_q_eff = K_INV_QUOTE * s ** INV_VOL_POW
        self._s_levels_eff = max(S_LEVELS, round(S_LEVELS * s ** S_LEVELS_VOL_POW))

    def _resize(self) -> None:
        """Maker safety switch: scale the passive opening size by how our recent limit-order fills marked out.
        (Those fills include crossing orders, which the fill events cannot tell apart from passive ones.)"""
        mk = self._mk.value()
        if mk is None:
            self._open_size = PASSIVE_OPEN
            return
        zero, full = MK_ZERO * self._scale, MK_FULL * self._scale
        frac = (mk - zero) / (full - zero) if full > zero else 1.0
        frac = min(1.0, max(MK_FLOOR, frac))
        self._open_size = round(PASSIVE_OPEN * frac)

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        """Per market event: read the touch, settle the shadow's pending IOC, update the signals,
        forecast, run the taker and the shadow, then place the quotes around fair."""
        bid, bid_qty = _touch(snapshot.bids)
        ask, ask_qty = _touch(snapshot.asks)
        if bid is None or ask is None:
            return
        ts = snapshot.timestamp_us
        mid = (bid + ask) / 2
        if self._vol.update(ts, mid):
            self._rescale()
        if MAKER_CHECK:
            self._mk.update(ts, mid)
            self._resize()
        self._shadow_fill(ts)
        self._update_signals(ts, bid, bid_qty, ask, ask_qty)
        forecast = self._forecast()
        if TAKE_ENABLED:
            self._run_taker(forecast, bid, ask)
            self._shadow_step(ts, bid, ask)
        if MAKE_ENABLED:
            fair = self._fair(snapshot, trades, bid, bid_qty, ask, ask_qty)
            fair_q = fair - (self._k_inv_q_eff - self._k_inv_eff) * (self._inventory - self._target)
            take = self._take_eff
            inv = self._inventory
            take_b = take * EXIT_MULT if inv < 0 else take
            take_a = take * EXIT_MULT if inv > 0 else take
            pb, pa = self._targets(fair, fair_q, bid, ask, take_b, take_a)
            if GUARD and trades:
                buy_vol = 0
                sell_vol = 0
                for t in trades:
                    if t.taker_owner == 'market':
                        if t.book_side == 'ask':
                            buy_vol += t.quantity
                        else:
                            sell_vol += t.quantity
                if buy_vol >= GUARD_LOTS:
                    self._guard_ask_until = ts + GUARD_US
                if sell_vol >= GUARD_LOTS:
                    self._guard_bid_until = ts + GUARD_US
            cross_b = pb >= ask
            cross_a = pa <= bid
            if cross_b:
                size_b = self._cross_size(fair - ask, take, inv)
            else:
                size_b = 0 if GUARD and ts < self._guard_bid_until else self._side_size(BUY, False)
            if cross_a:
                size_a = self._cross_size(bid - fair, take, -inv)
            else:
                size_a = 0 if GUARD and ts < self._guard_ask_until else self._side_size(SELL, False)
            self._manage_ladder(BUY, pb, fair_q, size_b)
            self._manage_ladder(SELL, pa, fair_q, size_a)

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        """Update inventory from fills, and drop orders that are filled, cancelled or rejected."""
        order = self._orders.get(event.order_id)
        if order is None:
            return
        status = event.status
        if status is OrderStatus.PARTIALLY_FILLED or status is OrderStatus.FILLED:
            self._inventory += event.filled_quantity if order[0] is BUY else -event.filled_quantity
            order[2] = event.remaining_quantity
            if MAKER_CHECK and not order[3]:
                self._mk.add_fill(event.timestamp_us, 1 if order[0] is BUY else -1, event.fill_price, event.filled_quantity)
            if status is OrderStatus.FILLED:
                self._forget(event.order_id, order[0])
        elif status is OrderStatus.MODIFIED:
            order[2] = event.remaining_quantity
            order[5] = False
        elif status is OrderStatus.REJECTED and order[5]:
            order[5] = False  # a shrink that lost the race to a fill: the order may still be live (even with a
            # cancel in flight), so keep tracking it; its CANCELED / FILLED event removes it
        elif status is OrderStatus.CANCELED or status is OrderStatus.REJECTED:
            self._forget(event.order_id, order[0])

    def _forget(self, order_id: str, side: Side) -> None:
        self._orders.pop(order_id, None)
        if self._taker_id == order_id:
            self._taker_id = None
