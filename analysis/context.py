"""Derived arrays shared by the report sections, computed once."""

from dataclasses import dataclass, field
from functools import cached_property
from math import gcd

import numpy as np
import pandas as pd

from .replay import Replay

GRID_STEP_US = 100_000  # fixed 100 ms sampling grid for depth and signals


@dataclass
class Context:
    tape: pd.DataFrame
    rp: Replay
    series_step_us: int
    bin_step_us: int
    cache: dict = field(default_factory=dict)

    @cached_property
    def tick(self) -> int:
        """Greatest common divisor of all distinct insert prices' gaps (1 when prices are in ticks)."""
        prices = np.unique(self.rp.od_price)
        if len(prices) < 2:
            return 1
        g = 0
        for d in np.diff(prices).tolist():
            g = gcd(g, int(d))
            if g == 1:
                break
        return max(g, 1)

    @cached_property
    def t0(self) -> int:
        return int(self.rp.ev_ts[0]) if len(self.rp.ev_ts) else 0

    @cached_property
    def t1(self) -> int:
        return int(self.rp.ev_ts[-1]) if len(self.rp.ev_ts) else 0

    @cached_property
    def duration_s(self) -> float:
        return (self.t1 - self.t0) / 1e6

    @cached_property
    def dur_us(self) -> np.ndarray:
        """How long each event's post-state lasted (time to the next event; 0 for the last)."""
        ts = self.rp.ev_ts
        d = np.zeros(len(ts), dtype=np.int64)
        if len(ts) > 1:
            d[:-1] = np.diff(ts)
        return d

    @cached_property
    def mid_raw_t(self) -> np.ndarray:
        """Mid in ticks per event, nan while a side is empty."""
        return self.rp.mid / self.tick

    @cached_property
    def mid_t(self) -> np.ndarray:
        """Mid in ticks per event, carrying the last two-sided mid through one-sided moments."""
        m = self.mid_raw_t
        ok = np.isfinite(m)
        idx = np.where(ok, np.arange(len(m)), -1)
        np.maximum.accumulate(idx, out=idx)
        out = np.full(len(m), np.nan)
        has = idx >= 0
        out[has] = m[idx[has]]
        return out

    @cached_property
    def spread_t(self) -> np.ndarray:
        return self.rp.spread / self.tick

    @cached_property
    def mid_before_event(self) -> np.ndarray:
        """Mid (ticks) just before each event."""
        m = np.full(len(self.mid_t), np.nan)
        m[1:] = self.mid_t[:-1]
        return m

    @cached_property
    def spread_before_event(self) -> np.ndarray:
        s = np.full(len(self.spread_t), np.nan)
        s[1:] = self.spread_t[:-1]
        return s

    def mid_at(self, t_us: np.ndarray) -> np.ndarray:
        """Mid in ticks (forward-filled) at the given times."""
        idx = self.rp.state_index(np.asarray(t_us))
        out = np.full(idx.shape, np.nan)
        ok = idx >= 0
        out[ok] = self.mid_t[idx[ok]]
        return out

    # Grid -----------------------------------------------------------------

    @cached_property
    def grid_idx(self) -> np.ndarray:
        return self.rp.state_index(self.rp.grid_ts)

    def _at_grid(self, arr: np.ndarray, fill=np.nan) -> np.ndarray:
        idx = self.grid_idx
        out = np.full(len(idx), fill, dtype=float)
        ok = idx >= 0
        out[ok] = arr[idx[ok]]
        return out

    @cached_property
    def grid_mid(self) -> np.ndarray:
        return self._at_grid(self.mid_t)

    @cached_property
    def grid_spread(self) -> np.ndarray:
        return self._at_grid(self.spread_t)

    @cached_property
    def grid_bid_qty(self) -> np.ndarray:
        return self._at_grid(self.rp.bid_qty.astype(float))

    @cached_property
    def grid_ask_qty(self) -> np.ndarray:
        return self._at_grid(self.rp.ask_qty.astype(float))

    # Trades ---------------------------------------------------------------

    @cached_property
    def cum_signed_volume(self) -> np.ndarray:
        """Cumulative signed trade volume, with a leading 0 (index = number of trades so far)."""
        return np.concatenate([[0], np.cumsum(self.rp.tr_qty * self.rp.tr_sign.astype(np.int64))])

    @cached_property
    def cum_volume(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.rp.tr_qty)])

    def trades_upto(self, t_us: np.ndarray) -> np.ndarray:
        """Number of trades with timestamp <= t."""
        return np.searchsorted(self.rp.tr_ts, t_us, side='right')

    @cached_property
    def taker_events(self) -> pd.DataFrame:
        """One row per aggressive order: the trades one taker produced on arrival."""
        rp = self.rp
        if len(rp.tr_ts) == 0:
            return pd.DataFrame(
                columns=['taker', 'ts', 'event', 'sign', 'qty', 'n_trades', 'levels', 'ioc', 'mid_before', 'spread_before', 'vwap']
            )
        start = np.flatnonzero(np.r_[True, rp.tr_taker[1:] != rp.tr_taker[:-1]])
        qty = np.add.reduceat(rp.tr_qty, start)
        notional = np.add.reduceat(rp.tr_qty * rp.tr_price, start)
        new_level = np.r_[True, (rp.tr_price[1:] != rp.tr_price[:-1]) | (rp.tr_taker[1:] != rp.tr_taker[:-1])]
        levels = np.add.reduceat(new_level.astype(np.int64), start)
        n_trades = np.diff(np.r_[start, len(rp.tr_ts)])
        taker = rp.tr_taker[start]
        event = rp.tr_event[start]
        return pd.DataFrame(
            {
                'taker': taker,
                'ts': rp.tr_ts[start],
                'event': event,
                'sign': rp.tr_sign[start].astype(np.int64),
                'qty': qty,
                'n_trades': n_trades,
                'levels': levels,
                'ioc': rp.od_ioc[taker],
                'mid_before': self.mid_before_event[event],
                'spread_before': self.spread_before_event[event],
                'vwap': notional / qty / self.tick,
            }
        )
