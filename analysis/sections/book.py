"""Order book shape and dynamics, time-weighted unless the key says otherwise."""

import numpy as np

from ..context import Context
from ..replay import DEPTH_LEVELS
from ..util import US, binned_shares, dist, safe_div

SPREAD_EDGES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 10.5, np.inf]
SPREAD_NAMES = ['1', '2', '3', '4', '5', '6_10', 'gt10']
LATENCY_US = 5_000


def _episodes(flag: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Durations (us) of the runs where `flag` holds, measured in event time."""
    if not len(flag):
        return np.array([])
    f = flag.astype(np.int8)
    d = np.diff(np.r_[0, f, 0])
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)  # first index after the run
    ends_ts = np.where(ends < len(ts), ts[np.minimum(ends, len(ts) - 1)], ts[-1])
    return (ends_ts - ts[starts]).astype(float)


def book_section(ctx: Context) -> dict:
    rp = ctx.rp
    w = ctx.dur_us.astype(float)
    both = np.isfinite(rp.best_bid) & np.isfinite(rp.best_ask)
    total_w = float(w.sum())
    spread = ctx.spread_t
    bq, aq = rp.bid_qty.astype(float), rp.ask_qty.astype(float)
    imb = np.where(both & ((bq + aq) > 0), (bq - aq) / np.maximum(bq + aq, 1), np.nan)

    # Depth profile from the 100 ms grid.
    gb, ga = rp.grid_bid_depth.astype(float), rp.grid_ask_depth.astype(float)
    g_ok = (gb[:, 0] > 0) & (ga[:, 0] > 0) if len(gb) else np.array([], dtype=bool)
    gb, ga = gb[g_ok], ga[g_ok]

    def depth_imb(k: int) -> np.ndarray:
        b, a = gb[:, :k].sum(1), ga[:, :k].sum(1)
        return (b - a) / np.maximum(b + a, 1)

    # Touch dynamics, event time.
    bb, ba = rp.best_bid, rp.best_ask
    bb_ch = np.r_[False, (bb[1:] != bb[:-1]) & ~(np.isnan(bb[1:]) & np.isnan(bb[:-1]))]
    ba_ch = np.r_[False, (ba[1:] != ba[:-1]) & ~(np.isnan(ba[1:]) & np.isnan(ba[:-1]))]
    touch_ch = bb_ch | ba_ch
    mid = ctx.mid_t
    mid_ch = np.r_[False, (mid[1:] != mid[:-1]) & np.isfinite(mid[1:]) & np.isfinite(mid[:-1])]
    qty_ch = np.r_[False, (rp.bid_qty[1:] != rp.bid_qty[:-1]) | (rp.ask_qty[1:] != rp.ask_qty[:-1])]
    ch_ts = rp.ev_ts[touch_ch]
    touch_life_ms = np.diff(ch_ts) / 1000 if len(ch_ts) > 1 else np.array([])

    # What changes during the 5 ms an order is in flight. Sent right after event i, it lands
    # just before the first event with ts >= ts_i + 5 ms, so it meets the state after event j-1.
    n = len(rp.ev_ts)
    if n:
        j = np.searchsorted(rp.ev_ts, rp.ev_ts + LATENCY_US, side='left') - 1
        j = np.clip(j, 0, n - 1)
        ok = both & both[j]
        moved_touch = ((bb[j] != bb) | (ba[j] != ba))[ok]
        moved_mid = np.abs(mid[j] - mid)[ok]
        moved_qty = ((rp.bid_qty[j] != rp.bid_qty) | (rp.ask_qty[j] != rp.ask_qty))[ok]
        # Same thing at uniform random times (grid), i.e. time-weighted.
        g = rp.grid_ts
        gi = ctx.grid_idx
        gj = rp.state_index(g + LATENCY_US)
        gok = (gi >= 0) & both[np.maximum(gi, 0)] & both[np.maximum(gj, 0)] & (gj >= 0)
        g_moved_touch = ((bb[gj] != bb[gi]) | (ba[gj] != ba[gi]))[gok] if len(g) else np.array([])
        g_moved_mid = np.abs(mid[gj] - mid[gi])[gok] if len(g) else np.array([])
    else:
        moved_touch = moved_mid = moved_qty = g_moved_touch = g_moved_mid = np.array([])

    modal_spread = None
    wide_ms = np.array([])
    if both.any():
        vals, inv = np.unique(spread[both], return_inverse=True)
        modal_spread = float(vals[np.argmax(np.bincount(inv, weights=w[both]))])
        wide_ms = _episodes(both & (spread > modal_spread), rp.ev_ts) / 1000

    dur = ctx.duration_s or None
    return {
        'time_share': {
            'both_sides': safe_div(float(w[both].sum()), total_w),
            'bid_empty': safe_div(float(w[np.isnan(bb)].sum()), total_w),
            'ask_empty': safe_div(float(w[np.isnan(ba)].sum()), total_w),
        },
        'spread_ticks': dist(spread[both], w[both]),
        'spread_ticks_event_weighted': dist(spread[both]),
        'spread_share': binned_shares(spread[both], SPREAD_EDGES, SPREAD_NAMES, w[both]),
        'modal_spread_ticks': modal_spread,
        'wide_spread_episode_ms': dist(wide_ms),
        'touch_qty': {
            'bid': dist(bq[np.isfinite(bb)], w[np.isfinite(bb)]),
            'ask': dist(aq[np.isfinite(ba)], w[np.isfinite(ba)]),
        },
        'touch_orders': {
            'bid': dist(rp.bid_orders[np.isfinite(bb)], w[np.isfinite(bb)]),
            'ask': dist(rp.ask_orders[np.isfinite(ba)], w[np.isfinite(ba)]),
        },
        'depth_profile': {
            'levels': DEPTH_LEVELS,
            'note': 'index k = k ticks away from own-side best; quantity sampled every 100 ms',
            'bid_mean': gb.mean(0) if len(gb) else [None] * DEPTH_LEVELS,
            'ask_mean': ga.mean(0) if len(ga) else [None] * DEPTH_LEVELS,
            'bid_median': np.median(gb, 0) if len(gb) else [None] * DEPTH_LEVELS,
            'ask_median': np.median(ga, 0) if len(ga) else [None] * DEPTH_LEVELS,
            'bid_nonempty_share': (gb > 0).mean(0) if len(gb) else [None] * DEPTH_LEVELS,
            'ask_nonempty_share': (ga > 0).mean(0) if len(ga) else [None] * DEPTH_LEVELS,
            'bid_cum_mean': gb.cumsum(1).mean(0) if len(gb) else [None] * DEPTH_LEVELS,
            'ask_cum_mean': ga.cumsum(1).mean(0) if len(ga) else [None] * DEPTH_LEVELS,
        },
        'imbalance': {
            'touch': dist(imb, w),
            'depth_5': dist(depth_imb(5)) if len(gb) else dist([]),
            'depth_20': dist(depth_imb(DEPTH_LEVELS)) if len(gb) else dist([]),
        },
        'dynamics': {
            'best_bid_changes_per_s': safe_div(int(bb_ch.sum()), dur),
            'best_ask_changes_per_s': safe_div(int(ba_ch.sum()), dur),
            'mid_changes_per_s': safe_div(int(mid_ch.sum()), dur),
            'share_events_moving_touch_price': float(touch_ch.mean()) if n else None,
            'share_events_changing_touch_qty': float(qty_ch.mean()) if n else None,
            'touch_price_lifetime_ms': dist(touch_life_ms),
        },
        'latency_5ms': {
            'note': 'state change between sending and landing of an order (5 ms latency)',
            'event_weighted': {
                'touch_price_moved_share': float(moved_touch.mean()) if len(moved_touch) else None,
                'touch_qty_changed_share': float(moved_qty.mean()) if len(moved_qty) else None,
                'mid_moved_share': float((moved_mid > 0).mean()) if len(moved_mid) else None,
                'mean_abs_mid_move_ticks': float(moved_mid.mean()) if len(moved_mid) else None,
            },
            'time_weighted': {
                'touch_price_moved_share': float(g_moved_touch.mean()) if len(g_moved_touch) else None,
                'mid_moved_share': float((g_moved_mid > 0).mean()) if len(g_moved_mid) else None,
                'mean_abs_mid_move_ticks': float(g_moved_mid.mean()) if len(g_moved_mid) else None,
            },
        },
        'grid_samples': int(len(rp.grid_ts)),
        'grid_step_s': (rp.grid_ts[1] - rp.grid_ts[0]) / US if len(rp.grid_ts) > 1 else None,
    }
