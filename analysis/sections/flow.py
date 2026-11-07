"""Tape composition and order lifecycle: who sends what, where, and what happens to it."""

import numpy as np

from ..context import Context
from ..replay import CANCELED, FILLED, IOC_DONE, RESTING_AT_END
from ..util import US, binned_shares, categorical_shares, corr, dist, safe_div

PLACEMENT_EDGES = [-np.inf, -20, -5, -1, 0, 1, np.inf]
PLACEMENT_NAMES = ['behind_gt20', 'behind_6_20', 'behind_2_5', 'behind_1', 'join', 'improve']


def tape_section(ctx: Context) -> dict:
    t = ctx.tape
    rp = ctx.rp
    ins = t['is_insert'].to_numpy()
    side = t['side_sign'].to_numpy()
    vol = t['Volume'].fillna(0).to_numpy(dtype=np.int64)
    ioc = t['ioc'].to_numpy()
    ts = rp.ev_ts
    n = len(t)

    dt = np.diff(ts) if n > 1 else np.array([], dtype=np.int64)
    sec = (ts - ctx.t0) // US
    per_sec = np.bincount(sec) if n else np.array([0])

    # Events within any 5 ms window: count of rows in [ts, ts + 5ms)
    if n:
        ahead = np.searchsorted(ts, ts + 5_000, side='left') - np.arange(n)
    else:
        ahead = np.array([0])

    ins_vol = vol[ins]
    ins_px = rp.od_price
    sizes, counts = np.unique(ins_vol, return_counts=True)
    top = np.argsort(-counts, kind='stable')[:10]
    rel_px = ins_px // ctx.tick

    ioc_idx = np.flatnonzero(rp.od_ioc)
    ioc_qty = rp.od_qty[ioc_idx]
    ioc_fill = rp.od_taker_fill[ioc_idx]

    return {
        'rows': n,
        'inserts': int(ins.sum()),
        'cancels': int((~ins).sum()),
        'limit_inserts': int((ins & ~ioc).sum()),
        'ioc_inserts': int(ioc.sum()),
        'start_s': ctx.t0 / US,
        'end_s': ctx.t1 / US,
        'duration_s': ctx.duration_s,
        'rows_per_s': safe_div(n, ctx.duration_s),
        'inserts_per_s': safe_div(int(ins.sum()), ctx.duration_s),
        'cancel_to_insert_ratio': safe_div(int((~ins).sum()), int(ins.sum())),
        'unique_order_ids': int(t['OrderId'].nunique()),
        'buy_inserts': int((ins & (side == 1)).sum()),
        'sell_inserts': int((ins & (side == -1)).sum()),
        'buy_insert_volume': int(vol[ins & (side == 1)].sum()),
        'sell_insert_volume': int(vol[ins & (side == -1)].sum()),
        'anomalies': {
            'cancel_not_resting': rp.n_cancel_not_resting,
            'cancel_unknown_id': rp.n_cancel_unknown,
            'cancel_after_full_fill': rp.n_cancel_after_fill,
            'cancel_twice': rp.n_cancel_twice,
            'duplicate_insert_while_resting': rp.n_duplicate_insert,
            'nonmonotonic_timestamps': int((dt < 0).sum()),
        },
        'timing': {
            'interarrival_us': dist(dt),
            'share_interarrival_zero': float((dt == 0).mean()) if len(dt) else None,
            'share_interarrival_lt_5ms': float((dt < 5_000).mean()) if len(dt) else None,
            'events_per_second': dist(per_sec),
            'busiest_second_events': int(per_sec.max()) if len(per_sec) else 0,
            'events_in_next_5ms': dist(ahead),
            'longest_gap_s': float(dt.max() / US) if len(dt) else None,
        },
        'insert_size': dist(ins_vol),
        'insert_size_top': [
            {'size': int(sizes[i]), 'count': int(counts[i]), 'share': float(counts[i] / len(ins_vol))} for i in top
        ],
        'price': {
            'tick_size': ctx.tick,
            'min': float(ins_px.min()) if len(ins_px) else None,
            'max': float(ins_px.max()) if len(ins_px) else None,
            'range_ticks': float((ins_px.max() - ins_px.min()) / ctx.tick) if len(ins_px) else None,
            'last_digit_share_by_volume': categorical_shares(rel_px % 10, list(range(10)), weights=rp.od_qty),
            'share_mult_5': float((rel_px % 5 == 0).mean()) if len(rel_px) else None,
            'share_mult_10': float((rel_px % 10 == 0).mean()) if len(rel_px) else None,
        },
        'ioc': {
            'count': int(len(ioc_idx)),
            'volume': int(ioc_qty.sum()),
            'share_any_fill': float((ioc_fill > 0).mean()) if len(ioc_idx) else None,
            'share_full_fill': float((ioc_fill == ioc_qty).mean()) if len(ioc_idx) else None,
            'volume_fill_ratio': safe_div(int(ioc_fill.sum()), int(ioc_qty.sum())),
            'priced_through_touch_share': float((rp.od_cross_gap[ioc_idx] > 0).mean()) if len(ioc_idx) else None,
        },
    }


def _bucket_stats(ctx: Context, mask: np.ndarray) -> dict:
    rp = ctx.rp
    m = mask
    cnt = int(m.sum())
    ended = m & (rp.od_end_ts >= 0)
    life_ms = (rp.od_end_ts[ended] - rp.od_ts[ended]) / 1000
    ff = m & (rp.od_first_maker_fill_ts >= 0)
    ttf_ms = (rp.od_first_maker_fill_ts[ff] - rp.od_ts[ff]) / 1000
    return {
        'count': cnt,
        'share': None,  # filled by caller
        'any_fill_prob': float((rp.od_maker_fill[m] > 0).mean()) if cnt else None,
        'full_fill_prob': float((rp.od_outcome[m] == FILLED).mean()) if cnt else None,
        'cancel_prob': float((rp.od_outcome[m] == CANCELED).mean()) if cnt else None,
        'volume_fill_ratio': safe_div(int(rp.od_maker_fill[m].sum()), int(rp.od_qty[m].sum())),
        'median_lifetime_ms': float(np.median(life_ms)) if len(life_ms) else None,
        'median_time_to_first_fill_ms': float(np.median(ttf_ms)) if len(ttf_ms) else None,
        'mean_size': float(rp.od_qty[m].mean()) if cnt else None,
    }


def orders_section(ctx: Context) -> dict:
    rp = ctx.rp
    n = len(rp.od_ts)
    out_names = {RESTING_AT_END: 'resting_at_end', FILLED: 'filled', CANCELED: 'canceled', IOC_DONE: 'ioc_unfilled_remainder'}
    outcome = {name: float((rp.od_outcome == code).mean()) if n else None for code, name in out_names.items()}

    marketable = rp.od_cross_gap >= 0
    rested = rp.od_queue_ahead >= 0
    passive = ~marketable & ~rp.od_ioc
    rel = rp.od_rel_touch / ctx.tick
    side_empty = np.isnan(rel)

    by_placement = {}
    for name, lo, hi in zip(PLACEMENT_NAMES, PLACEMENT_EDGES[:-1], PLACEMENT_EDGES[1:], strict=True):
        m = passive & (rel >= lo) & (rel < hi)
        by_placement[name] = _bucket_stats(ctx, m)
    by_placement['side_empty'] = _bucket_stats(ctx, passive & side_empty)
    n_passive = int(passive.sum())
    for v in by_placement.values():
        v['share'] = safe_div(v['count'], n_passive)

    canceled = rp.od_outcome == CANCELED
    c_life_ms = (rp.od_end_ts[canceled] - rp.od_ts[canceled]) / 1000
    filled_rest = (rp.od_outcome == FILLED) & rested
    f_life_ms = (rp.od_end_ts[filled_rest] - rp.od_ts[filled_rest]) / 1000
    ended_rest = rested & (rp.od_end_ts >= 0)
    life_ms_all = (rp.od_end_ts[ended_rest] - rp.od_ts[ended_rest]) / 1000

    # Fill probability of touch joiners by queue ahead quartile.
    joins = passive & (rel == 0) & rested
    qa = rp.od_queue_ahead[joins].astype(float)
    filled_any = rp.od_maker_fill[joins] > 0
    queue_buckets = []
    if len(qa) >= 20:
        edges = np.unique(np.quantile(qa, [0, 0.25, 0.5, 0.75, 1.0]))
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            m = (qa >= lo) & ((qa < hi) if hi < edges[-1] else (qa <= hi))
            queue_buckets.append(
                {'queue_ahead_lo': float(lo), 'queue_ahead_hi': float(hi), 'n': int(m.sum()), 'any_fill_prob': float(filled_any[m].mean()) if m.any() else None}
            )

    cancel_rel = rp.od_cancel_rel_touch[canceled] / ctx.tick
    life_all = np.where(rp.od_end_ts >= 0, rp.od_end_ts - rp.od_ts, np.nan)

    return {
        'count': n,
        'outcome_share': outcome,
        'marketable_share': float(marketable.mean()) if n else None,
        'marketable_limit_share': float((marketable & ~rp.od_ioc).mean()) if n else None,
        'taker_fill_volume_share': safe_div(int(rp.od_taker_fill.sum()), int(rp.od_qty.sum())),
        'passive_placement': by_placement,
        'lifetime_ms': {
            'all_rested': dist(life_ms_all),
            'canceled': dist(c_life_ms),
            'filled': dist(f_life_ms),
        },
        'cancel_within_share': {
            label: float((c_life_ms < lim).mean()) if len(c_life_ms) else None
            for label, lim in [('5ms', 5), ('50ms', 50), ('500ms', 500), ('1s', 1000), ('10s', 10_000), ('60s', 60_000)]
        },
        'cancel_placement_share': binned_shares(cancel_rel, PLACEMENT_EDGES, PLACEMENT_NAMES),
        'join_fill_prob_by_queue_ahead': queue_buckets,
        'queue_ahead_on_join': dist(qa),
        'size_vs_lifetime': corr(rp.od_qty.astype(float), life_all),
        'median_size_by_outcome': {
            name: float(np.median(rp.od_qty[rp.od_outcome == code])) if (rp.od_outcome == code).any() else None
            for code, name in out_names.items()
        },
    }
