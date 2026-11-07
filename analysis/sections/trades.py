"""Trades, aggressive flow, price impact and what passive fills earn (adverse selection)."""

import numpy as np
import pandas as pd

from ..context import Context
from ..util import US, autocorr, categorical_shares, dist, label_s, ols, safe_div

IMPACT_HORIZONS_S = (0.005, 0.1, 1, 5, 30, 60)
MARKOUT_HORIZONS_S = (0.0, 0.005, 0.1, 1, 5, 30, 60)
SIZE_BUCKETS = 5


def _size_buckets(qty: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
    """Up to SIZE_BUCKETS quantile buckets of `qty` as (lo, hi, mask)."""
    if len(qty) == 0:
        return []
    edges = np.unique(np.quantile(qty, np.linspace(0, 1, SIZE_BUCKETS + 1)))
    out = []
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        last = k == len(edges) - 2
        m = (qty >= lo) & ((qty <= hi) if last else (qty < hi))
        out.append((float(lo), float(hi), m))
    if not out:  # all equal
        out.append((float(edges[0]), float(edges[0]), np.ones(len(qty), dtype=bool)))
    return out


def _wmean(x: np.ndarray, w: np.ndarray) -> float | None:
    ok = np.isfinite(x)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() and w[ok].sum() > 0 else None


def trades_section(ctx: Context) -> dict:
    rp = ctx.rp
    te: pd.DataFrame = ctx.taker_events
    qty = rp.tr_qty
    n_tr = len(qty)
    vol = int(qty.sum())
    te_qty = te['qty'].to_numpy(dtype=np.int64)
    te_sign = te['sign'].to_numpy(dtype=np.int64)
    te_ts = te['ts'].to_numpy(dtype=np.int64)
    te_ioc = te['ioc'].to_numpy(dtype=bool)
    mid_before = te['mid_before'].to_numpy(dtype=float)

    # Impact of aggressive orders: signed mid move from just before the order.
    impact = {}
    imp_arrays = {}
    for h in IMPACT_HORIZONS_S:
        mh = ctx.mid_at(te_ts + int(h * US))
        x = te_sign * (mh - mid_before)
        imp_arrays[h] = x
        ok = np.isfinite(x)
        impact[label_s(h)] = {
            'mean_ticks': float(x[ok].mean()) if ok.any() else None,
            'volume_weighted_ticks': _wmean(x, te_qty.astype(float)),
            'n': int(ok.sum()),
        }
    impact_by_size = []
    for lo, hi, m in _size_buckets(te_qty.astype(float)):
        row = {'size_lo': lo, 'size_hi': hi, 'n': int(m.sum())}
        for h in IMPACT_HORIZONS_S:
            x = imp_arrays[h][m]
            x = x[np.isfinite(x)]
            row[label_s(h)] = float(x.mean()) if len(x) else None
        impact_by_size.append(row)

    # Kyle lambda: mid change vs net signed aggressive volume per bucket.
    kyle = {}
    for h in (10, 60):
        step = h * US
        edges = np.arange(ctx.t0, ctx.t1 + step, step, dtype=np.int64)
        if len(edges) < 3:
            kyle[label_s(h)] = ols(np.array([]), np.array([]))
            continue
        m = ctx.mid_at(edges)
        cs = ctx.cum_signed_volume[ctx.trades_upto(edges)]
        kyle[label_s(h)] = ols(np.diff(cs).astype(float), np.diff(m))

    # Largest aggressive orders.
    top = np.argsort(-te_qty, kind='stable')[:10]
    imp1 = imp_arrays[1]
    largest = [
        {
            't_s': float((te_ts[i] - ctx.t0) / US),
            'size': int(te_qty[i]),
            'sign': int(te_sign[i]),
            'levels': int(te['levels'].iat[i]),
            'ioc': bool(te_ioc[i]),
            'impact_1s_ticks': float(imp1[i]) if np.isfinite(imp1[i]) else None,
        }
        for i in top
    ]

    maker_age_ms = (rp.tr_ts - rp.od_ts[rp.tr_maker]) / 1000
    at_touch = te['levels'].to_numpy() == 1
    dur = ctx.duration_s or None
    return {
        'count': n_tr,
        'volume': vol,
        'trades_per_s': safe_div(n_tr, dur),
        'volume_per_s': safe_div(vol, dur),
        'volume_per_minute': safe_div(vol * 60, dur),
        'taker_events': int(len(te)),
        'taker_events_per_s': safe_div(len(te), dur),
        'buy_initiated_share': {
            'by_count': float((te_sign > 0).mean()) if len(te) else None,
            'by_volume': safe_div(int(te_qty[te_sign > 0].sum()), int(te_qty.sum())),
        },
        'trade_size': dist(qty),
        'taker_event_size': dist(te_qty),
        'levels_swept_share': categorical_shares(np.minimum(te['levels'].to_numpy(), 4), [1, 2, 3, 4]),
        'levels_swept_note': 'key 4 means 4 or more price levels',
        'single_level_share': float(at_touch.mean()) if len(te) else None,
        'taker_ioc_share': {
            'by_count': float(te_ioc.mean()) if len(te) else None,
            'by_volume': safe_div(int(te_qty[te_ioc].sum()), int(te_qty.sum())),
        },
        'taker_interarrival_ms': dist(np.diff(te_ts) / 1000 if len(te_ts) > 1 else []),
        'sign_autocorr': autocorr(te_sign.astype(float), (1, 2, 5, 10, 20, 50, 100)),
        'same_sign_run_length': dist(_run_lengths(te_sign)),
        'maker_age_at_fill_ms': dist(maker_age_ms),
        'spread_before_taker_ticks': dist(te['spread_before'].to_numpy(dtype=float)),
        'impact_ticks': impact,
        'impact_by_size': impact_by_size,
        'kyle_lambda': kyle,
        'largest_taker_events': largest,
    }


def _run_lengths(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.array([])
    change = np.flatnonzero(np.r_[True, x[1:] != x[:-1], True])
    return np.diff(change)


def maker_section(ctx: Context) -> dict:
    """Markout of every passive fill from the maker's side, per lot, in ticks.

    markout(h) = maker_side * (mid(t + h) - fill_price). h = 0 uses the mid just
    before the aggressive order, so it is the half-spread captured. A touch market
    maker earns roughly markout(h) per lot it gets filled, before inventory costs.
    """
    rp = ctx.rp
    qty = rp.tr_qty.astype(float)
    side = -rp.tr_sign.astype(float)
    price = rp.tr_price / ctx.tick
    spread_before = ctx.spread_before_event[rp.tr_event]
    taker_ioc = rp.od_ioc[rp.tr_taker]
    first_level = np.r_[True, (rp.tr_taker[1:] != rp.tr_taker[:-1])]
    # A fill is "at the touch" if its price equals the first price of its sweep.
    first_price = price[np.maximum.accumulate(np.where(first_level, np.arange(len(price)), 0))] if len(price) else price
    at_touch = price == first_price

    mk = {}
    for h in MARKOUT_HORIZONS_S:
        if h == 0:
            m = ctx.mid_before_event[rp.tr_event]
        else:
            m = ctx.mid_at(rp.tr_ts + int(h * US))
        mk[h] = side * (m - price)

    def summary(mask: np.ndarray) -> dict:
        row: dict = {'fills': int(mask.sum()), 'volume': int(qty[mask].sum())}
        for h in MARKOUT_HORIZONS_S:
            row[label_s(h)] = _wmean(mk[h][mask], qty[mask])
        return row

    allm = np.ones(len(qty), dtype=bool)
    by_size = []
    for lo, hi, m in _size_buckets(qty):
        by_size.append({'size_lo': lo, 'size_hi': hi, **summary(m)})

    profitable = {}
    for h in MARKOUT_HORIZONS_S:
        x = mk[h]
        ok = np.isfinite(x)
        profitable[label_s(h)] = float((x[ok] > 0).mean()) if ok.any() else None

    # Volume available to a touch maker per second (fills at the touch).
    dur = ctx.duration_s or None
    return {
        'note': 'ticks per lot from the passive side; positive = maker profits; volume-weighted',
        'markout_ticks': summary(allm),
        'profitable_share': profitable,
        'at_touch': summary(at_touch),
        'through_touch': summary(~at_touch),
        'by_spread_before': {
            '1': summary(spread_before == 1),
            '2': summary(spread_before == 2),
            'gt2': summary(spread_before > 2),
        },
        'by_taker_type': {
            'ioc': summary(taker_ioc),
            'limit': summary(~taker_ioc),
        },
        'by_maker_side': {
            'bid': summary(side > 0),
            'ask': summary(side < 0),
        },
        'by_fill_size': by_size,
        'touch_fill_volume_per_s': safe_div(float(qty[at_touch].sum()), dur),
    }
