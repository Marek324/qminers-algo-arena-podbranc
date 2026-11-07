"""Time series for plotting: a fine series (default 1 s) and coarser bins (default 60 s) with per-bin stats.

Both are columnar: {'columns': [...], 'data': {column: [values...]}}. Row k covers the
interval (t_k - step, t_k]; state columns (bid, ask, ...) are the book at t_k.
"""

import numpy as np

from ..context import Context
from ..replay import DEPTH_LEVELS
from ..util import US, autocorr, corr


def _edges(ctx: Context, step_us: int) -> np.ndarray:
    k = max(int(np.ceil((ctx.t1 - ctx.t0) / step_us)), 1)
    e = ctx.t0 + step_us * np.arange(k + 1, dtype=np.int64)
    e[0] = ctx.t0 - 1  # first interval includes the first event
    return e


def _interval_sums(ts: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    cum = np.concatenate([[0], np.cumsum(values)])
    idx = np.searchsorted(ts, edges, side='right')
    return np.diff(cum[idx])


def _flows(ctx: Context, edges: np.ndarray) -> dict[str, np.ndarray]:
    rp, t = ctx.rp, ctx.tape
    ins = t['is_insert'].to_numpy()
    ioc = t['ioc'].to_numpy()
    ones = np.ones(len(rp.ev_ts), dtype=np.int64)
    q = rp.tr_qty
    buy = np.where(rp.tr_sign > 0, q, 0)
    return {
        'events': _interval_sums(rp.ev_ts, ones, edges),
        'inserts': _interval_sums(rp.ev_ts, ins.astype(np.int64), edges),
        'cancels': _interval_sums(rp.ev_ts, (~ins).astype(np.int64), edges),
        'ioc_inserts': _interval_sums(rp.ev_ts, ioc.astype(np.int64), edges),
        'trades': _interval_sums(rp.tr_ts, np.ones(len(q), dtype=np.int64), edges),
        'volume': _interval_sums(rp.tr_ts, q, edges),
        'buy_volume': _interval_sums(rp.tr_ts, buy, edges),
        'sell_volume': _interval_sums(rp.tr_ts, q - buy, edges),
    }


def _state(ctx: Context, times: np.ndarray) -> dict[str, np.ndarray]:
    rp = ctx.rp
    idx = rp.state_index(times)
    ok = idx >= 0
    i = np.maximum(idx, 0)

    def take(arr):
        out = np.full(len(times), np.nan)
        out[ok] = arr[i[ok]]
        return out

    n_tr = ctx.trades_upto(times)
    last_px = np.full(len(times), np.nan)
    has = n_tr > 0
    last_px[has] = rp.tr_price[n_tr[has] - 1] / ctx.tick
    bq, aq = take(rp.bid_qty.astype(float)), take(rp.ask_qty.astype(float))
    return {
        'best_bid': take(rp.best_bid) / ctx.tick,
        'best_ask': take(rp.best_ask) / ctx.tick,
        'mid': take(ctx.mid_t),
        'spread': take(ctx.spread_t),
        'bid_qty': bq,
        'ask_qty': aq,
        'imbalance': np.where((bq + aq) > 0, (bq - aq) / np.maximum(bq + aq, 1), np.nan),
        'last_trade_price': last_px,
    }


def _columnar(data: dict[str, np.ndarray]) -> dict:
    return {'columns': list(data), 'data': data}


def series_section(ctx: Context) -> dict:
    step = ctx.series_step_us
    edges = _edges(ctx, step)
    times = edges[1:]
    data = {'t_s': (times - ctx.t0) / US}
    data.update(_state(ctx, times))
    data.update(_flows(ctx, edges))
    data['net_flow'] = data['buy_volume'] - data['sell_volume']
    return {'step_s': step / US, 'rows': int(len(times)), **_columnar(data)}


def bins_section(ctx: Context) -> tuple[dict, dict]:
    """Per-bin stats and the cross-bin relationships (stationarity, clustering)."""
    rp = ctx.rp
    step = ctx.bin_step_us
    edges = _edges(ctx, step)
    k = len(edges) - 1
    flows = _flows(ctx, edges)

    # OHLC of mid within each bin, event time.
    mid = ctx.mid_t
    ev_bin = np.clip(np.searchsorted(edges, rp.ev_ts, side='left') - 1, 0, k - 1)
    o = np.full(k, np.nan)
    h = np.full(k, np.nan)
    lo = np.full(k, np.nan)
    c = np.full(k, np.nan)
    fin = np.isfinite(mid)
    if fin.any():
        b, m = ev_bin[fin], mid[fin]
        starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
        ends = np.r_[starts[1:], len(b)] - 1
        ub = b[starts]
        o[ub] = m[starts]
        c[ub] = m[ends]
        h[ub] = np.maximum.reduceat(m, starts)
        lo[ub] = np.minimum.reduceat(m, starts)

    # Realized vol from 1 s mid changes inside each bin.
    t1s = np.arange(ctx.t0, edges[-1] + 1, US, dtype=np.int64)
    m1 = ctx.mid_at(t1s)
    r1 = np.diff(m1)
    r1_bin = np.clip(np.searchsorted(edges, t1s[1:], side='left') - 1, 0, k - 1)
    ok = np.isfinite(r1)
    rv = np.sqrt(np.bincount(r1_bin[ok], weights=r1[ok] ** 2, minlength=k))

    # Grid (100 ms) averages per bin.
    g = rp.grid_ts
    g_bin = np.clip(np.searchsorted(edges, g, side='left') - 1, 0, k - 1)

    def gmean(x):
        okx = np.isfinite(x)
        s = np.bincount(g_bin[okx], weights=x[okx], minlength=k)
        n = np.bincount(g_bin[okx], minlength=k)
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)

    bq, aq = ctx.grid_bid_qty, ctx.grid_ask_qty
    gimb = np.where((bq + aq) > 0, (bq - aq) / np.maximum(bq + aq, 1), np.nan)
    depth_b = np.stack([gmean(rp.grid_bid_depth[:, j].astype(float)) for j in range(DEPTH_LEVELS)], 1) if len(g) else np.full((k, DEPTH_LEVELS), np.nan)
    depth_a = np.stack([gmean(rp.grid_ask_depth[:, j].astype(float)) for j in range(DEPTH_LEVELS)], 1) if len(g) else np.full((k, DEPTH_LEVELS), np.nan)

    data = {
        't_start_s': (np.r_[ctx.t0, edges[1:-1]] - ctx.t0) / US,
        't_end_s': (edges[1:] - ctx.t0) / US,
        'mid_open': o,
        'mid_high': h,
        'mid_low': lo,
        'mid_close': c,
        'return_ticks': c - o,
        'realized_vol_ticks': rv,
        'spread_mean': gmean(ctx.grid_spread),
        'bid_qty_mean': gmean(bq),
        'ask_qty_mean': gmean(aq),
        'imbalance_mean': gmean(gimb),
        **flows,
    }
    data['net_flow'] = data['buy_volume'] - data['sell_volume']
    data['cancel_insert_ratio'] = np.where(data['inserts'] > 0, data['cancels'] / np.maximum(data['inserts'], 1), np.nan)
    data['bid_depth_mean'] = depth_b
    data['ask_depth_mean'] = depth_a

    rel = {
        'bins': int(k),
        'corr_vol_spread': corr(rv, data['spread_mean']),
        'corr_vol_volume': corr(rv, data['volume'].astype(float)),
        'corr_vol_events': corr(rv, data['events'].astype(float)),
        'corr_absret_volume': corr(np.abs(data['return_ticks']), data['volume'].astype(float)),
        'corr_netflow_return': corr(data['net_flow'].astype(float), data['return_ticks']),
        'corr_netflow_next_return': corr(data['net_flow'][:-1].astype(float), data['return_ticks'][1:]),
        'corr_return_next_return': corr(data['return_ticks'][:-1], data['return_ticks'][1:]),
        'autocorr_vol': autocorr(rv, (1, 2, 5)),
        'autocorr_volume': autocorr(data['volume'].astype(float), (1, 2, 5)),
        'autocorr_spread': autocorr(data['spread_mean'], (1, 2, 5)),
        'autocorr_return': autocorr(data['return_ticks'], (1, 2, 5)),
        'cv_volume': float(np.std(data['volume']) / np.mean(data['volume'])) if np.mean(data['volume']) > 0 else None,
        'cv_realized_vol': float(np.nanstd(rv) / np.nanmean(rv)) if np.nanmean(rv) > 0 else None,
    }
    return {'step_s': step / US, 'rows': int(k), **_columnar(data)}, rel
