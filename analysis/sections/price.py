"""Price behaviour: returns, volatility across horizons, mean reversion vs trend."""

import numpy as np

from ..context import Context
from ..util import US, autocorr, categorical_shares, dist, label_s, moments, safe_div

RETURN_HORIZONS_S = (1, 5, 10, 30, 60, 300)


def _grid_mid(ctx: Context, step_s: float) -> np.ndarray:
    step = int(step_s * US)
    if ctx.t1 <= ctx.t0:
        return np.array([])
    t = np.arange(ctx.t0, ctx.t1 + 1, step, dtype=np.int64)
    return ctx.mid_at(t)


def price_section(ctx: Context) -> dict:
    rp = ctx.rp
    mid = ctx.mid_t
    fin = np.isfinite(mid)
    mid_f = mid[fin]

    returns = {}
    var_by_h = {}
    for h in RETURN_HORIZONS_S:
        m = _grid_mid(ctx, h)
        r = np.diff(m)
        r = r[np.isfinite(r)]
        mo = moments(r)
        returns[label_s(h)] = {
            **mo,
            'abs_mean': float(np.abs(r).mean()) if len(r) else None,
            'zero_share': float((r == 0).mean()) if len(r) else None,
            'up_share': float((r > 0).mean()) if len(r) else None,
            'quantiles': dist(r),
            'autocorr': autocorr(r, (1, 2, 3)),
        }
        var_by_h[h] = float(r.var(ddof=1)) if len(r) > 2 else np.nan

    v1 = var_by_h[1]
    variance_ratio = {label_s(h): safe_div(var_by_h[h], h * v1) if np.isfinite(v1) and v1 > 0 else None for h in RETURN_HORIZONS_S}
    vol_per_sqrt_s = {label_s(h): float(np.sqrt(var_by_h[h] / h)) if np.isfinite(var_by_h[h]) else None for h in RETURN_HORIZONS_S}
    hs = np.array([h for h in RETURN_HORIZONS_S if np.isfinite(var_by_h[h]) and var_by_h[h] > 0], dtype=float)
    hurst = None
    if len(hs) >= 3:
        stds = np.sqrt([var_by_h[int(h)] for h in hs])
        hurst = float(np.polyfit(np.log(hs), np.log(stds), 1)[0])

    # Event-time mid changes.
    dm = np.diff(mid_f)
    dm = dm[dm != 0]
    t_mid = rp.ev_ts[fin]
    ch_ts = t_mid[1:][np.diff(mid_f) != 0]
    s = np.sign(dm)
    reversal = float((s[1:] != s[:-1]).mean()) if len(s) > 1 else None

    # Trend & drawdown on the 1 s grid.
    m1 = _grid_mid(ctx, 1)
    m1 = m1[np.isfinite(m1)]
    dd = float((np.maximum.accumulate(m1) - m1).max()) if len(m1) else None
    ru = float((m1 - np.minimum.accumulate(m1)).max()) if len(m1) else None
    r60 = np.diff(_grid_mid(ctx, 60))
    r60 = r60[np.isfinite(r60)]
    t60 = float(r60.mean() / r60.std(ddof=1) * np.sqrt(len(r60))) if len(r60) > 2 and r60.std() > 0 else None
    r1 = np.diff(m1)
    sd1 = r1.std() if len(r1) > 2 else 0.0

    # Trade prices.
    tp = rp.tr_price / ctx.tick
    dp = np.diff(tp)
    roll = None
    if len(dp) > 10:
        c = float(np.cov(dp[1:], dp[:-1])[0, 1])
        roll = float(2 * np.sqrt(-c)) if c < 0 else 0.0
    w = ctx.dur_us.astype(float)

    hours = ctx.duration_s / 3600 if ctx.duration_s else None
    return {
        'mid': {
            'first': float(mid_f[0]) if len(mid_f) else None,
            'last': float(mid_f[-1]) if len(mid_f) else None,
            'min': float(mid_f.min()) if len(mid_f) else None,
            'max': float(mid_f.max()) if len(mid_f) else None,
            'change_ticks': float(mid_f[-1] - mid_f[0]) if len(mid_f) else None,
            'range_ticks': float(mid_f.max() - mid_f.min()) if len(mid_f) else None,
            'twap': float(np.average(mid[fin], weights=w[fin])) if fin.any() and w[fin].sum() > 0 else None,
        },
        'trade_vwap': float((tp * rp.tr_qty).sum() / rp.tr_qty.sum()) if len(tp) else None,
        'last_trade_price': float(tp[-1]) if len(tp) else None,
        'max_drawdown_ticks': dd,
        'max_runup_ticks': ru,
        'trend': {
            'drift_ticks_per_hour': safe_div(float(mid_f[-1] - mid_f[0]), hours) if len(mid_f) and hours else None,
            't_stat_60s_returns': t60,
            'up_share_60s': float((r60 > 0).mean()) if len(r60) else None,
        },
        'returns_ticks': returns,
        'vol_ticks_per_sqrt_s': vol_per_sqrt_s,
        'variance_ratio_vs_1s': variance_ratio,
        'hurst_estimate': hurst,
        'jumps_1s': {
            'count_gt_5sd': int((np.abs(r1) > 5 * sd1).sum()) if sd1 > 0 else 0,
            'largest_abs_ticks': float(np.abs(r1).max()) if len(r1) else None,
        },
        'event_time': {
            'mid_changes': int(len(dm)),
            'mid_change_size_share': categorical_shares(np.minimum(np.abs(dm), 2.5), [0.5, 1.0, 1.5, 2.0, 2.5]),
            'mid_change_size_note': 'keys are |change| in ticks; 2.5 means >= 2.5',
            'autocorr_nonzero_changes': autocorr(dm, (1, 2, 3, 4, 5)),
            'reversal_prob': reversal,
            'time_between_mid_changes_ms': dist(np.diff(ch_ts) / 1000 if len(ch_ts) > 1 else []),
        },
        'trade_price': {
            'roll_spread_ticks': roll,
            'autocorr_changes': autocorr(dp[dp != 0], (1, 2, 3)) if len(dp) else autocorr(np.array([]), (1, 2, 3)),
        },
    }
