"""Does anything observable predict the next mid move? Features vs forward returns on the 100 ms grid."""

import numpy as np

from ..context import Context
from ..util import US, corr, dist, label_s, ols

FORWARD_HORIZONS_S = (0.1, 1, 5, 30)
DECILE_HORIZONS_S = (1, 5)
DIRECTIONAL = (
    'imbalance_touch',
    'imbalance_depth5',
    'imbalance_depth20',
    'microprice_offset',
    'flow_1s',
    'flow_10s',
    'flow_60s',
    'past_return_1s',
    'past_return_10s',
    'past_return_60s',
    'last_trade_sign',
)
ACTIVITY = ('spread', 'touch_qty_total', 'trades_10s', 'volume_10s', 'abs_past_return_10s')
N_DECILES = 10


def _features(ctx: Context) -> tuple[np.ndarray, dict[str, np.ndarray], dict[float, np.ndarray]]:
    rp = ctx.rp
    g = rp.grid_ts
    mid = ctx.grid_mid
    bq, aq = ctx.grid_bid_qty, ctx.grid_ask_qty
    gb, ga = rp.grid_bid_depth.astype(float), rp.grid_ask_depth.astype(float)
    spread = ctx.grid_spread

    def imb(b, a):
        return np.where((b + a) > 0, (b - a) / np.maximum(b + a, 1), np.nan)

    def past(h):
        return mid - ctx.mid_at(g - int(h * US))

    def flow(h):
        return (ctx.cum_signed_volume[ctx.trades_upto(g)] - ctx.cum_signed_volume[ctx.trades_upto(g - int(h * US))]).astype(float)

    n_now = ctx.trades_upto(g)
    n_10 = ctx.trades_upto(g - 10 * US)
    last_sign = np.zeros(len(g))
    has = n_now > 0
    last_sign[has] = rp.tr_sign[n_now[has] - 1]

    bid_px = mid - spread / 2
    ask_px = mid + spread / 2
    micro = np.where((bq + aq) > 0, (bid_px * aq + ask_px * bq) / np.maximum(bq + aq, 1), np.nan) - mid

    p10 = past(10)
    feats = {
        'imbalance_touch': imb(bq, aq),
        'imbalance_depth5': imb(gb[:, :5].sum(1), ga[:, :5].sum(1)),
        'imbalance_depth20': imb(gb.sum(1), ga.sum(1)),
        'microprice_offset': micro,
        'flow_1s': flow(1),
        'flow_10s': flow(10),
        'flow_60s': flow(60),
        'past_return_1s': past(1),
        'past_return_10s': p10,
        'past_return_60s': past(60),
        'last_trade_sign': last_sign,
        'spread': spread,
        'touch_qty_total': bq + aq,
        'trades_10s': (n_now - n_10).astype(float),
        'volume_10s': (ctx.cum_volume[n_now] - ctx.cum_volume[n_10]).astype(float),
        'abs_past_return_10s': np.abs(p10),
    }
    fwd = {h: ctx.mid_at(g + int(h * US)) - mid for h in FORWARD_HORIZONS_S}
    valid = np.isfinite(mid) & (g - ctx.t0 >= 60 * US)
    return valid, feats, fwd


def _deciles(x: np.ndarray, y: np.ndarray) -> list[dict]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < N_DECILES * 10:
        return []
    order = np.argsort(x, kind='stable')
    out = []
    for chunk in np.array_split(order, N_DECILES):
        xs, ys = x[chunk], y[chunk]
        out.append(
            {
                'x_lo': float(xs.min()),
                'x_hi': float(xs.max()),
                'n': int(len(chunk)),
                'mean_fwd_ticks': float(ys.mean()),
                'up_share': float((ys > 0).mean()),
                'down_share': float((ys < 0).mean()),
            }
        )
    return out


def _hit_rate(x: np.ndarray, y: np.ndarray) -> float | None:
    ok = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
    return float((np.sign(x[ok]) == np.sign(y[ok])).mean()) if ok.sum() >= 10 else None


def signals_section(ctx: Context) -> dict:
    valid, feats, fwd = _features(ctx)
    feats = {k: v[valid] for k, v in feats.items()}
    fwd = {h: v[valid] for h, v in fwd.items()}

    directional = {}
    for name in DIRECTIONAL:
        x = feats[name]
        row = {'feature': dist(x)}
        for h in FORWARD_HORIZONS_S:
            c = corr(x, fwd[h])
            row[label_s(h)] = {**c, 'hit_rate': _hit_rate(x, fwd[h])}
        row['deciles'] = {label_s(h): _deciles(x, fwd[h]) for h in DECILE_HORIZONS_S}
        directional[name] = row

    activity = {}
    for name in ACTIVITY:
        x = feats[name]
        row = {'feature': dist(x)}
        for h in FORWARD_HORIZONS_S:
            row[label_s(h)] = corr(x, np.abs(fwd[h]))
        activity[name] = row

    # Linear combination of directional features, in-sample and train-first-half / test-second-half.
    X = np.column_stack([feats[n] for n in DIRECTIONAL]) if len(feats['spread']) else np.empty((0, len(DIRECTIONAL)))
    combined = {}
    for h in DECILE_HORIZONS_S:
        y = fwd[h]
        fit = ols(X, y)
        half = len(y) // 2
        train = ols(X[:half], y[:half])
        oos = None
        if train['intercept'] is not None:
            Xt, yt = X[half:], y[half:]
            ok = np.isfinite(yt) & np.all(np.isfinite(Xt), axis=1)
            pred = train['intercept'] + Xt[ok] @ np.array(train['coef'])
            sst = float(((yt[ok] - yt[ok].mean()) ** 2).sum())
            oos = 1 - float(((yt[ok] - pred) ** 2).sum()) / sst if sst > 0 else None
        combined[label_s(h)] = {
            'features': list(DIRECTIONAL),
            'in_sample': fit,
            'r2_out_of_sample': oos,
        }

    fwd_summary = {label_s(h): dist(fwd[h]) for h in FORWARD_HORIZONS_S}
    return {
        'note': 'grid every 100 ms, first 60 s skipped; forward return = mid(t+h) - mid(t) in ticks; overlapping samples',
        'samples': int(valid.sum()),
        'forward_returns': fwd_summary,
        'directional': directional,
        'activity_vs_abs_move': activity,
        'linear_combination': combined,
    }
