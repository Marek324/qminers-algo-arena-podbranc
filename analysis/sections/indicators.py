"""Technical indicators on the mid price, sampled on the series grid (default 1 s), and how well each predicts the next move.

Windows are in seconds and converted to rows of the series grid, so names stay the same whatever --series-step is.
Every indicator here can be updated in O(1) per tick inside a strategy (EMAs, running sums, rolling windows).
`features` are the stationary forms used for prediction (distance from price, oscillator values), in ticks or units.
"""

import numpy as np
import pandas as pd

from ..context import Context
from ..util import US, corr, dist, label_s

FORWARD_S = (1, 5, 30, 60)
MA_S = (10, 30, 60, 300)
N_DECILES = 10


def _rows(sec: float, step_s: float) -> int:
    return max(1, round(sec / step_s))


def _grid(ctx: Context) -> tuple[np.ndarray, pd.DataFrame]:
    """Per row of the series grid: mid OHLC over the row (event time), trade volume, signed volume and notional."""
    rp = ctx.rp
    step = ctx.series_step_us
    k = max(int(np.ceil((ctx.t1 - ctx.t0) / step)), 1)
    edges = ctx.t0 + step * np.arange(k + 1, dtype=np.int64)
    edges[0] = ctx.t0 - 1
    t = edges[1:]
    close = ctx.mid_at(t)

    mid = ctx.mid_t
    ok = np.isfinite(mid)
    b = np.clip(np.searchsorted(edges, rp.ev_ts[ok], side='left') - 1, 0, k - 1)
    m = mid[ok]
    high = close.copy()
    low = close.copy()
    if len(m):
        starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
        ub = b[starts]
        high[ub] = np.maximum(high[ub], np.maximum.reduceat(m, starts))
        low[ub] = np.minimum(low[ub], np.minimum.reduceat(m, starts))
    prev_close = np.r_[close[0], close[:-1]]
    high = np.fmax(high, prev_close)  # a row's range includes where it opened
    low = np.fmin(low, prev_close)

    ti = np.searchsorted(rp.tr_ts, edges, side='right')

    def sums(x):
        c = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
        return np.diff(c[ti])

    qty = rp.tr_qty.astype(float)
    df = pd.DataFrame({
        'close': close,
        'high': high,
        'low': low,
        'volume': sums(qty),
        'signed_volume': sums(qty * rp.tr_sign),
        'notional': sums(qty * rp.tr_price / ctx.tick),
    })
    return t, df


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).where(dn > 0, 100.0).where(up + dn > 0, 50.0)


def _compute(df: pd.DataFrame, step_s: float) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """(series to plot, in price units where applicable) and (stationary features for prediction)."""
    r = lambda s: _rows(s, step_s)  # noqa: E731
    c, h, lo = df['close'], df['high'], df['low']
    out: dict[str, pd.Series] = {}
    feat: dict[str, pd.Series] = {}

    for s in MA_S:
        ema = c.ewm(span=r(s), adjust=False).mean()
        sma = c.rolling(r(s), min_periods=1).mean()
        out[f'ema_{s}s'] = ema
        out[f'sma_{s}s'] = sma
        feat[f'mid_minus_ema_{s}s'] = c - ema
        feat[f'mid_minus_sma_{s}s'] = c - sma
    feat['ema_10s_minus_ema_60s'] = out['ema_10s'] - out['ema_60s']
    feat['ema_30s_minus_ema_300s'] = out['ema_30s'] - out['ema_300s']

    macd = c.ewm(span=r(12), adjust=False).mean() - c.ewm(span=r(26), adjust=False).mean()
    sig = macd.ewm(span=r(9), adjust=False).mean()
    out['macd'], out['macd_signal'], out['macd_hist'] = macd, sig, macd - sig
    feat['macd'], feat['macd_hist'] = macd, macd - sig

    for n in (14, 60):
        out[f'rsi_{n}s'] = _rsi(c, r(n))
        feat[f'rsi_{n}s'] = out[f'rsi_{n}s']

    n = r(60)
    mean, sd = c.rolling(n, min_periods=2).mean(), c.rolling(n, min_periods=2).std()
    out['bb_upper_60s'], out['bb_lower_60s'] = mean + 2 * sd, mean - 2 * sd
    width = (4 * sd).replace(0, np.nan)
    feat['bb_pctb_60s'] = (c - out['bb_lower_60s']) / width
    feat['bb_width_60s'] = 4 * sd
    feat['zscore_60s'] = (c - mean) / sd.replace(0, np.nan)
    out['zscore_60s'] = feat['zscore_60s']

    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / r(14), adjust=False).mean()
    out['atr_14s'] = atr
    feat['atr_14s'] = atr
    ema20 = c.ewm(span=r(20), adjust=False).mean()
    out['keltner_upper_20s'], out['keltner_lower_20s'] = ema20 + 2 * atr, ema20 - 2 * atr
    feat['keltner_pos_20s'] = (c - ema20) / (2 * atr).replace(0, np.nan)

    n = r(14)
    hh, ll = h.rolling(n, min_periods=1).max(), lo.rolling(n, min_periods=1).min()
    rng = (hh - ll).replace(0, np.nan)
    k = 100 * (c - ll) / rng
    out['stoch_k_14s'], out['stoch_d_14s'] = k, k.rolling(r(3), min_periods=1).mean()
    out['williams_r_14s'] = -100 * (hh - c) / rng
    feat['stoch_k_14s'], feat['stoch_d_14s'], feat['williams_r_14s'] = out['stoch_k_14s'], out['stoch_d_14s'], out['williams_r_14s']

    n = r(20)
    tp = (h + lo + c) / 3
    md = tp.rolling(n, min_periods=2).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    out['cci_20s'] = (tp - tp.rolling(n, min_periods=2).mean()) / (0.015 * md.replace(0, np.nan))
    feat['cci_20s'] = out['cci_20s']

    n = r(60)
    out['donchian_high_60s'], out['donchian_low_60s'] = h.rolling(n, min_periods=1).max(), lo.rolling(n, min_periods=1).min()
    feat['donchian_pos_60s'] = (c - out['donchian_low_60s']) / (out['donchian_high_60s'] - out['donchian_low_60s']).replace(0, np.nan)

    for s in (10, 60):
        out[f'roc_{s}s'] = c - c.shift(r(s))
        feat[f'roc_{s}s'] = out[f'roc_{s}s']

    # ADX / DMI
    n = r(14)
    up, dn = h.diff(), -lo.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    atr_n = tr.ewm(alpha=1 / n, adjust=False).mean().replace(0, np.nan)
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_n
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_n
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    out['adx_14s'], out['plus_di_14s'], out['minus_di_14s'] = dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi
    feat['di_diff_14s'] = pdi - mdi
    feat['adx_14s'] = out['adx_14s']

    # Volume-based
    vol, notional = df['volume'], df['notional']
    out['vwap_session'] = notional.cumsum() / vol.cumsum().replace(0, np.nan)
    n = r(60)
    out['vwap_60s'] = notional.rolling(n, min_periods=1).sum() / vol.rolling(n, min_periods=1).sum().replace(0, np.nan)
    feat['mid_minus_vwap_session'] = c - out['vwap_session']
    feat['mid_minus_vwap_60s'] = c - out['vwap_60s']
    out['cvd'] = df['signed_volume'].cumsum()
    for s in (10, 60):
        feat[f'cvd_change_{s}s'] = out['cvd'] - out['cvd'].shift(r(s))
    out['realized_vol_60s'] = c.diff().rolling(r(60), min_periods=2).std()
    feat['realized_vol_60s'] = out['realized_vol_60s']
    return out, feat


def _deciles(x: np.ndarray, y: np.ndarray) -> list[dict]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < N_DECILES * 10:
        return []
    order = np.argsort(x, kind='stable')
    return [
        {'x_lo': float(x[ch].min()), 'x_hi': float(x[ch].max()), 'n': int(len(ch)), 'mean_fwd_ticks': float(y[ch].mean()),
         'up_share': float((y[ch] > 0).mean()), 'down_share': float((y[ch] < 0).mean())}
        for ch in np.array_split(order, N_DECILES)
    ]


def indicators_section(ctx: Context) -> dict:
    step_s = ctx.series_step_us / US
    t, df = _grid(ctx)
    out, feat = _compute(df, step_s)
    close = df['close'].to_numpy()
    warm = _rows(300, step_s)  # skip until the slowest window is filled
    valid = np.arange(len(t)) >= warm

    evaluation = {}
    for name, s in feat.items():
        x = s.to_numpy(dtype=float)
        row: dict = {'feature': dist(x[valid])}
        for h in FORWARD_S:
            fwd = ctx.mid_at(t + h * US) - close
            c = corr(x[valid], fwd[valid])
            xs, ys = x[valid], fwd[valid]
            ok = np.isfinite(xs) & np.isfinite(ys) & (xs != 0) & (ys != 0)
            # Oscillators centred at 50 (RSI, stochastic, %B) are judged on the sign of their distance from the centre.
            if name.startswith(('rsi', 'stoch')):
                xs = xs - 50
            elif name.startswith(('bb_pctb', 'donchian_pos')):
                xs = xs - 0.5
            elif name.startswith('williams'):
                xs = xs + 50
            ok = np.isfinite(xs) & np.isfinite(ys) & (xs != 0) & (ys != 0)
            hit = float((np.sign(xs[ok]) == np.sign(ys[ok])).mean()) if ok.sum() >= 10 else None
            row[label_s(h)] = {**c, 'hit_rate': hit}
        fwd5 = ctx.mid_at(t + 5 * US) - close
        row['deciles_5s'] = _deciles(x[valid], fwd5[valid])
        evaluation[name] = row

    ranking = sorted(
        ({'indicator': k, 'spearman_5s': v['5s']['spearman'], 'spearman_30s': v['30s']['spearman'], 'hit_rate_5s': v['5s']['hit_rate']}
         for k, v in evaluation.items()),
        key=lambda r: -abs(r['spearman_5s'] or 0),
    )
    data = {'t_s': (t - ctx.t0) / US, 'mid': close}
    data.update({k: v.to_numpy(dtype=float) for k, v in out.items()})
    return {
        'note': f'on the {step_s:g} s series grid; windows in seconds; first 300 s skipped for evaluation; '
        'forward move = mid(t+h) - mid(t) in ticks; centred oscillators (RSI, stochastic, %B, Williams) hit rates use distance from centre',
        'samples': int(valid.sum()),
        'evaluation': evaluation,
        'ranking_by_abs_spearman_5s': ranking,
        'series': {'step_s': step_s, 'rows': int(len(t)), 'columns': list(data), 'data': data},
    }
