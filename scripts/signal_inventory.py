"""Inventory of short-horizon signals: every candidate computed causally on a 50 ms grid, scored by Spearman IC against
the future mid move at 100 ms / 500 ms / 1 s / 5 s, overall and per regime (calm / storm from
estimate_model.detect_regimes), plus predictors of the size of the next move (|move| 1 s).

uv run python scripts/signal_inventory.py data/round2_public.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from analysis.replay import replay  # noqa: E402
from analysis.tape import load_tape  # noqa: E402
from estimate_model import REGIME_WINDOW_S, detect_jumps, detect_regimes  # noqa: E402

US = 1_000_000
STEP = 50_000
HORIZONS = (100_000, 500_000, 1_000_000, 5_000_000)


def window_sum(ts, w, g, width):
    """Sum of w over events with ts in (g - width, g]."""
    cs = np.r_[0.0, np.cumsum(w)]
    return cs[np.searchsorted(ts, g, 'right')] - cs[np.searchsorted(ts, g - width, 'right')]


def at(ev_ts, arr, g):
    return arr[np.maximum(np.searchsorted(ev_ts, g, 'right') - 1, 0)]


def main(path: str) -> None:
    rp = replay(load_tape(Path(path)), 10**9)
    jumps = detect_jumps(rp)
    reg = detect_regimes(rp, jumps)
    t0, t1 = rp.ev_ts[0], rp.ev_ts[-1]
    g = np.arange(t0 + 10 * US, t1 - 6 * US, STEP)
    mid = pd.Series(rp.mid).ffill().bfill().to_numpy()
    m0 = at(rp.ev_ts, mid, g)
    bid, ask = at(rp.ev_ts, rp.best_bid, g), at(rp.ev_ts, rp.best_ask, g)
    bq, aq = at(rp.ev_ts, rp.bid_qty, g).astype(float), at(rp.ev_ts, rp.ask_qty, g).astype(float)
    spread = np.nan_to_num(ask - bid, nan=1.0)

    lim, ioc = ~rp.od_ioc, rp.od_ioc
    lt, lp, ls = rp.od_ts[lim], rp.od_price[lim].astype(float), rp.od_side[lim].astype(float)
    gap = rp.od_cross_gap
    it, iside, iq = rp.od_ts[ioc], rp.od_side[ioc].astype(float), rp.od_qty[ioc].astype(float)
    igap = gap[ioc]

    S = {}
    # 1. hidden price: mean of new limit prices per side over the last 300 ms, averaged, minus the mid
    for name, width, sel in (('hp_delta_300ms', 300_000, np.ones(len(lt), bool)),
                             ('hp_delta_near_300ms', 300_000, np.abs(rp.od_rel_touch[lim]) <= 2),
                             ('hp_delta_1s', US, np.ones(len(lt), bool))):
        b = (ls > 0) & sel
        a = (ls < 0) & sel
        nb, na = window_sum(lt[b], np.ones(b.sum()), g, width), window_sum(lt[a], np.ones(a.sum()), g, width)
        mb = window_sum(lt[b], lp[b], g, width) / np.maximum(nb, 1)
        ma = window_sum(lt[a], lp[a], g, width) / np.maximum(na, 1)
        ok = (nb > 0) & (na > 0)
        S[name] = np.where(ok, (mb + ma) / 2 - m0, 0.0)
    # 2. far orders: buy placed far below the touch = S fell (negative), sell far above = S rose
    far = lim & np.isfinite(gap) & (gap <= -7)
    fdepth = (-gap[far] - 1) * -rp.od_side[far]
    S['far_signed_500ms'] = window_sum(rp.od_ts[far], fdepth, g, 500_000)
    S['far_signed_2s'] = window_sum(rp.od_ts[far], fdepth, g, 2 * US)
    # 3. trades
    k = np.searchsorted(rp.tr_ts, g, 'right') - 1
    S['last_trade_sign'] = np.where(k >= 0, rp.tr_sign[np.maximum(k, 0)], 0).astype(float)
    S['trade_flow_100ms'] = window_sum(rp.tr_ts, rp.tr_sign * rp.tr_qty.astype(float), g, 100_000)
    S['trade_flow_1s'] = window_sum(rp.tr_ts, rp.tr_sign * rp.tr_qty.astype(float), g, US)
    # 4. IOCs: deep sweeps (priced >= 2 through the opposite best, so S is beyond it), predicted permanent impact
    deep = np.isfinite(igap) & (igap >= 2)
    S['ioc_deep_signed_500ms'] = window_sum(it[deep], iside[deep], g, 500_000)
    S['ioc_impact_2s'] = window_sum(it, iside * 0.0158 * iq**0.61, g, 2 * US)
    S['ioc_net_qty_1s'] = window_sum(it, iside * iq, g, US)
    # 5. book
    S['imbalance'] = np.where(bq + aq > 0, (bq - aq) / np.maximum(bq + aq, 1), 0.0)
    S['microprice_off'] = S['imbalance'] * spread / 2
    # 6. momentum
    S['mom_1s'] = m0 - at(rp.ev_ts, mid, g - US)
    S['mom_5s'] = m0 - at(rp.ev_ts, mid, g - 5 * US)
    S['mom_100ms'] = m0 - at(rp.ev_ts, mid, g - 100_000)

    F = pd.DataFrame(S)
    tgt = {h: at(rp.ev_ts, mid, g + h) - m0 for h in HORIZONS}
    storm = np.zeros(len(g), bool)
    if reg:
        sp = np.array(reg['storm_prob'])
        storm = sp[np.clip(((g - t0) // (REGIME_WINDOW_S * US)).astype(int), 0, len(sp) - 1)] > 0.5
    print(f'{path}: {len(g):,} grid points (50 ms), storm share {storm.mean():.2f}')

    rows = []
    for c in F.columns:
        x = F[c].to_numpy()
        r = {'signal': c, 'nonzero': float((x != 0).mean())}
        for h in HORIZONS:
            y = tgt[h]
            for name, m in (('all', np.ones(len(g), bool)), ('calm', ~storm), ('storm', storm)):
                if m.sum() > 1000:
                    r[f'{name} {h // 1000}ms'] = spearmanr(x[m], y[m]).statistic
        rows.append(r)
    T = pd.DataFrame(rows)
    cols = ['signal', 'nonzero'] + [f'all {h // 1000}ms' for h in HORIZONS] + [f'calm {h // 1000}ms' for h in (500_000, 1_000_000)] + \
           [f'storm {h // 1000}ms' for h in (500_000, 1_000_000)]
    print('\nSpearman IC with the future mid move (sign: + = predicts the direction):')
    print(T[cols].sort_values('all 1000ms', key=np.abs, ascending=False).round(3).to_string(index=False))

    # combined linear model, time split
    X = F.to_numpy()
    y = tgt[1_000_000]
    cut = int(0.6 * len(g))
    Xs = (X - X[:cut].mean(0)) / (X[:cut].std(0) + 1e-9)
    w, *_ = np.linalg.lstsq(np.c_[np.ones(cut), Xs[:cut]], y[:cut], rcond=None)
    pred = np.c_[np.ones(len(g) - cut), Xs[cut:]] @ w
    print(f'\nall signals, linear, 1 s target: out-of-sample IC {spearmanr(pred, y[cut:]).statistic:.3f}, '
          f'R2 {1 - np.var(y[cut:] - pred) / np.var(y[cut:]):.3f}')
    print('  weights (standardized):', {c: round(float(v), 3) for c, v in zip(F.columns, w[1:])})

    # size of the next move
    av = np.abs(tgt[1_000_000])
    vol = {
        'spread': spread,
        'far_count_2s': window_sum(rp.od_ts[far], np.ones(far.sum()), g, 2 * US),
        'ioc_count_2s': window_sum(it, np.ones(len(it)), g, 2 * US),
        'abs_mom_1s': np.abs(S['mom_1s']),
        'touch_qty': bq + aq,
        'limits_2s': window_sum(lt, np.ones(len(lt)), g, 2 * US),
    }
    print('\nSpearman with |mid move over 1 s| (size of the next move):')
    for c, x in vol.items():
        print(f'  {c:<14} {spearmanr(x, av).statistic:+.3f}')


if __name__ == '__main__':
    main(sys.argv[1])
