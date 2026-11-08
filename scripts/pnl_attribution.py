"""Where does a strategy earn and leak? Attribution of a full-logs run by regime, fill type and holding horizon.

usage: .venv/bin/python scripts/pnl_attribution.py logs/<full-logs run> data/<tape>.csv

Per own fill: passive (we were the maker) or taking, the 1 s realized volatility regime around it (calm / storm by the
tape's 1 s |mid move| over the surrounding 30 s against its median), and the markout of the fill against the mid
after 0.1 / 1 / 5 / 30 s (ticks x lots, signed so + is good). Also the run's PnL earned in calm vs storm seconds
(from pnl.csv) and the hindsight-best PnL per regime (max of |30 s mid move| x 100 lots, a crude ceiling).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.replay import replay  # noqa: E402
from analysis.tape import load_tape  # noqa: E402

US = 1_000_000
HORIZONS = (0.1, 1, 5, 30)


def main() -> None:
    run, tape_path = Path(sys.argv[1]), Path(sys.argv[2])
    rp = replay(load_tape(tape_path), grid_step_us=US)
    ts = rp.ev_ts
    mid = (rp.best_bid + rp.best_ask) / 2
    ok = np.isfinite(mid)
    ts, mid = ts[ok], mid[ok]

    def mid_at(t):
        i = np.clip(np.searchsorted(ts, t, 'right') - 1, 0, len(ts) - 1)
        return mid[i]

    t0, t1 = ts[0], ts[-1]
    sec = np.arange(t0, t1, US)
    m1 = mid_at(sec)
    dm = np.abs(np.diff(m1, prepend=m1[0]))
    vol30 = pd.Series(dm).rolling(30, center=True, min_periods=1).mean().to_numpy()
    thr = 2.0 * np.median(vol30)
    storm = vol30 > thr
    print(f'storm seconds (30 s mean |1 s move| > 2x median {thr:.2f}): {storm.mean():.1%}')

    tr = pd.read_csv(run / 'trades.csv')
    own = tr[(tr.taker_owner == 'user') | (tr.maker_owner == 'user')].copy()
    taker_buys = own.taker_side == 'B'   # B: the taker bought (checked against result.json cash)
    own['taking'] = own.taker_owner == 'user'
    own['sign'] = np.where(own.taking, np.where(taker_buys, 1, -1), np.where(taker_buys, -1, 1))
    t = own.timestamp_us.to_numpy()
    si = np.clip(((t - t0) // US).astype(int), 0, len(storm) - 1)
    own['storm'] = storm[si]
    for h in HORIZONS:
        own[f'mo_{h}'] = own.sign * (mid_at(t + int(h * US)) - own.price) * own.quantity
    own['lots'] = own.quantity
    g = own.groupby(['storm', 'taking'])
    out = g[['lots'] + [f'mo_{h}' for h in HORIZONS]].sum()
    out['fills'] = g.size()
    for h in HORIZONS:
        out[f'tick/lot {h}s'] = out[f'mo_{h}'] / out['lots']
    pd.set_option('display.width', 200)
    print('\nmarkout of own fills (ticks x lots, + = good), by regime and fill type:')
    print(out.round(2).to_string())

    # decision types. Taking: level 0 = best price of that taker order, deeper = sweep levels behind it.
    # Passive: the fill price vs the mid at fill time (half a tick = at the touch, more = deeper / improved).
    own = own.sort_values(['timestamp_us']).reset_index(drop=True)
    tk = own.taking
    first_px = own[tk].groupby('taker_order_id').price.transform(lambda p: p.iloc[0])
    own.loc[tk, 'level'] = (own.loc[tk, 'price'] - first_px).abs()
    own.loc[tk, 'kind'] = np.where(own.loc[tk, 'level'] == 0, 'take_L0', np.where(own.loc[tk, 'level'] <= 2, 'take_L1-2', 'take_L3+'))
    m_fill = mid_at(own.timestamp_us.to_numpy() - 1)
    dist = own.sign * (m_fill - own.price)   # + = we bought below / sold above the mid
    own.loc[~tk, 'kind'] = np.where(dist[~tk] <= 0.5, 'pass_touch', 'pass_deep')
    inv = np.cumsum(own.sign * own.quantity)
    own['inv_after'] = inv
    own['to_cap'] = np.abs(inv) >= 100
    print('\nby decision type (ticks x lots at 5 s / 30 s; per lot):')
    g = own.groupby(['storm', 'kind'])
    t2 = g[['lots', 'mo_1', 'mo_5', 'mo_30']].sum()
    t2['fills'] = g.size()
    t2['t/lot 1s'] = t2.mo_1 / t2.lots
    t2['t/lot 5s'] = t2.mo_5 / t2.lots
    t2['t/lot 30s'] = t2.mo_30 / t2.lots
    print(t2.round(2).to_string())
    print('\nfills that put us at the +-100 cap vs not (5 s markout per lot):')
    g = own.groupby(['storm', 'taking', 'to_cap'])
    t3 = g[['lots', 'mo_5']].sum()
    t3['t/lot 5s'] = t3.mo_5 / t3.lots
    print(t3.round(2).to_string())
    # worst 1-minute windows of the run
    pnl = pd.read_csv(run / 'pnl.csv')
    pcol = [c for c in pnl.columns if 'pnl' in c.lower()][0]
    tcol = pnl.columns[0]
    p = pnl[pcol].to_numpy()
    pt = pnl[tcol].to_numpy() * US + t0 if tcol == 'elapsed_s' else pnl[tcol].to_numpy()
    dp = np.diff(p)
    pi = np.clip(((pt[1:] - t0) // US).astype(int), 0, len(storm) - 1)
    print(f'\nrun PnL from {tcol}/{pcol}: total {p[-1]:,.0f}; earned in storm seconds {dp[storm[pi]].sum():,.0f} '
          f'({storm[pi].mean():.1%} of samples), calm {dp[~storm[pi]].sum():,.0f}')
    w = 30
    mv = np.abs(m1[w:] - m1[:-w])
    ceil = mv[::w] * 100
    st = storm[w::w][:len(ceil)]
    print(f'crude directional ceiling (100 lots, |30 s move|, non-overlapping): storm {ceil[st].sum():,.0f} calm {ceil[~st].sum():,.0f}')
    # per-hour of the tape
    hr = ((pt[1:] - t0) // (600 * US)).astype(int)
    print('\nPnL per 10 min block:', ' '.join(f'{dp[hr == k].sum() / 1000:+.0f}k' for k in range(hr.max() + 1)))
    mn = ((pt[1:] - t0) // (60 * US)).astype(int)
    per_min = pd.Series(dp).groupby(mn).sum()
    worst = per_min.nsmallest(10)
    print('worst 10 minutes (start s: PnL, storm share):', ', '.join(
        f'{k * 60}: {v / 1000:+.1f}k ({storm[k * 60:(k + 1) * 60].mean():.0%})' for k, v in worst.items()))
    print(f'minutes negative: {(per_min < 0).mean():.0%}; sum of negative minutes {per_min[per_min < 0].sum():,.0f}')


if __name__ == '__main__':
    main()
