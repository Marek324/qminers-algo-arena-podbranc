"""Where does a strategy lose? The explorer's "click a PnL drop", done in bulk from a full-logs run.

1. The biggest 30 s PnL drops (non-overlapping): mid move, our inventory before, our fills in the window (passive /
   taking, buys / sells), jumps and the regime.
2. PnL by regime (calm / storm seconds, estimate_model.detect_regimes).
3. Markout of every fill of ours at 1 s / 5 s / 30 s (ticks per lot, + = good), by passive / taking and by regime.

uv run python scripts/pnl_drops.py logs/<run> data/<tape>.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from analysis.replay import replay  # noqa: E402
from analysis.tape import load_tape  # noqa: E402
from estimate_model import REGIME_WINDOW_S, detect_jumps, detect_regimes  # noqa: E402

US = 1_000_000
WIN = 30


def main(run: str, tape: str) -> None:
    run = Path(run)
    pnl = pd.read_csv(run / 'pnl.csv')
    tr = pd.read_csv(run / 'trades.csv')
    rp = replay(load_tape(Path(tape)), 10**9)
    jumps = detect_jumps(rp)
    reg = detect_regimes(rp, jumps)
    t0 = rp.ev_ts[0]
    mid = pd.Series(rp.mid).ffill().bfill().to_numpy()

    def mid_at(ts):
        return mid[np.clip(np.searchsorted(rp.ev_ts, ts, 'right') - 1, 0, len(mid) - 1)]

    owners = set(tr.taker_owner) | set(tr.maker_owner)
    owners.discard('market')
    me = owners.pop() if owners else None
    ours = tr[(tr.taker_owner == me) | (tr.maker_owner == me)].copy()
    ours['taking'] = ours.taker_owner == me
    # our side: if we took, our side is the taker side (B buys); if we made, the opposite
    taker_buy = ours.taker_side.astype(str).str.upper().str.startswith('B')
    ours['sign'] = np.where(ours.taking, np.where(taker_buy, 1, -1), np.where(taker_buy, -1, 1))
    ours['inv'] = (ours.sign * ours.quantity).cumsum()
    ts = ours.timestamp_us.to_numpy()
    storm_prob = np.array(reg['storm_prob']) if reg else np.zeros(1)
    storm_at = lambda t: storm_prob[np.clip(((t - t0) // (REGIME_WINDOW_S * US)).astype(int), 0, len(storm_prob) - 1)] > 0.5  # noqa: E731
    ours['storm'] = storm_at(ts)
    print(f'{run}: {len(ours):,} fills of ours ({me}), final pnl {pnl.pnl.iloc[-1]:,.0f}; {len(jumps["t_us"])} jumps')

    # 1. biggest drops
    p = pnl.set_index('elapsed_s').pnl
    p = p[~p.index.duplicated()].reindex(np.arange(0, int(p.index.max()) + 1), method='ffill')
    d = (p.shift(-WIN) - p).dropna()
    picked = []
    for t in d.sort_values().index:
        if all(abs(t - u) > WIN for u in picked):
            picked.append(t)
        if len(picked) == 10:
            break
    jt = (jumps['t_us'] - t0) / US
    rows = []
    for t in sorted(picked):
        a, b = t0 + int(t * US), t0 + int((t + WIN) * US)
        w = ours[(ours.timestamp_us >= a) & (ours.timestamp_us < b)]
        before = ours[ours.timestamp_us < a]
        inv0 = int(before.inv.iloc[-1]) if len(before) else 0
        jw = (jt >= t - 2) & (jt < t + WIN)
        rows.append({'t_s': int(t), 'pnl_drop': d[t], 'mid_move': mid_at(b) - mid_at(a), 'inv_before': inv0,
                     'fills': len(w), 'took_lots': int(w[w.taking].quantity.sum()), 'made_lots': int(w[~w.taking].quantity.sum()),
                     'net_lots': int((w.sign * w.quantity).sum()), 'jumps': int(jw.sum()),
                     'jump_ticks': float(np.abs(jumps['size'][jw]).sum()), 'storm': bool(storm_at(np.array([a]))[0])})
    print(f'\nbiggest {WIN} s PnL drops:')
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    loss = sum(r['pnl_drop'] for r in rows)
    print(f'  sum of these 10 drops {loss:,.0f}; held inventory x mid move explains '
          f'{sum(r["inv_before"] * r["mid_move"] for r in rows):,.0f}')

    # 2. PnL by regime
    secs = np.arange(len(p) - 1)
    dp = np.diff(p.to_numpy())
    st = storm_at(t0 + secs * US)
    print(f'\nPnL in calm seconds {dp[~st].sum():,.0f} ({(~st).mean():.0%} of time), in storm seconds {dp[st].sum():,.0f} ({st.mean():.0%})')

    # 3. markouts
    px = ours.price.to_numpy().astype(float)
    sg = ours.sign.to_numpy()
    for h in (1, 5, 30):
        ours[f'mo{h}s'] = sg * (mid_at(ts + h * US) - px)
    print('\nmarkout per lot (ticks, + = good), lots-weighted:')
    g = ours.groupby(['storm', 'taking'])
    out = g.apply(lambda x: pd.Series({'lots': x.quantity.sum(), **{f'mo{h}s': np.average(x[f'mo{h}s'], weights=x.quantity) for h in (1, 5, 30)},
                                       'pnl_30s': (x[f'mo30s'] * x.quantity).sum()}), include_groups=False)
    print(out.round(3).to_string())


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
