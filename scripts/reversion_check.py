"""Is there mean reversion, and on what horizon? Impact decay after taker orders, jump follow-through, and the IC of
reversion features (z-score of the mid over 30/60/120 s, momentum over 5/30/60 s) with the future mid move at
5/10/30/60 s, per regime. Round 2 had permanent impact and no reversion; round 3's analysis hints at reversion.

uv run python scripts/reversion_check.py data/round3_public.csv [data/round2_public.csv]
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


def main(path: str) -> None:
    rp = replay(load_tape(Path(path)), 10**9)
    J = detect_jumps(rp)
    reg = detect_regimes(rp, J)
    mid = pd.Series(rp.mid).ffill().bfill().to_numpy()
    mid_at = lambda t: mid[np.clip(np.searchsorted(rp.ev_ts, t, 'right') - 1, 0, len(mid) - 1)]  # noqa: E731
    t0, t1 = rp.ev_ts[0], rp.ev_ts[-1]
    print(f'===== {path}')

    # 1. impact decay by IOC size
    ioc = rp.od_ioc
    it, sd, q = rp.od_ts[ioc], rp.od_side[ioc].astype(float), rp.od_qty[ioc].astype(float)
    ok = (it > t0 + 2 * US) & (it < t1 - 65 * US)
    it, sd, q = it[ok], sd[ok], q[ok]
    base = mid_at(it - 1)
    H = (1, 5, 10, 30, 60)
    mv = {h: sd * (mid_at(it + h * US) - base) for h in H}
    edges = np.quantile(q, [0, .5, .8, .95, 1])
    print('signed mid move after a taker order (ticks), by size; permanent impact = flat across horizons')
    print('  size          ' + ''.join(f'{h:>8}s' for h in H))
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (q >= lo) & (q <= hi)
        print(f'  {lo:>5.0f}-{hi:<7.0f}' + ''.join(f'{mv[h][m].mean():>9.3f}' for h in H))

    # 2. jump follow-through
    jt, js = J['t_us'], J['size']
    m = (jt > t0 + 2 * US) & (jt < t1 - 125 * US)
    jt, js = jt[m], js[m]
    if len(jt):
        b0 = mid_at(jt - 200_000)
        print(f'\njumps (>= 5 ticks / 100 ms): {len(jt)}; signed move from 0.2 s before, as a share of the jump size:')
        for h in (1, 5, 30, 60, 120):
            sh = np.sign(js) * (mid_at(jt + h * US) - b0) / np.abs(js)
            print(f'  +{h:>3} s: median {np.median(sh):.2f}, mean {np.mean(sh):.2f}')

    # 3. reversion features vs future moves, 1 s grid
    g = np.arange(t0 + 130 * US, t1 - 65 * US, US)
    m0 = mid_at(g)
    S = {}
    for w in (30, 60, 120):
        past = np.stack([mid_at(g - k * US) for k in range(0, w, 2)])
        S[f'zscore_{w}s'] = (m0 - past.mean(0)) / (past.std(0) + 0.5)
    for w in (5, 30, 60):
        S[f'mom_{w}s'] = m0 - mid_at(g - w * US)
    storm = np.zeros(len(g), bool)
    if reg:
        sp = np.array(reg['storm_prob'])
        storm = sp[np.clip(((g - t0) // (REGIME_WINDOW_S * US)).astype(int), 0, len(sp) - 1)] > 0.5
    print('\nSpearman IC with the future mid move (negative = reversion):')
    print(f'  {"feature":<12}' + ''.join(f'{f"{h}s all":>10}' for h in (5, 10, 30, 60)) + f'{"30s calm":>10}{"30s storm":>10}')
    for name, x in S.items():
        row = [spearmanr(x, mid_at(g + h * US) - m0).statistic for h in (5, 10, 30, 60)]
        y30 = mid_at(g + 30 * US) - m0
        c = spearmanr(x[~storm], y30[~storm]).statistic if (~storm).sum() > 100 else np.nan
        s_ = spearmanr(x[storm], y30[storm]).statistic if storm.sum() > 100 else np.nan
        print(f'  {name:<12}' + ''.join(f'{v:>10.3f}' for v in row) + f'{c:>10.3f}{s_:>10.3f}')
    # variance ratios
    r1 = np.diff(mid_at(np.arange(t0, t1, US)))
    for k in (5, 30, 60, 300):
        rk = mid_at(np.arange(t0 + k * US, t1, k * US)) - mid_at(np.arange(t0, t1 - k * US, k * US))
        print(f'  variance ratio {k:>3} s / 1 s: {rk.var() / (k * r1.var()):.2f}')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        main(p)
