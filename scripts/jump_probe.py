"""What is a jump? Prints 100 ms bins around the largest mid moves of a tape: mid, mean new-limit price per side
(the side away from the move is not clamped, so it shows S), IOC count and deepest IOC price through the touch.

uv run python scripts/jump_probe.py data/round2_public.csv [n_episodes]
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.replay import replay  # noqa: E402
from analysis.tape import load_tape  # noqa: E402

US = 1_000_000


def main(path: str, n: int = 6) -> None:
    rp = replay(load_tape(Path(path)), 10**9)
    ev_ts, mid = rp.ev_ts, rp.mid
    # 1 s mid grid to locate episodes
    grid = np.arange(ev_ts[0], ev_ts[-1], US)
    m1 = mid[np.maximum(np.searchsorted(ev_ts, grid, 'right') - 1, 0)]
    r = np.abs(np.diff(m1))
    picked = []
    for i in np.argsort(-r):
        if all(abs(i - j) > 20 for j in picked):
            picked.append(i)
        if len(picked) == n:
            break
    lim, ioc = ~rp.od_ioc, rp.od_ioc
    for i in sorted(picked):
        t0 = grid[i] - 2 * US
        print(f'\n=== episode at {grid[i] / US:.0f}s, 1 s move {m1[i + 1] - m1[i]:+.1f}')
        print('   t(s)     mid   newbuy  newsell  nIOC  maxthru  ioc_net')
        for k in range(0, 60):
            a, b = t0 + k * 100_000, t0 + (k + 1) * 100_000
            mm = mid[max(np.searchsorted(ev_ts, b, 'right') - 1, 0)]
            sel = (rp.od_ts >= a) & (rp.od_ts < b)
            bp = rp.od_price[sel & lim & (rp.od_side == 1)]
            sp = rp.od_price[sel & lim & (rp.od_side == -1)]
            ii = sel & ioc
            gap = rp.od_cross_gap[ii]
            gap = gap[np.isfinite(gap)]
            net = int((rp.od_side[ii] * rp.od_qty[ii]).sum())
            print(f'{(a - grid[i]) / US:+6.1f} {mm:8.1f} {bp.mean() if len(bp) else np.nan:8.1f} '
                  f'{sp.mean() if len(sp) else np.nan:8.1f} {ii.sum():5d} {gap.max() if len(gap) else np.nan:8.0f} {net:8d}')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
