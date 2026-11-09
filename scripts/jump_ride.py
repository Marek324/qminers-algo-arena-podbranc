"""What happens after the big moves of a tape, and what riding them with leverage would earn.

For every move of >= THR ticks in the 100 ms mid within WIN_S seconds (merged into one episode), prints its start,
size, how long it took, and the mid path after it (continuation vs reversion). Then the value of a leveraged ride:
enter LOTS in the move direction DELAY_S after the episode is first detectable (mid already THR/2 through), at the
mid +- half spread, exit after H seconds, minus the flat 10,000 / s penalty while over 100 lots.

.venv/bin/python scripts/jump_ride.py data/round2_train.csv data/round3_train.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.replay import replay  # noqa: E402
from analysis.tape import load_tape  # noqa: E402

US = 1_000_000
BIN = 100_000
THR = 15
WIN_S = 2.0
LOTS = 1000
DELAY_S = 0.1
HOLDS = (0.5, 1, 2, 5, 10, 30)
AFTER = (1, 2, 5, 10, 30, 60, 120, 300)


def main() -> None:
    for path in sys.argv[1:]:
        tape = load_tape(Path(path))
        rp = replay(tape, grid_step_us=US)
        t0, t1 = rp.ev_ts[0], rp.ev_ts[-1]
        edges = t0 + np.arange(int((t1 - t0) // BIN) + 1) * BIN
        idx = np.maximum(np.searchsorted(rp.ev_ts, edges, 'right') - 1, 0)
        bb, ba = rp.best_bid[idx], rp.best_ask[idx]
        mid = (bb + ba) / 2
        ok = np.isfinite(mid)
        mid = np.interp(np.arange(len(mid)), np.flatnonzero(ok), mid[ok])
        spr = np.where(np.isfinite(ba - bb), ba - bb, 2.0)
        w = int(WIN_S * US / BIN)
        n = len(mid)
        print(f'\n=== {path}: {(t1 - t0) / US:.0f} s, mid {mid[0]:.1f} -> {mid[-1]:.1f}, min {mid.min():.0f} max {mid.max():.0f}')
        # episodes: |mid[i+w] - mid[i]| >= THR, greedy, non-overlapping
        eps = []
        i = 0
        while i < n - w:
            d = mid[i + 1:i + w + 1] - mid[i]
            j = int(np.argmax(np.abs(d)))
            if abs(d[j]) >= THR:
                sg = np.sign(d[j])
                # extend while it keeps going the same way within the window
                end = i + 1 + j
                while end + 1 < n:
                    d2 = mid[end + 1:min(n, end + w + 1)] - mid[end]
                    k = int(np.argmax(sg * d2))
                    if sg * d2[k] >= 3:
                        end = end + 1 + k
                    else:
                        break
                # detection: first bin where mid is THR/2 through from the start
                det = i + 1 + int(np.argmax(sg * (mid[i + 1:end + 1] - mid[i]) >= THR / 2))
                eps.append((i, end, det, sg))
                i = end + 1
            else:
                i += 1
        print(f'{len(eps)} episodes of >= {THR} ticks within {WIN_S} s')
        hdr = ' '.join(f'+{a}s' for a in AFTER)
        print(f'{"t_s":>7} {"size":>6} {"dur_s":>6} {"det_s":>6} {"pre":>5} | mid after end, signed rel. end: {hdr}')
        rides = {h: [] for h in HOLDS}
        for i, end, det, sg in eps:
            size = mid[end] - mid[i]
            after = []
            for a in AFTER:
                k = end + int(a * US / BIN)
                after.append(f'{sg * (mid[k] - mid[end]):+6.1f}' if k < n else '     -')
            pre = sg * (mid[i] - mid[max(0, i - int(10 * US / BIN))])
            print(f'{(edges[i] - t0) / US:7.1f} {size:+6.0f} {(end - i) * BIN / US:6.1f} {(det - i) * BIN / US:6.1f} {pre:+5.0f} | ' + ' '.join(after))
            e = det + int(DELAY_S * US / BIN)
            if e >= n:
                continue
            entry = mid[e] + sg * spr[e] / 2
            for h in HOLDS:
                x = e + int(h * US / BIN)
                if x >= n:
                    continue
                exitp = mid[x] - sg * spr[x] / 2
                rides[h].append(LOTS * sg * (exitp - entry) - 10_000 * h)
        print(f'leveraged ride: {LOTS} lots, entry {DELAY_S}s after detection (mid {THR / 2} through), no slippage beyond half spread')
        for h in HOLDS:
            r = np.array(rides[h])
            if len(r):
                print(f'  hold {h:>4}s: n {len(r)}  mean {r.mean():+9.0f}  total {r.sum():+11.0f}  win {np.mean(r > 0):.2f}  '
                      f'first3 {[round(v) for v in r[:3]]}')


if __name__ == '__main__':
    main()
