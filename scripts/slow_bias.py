"""Slow biases a short-horizon signal would miss: anchor reversion, long-lag autocorrelation, jump-sign balance,
time-of-session effects, round-number magnets, taker-side balance.

usage: .venv/bin/python scripts/slow_bias.py data/<tape>.csv
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
tape = Path(sys.argv[1])
rp = replay(load_tape(tape), grid_step_us=US)
ts = rp.ev_ts
mid = (rp.best_bid + rp.best_ask) / 2
ok = np.isfinite(mid)
ts, mid = ts[ok], mid[ok]
sec = np.arange(ts[0], ts[-1], US)
m = mid[np.clip(np.searchsorted(ts, sec, 'right') - 1, 0, len(ts) - 1)]
n = len(m)
print(f'{tape.name}: {n} s, start {m[0]:.1f}, end {m[-1]:.1f}, min {m.min():.1f}, max {m.max():.1f}')


def corr_t(x, y, step):
    """Pearson r on non-overlapping samples (every `step` s) and its t-stat."""
    x, y = x[::step], y[::step]
    k = np.isfinite(x) & np.isfinite(y)
    x, y = x[k], y[k]
    r = np.corrcoef(x, y)[0, 1]
    return r, r * np.sqrt((len(x) - 2) / max(1e-12, 1 - r * r)), len(x)


print('\n1) anchor reversion: future move over h vs (mid - 10000) and vs (mid - trailing mean over W)')
for h in (60, 300, 600):
    fut = np.r_[m[h:] - m[:-h], np.full(h, np.nan)]
    r, t, k = corr_t(m - 10000.0, fut, h)
    print(f'   h {h:4d}s  vs mid-10000: r {r:+.3f} t {t:+.2f} n {k}', end='')
    for w in (300, 1200):
        ema = pd.Series(m).rolling(w, min_periods=w).mean().to_numpy()
        r, t, k = corr_t(m - ema, fut, h)
        print(f' | vs mid-mean{w}: r {r:+.3f} t {t:+.2f} n {k}', end='')
    print()

print('\n2) return autocorrelation of non-overlapping h-second returns (lag 1)')
for h in (1, 5, 10, 30, 60, 120, 300, 600):
    r_ = m[h::h] - m[:-h:h]
    if len(r_) > 10:
        r = np.corrcoef(r_[:-1], r_[1:])[0, 1]
        print(f'   h {h:4d}s: r {r:+.3f} (n {len(r_)}, se {1 / np.sqrt(len(r_)):.3f})')

print('\n3) jumps (|mid move| >= 20 ticks within 0.5 s, 30 s apart): sign balance and size')
d = np.diff(mid)
big = []
last = -10**18
i0 = 0
for i in range(len(ts)):
    while ts[i] - ts[i0] > 500_000:
        i0 += 1
    mv = mid[i] - mid[i0]
    if abs(mv) >= 20 and ts[i] - last > 30 * US:
        big.append((ts[i], mv))
        last = ts[i]
if big:
    s = np.array([b[1] for b in big])
    print(f'   {len(s)} jumps, up {int((s > 0).sum())} down {int((s < 0).sum())}, sum up {s[s > 0].sum():+.0f} down {s[s < 0].sum():+.0f}')
    tj = np.array([(b[0] - ts[0]) / US for b in big])
    print('   jump times (s):', ' '.join(f'{x:.0f}{"+" if v > 0 else "-"}' for x, v in zip(tj, s)))
    after = []
    for t0, mv in big:
        j = np.searchsorted(sec, t0)
        if j + 60 < n:
            after.append((np.sign(mv), m[j + 60] - m[j], m[min(n - 1, j + 300)] - m[j]))
    a = np.array(after)
    print(f'   move after jump, signed by jump: +60 s {np.mean(a[:, 0] * a[:, 1]):+.1f}, +300 s {np.mean(a[:, 0] * a[:, 2]):+.1f} ticks (n {len(a)})')

print('\n4) time of session: per 600 s block, net move, 1 s vol, |move| share up')
for b in range(0, n, 600):
    seg = m[b:b + 600]
    if len(seg) > 10:
        r1 = np.diff(seg)
        print(f'   {b:5d}-{b + 600:5d}: net {seg[-1] - seg[0]:+7.1f}  vol1s {r1.std():5.2f}  up-share {np.mean(r1 > 0) / max(1e-9, np.mean(r1 != 0)):.2f}')

print('\n5) round-number magnets: future 60 s move toward the nearest multiple of 50 / 100, by distance')
for base in (50, 100):
    near = np.round(m / base) * base
    dist = m - near
    fut = np.r_[m[60:] - m[:-60], np.full(60, np.nan)]
    r, t, k = corr_t(dist, fut, 60)
    print(f'   base {base}: corr(mid - nearest, next 60 s) r {r:+.3f} t {t:+.2f} n {k} (negative = pull)')

print('\n6) anchor reversion null: paths rebuilt from 10000 out of shuffled 60 s return blocks (keeps jumps and storms)')
rng = np.random.default_rng(0)
r1 = np.diff(m)
nb = len(r1) // 60
blocks = r1[:nb * 60].reshape(nb, 60)


def anchor_stats(path):
    out = []
    for h in (60, 300, 600):
        fut = np.r_[path[h:] - path[:-h], np.full(h, np.nan)]
        x, y = (path - 10000.0)[::h], fut[::h]
        k = np.isfinite(y)
        out.append(np.corrcoef(x[k], y[k])[0, 1])
    return np.array(out)


obs = anchor_stats(m)
null = np.array([anchor_stats(np.r_[10000.0, 10000.0 + np.cumsum(blocks[rng.permutation(nb)].ravel())]) for _ in range(2000)])
for i, h in enumerate((60, 300, 600)):
    print(f'   h {h:4d}s: observed r {obs[i]:+.3f}, null mean {null[:, i].mean():+.3f}, sd {null[:, i].std():.3f}, '
          f'P(null <= observed) {np.mean(null[:, i] <= obs[i]):.3f}')
