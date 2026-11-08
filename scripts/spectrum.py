"""Power spectrum of a tape's 1 s mid after removing the lowest-frequency wave (the session-long cubic swing),
and of 1 s volatility |dmid|, each against a null of 60 s blocks shuffled in time (keeps the short-term structure,
destroys any periodicity longer than a block).

usage: .venv/bin/python scripts/spectrum.py data/<tape>.csv out.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.replay import replay  # noqa: E402
from analysis.tape import load_tape  # noqa: E402

tape, out = Path(sys.argv[1]), Path(sys.argv[2])
rp = replay(load_tape(tape), grid_step_us=1_000_000)
ts = rp.ev_ts
mid = (rp.best_bid + rp.best_ask) / 2
ok = np.isfinite(mid)
ts, mid = ts[ok], mid[ok]
sec = np.arange(ts[0], ts[-1], 1_000_000)
m = mid[np.clip(np.searchsorted(ts, sec, 'right') - 1, 0, len(ts) - 1)]
n = len(m)
u = np.arange(n) / n


def detrend(x):
    return x - np.polyval(np.polyfit(u[:len(x)], x, 3), u[:len(x)])


def psd(x):
    x = x - x.mean()
    w = np.hanning(len(x))
    p = np.abs(np.fft.rfft(x * w)) ** 2 / (w ** 2).sum()
    return np.fft.rfftfreq(len(x), 1.0), p


edges = np.logspace(np.log10(1 / 3000), np.log10(0.5), 40)


def binned(f, p):
    idx = np.digitize(f, edges)
    fc, pc = [], []
    for k in range(1, len(edges)):
        s = idx == k
        if s.sum():
            fc.append(np.exp(np.log(f[s]).mean()))
            pc.append(p[s].mean())
    return np.array(fc), np.array(pc)


rng = np.random.default_rng(0)
B = 60
nb = n // B
r = np.diff(m)[: nb * B - 1]
series = {
    'mid minus the session-long cubic (yellow wave)': (detrend(m), lambda: detrend(np.r_[0.0, np.cumsum(np.r_[r, 0.0].reshape(nb, B)[rng.permutation(nb)].ravel())][:n])),
    'volatility |1 s mid move|': (np.abs(np.diff(m)), lambda: np.abs(np.r_[r, 0.0]).reshape(nb, B)[rng.permutation(nb)].ravel()),
}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor='#fcfcfb')
for ax, (name, (x, gen)) in zip(axes, series.items()):
    f, p = psd(x)
    fc, pc = binned(f[1:], p[1:])
    nulls = np.array([binned(*[a[1:] for a in psd(gen())])[1] for _ in range(300)])
    lo, med, hi = np.percentile(nulls, [2.5, 50, 97.5], axis=0)
    ax.set_facecolor('#fcfcfb')
    ax.fill_between(1 / fc, lo, hi, color='#86b6ef', alpha=0.35, lw=0, label='shuffled 60 s blocks, 95% band')
    ax.plot(1 / fc, med, color='#2a78d6', lw=1.2, ls='--', label='shuffled, median')
    ax.plot(1 / fc, pc, color='#eb6834', lw=2, marker='o', ms=4, label='round4_public' if 'round4' in tape.name else tape.stem)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.set_xlabel('period (s)', color='#52514e')
    ax.set_ylabel('power (log-binned)', color='#52514e')
    ax.set_title(name, color='#0b0b0b', fontsize=11, loc='left')
    ax.grid(True, which='major', color='#e6e5e1', lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#bdbcb6')
    ax.tick_params(colors='#52514e')
    ax.legend(frameon=False, fontsize=8, loc='lower left' if 'mid' in name else 'upper right')
    above = [(1 / a, b / c) for a, b, c, h in zip(fc, pc, med, hi) if b > h]
    print(f'{name}: bins above the 97.5% null band (period s, power / null median):',
          ', '.join(f'{a:.0f}s x{b:.1f}' for a, b in above) or 'none')
    fr = f[1:]
    pr = p[1:]
    top = np.argsort(pr[(fr > 1 / 3000) & (fr < 1 / 10)])[::-1][:5]
    sel = np.where((fr > 1 / 3000) & (fr < 1 / 10))[0][top]
    print('   strongest raw lines (period s):', ', '.join(f'{1 / fr[i]:.0f}' for i in sel))
fig.suptitle(f'{tape.name}: spectrum after removing the lowest-frequency wave', x=0.01, ha='left', fontsize=12, color='#0b0b0b')
fig.tight_layout()
fig.savefig(out, dpi=130)
print('saved', out)
