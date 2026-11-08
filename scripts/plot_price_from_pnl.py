"""Rebuild the price chart from a hold-probe PnL curve alone (pnl.csv only).

strategies/round2/simple_probe_mm.py buys HOLD lots once and holds them, so
PnL(t) = HOLD * (price(t) - entry): the price relative to the entry is PnL / HOLD.
Pass the entry price to get absolute ticks (it is the first trade's price; leave
it at 0 for a relative chart). Works on a local run or an arena download.

    MPLCONFIGDIR=$TMPDIR python3 scripts/plot_price_from_pnl.py logs/probe_experiment/pnl.csv --hold 10 --entry 10001
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def hms(ticks):
    return [f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:00" for s in ticks]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pnl_csv", nargs="?", default="logs/probe_experiment/pnl.csv")
    ap.add_argument("--hold", type=float, default=10, help="lots the probe holds (HOLD in the strategy)")
    ap.add_argument("--entry", type=float, default=0, help="entry price; 0 = chart relative to entry")
    ap.add_argument("--candle", type=float, default=30, help="candle width in seconds")
    ap.add_argument("--out", default="data_visualization/price_from_pnl.png")
    a = ap.parse_args()

    with open(a.pnl_csv) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["elapsed_s"]) for r in rows])
    price = np.array([float(r["pnl"]) for r in rows]) / a.hold + a.entry

    # candles: open/high/low/close per bucket
    edges = np.arange(0, t[-1] + a.candle, a.candle)
    idx = np.digitize(t, edges) - 1
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.4, 1.2])
    ax = fig.add_subplot(gs[0, :])
    for k in np.unique(idx):
        p = price[idx == k]
        o, c = p[0], p[-1]
        col = "#26a69a" if c >= o else "#ef5350"
        x = edges[k] + a.candle / 2
        ax.vlines(x, p.min(), p.max(), color=col, lw=0.7)
        ax.vlines(x, min(o, c), max(o, c) + 1e-9, color=col, lw=2.2)
    ticks = np.arange(0, t[-1] + 1, 600)
    ax.set_xticks(ticks, hms(ticks))
    ax.set_xlim(0, t[-1])
    ax.yaxis.tick_right()
    ax.grid(axis="y", color="#eee")
    for s in ("top", "left", "right"):
        ax.spines[s].set_visible(False)
    unit = "ticks" if a.entry else "ticks from entry"
    ax.set_title(f"Price rebuilt from PnL only (PnL / {a.hold:g} lots{f' + {a.entry:g}' if a.entry else ''}, "
                 f"{a.candle:g} s candles, {unit})")

    # price increment distributions, still from PnL only
    sec = np.searchsorted(t, np.arange(0, t[-1], 1.0))
    p1 = price[np.clip(sec, 0, len(price) - 1)]
    for j, lag in enumerate((1, 10, 60)):
        d = np.diff(p1[::lag])
        ax = fig.add_subplot(gs[1, j])
        lim = max(np.percentile(np.abs(d), 99.9), 1)
        ax.hist(d, bins=np.linspace(-lim, lim, 81), color="#26a69a", alpha=0.6, density=True)
        g = np.linspace(-lim, lim, 400)
        sd = d.std()
        ax.plot(g, np.exp(-g**2 / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi)), color="#ef5350", lw=1, label="normal, same std")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1 / (len(d) * 10))
        kurt = ((d - d.mean()) ** 4).mean() / sd**4 - 3
        ax.set_title(f"{lag} s price changes: std {sd:.2f}, excess kurtosis {kurt:.1f}")
        ax.set_xlabel("ticks")
        if j == 0:
            ax.legend()

    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"{len(t)} samples, price {price.min():.1f}..{price.max():.1f}, wrote {out}")


if __name__ == "__main__":
    main()
