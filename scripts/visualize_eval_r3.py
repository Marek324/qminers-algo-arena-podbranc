"""Round 3 evaluation tape, reconstructed from the three arena probe runs (PnL only) and checked against its twin.

    MPLCONFIGDIR=$TMPDIR python3 scripts/visualize_eval_r3.py

Inputs (data/probes_r3/): r3_simple_probe (+10 lots -> best bid), r3_probe_short_spread (-10 lots -> best ask),
r3_depth (+100 lots -> 100-lot sweep price). Twin check: logs/r3twin_simple_probe_mm (the long probe re-run on
data/r3_eval_twin.csv.gz). Price held by a probe = (pnl - cash) / inventory.
Output: data_visualization/r3_eval_reconstruction.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "data/probes_r3"
OUT = ROOT / "data_visualization/r3_eval_reconstruction.png"
TWIN = ROOT / "logs/r3twin_simple_probe_mm"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
JUMP_T, WIN_S = 20, 30


def probe(d: Path):
    r = json.loads((d / "result.json").read_text())
    a = np.genfromtxt(d / "pnl.csv", delimiter=",", names=True)
    t = a["elapsed_s"]
    px = (a["pnl"] - r["cash"]) / r["inventory"]
    keep = np.r_[True, np.diff(t) > 1e-6] & (t >= 1)
    return t[keep], px[keep]


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def main() -> None:
    tb, bid = probe(PROBES / "r3_simple_probe")
    ta, ask = probe(PROBES / "r3_probe_short_spread")
    td, d100 = probe(PROBES / "r3_depth")
    ask = np.interp(tb, ta, ask)
    d100 = np.interp(tb, td, d100)
    t, mid, spread, slip = tb, (bid + ask) / 2, ask - bid, bid - d100

    # storms: 30 s windows with a >= JUMP_T 1 s move or 1 s vol above the calm/storm midpoint (round3 fit)
    sec = np.interp(np.arange(1, int(t[-1])), t, mid)
    dm = np.diff(sec)
    w = np.arange(len(dm)) // WIN_S
    nw = w.max() + 1
    dc = np.where(abs(dm) >= JUMP_T, 0, dm)
    vol = np.sqrt(np.bincount(w, dc**2, nw) / np.bincount(w, minlength=nw))
    storm = vol > (1.295 + 4.218) / 2
    storm[np.unique(w[abs(dm) >= JUMP_T])] = True

    fig = plt.figure(figsize=(18, 14), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1, 1], hspace=0.38, wspace=0.18)

    ax = fig.add_subplot(gs[0, :])
    style(ax, f"A. Mid price over the 12,000 s evaluation tape (storm windows shaded: {storm.mean():.0%} of time)")
    for i in np.flatnonzero(storm):
        ax.axvspan(1 + i * WIN_S, 1 + (i + 1) * WIN_S, color=GRID, lw=0)
    ax.plot(t, mid, color=BLUE, lw=1.2)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("elapsed s", color=INK2)
    ax.set_ylabel("ticks", color=INK2)
    ax.annotate(f"start {mid[0]:,.1f}", (t[0], mid[0]), xytext=(8, 8), textcoords="offset points", color=INK2)
    ax.annotate(f"end {mid[-1]:,.1f}", (t[-1], mid[-1]), xytext=(-70, 8), textcoords="offset points", color=INK2)

    ax = fig.add_subplot(gs[1, 0])
    style(ax, "B. Price-change distribution, scaled by its std (log density)")
    edges = np.linspace(-8, 8, 81)
    for lag, col in ((1, BLUE), (10, ORANGE), (60, AQUA)):
        x = np.diff(sec[::lag])
        h, _ = np.histogram(x / x.std(), edges, density=True)
        c = (edges[1:] + edges[:-1]) / 2
        ax.semilogy(c[h > 0], h[h > 0], color=col, lw=2,
                    label=f"{lag:>2} s: std {x.std():.1f} t, kurtosis {((x - x.mean())**4).mean() / x.var()**2 - 3:.0f}")
    c = (edges[1:] + edges[:-1]) / 2
    ax.semilogy(c, np.exp(-c**2 / 2) / np.sqrt(2 * np.pi), color=INK2, lw=1.2, ls="--", label="normal")
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel("change / std", color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    ax = fig.add_subplot(gs[1, 1])
    style(ax, "C. Volatility by regime: 1 s changes in calm vs storm windows")
    ws = storm[np.minimum(w, nw - 1)]
    e2 = np.linspace(-15, 15, 61)
    for m, col, lab in ((~ws, BLUE, "calm"), (ws, ORANGE, "storm")):
        x = dm[m & (abs(dm) < JUMP_T)]
        ax.hist(x, e2, density=True, histtype="step", lw=2, color=col,
                label=f"{lab}: σ {x.std():.2f} t/√s, n {m.sum():,} s")
    nj = int((abs(dm) >= JUMP_T).sum())
    ax.set_xlabel("1 s mid change, ticks (jumps ≥ 20 t excluded)", color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, title=f"{nj} jumps ≥ {JUMP_T} t", title_fontsize=9)

    ax = fig.add_subplot(gs[2, 0])
    style(ax, "D. Spread (ask − bid) and 100-lot slippage")
    vals = np.round(spread * 2) / 2
    u, cnt = np.unique(np.clip(vals, 0.5, 5), return_counts=True)
    ax.bar(u - 0.1, cnt / cnt.sum(), width=0.2, color=BLUE, label=f"spread, mean {spread.mean():.2f} t")
    su, sc = np.unique(np.clip(np.round(slip * 2) / 2, 0, 5), return_counts=True)
    ax.bar(su + 0.1, sc / sc.sum(), width=0.2, color=ORANGE,
           label=f"100-lot slippage, mean {slip.mean():.2f} t, >0 in {(slip > 0.01).mean():.0%}")
    ax.set_xlabel("ticks (capped at 5)", color=INK2)
    ax.set_ylabel("share of seconds", color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    ax = fig.add_subplot(gs[2, 1])
    tt, tw = probe(TWIN)
    k = np.argmax(np.convolve(storm, np.ones(10), "valid"))
    lo = 1 + k * WIN_S
    hi = lo + 600
    m1, m2 = (t >= lo) & (t <= hi), (tt >= lo) & (tt <= hi)
    e = np.interp(tt, t, bid) - tw
    style(ax, f"E. Twin check: best bid, arena vs twin (stormiest 600 s; full-tape median |err| {np.median(abs(e)):.1f} t)")
    ax.plot(t[m1], bid[m1], color=BLUE, lw=2, label="arena (probe PnL)")
    ax.plot(tt[m2], tw[m2], color=ORANGE, lw=1.2, label="twin r3_eval_twin")
    ax.set_xlabel("elapsed s", color=INK2)
    ax.set_ylabel("ticks", color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    fig.suptitle("Round 3 evaluation tape, rebuilt from the PnL of three hold probes", fontsize=16,
                 fontweight="bold", color=INK, x=0.06, ha="left", y=0.955)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"storm {storm.mean():.1%}  jumps>={JUMP_T}t {nj}  spread {spread.mean():.2f}  slip {slip.mean():.3f}")


if __name__ == "__main__":
    main()
