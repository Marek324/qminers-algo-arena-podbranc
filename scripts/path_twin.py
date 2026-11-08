"""Synthetic twin of a tape we can't see, from a hold-probe PnL curve (strategies/round2/simple_probe_mm.py).

The probe holds HOLD lots, so PnL / HOLD + entry is the best bid each second; S ~ that + HALF (half spread).
The generator (analysis/generator.py) then rebuilds a full order flow around that S path: the path's own jumps,
calm/storm regimes read off the path (30 s windows with a jump or high volatility are storms), and the fitted
round's rates, sizes, X and impact.

    .venv/bin/python scripts/path_twin.py "docs/results (2)/pnl.csv" data/synthetic/eval_twin_1.csv --seed 1
    .venv/bin/python scripts/path_twin.py logs/probe_experiment/pnl.csv data/synthetic/pub_twin_1.csv --seed 1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.generator import PARAM_DIR, GenParams, generate, write_tape  # noqa: E402

WIN_S = 30


def path_from_pnl(pnl_csv: Path, hold: float, entry: float, half: float) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(pnl_csv)
    t = df.elapsed_s.to_numpy()
    s = df.pnl.to_numpy() / hold + entry + half
    s[t < 1.0] = s[np.argmax(t >= 1.0)]  # before the entry fill PnL is 0
    keep = np.r_[True, np.diff(t) > 1e-6]
    return t[keep], s[keep]


def probe_price(log_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Price held by a probe run (pnl.csv + result.json): pnl / inventory - cash / inventory (bid if long, ask if short)."""
    r = json.loads((log_dir / "result.json").read_text())
    df = pd.read_csv(log_dir / "pnl.csv")
    t = df.elapsed_s.to_numpy()
    p = (df.pnl.to_numpy() - r["cash"]) / r["inventory"]
    p[t < 1.0] = p[np.argmax(t >= 1.0)]
    keep = np.r_[True, np.diff(t) > 1e-6]
    return t[keep], p[keep]


def mid_from_probes(long_dir: Path, short_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    tb, bid = probe_price(long_dir)
    ta, ask = probe_price(short_dir)
    ask = np.interp(tb, ta, ask)
    ask = np.maximum(ask, bid + 1)  # sampling mismatch: the book never crosses
    return tb, (bid + ask) / 2


def regimes_from_path(t: np.ndarray, s: np.ndarray, p: GenParams, jump_ticks: float):
    """Storm = a window with a >= jump_ticks move or 1 s volatility above the calm/storm midpoint."""
    edges = np.arange(0, t[-1] + WIN_S, WIN_S)
    sec = np.interp(np.arange(0, t[-1]), t, s)
    d = np.diff(sec)
    w = np.arange(len(d)) // WIN_S
    jump = np.zeros(len(edges) - 1, bool)
    jump[np.unique(w[np.abs(d) >= jump_ticks])] = True
    dc = np.where(np.abs(d) >= jump_ticks, 0.0, d)
    vol = np.sqrt(np.bincount(w, dc**2, minlength=len(edges) - 1) / np.maximum(np.bincount(w, minlength=len(edges) - 1), 1))
    cut = (p.sigma_s + p.sigma_storm) / 2 if p.sigma_storm > 0 else np.inf
    states = (jump | (vol[: len(jump)] > cut)).astype(int)
    edges[-1] = max(edges[-1], t[-1] + 1)
    return edges, states


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pnl_csv", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--params", default="round2_v2", help="logs/generator/NAME_params.json")
    ap.add_argument("--hold", type=float, default=10)
    ap.add_argument("--entry", type=float, default=10001)
    ap.add_argument("--half", type=float, default=0.5)
    ap.add_argument("--jump-ticks", type=float, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--short-dir", type=Path, default=None,
                    help="short probe run dir; then pnl_csv is the long probe run dir and S = mid of the two")
    a = ap.parse_args()

    p = GenParams.load(PARAM_DIR / f"{a.params}_params.json")
    if a.short_dir is not None:
        t, s = mid_from_probes(a.pnl_csv, a.short_dir)
    else:
        t, s = path_from_pnl(a.pnl_csv, a.hold, a.entry, a.half)
    reg = regimes_from_path(t, s, p, a.jump_ticks) if p.storm_dur_s > 0 else None
    if reg is not None:
        print(f"storm share {reg[1].mean():.1%} of {len(reg[1])} windows")
    df, ts, S = generate(p, seed=a.seed, duration_s=float(t[-1]), s_path=(t, s), jump_ticks=a.jump_ticks, regimes=reg)
    write_tape(df, a.out)
    print(f"{len(df)} rows, {t[-1]:.0f} s, S {S.min():.0f}..{S.max():.0f}, wrote {a.out}")


if __name__ == "__main__":
    main()
