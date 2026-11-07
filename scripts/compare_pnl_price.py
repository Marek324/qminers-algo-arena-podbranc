"""How close is the price rebuilt from a hold-probe PnL to the tape's real price?

Replays the tape's book (Insert G = limit, Insert F = IOC, Cancel) to get the
real best bid / ask / mid once per PnL sample, rebuilds price = PnL / hold + entry
from the probe run, and compares levels and 1 s increment distributions.
Writes <log_dir>/price_compare.npz for plotting.

    .venv/bin/python scripts/compare_pnl_price.py data/round2_public.csv logs/probe_experiment
"""

import bisect
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def replay(tape: Path, sample_t: np.ndarray):
    df = pd.read_csv(tape, dtype={"Lifespan": "string"})
    tm, op, oid = df.Time.to_numpy(), df.Operation.to_numpy(), df.OrderId.to_numpy()
    side, vol, px, life = df.Side.to_numpy(), df.Volume.to_numpy(), df.Price.to_numpy(), df.Lifespan.to_numpy()
    orders = {}  # oid -> [is_bid, price, remaining]
    levels = ({}, {})  # ask, bid: price -> deque of oids
    prices = ([], [])  # sorted price lists
    out = np.full((len(sample_t), 2), np.nan)
    k = 0

    def drop_level(b, p):
        del levels[b][p]
        prices[b].pop(bisect.bisect_left(prices[b], p))

    for i in range(len(df)):
        while k < len(sample_t) and tm[i] > sample_t[k]:
            out[k] = (prices[1][-1] if prices[1] else np.nan, prices[0][0] if prices[0] else np.nan)
            k += 1
        if op[i] == "Cancel":
            o = orders.pop(oid[i], None)
            if o is not None:
                q = levels[o[0]].get(o[1])
                if q is not None:
                    q.remove(oid[i])
                    if not q:
                        drop_level(o[0], o[1])
            continue
        b = 1 if side[i] == "B" else 0
        rem = int(vol[i])
        opp = 1 - b
        # match against the opposite side while it crosses
        while rem > 0 and prices[opp]:
            best = prices[opp][0] if opp == 0 else prices[opp][-1]
            if (b == 1 and best > px[i]) or (b == 0 and best < px[i]):
                break
            q = levels[opp][best]
            o = orders[q[0]]
            f = min(rem, o[2])
            rem -= f
            o[2] -= f
            if o[2] == 0:
                orders.pop(q.popleft())
                if not q:
                    drop_level(opp, best)
        if rem > 0 and life[i] == "G":
            orders[oid[i]] = [b, px[i], rem]
            if px[i] not in levels[b]:
                levels[b][px[i]] = deque()
                bisect.insort(prices[b], px[i])
            levels[b][px[i]].append(oid[i])
    while k < len(sample_t):
        out[k] = (prices[1][-1], prices[0][0])
        k += 1
    return out


def main() -> None:
    tape = Path(sys.argv[1] if len(sys.argv) > 1 else "data/round2_public.csv")
    log_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "logs/probe_experiment")
    res = json.loads((log_dir / "result.json").read_text())
    hold = res["inventory"]
    entry = -res["cash"] / hold
    p = pd.read_csv(log_dir / "pnl.csv")
    t = p.elapsed_s.to_numpy()
    rebuilt = p.pnl.to_numpy() / hold + entry
    bb = replay(tape, t)
    bid, ask = bb[:, 0], bb[:, 1]
    mid = (bid + ask) / 2
    ok = ~np.isnan(mid) & (t > 1)

    print(f"hold={hold} entry={entry:.2f} samples={ok.sum()}")
    for name, real in (("best bid", bid), ("mid", mid)):
        e = rebuilt[ok] - real[ok]
        print(f"vs {name:8s}: mean err {e.mean():+.3f}  rmse {np.sqrt((e**2).mean()):.3f}  "
              f"max |err| {np.abs(e).max():.1f}  corr {np.corrcoef(rebuilt[ok], real[ok])[0, 1]:.6f}")
    for lag in (1, 10, 60):
        a = np.diff(rebuilt[ok][::lag])
        r = np.diff(mid[ok][::lag])
        ks = stats.ks_2samp(a, r)
        print(f"{lag:3d}s increments: std rebuilt {a.std():.3f} real mid {r.std():.3f}  "
              f"kurt {stats.kurtosis(a):.1f} / {stats.kurtosis(r):.1f}  KS {ks.statistic:.4f} (p {ks.pvalue:.3f})  "
              f"corr {np.corrcoef(a, r)[0, 1]:.4f}")
    np.savez(log_dir / "price_compare.npz", t=t, rebuilt=rebuilt, bid=bid, ask=ask, mid=mid)
    print(f"wrote {log_dir / 'price_compare.npz'}")


if __name__ == "__main__":
    main()
