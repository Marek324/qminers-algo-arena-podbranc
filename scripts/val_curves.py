import pandas as pd, numpy as np

base = "data/validation_round3"
subs = {
    "final_hmm_mm": "0138_907826f2",   # tuned hmm_mm  1,076,934
    "hmm_617_fit": "0226_1e79c8a0",    # 1,091,395
    "cand_nm6":     "0247_63d7df2e",    # 1,185,019  FINAL
    "deep_nm6":     "0256_e9ace9e5",    # 1,162,363
    "bujza":        "0033_a38b1737",    # teammate 1,028,431
}
curves = {}
for name, d in subs.items():
    df = pd.read_csv(f"{base}/{d}/pnl.csv")
    curves[name] = df.set_index("elapsed_s")["pnl"]

C = pd.DataFrame(curves)
print("final PnL:", {k: int(v.iloc[-1]) for k, v in C.items()})
print()

# PnL per 1000 s block: where is money made on the hidden tape?
blk = C.groupby(C.index // 1000 * 1000).last().diff().fillna(C.groupby(C.index // 1000 * 1000).last().iloc[0])
print("PnL per 1000s block (k$):")
print((blk / 1000).round(0).astype(int).to_string())
print()

# first vs second half
for half, lo, hi in [("h1 0-6ks", 0, 6000), ("h2 6-12ks", 6000, 12000)]:
    print(half, {k: int(v.iloc[hi - 1] - v.iloc[lo]) for k, v in C.items()})
print()

# where does the FINALIST beat the old final? per-1000s diff
d = C["cand_nm6"] - C["final_hmm_mm"]
dblk = d.groupby(d.index // 1000 * 1000).last().diff().fillna(0)
print("cand_nm6 minus final_hmm_mm per 1000s (k$):")
print((dblk / 1000).round(0).astype(int).to_string())
print()

# storm signature: 10s PnL bursts of the finalist
g = C["cand_nm6"].groupby(C.index // 10 * 10).last()
r = g.diff().fillna(0)
top = r.sort_values(ascending=False).head(15)
print("top 15 10-second PnL bursts of finalist (storm takes):")
print(top.round(0).astype(int).to_string())
print()
tot = C["cand_nm6"].iloc[-1]
print(f"sum of top 15 bursts: {int(top.sum())} = {100*top.sum()/tot:.0f}% of final PnL")
