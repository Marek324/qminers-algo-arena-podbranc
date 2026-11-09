"""Combine Depth 1 (results (2), 10 lots) and Depth 3 (results3, 100 lots) to analyze book thickness."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    out_dir = Path("logs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Depth 1 (results (2), 10 lots) and Depth 3 (results3, 100 lots)
    df10 = pd.read_csv("docs/results (2)/pnl.csv")
    df100 = pd.read_csv("docs/results3/pnl.csv")

    t = df10["elapsed_s"].values
    p10 = df10["pnl"].values
    p100 = df100["pnl"].values

    # Price per lot from 10-lot probe (cash = -100,010.0)
    p_bid_10 = (p10 + 100010.0) / 10.0

    # Price per lot from 100-lot probe (cash = -1,000,100.0)
    p_bid_100 = (p100 + 1000100.0) / 100.0

    # Slippage per lot due to depth exhaustion across 100 lots
    slippage = p_bid_10 - p_bid_100
    slippage_cash = 10.0 * (p10 + 100010.0) - (p100 + 1000100.0)

    # Price jumps
    dp = np.diff(p_bid_10)
    dp_1s = np.zeros(len(t))
    dp_1s[1:] = dp
    major_jumps = np.abs(dp_1s) >= 20.0
    jump_times = t[major_jumps]

    # Style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(20, 15), dpi=150)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # ==========================================================
    # Panel 1: Reconstructed Price & Order Book Liquidity Drains
    # ==========================================================
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, p_bid_10, color="#1f77b4", lw=1.2, label="Reconstructed Market Price S(t) (from Depth 1 Probe)")
    
    # Highlight areas where depth collapsed (> 2 ticks slippage)
    severe_depletion = slippage > 2.0
    ax1.scatter(t[severe_depletion], p_bid_10[severe_depletion], color="#d62728", s=18, alpha=0.8, zorder=4,
                label=f"Severe Liquidity Collapses (Slippage > 2 ticks, N={severe_depletion.sum()})")

    ax1.set_title("A. 12,000s Market Price S(t) with Order Book Liquidity Holes Identified by Combining Depth 1 and Depth 3",
                  fontsize=14, fontweight="bold", pad=10)
    ax1.set_xlabel("Elapsed Time (seconds)", fontsize=11)
    ax1.set_ylabel("Asset Price (ticks)", fontsize=11)
    ax1.set_xlim(0, 12000)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)

    # ==========================================================
    # Panel 2: PnL Comparison (Depth 1: 10 lots vs Depth 3: 100 lots)
    # ==========================================================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, p100 / 1000.0, color="#2ca02c", lw=1.5, label="Depth 3 Probe (Hold 100 lots: Final +$47.9k)")
    ax2.plot(t, (p10 * 10.0) / 1000.0, color="#1f77b4", lw=1.2, ls="--", label="Depth 1 Probe Scaled x10 (Theoretical Zero Slippage: +$48.0k)")
    ax2.plot(t, p10 / 1000.0, color="#ff7f0e", lw=1.0, label="Depth 1 Probe Raw (Hold 10 lots: Final +$4.8k)")

    ax2.set_title("B. Cumulative PnL Trajectories: Depth 3 (100 lots) vs Scaled Depth 1 (10 lots)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Elapsed Time (seconds)", fontsize=10)
    ax2.set_ylabel("PnL ($ Thousands)", fontsize=10)
    ax2.set_xlim(0, 12000)
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    # ==========================================================
    # Panel 3: Order Book Depth Slippage Time Series
    # ==========================================================
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, slippage, color="#d62728", lw=0.7, alpha=0.8, label="Depth Slippage: P_bid(10) - P_bid(100) (ticks)")
    ax3.axhline(0, color="gray", lw=0.8, ls="--")
    ax3.axhline(np.mean(slippage[slippage > 0.001]), color="#ff7f0e", ls=":", lw=1.5,
                label=f"Avg Slippage when depleted: {np.mean(slippage[slippage > 0.001]):.2f} ticks")

    # Mark max spike
    max_idx = np.argmax(slippage)
    ax3.annotate(f"Max Collapse: {slippage[max_idx]:.1f} ticks\n(Loss: ${slippage_cash[max_idx]:,.0f})",
                 xy=(t[max_idx], slippage[max_idx]), xytext=(t[max_idx] + 400, slippage[max_idx] - 5),
                 arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
                 fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#ffdddd", ec="red"))

    ax3.set_title("C. Order Book Depth Slippage Across 100 Lots (Liquidity Hollows)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Elapsed Time (seconds)", fontsize=10)
    ax3.set_ylabel("Execution Slippage (ticks)", fontsize=10)
    ax3.set_xlim(0, 12000)
    ax3.set_ylim(-1, 42)
    ax3.legend(loc="upper right", frameon=True, fontsize=9)

    # ==========================================================
    # Panel 4: Zoomed Case Study Around Severe Jump / Liquidity Drain
    # ==========================================================
    ax4 = fig.add_subplot(gs[2, 0])
    # Find severe event around max slippage
    t_center = t[max_idx]
    zoom_mask = (t >= t_center - 100) & (t <= t_center + 100)

    ax4_sub = ax4.twinx()
    l1 = ax4.plot(t[zoom_mask], p_bid_10[zoom_mask], color="#1f77b4", lw=2.0, label="Price S(t) (Left Axis)")
    l2 = ax4_sub.plot(t[zoom_mask], slippage[zoom_mask], color="#d62728", lw=2.0, ls="--", label="Depth Slippage (Right Axis)")
    ax4_sub.fill_between(t[zoom_mask], 0, slippage[zoom_mask], color="#d62728", alpha=0.15)

    ax4.set_title(f"D. Microstructure Zoom: Liquidity Evaporation at t={t_center:.0f}s", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Elapsed Time (seconds)", fontsize=10)
    ax4.set_ylabel("Price S(t) (ticks)", color="#1f77b4", fontsize=10)
    ax4_sub.set_ylabel("Slippage (ticks)", color="#d62728", fontsize=10)
    ax4.set_xlim(t_center - 100, t_center + 100)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc="upper left", frameon=True, fontsize=9)

    # ==========================================================
    # Panel 5: Liquidity Availability Breakdown (Calm vs Jump)
    # ==========================================================
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Calculate depth status
    full_depth_pct = np.mean(slippage <= 0.001) * 100
    mild_depletion_pct = np.mean((slippage > 0.001) & (slippage <= 1.0)) * 100
    severe_depletion_pct = np.mean(slippage > 1.0) * 100

    categories = ["Full Depth\n(>=100 lots at touch)", "Mild Depletion\n(Slippage <= 1 tick)", "Severe Liquidity Hole\n(Slippage > 1 tick)"]
    pcts = [full_depth_pct, mild_depletion_pct, severe_depletion_pct]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    bars = ax5.bar(categories, pcts, color=colors, edgecolor="black", width=0.55)
    for bar, pct in zip(bars, pcts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1.0, f"{pct:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax5.set_title("E. Order Book Liquidity Availability on 12,000s Evaluation Tape", fontsize=13, fontweight="bold")
    ax5.set_ylabel("Percentage of Simulation Time (%)", fontsize=10)
    ax5.set_ylim(0, 100)

    plt.suptitle("Combined Microstructure Analysis: Depth 1 (10 lots) + Depth 3 (100 lots) on 12,000s Private Evaluation Tape",
                 fontsize=16, fontweight="bold", y=0.99)

    fig_path = out_dir / "results_depth_combined.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Combined visualization saved successfully to {fig_path}")

if __name__ == "__main__":
    main()
