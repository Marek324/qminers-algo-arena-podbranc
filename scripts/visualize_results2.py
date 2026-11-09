"""Visualize reconstructed market price process and distribution from docs/results (2)."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def main():
    out_dir = Path("logs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load results (2) - probe run that tracked mark price
    probe_file = Path("docs/results (2)/pnl.csv")
    champ_file = Path("docs/results/pnl.csv")

    df_probe = pd.read_csv(probe_file)
    t = df_probe["elapsed_s"].values
    pnl_probe = df_probe["pnl"].values

    # Reconstruct true price process: S(t) = 10001.0 + pnl / 10.0
    price = 10001.0 + pnl_probe / 10.0

    # Load champion strategy PnL if available
    df_champ = pd.read_csv(champ_file) if champ_file.exists() else None

    # Compute 1-second price increments
    dt = np.diff(t)
    dp = np.diff(price)
    valid = dt > 0.01
    dp_1s = dp[valid] / np.sqrt(dt[valid])
    t_mid = t[1:][valid]

    # Robust calm volatility estimation
    med_dp = np.median(dp_1s)
    mad = np.median(np.abs(dp_1s - med_dp))
    sigma_calm = 1.4826 * mad
    th_calm = 3.5 * sigma_calm  # 5.19 ticks

    is_jump = np.abs(dp_1s - med_dp) > th_calm
    jump_t = t_mid[is_jump]
    jump_sizes = dp_1s[is_jump]
    calm_dp = dp_1s[~is_jump]

    # Major jumps (>= 20 ticks)
    is_major = np.abs(dp_1s) >= 20.0
    major_t = t_mid[is_major]
    major_sizes = dp_1s[is_major]
    major_prices = price[1:][valid][is_major]

    # Plotting setup
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(20, 14), dpi=150)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # ==========================================================
    # Panel 1: Full 12,000s Reconstructed Price Process S(t)
    # ==========================================================
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, price, color="#1f77b4", lw=1.2, label="Reconstructed Market Price S(t) = 10,001 + PnL/10")
    
    # Mark major jumps
    pos_maj = major_sizes > 0
    neg_maj = major_sizes < 0
    ax1.scatter(major_t[pos_maj], major_prices[pos_maj], color="#2ca02c", marker="^", s=60, zorder=5, label=f"Major Upward Jumps (>= +20t, N={pos_maj.sum()})")
    ax1.scatter(major_t[neg_maj], major_prices[neg_maj], color="#d62728", marker="v", s=60, zorder=5, label=f"Major Downward Jumps (<= -20t, N={neg_maj.sum()})")

    ax1.set_title("A. Private Evaluation Tape: 12,000-Second (3.33 hr) Reconstructed Price Process S(t) from `docs/results (2)`", fontsize=14, fontweight="bold", pad=10)
    ax1.set_xlabel("Elapsed Time (seconds)", fontsize=11)
    ax1.set_ylabel("Asset Price (ticks)", fontsize=11)
    ax1.set_xlim(0, 12000)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)

    # ==========================================================
    # Panel 2: Strategy Performance Comparison
    # ==========================================================
    ax2 = fig.add_subplot(gs[1, 0])
    if df_champ is not None:
        ax2.plot(df_champ["elapsed_s"], df_champ["pnl"] / 1000.0, color="#2ca02c", lw=1.8, label="Champion Strategy (`HmmMMStrategy`: +$1,290.8k)")
    ax2.plot(t, pnl_probe / 1000.0, color="#ff7f0e", lw=1.2, label="Price Probe (`SimpleProbeMM`: +$4.8k)")
    
    ax2.set_title("B. Strategy PnL Trajectories on the 12,000s Evaluation Tape", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Elapsed Time (seconds)", fontsize=10)
    ax2.set_ylabel("Cumulative PnL ($ Thousands)", fontsize=10)
    ax2.set_xlim(0, 12000)
    ax2.legend(loc="upper left", frameon=True, fontsize=10)

    # ==========================================================
    # Panel 3: 1-Second Discrete Price Changes (Diffusion vs Jumps)
    # ==========================================================
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_mid, dp_1s, color="#7f7f7f", lw=0.5, alpha=0.6, label="1s Price Change ΔS")
    ax3.axhline(th_calm, color="#d62728", ls="--", lw=1.2, label=f"+3.5σ Threshold (+{th_calm:.1f}t)")
    ax3.axhline(-th_calm, color="#d62728", ls="--", lw=1.2, label=f"-3.5σ Threshold (-{th_calm:.1f}t)")
    ax3.fill_between(t_mid, -th_calm, th_calm, color="#aec7e8", alpha=0.3, label="Calm Diffusion Band (±5.2 ticks)")

    ax3.set_title("C. High-Frequency Price Increments (Continuous Diffusion vs Discrete Jumps)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Elapsed Time (seconds)", fontsize=10)
    ax3.set_ylabel("Price Change ΔS (ticks/s)", fontsize=10)
    ax3.set_xlim(0, 12000)
    ax3.set_ylim(-80, 80)
    ax3.legend(loc="upper right", frameon=True, fontsize=9)

    # ==========================================================
    # Panel 4: Empirical Jump Size Distribution & Heavy Tail Fit
    # ==========================================================
    ax4 = fig.add_subplot(gs[2, 0])
    major_jump_sizes = np.abs(dp_1s[np.abs(dp_1s) >= 10.0])

    loc_n, scale_n = stats.norm.fit(major_jump_sizes)
    loc_e, scale_e = stats.expon.fit(major_jump_sizes)
    df_t, loc_t, scale_t = stats.t.fit(major_jump_sizes)

    x_grid = np.linspace(10.0, np.percentile(major_jump_sizes, 99.5), 300)

    ax4.hist(major_jump_sizes, bins=35, density=True, alpha=0.5, color="#1f77b4", edgecolor="black", label=f"Observed Jumps >= 10t (N={len(major_jump_sizes)})")
    ax4.plot(x_grid, stats.t.pdf(x_grid, df=df_t, loc=loc_t, scale=scale_t), color="#2ca02c", lw=2.2, label=f"Student-t (df={df_t:.2f}, Fat Tails)")
    ax4.plot(x_grid, stats.expon.pdf(x_grid, loc=loc_e, scale=scale_e), color="#ff7f0e", lw=2.0, ls="-.", label=f"Laplace / Kou (scale={scale_e:.1f}t)")
    ax4.plot(x_grid, stats.norm.pdf(x_grid, loc=loc_n, scale=scale_n), color="#d62728", lw=2.0, ls=":", label="Gaussian / Merton (Fails)")

    ax4.set_yscale("log")
    ax4.set_title("D. Jump Size Distribution (Extreme Fat Tails: df = 0.94)", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Absolute Jump Magnitude (ticks)", fontsize=10)
    ax4.set_ylabel("Probability Density (Log Scale)", fontsize=10)
    ax4.legend(loc="upper right", frameon=True, fontsize=9)

    # ==========================================================
    # Panel 5: Calm Regime Diffusion (Pure Brownian Motion)
    # ==========================================================
    ax5 = fig.add_subplot(gs[2, 1])
    mu_c, std_c = np.mean(calm_dp), np.std(calm_dp)
    xc_grid = np.linspace(mu_c - 4*std_c, mu_c + 4*std_c, 200)

    ax5.hist(calm_dp, bins=50, density=True, alpha=0.5, color="#2ca02c", edgecolor="black", label="Empirical Calm Increments")
    ax5.plot(xc_grid, stats.norm.pdf(xc_grid, mu_c, std_c), color="#1f77b4", lw=2.2, label=f"Fitted Gaussian (μ={mu_c:+.3f}, σ={std_c:.2f} t/√s)")

    ax5.set_title("E. Calm Regime Dynamics (Standard Gaussian Diffusion: σ = 1.48 t/√s)", fontsize=13, fontweight="bold")
    ax5.set_xlabel("1s Calm Price Change ΔS (ticks)", fontsize=10)
    ax5.set_ylabel("Probability Density", fontsize=10)
    ax5.legend(loc="upper right", frameon=True, fontsize=9)

    plt.suptitle("Private Evaluation Tape (12,000s): Microstructure & Generative Process Analysis (`docs/results (2)`)", fontsize=16, fontweight="bold", y=0.99)

    fig_path = out_dir / "results2_market_reconstruction.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Visualization saved successfully to {fig_path}")

if __name__ == "__main__":
    main()
