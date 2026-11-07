"""Infer underlying price process and distribution shape PURELY from (timestamp, PnL) curve."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def main():
    pnl_file = Path("logs/probe_experiment/pnl.csv")
    if not pnl_file.exists():
        print(f"Error: {pnl_file} not found.")
        return

    df = pd.read_csv(pnl_file)
    print(f"Loaded PnL curve with {len(df)} points over {df['elapsed_s'].iloc[-1]:.1f} seconds.")

    # 1. Compute 1-second discrete increments
    t = df['elapsed_s'].values
    pnl = df['pnl'].values
    dp = np.diff(pnl)
    dt = np.diff(t)

    # Normalize to 1-second increments (where dt > 0)
    valid = dt > 0.01
    dp_1s = dp[valid] / np.sqrt(dt[valid])  # scaled increments
    t_mid = t[1:][valid]

    # 2. High-Frequency Econometric Jump Decomposition
    # Realized Variance (RV) includes continuous + jumps
    rv = np.sum(dp_1s ** 2)

    # Bi-Power Variation (BV) asymptotically contains ONLY continuous diffusion
    abs_dp = np.abs(dp_1s)
    bv = (np.pi / 2.0) * np.sum(abs_dp[1:] * abs_dp[:-1])

    jump_variance_ratio = max(0.0, (rv - bv) / rv)
    print("\n" + "=" * 60)
    print("1. VARIANCE DECOMPOSITION (Bi-Power Variation Test)")
    print("=" * 60)
    print(f"Total Realized Variance (RV):     {rv:>12,.1f}")
    print(f"Bi-Power Variation (BV - Calm):   {bv:>12,.1f}")
    print(f"Jump Variance Ratio (RV - BV)/RV: {jump_variance_ratio:>11.2%}")
    if jump_variance_ratio > 0.15:
        print("=> CONCLUSION: Strong, unmistakable presence of DISCONTINUOUS JUMPS!")
    else:
        print("=> CONCLUSION: Pure continuous diffusion dominant.")

    # 3. Robust Jump Separation via Median Absolute Deviation (MAD)
    # Estimate calm diffusion scale robustly
    med_dp = np.median(dp_1s)
    mad = np.median(np.abs(dp_1s - med_dp))
    sigma_calm = 1.4826 * mad
    print(f"\nRobust Calm Volatility (MAD sigma): {sigma_calm:.2f} $/sqrt(s)")

    # Jump detection threshold: 3.5 * sigma_calm
    threshold = 3.5 * sigma_calm
    is_jump = np.abs(dp_1s - med_dp) > threshold
    jump_indices = np.where(is_jump)[0]
    calm_indices = np.where(~is_jump)[0]

    jump_times = t_mid[jump_indices]
    jump_sizes = dp_1s[jump_indices]
    calm_dp = dp_1s[calm_indices]

    print("\n" + "=" * 60)
    print(f"2. JUMP IDENTIFICATION (Threshold = {threshold:.1f} $)")
    print("=" * 60)
    print(f"Detected Jumps in PnL:        {len(jump_sizes)}")
    print(f"Estimated Jump Intensity lambda: {len(jump_sizes) / (t[-1] - t[0]):.4f} jumps/sec (~{len(jump_sizes) / ((t[-1] - t[0])/60):.2f} jumps/min)")
    print(f"Positive Jumps:               {(jump_sizes > 0).sum()} ({np.mean(jump_sizes > 0):.1%})")
    print(f"Negative Jumps:               {(jump_sizes < 0).sum()} ({np.mean(jump_sizes < 0):.1%})")
    print(f"Average Absolute Jump Size:   {np.mean(np.abs(jump_sizes)):.1f} $")
    print(f"Max Jump Shock:               {np.max(np.abs(jump_sizes)):.1f} $")

    # 4. Distribution Shape Analysis on Jumps
    print("\n" + "=" * 60)
    print("3. JUMP DISTRIBUTION SHAPE FITTING")
    print("=" * 60)

    abs_jumps = np.abs(jump_sizes)

    # Goodness of fit against various distributions
    # Normal (Merton model assumption)
    loc_n, scale_n = stats.norm.fit(abs_jumps)
    ks_norm = stats.kstest(abs_jumps, stats.norm(loc=loc_n, scale=scale_n).cdf)

    # Exponential / Laplace (Kou Double-Exponential model assumption)
    loc_e, scale_e = stats.expon.fit(abs_jumps)
    ks_expon = stats.kstest(abs_jumps, stats.expon(loc=loc_e, scale=scale_e).cdf)

    # Student-t (Heavy-tailed power law)
    df_t, loc_t, scale_t = stats.t.fit(abs_jumps)
    ks_t = stats.kstest(abs_jumps, stats.t(df=df_t, loc=loc_t, scale=scale_t).cdf)

    # Uniform (Bounded jumps)
    loc_u, scale_u = stats.uniform.fit(abs_jumps)
    ks_uniform = stats.kstest(abs_jumps, stats.uniform(loc=loc_u, scale=scale_u).cdf)

    fits = [
        ("Laplace / Double-Exponential (Kou Model)", ks_expon.pvalue, ks_expon.statistic, f"scale={scale_e:.1f}"),
        ("Student-t (Heavy-Tailed Power Law)", ks_t.pvalue, ks_t.statistic, f"df={df_t:.2f}, scale={scale_t:.1f}"),
        ("Normal / Gaussian (Merton Model)", ks_norm.pvalue, ks_norm.statistic, f"mean={loc_n:.1f}, std={scale_n:.1f}"),
        ("Uniform (Flat Bounded)", ks_uniform.pvalue, ks_uniform.statistic, f"min={loc_u:.1f}, max={loc_u+scale_u:.1f}"),
    ]
    fits.sort(key=lambda x: x[2])  # lower KS statistic is better fit

    print(f"{'Candidate Distribution Family':<40} | {'KS Stat':<8} | {'p-value':<8} | {'Fitted Parameters'}")
    print("-" * 80)
    for name, pval, stat_val, param_str in fits:
        print(f"{name:<40} | {stat_val:<8.4f} | {pval:<8.4f} | {param_str}")

    best_fit = fits[0][0]
    print(f"\n=> BEST-FITTING JUMP DISTRIBUTION SHAPE: {best_fit}")

    # 5. Jump Arrival Process (Poisson vs Clustered/Hawkes)
    jump_intervals = np.diff(jump_times)
    # For Poisson process, coefficient of variation (std / mean) should be 1.0.
    # CV > 1.0 indicates clustering / self-excitation (Hawkes).
    cv_intervals = np.std(jump_intervals) / np.mean(jump_intervals)
    print(f"\nJump Inter-Arrival Dispersion (Std / Mean): {cv_intervals:.2f}")
    if cv_intervals > 1.2:
        print("=> ARRIVAL PROCESS: Clustered / Self-Exciting (Hawkes/Burst-like)")
    elif 0.8 <= cv_intervals <= 1.2:
        print("=> ARRIVAL PROCESS: Independent Poisson Process (Memoryless)")
    else:
        print("=> ARRIVAL PROCESS: Periodic / Sub-Poisson")

    # 6. Calm Diffusion Properties
    print("\n" + "=" * 60)
    print("4. CALM REGIME PROPERTIES (Maker Edge & Noise)")
    print("=" * 60)
    calm_drift = np.mean(calm_dp)
    calm_std = np.std(calm_dp)
    calm_skew = stats.skew(calm_dp)
    calm_kurt = stats.kurtosis(calm_dp)

    print(f"Calm Drift Rate (Maker spread capture): {calm_drift:+.3f} $/sec ({calm_drift*3600:+.1f} $/hr)")
    print(f"Calm Volatility (Inventory diffusion):   {calm_std:.3f} $/sqrt(s)")
    print(f"Calm Skewness:                          {calm_skew:.3f}")
    print(f"Calm Excess Kurtosis:                   {calm_kurt:.3f} (Gaussian = 0.0)")

    # 7. Ground Truth Verification against actual round2 catalog!
    gt_file = Path("findings/reports/round2_jumps_catalog.json")
    if gt_file.exists():
        with open(gt_file) as f:
            gt_jumps = json.load(f)
        print("\n" + "=" * 60)
        print("5. GROUND TRUTH VALIDATION (Comparing against true Round 2 jumps)")
        print("=" * 60)
        print(f"True Market Discontinuities in Tape: {len(gt_jumps)} jumps")
        gt_times = np.array([j['time_s'] for j in gt_jumps])
        gt_sizes = np.array([j['size'] for j in gt_jumps])

        # Match detected PnL jumps with true market jumps (within 3s window)
        matched = 0
        pnl_to_true_size = []
        for jt, js in zip(jump_times, jump_sizes):
            dt_match = np.abs(gt_times - jt)
            min_idx = np.argmin(dt_match)
            if dt_match[min_idx] <= 3.0:
                matched += 1
                pnl_to_true_size.append((js, gt_sizes[min_idx]))

        print(f"PnL Jump Detections Matching True Market Jumps: {matched} / {len(jump_sizes)} ({matched / len(jump_sizes):.1%})")
        recall = matched / len(gt_jumps)
        print(f"Ground Truth Market Jumps Recovered from PnL:   {matched} / {len(gt_jumps)} ({recall:.1%})")

        if pnl_to_true_size:
            pnl_s, gt_s = zip(*pnl_to_true_size)
            corr = np.corrcoef(np.abs(pnl_s), np.abs(gt_s))[0, 1]
            print(f"Correlation between PnL shock size and true price jump size: {corr:.3f}")

if __name__ == "__main__":
    main()
