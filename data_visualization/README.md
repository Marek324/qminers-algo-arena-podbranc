# Data Visualization: Reverse Engineering the Generative Distribution from PnL

This directory contains visual artifacts demonstrating how the mathematical shape of the market's price process can be recovered solely from the strategy's cumulative PnL time series $\Pi(t)$, without access to the order book or trade tape.

## Generated Visualizations

1. **Private Evaluation Tape Reconstruction (12,000s = 3.33 hr)**:
   - File: [`results2_market_reconstruction.png`](results2_market_reconstruction.png) (20×14 inches, 150 DPI)
   - Source Data: [`docs/results (2)/pnl.csv`](file:///home/alcrid/Projects/Hackatons/Qminers/docs/results%20(2)/pnl.csv) and [`docs/results/pnl.csv`](file:///home/alcrid/Projects/Hackatons/Qminers/docs/results/pnl.csv)
   - Shows the full reconstructed price path $S(t) = 10,001 + \Pi(t)/10$, empirical jump distribution fitting, calm Brownian diffusion, and performance comparison against the +$1.29M champion run.

2. **6,000s Diagnostic Calibration Experiment**:
   - File: [`pnl_distribution_inference.png`](pnl_distribution_inference.png) (18×12 inches, 150 DPI)
   - Source Data: [`logs/probe_experiment/pnl.csv`](file:///home/alcrid/Projects/Hackatons/Qminers/logs/probe_experiment/pnl.csv)
   - Diagnostic validation matching inferred jumps against the ground-truth catalog.

---

## Detailed Breakdown of `results2_market_reconstruction.png`

```
+---------------------------------------------------------------------------------------------------+
| Panel A: 12,000-Second (3.33 hr) Reconstructed Price Process S(t) from `docs/results (2)`        |
+-------------------------------------------------+-------------------------------------------------+
| Panel B: Strategy PnL Trajectories              | Panel C: High-Frequency Price Increments        |
| (Champion +$1,290.8k vs Probe +$4.8k)           | (Calm Diffusion Band ±5.2t vs Outlier Spikes)   |
+-------------------------------------------------+-------------------------------------------------+
| Panel D: Jump Size Distribution (Log Scale)     | Panel E: Calm Regime Dynamics                   |
| (Extreme Fat Tails: Student-t df = 0.94)        | (Standard Gaussian Diffusion: σ = 1.48 t/√s)    |
+-------------------------------------------------+-------------------------------------------------+
```

### Statistical Findings from `docs/results (2)`:
- **Total Realized Variance (RV)**: $323,826.1$
- **Bi-Power Variation (BV)**: $157,799.3$
- **Jump Variance Contribution**: **51.27%** of all market movement comes from discontinuous jumps.
- **Calm Diffusion Volatility**: $\sigma = 1.483\text{ ticks}/\sqrt{\text{s}}$ (exact match with Round 2 Public).
- **Major Jumps ($\ge 20$ ticks)**: **51 jumps** identified across the 12,000-second session (0.26 / min).
- **Best-Fitting Jump Distribution**:
  - **Student-t Power Law**: Degrees of freedom **$\text{df} = 0.94$** (extreme Cauchy-like fat tails).
  - **Laplace / Kou Double-Exponential**: Scale parameter $6.79\text{ ticks}$.
  - **Gaussian / Merton Jump-Diffusion**: Overwhelmingly rejected ($p = 0.0000$, KS stat $0.3428$).
- **Calm Regime Kurtosis**: Excess kurtosis $\approx 0.0$, proving pure Gaussian Brownian motion between jumps.
