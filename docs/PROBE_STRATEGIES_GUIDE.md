# Diagnostic Probe Strategies: Platform Execution & Reconstruction Guide

This document provides instructions for deploying our three diagnostic probe strategies on the competition evaluation sandbox, and how to decode the resulting output data.

---

## 1. Overview of the Three Diagnostic Probes

Because the evaluation sandbox does not provide full public order book or trade tape logs, each probe is mathematically designed to exploit the exchange liquidation engine to reconstruct the latent market variables:

| Probe Strategy | Source File | Position Held | What It Extracts | Mathematical Reconstruction Formula |
| :--- | :--- | :---: | :--- | :--- |
| **Probe 1 (Bid / Mid Price)** | [`strategies/round2/simple_probe_mm.py`](../strategies/round2/simple_probe_mm.py) | **$+10\text{ lots}$** | **Best Bid $P_{\text{bid}}(t)$** & Macro Midprice | $P_{\text{bid}}(t) = 10,001.0 + \frac{\Pi_1(t)}{10.0}$ |
| **Probe 2 (Ask / Spread Clock)** | [`strategies/round2/probe_short_spread.py`](../strategies/round2/probe_short_spread.py) | **$-10\text{ lots}$** | **Best Ask $P_{\text{ask}}(t)$** & Continuous Spread | $P_{\text{ask}}(t) = \frac{\text{Cash} - \Pi_2(t)}{10.0}$<br>$s(t) = P_{\text{ask}}(t) - P_{\text{bid}}(t)$ |
| **Probe 3 (Depth / Queue Size)** | [`strategies/round2/probe_depth_flow.py`](../strategies/round2/probe_depth_flow.py) | **$+100\text{ lots}$** | **Queue Thickness $Q_{\text{touch}}(t)$** & Slippage | $\text{Slippage}(t) = P_{\text{bid}, 10}(t) - \frac{\text{Cash} + \Pi_3(t)}{100.0}$ |

---

## 2. Submission & Platform Execution Instructions

### A. Submitting to Arena Sandbox
1. Go to **arena.qminers.com** $\to$ **Dashboard** $\to$ **Submit strategy**.
2. Select one of the probe files:
   - To get the **Best Ask & Continuous Spread**: upload `strategies/round2/probe_short_spread.py`.
   - To get the **Queue Thickness & Book Depth**: upload `strategies/round2/probe_depth_flow.py`.
   - To get the **Baseline Bid & Price**: upload `strategies/round2/simple_probe_mm.py`.
3. Submit and wait for evaluation completion. All three probes execute exactly 1 trade at $t = 0.005\text{s}$ and hold passively, guaranteeing **$0.00 breach penalty**.

### B. Downloading Results
1. Once the run finishes, click **Download results** (downloads a `.zip` archive containing `metadata.json`, `result.json`, and `pnl.csv`).
2. Move the `.zip` file into the `docs/` folder:
   - For Probe 1: `docs/results.zip` (unzipped into `docs/results (2)/`).
   - For Probe 2: `docs/results_short.zip` (unzip into `docs/results_short/`).
   - For Probe 3: `docs/results3.zip` (unzipped into `docs/results3/`).

---

## 3. Mathematical Decoding Formulas

### A. Reconstructing the Continuous Bid-Ask Spread $s(t)$
When running **Probe 1** ($+10\text{ lots}$) and **Probe 2** ($-10\text{ lots}$):
- From Probe 1 PnL ($\Pi_1$ with Cash $= -\$100,010$):
  $$P_{\text{bid}}(t) = \frac{\Pi_1(t) + 100,010.0}{10.0}$$
- From Probe 2 PnL ($\Pi_2$ with Cash $= +\$99,990$):
  $$P_{\text{ask}}(t) = \frac{99,990.0 - \Pi_2(t)}{10.0}$$
- **Continuous Bid-Ask Spread**:
  $$s(t) = P_{\text{ask}}(t) - P_{\text{bid}}(t)$$
- **Exact True Midpoint**:
  $$M(t) = \frac{P_{\text{bid}}(t) + P_{\text{ask}}(t)}{2.0}$$

### B. Reconstructing Order Book Depth $Q_{\text{touch}}(t)$
When running **Probe 1** ($+10\text{ lots}$) and **Probe 3** ($+100\text{ lots}$):
- Average price across 100 lots:
  $$P_{\text{bid}, 100}(t) = \frac{\Pi_3(t) + 1,000,100.0}{100.0}$$
- **Depth Slippage**:
  $$\text{Slippage}(t) = P_{\text{bid}, 10}(t) - P_{\text{bid}, 100}(t)$$
  - When $\text{Slippage}(t) = 0.0$: Touch queue has $\ge 100\text{ lots}$.
  - When $\text{Slippage}(t) > 0.0$: Touch queue is depleted ($< 100\text{ lots}$).
- **Exact Touch Queue Lots**:
  $$Q_{\text{touch}}(t) = 100 - \left(100 \times \text{Slippage}(t)\right)$$

---

## 4. Running the Automated Visualization Scripts

Once the results are unzipped in `docs/`:

1. **To regenerate the full 12,000s price reconstruction & distribution fit**:
   ```bash
   uv run --with matplotlib python scripts/visualize_results2.py
   ```
   *Outputs*: `data_visualization/results2_market_reconstruction.png`

2. **To regenerate the combined depth depletion analysis**:
   ```bash
   uv run --with matplotlib python scripts/visualize_depth_combined.py
   ```
   *Outputs*: `data_visualization/results_depth_combined.png`
