# Round 3: what the market is, what the strategy's model says, and what moved PnL

A reference for the next round's tuning. It covers the structure of the generator, how `hmm_imm_*` models it, what
each constant means, which values the data supports, and every experiment of 2026-11-08 with its numbers. Read this
before changing values, so a new tape is tuned from understanding rather than blind search.

## 1. The market (round 3 generator, `logs/generator/round3_public_params.json`)

The tapes are synthetic (docs/DATA_ANALYSIS.md). One engine, refitted per round:

- **Hidden price S.** Brownian with a calm or a storm volatility (fit: 1.30 and 4.22 ticks/sqrt(s)), plus Pareto
  jumps (tail alpha 1.8, about 0.0065/s calm and 0.07/s in storms, 54% up). Jumps are permanent. IOC impact on S is
  partly transient on round 3 (the 10-30 s mid z-score has IC -0.17), but trading against it hurt.
- **Regimes.** Calm spells last about 182 s, storms about 91 s (35% of the time on round3_public). In storms there are
  more IOCs (11.6/s vs 8.3/s), new orders land further from S, and the jumps cluster.
- **Limit orders** (236/s): buys at floor(S - X), sells at ceil(S + X), with X gamma-distributed (wider in storms),
  capped so they never cross the book (a capped order sits at ask - 1 / bid + 1 and only bounds S). They are cancelled
  after an exponential life of about 0.57 s. These new orders are the readings of S.
- **IOCs** are priced relative to S with a fat-tailed offset (5-14 ticks through the touch now and then) and move S
  by impact_k * q^beta (k 0.026, beta 0.49).
- **Exchange** (docs/MARKET.md): 5 ms latency, price-time priority, flat 10,000 $/s penalty while |inventory| > 100
  (the same for 101 and 1000 lots), orders rejected past 1000.

## 2. Where the money is (attribution, `scripts/pnl_attribution.py`)

hmm_deep_clear_ks on round3_public (713,244), markout of own fills in ticks per lot:

| regime | fill type | lots | 1 s | 5 s | 30 s |
|---|---|---|---|---|---|
| storm | take at best | 278k | +1.36 | +1.65 | +1.56 |
| calm | passive at touch | 527k | +0.16 | +0.21 | +0.40 |
| calm | take at best | 154k | +0.73 | +0.69 | -0.12 |
| storm | passive at touch | 69k | -0.05 | +0.44 | +0.82 |

- Storm taking earns about 70% of the PnL in 35% of the time.
- Inventory sits at +-100 almost always: the strategy is capacity-bound. `scripts/missed_opps.py` puts about 135k of
  hindsight value behind the cap and about 170k missed with room left.
- On the eval twin the storm-passive leak is much larger (-1.3 t/lot at 5 s). The twin's storms are known to be too
  harsh (docs/EVALUATION_DATASET_ANALYSIS.md), so trust round3_public more for storm passive fills.

## 3. The strategy structure (`strategies/round3/hmm_imm_b2.py`)

1. **Readings.** Every new market limit order within KF_LEVELS (calm) / KF_LEVELS_V (storm) ticks of its own best
   price is a reading y = price -+ c of S (c = learned average outward distance, per regime). Deeper orders (up to
   DEEP_MAX) count with noise r * DEEP_R * extra_depth, so the run of far orders after a jump pulls S_hat over within a
   few orders, while one stray far order barely moves it.
2. **IMM filter** (replaces the HMM + single Kalman of hmm_mm). Three Kalman filters (calm Q, storm V, jump J) mixed by
   a per-order Markov switch. Mode evidence per new order:
   - placement distance n (half-ticks beyond the first, capped at H_NMAX), geometric per mode with means learned
     online. This is the old HMM likelihood;
   - price innovation, used only for jump vs no jump (IMM_WP_J). Using it for calm vs storm (IMM_WP > 0) hurts badly.
   p_vol = mu_V + mu_J, S_hat = sum mu_i m_i.
3. **Quiet policy** (p_vol < P_ON): fair = mid + K_S (S_hat - mid) + K_IMB * imbalance + K_FLOW * flow/depth - K_INV *
   inventory. Cross at the touch when fair is EDGE_TAKE through it, else quote at the touch (improve by one when the
   spread allows and EDGE_IN fits), QTY per side, capped by MAX_POS.
4. **Volatile policy**: fair with V_K_S, V_K_FLOW. Take at the best opposite price when fair is V_EDGE_TAKE through it
   (V_QTY), no resting quotes.

## 4. Constants of hmm_imm_b2 and what they say

| constant | value | meaning | evidence |
|---|---|---|---|
| SIG_Q / SIG_V | 0.7 / 11 | S volatility per mode, ticks/sqrt(s) | fit says 1.3 / 4.2; SIG_V far above it, see section 6 |
| H_PQV / H_PVQ | 0.00003 / 0.0003 | per new order calm->storm / storm->calm | 236 orders/s: about 140 s / 14 s expected durations |
| H_NMAX | 5 | cap on one order's regime evidence | 4: -5k, 6: -11k, 20: -50k+ |
| P_ON | 0.2 | p_vol that switches to the volatile policy | 0.18: -3k, 0.22: -12k |
| KF_LEVELS / _V | 2 / 3 | readings used up to this depth | 1: -26k, 3: -18k |
| DEEP_R | 2.5 | noise multiplier per extra tick of depth | 2.5-3 flat |
| IMM_JVAR | 25 | one-step S variance in the jump mode (5 ticks sd) | 15: -5k, 35: -12k |
| IMM_WP / IMM_WP_J | 0 / 1 | weight of price evidence calm-vs-storm / jump | IMM_WP 0.1: -90k on the twin |
| K_S / V_K_S | 0.85 / 0.7 | weight of S_hat - mid in fair | flat within +-0.05 |
| EDGE_TAKE / V_EDGE_TAKE | 0.25 / 0.5 | ticks fair must be through the touch to take | 0.2 / 0.3: -6k / -8k |
| K_INV | 0.0015 | ticks of fair per lot of inventory | 0.15 ticks at the cap |
| QTY / V_QTY | 150 / 200 | order size; above MAX_POS so a cross can flip | |

## 5. Experiments and results (2026-11-08)

Scores are PnL on round3_public / r3_eval_twin. Sweeps ran on Barbora (s1-s15, c1-c4) and on Lambda
185.175.110.225 (imm1-imm3, p1-p4, x1).

| strategy / change | public | twin | verdict |
|---|---|---|---|
| hmm_mm (round 2) | 568,382 | 731,771 | baseline |
| hmm_deep_clear_ks | 713,244 | 925,207 | best HMM |
| hmm_imm_r3 (IMM, jump rule on) | 704,572 | 969,300 | best on the twin |
| hmm_imm_pon02 (P_ON 0.2) | 710,629 | 939,161 | |
| hmm_imm_sv11 (+ SIG_V 11) | 721,624 | | public only |
| hmm_imm_b2 (7 values) | 728,501 | | public only, local optimum |

Rejected (lost on the tapes that matter):
- leverage over 100 in every form tried: jump sniping (0 of 192 settings beat base on both tapes, worst -30.9M on
  the twin), small snipes 130-500 lots, deep sweep catchers (-115k to -170k). The flat penalty (10 $/ms) is more than
  the edge except in the jump instant, and round 3 leaves few stale lots at a jump (lever_econ: about 16k/h max).
- pulling quotes before storms (far-order, big-IOC and p_vol guards), mean-reversion skew, laddered quotes, waiting on
  exits, tighter spreads, the IOC-sweep bound on S.
- on hmm_imm_b2 (sweep x1, `strategies/round3/hmm_imm_x.py`, public): jump-mode leverage (cap 300-1000 while the
  IMM jump mode >= 0.1-0.5 and fair 1-3 ticks through) 724.8k-728.5k with 50-1,600 penalty, never better; sweeping
  several levels when taking (Q_SWEEP / V_SWEEP) 728.4k / 728.5k; size x1.5-2 on the inventory-reducing side 715.9k;
  KEEP_LEVELS 0 719.5k; MIN_NEW 30 717.5k; MIN_NEW 1 729.5k (+1k, noise).

## 6. How to tune for a new round

1. **Refit the generator** (skill `estimate-market-model`), but use it to understand the tape, not to set constants.
   Every textbook value tried on round 3 traded worse than the tuned knobs (section 8).
2. **Start from hmm_imm_b2's values and sweep one step either side.** The sensitive knobs, in order: KF_LEVELS,
   H_NMAX, P_ON, SIG_V, IMM_JVAR, SIG_Q, K_S, EDGE_TAKE, H_PQV.
   - Expect a plateau of about 1% around the optimum. Gains under about 30k on one tape (the pnl_se) are noise.
   - Check any winner on a second tape (the previous round's public tape, or a twin) before trusting it.
   - SIG_V and the storm noise trade off against each other (only sigma^2 / r sets the gain). If the fitted storm
     sigma changes, scale SIG_V by about sigma_new / sigma_old.
3. **Things that never worked on rounds 2-3:** leverage over 100, pulling quotes, ladders, close-only takes,
   storm passive quotes, cleaner regime models.
4. **Where the remaining money is:** storm takes (capacity-bound at +-100) and calm takes held through their decay
   (-1 t/lot from 5 s to 30 s). Both were attacked here without success.

Open questions:
- Why is the chattery regime switch better? Measure the volatile policy's PnL per switch episode by duration.
- The eval twin disagrees with public on storm passive fills. A better-calibrated twin (fix the n pmf and the
  drop-mode imbalance found in section 7) would make two-tape checks trustworthy.

## 7. Expert reviews (2026-11-08)

Three Opus reviewers read hmm_imm_b2.py against the generator (`analysis/generator.py` is the round 3 engine;
`scripts/gen_tape.py` is round 1). Their claims are below. The A/B results in section 8 decide which hold.

### Mathematics (filtering)

Generator facts the strategy ignores:
- X is the same in calm and storm (gamma shape 0.43, scale 2.69, shift 0.05). D = S - bid = X + U(0,1) has mean
  1.71 and variance 3.2 overall, and mean 1.1 and variance 0.47 within 3 ticks of the touch.
- Crossing orders are dropped, not capped (`cross_mode: drop`), and D >= 0.05: S lies strictly above a new bid and
  below a new ask.
- IOC limit price is round(S +- 1), so a market buy filling at p means S >= p - 1.5.
- Every IOC moves S by 0.0258 q^0.49.

What the tuned values are really saying:
- **SIG_V 11 is the right gain in the wrong units.** The storm reading noise r_v = 0.25 n_v (1 + n_v) + 0.1, about 4.0,
  is about 6.7x r_q, because storm placements look wide (the book is stale) even though X does not change. The Kalman
  gain depends only on sigma^2 / r, and 11 * sqrt(r_q / r_v) = 4.2 = the fitted storm sigma. The search rediscovered
  the generator.
- **H_PVQ 0.0003 per order is not a model parameter.** It implies 14 s storms (the fit says 91 s) and tracks the jump
  excitation bursts (tau 3.4 s). It is a policy knob for leaving the storm policy fast.
- **SIG_Q 0.7 is about half the fit** (1.3 diffusion, about 2.5 with calm jumps and impact): the fair value prefers a
  smoother S_hat.
- **IMM_PJ 0.0002 is right from storms and 7x too high from calm.**
- **With DEEP_R > 0 and DEEP_MAX 1000 every order is used.** KF_LEVELS only sets where the noise inflation starts.
  The inflation is measured from the stale best, so right after a jump it down-weights the fresh, accurate readings.
- **IMM_WP must be 0** because a Gaussian likelihood against the gamma tail (P(X > 5) = 4.4%) turns tail orders into
  mode flips.

Implementation issues:
- The capped bin n = H_NMAX should get the tail probability r^n.
- One depth limit is used for all modes.
- Depth is measured from a best price that includes our own quotes.
- The hard support (S above a new bid) is unused, and so is the IOC information.
- `self.P` is unused.

### Economics (decision layer)

- **A lot of capacity is worth about 3x more in a storm** (2.1 $/lot/s) than in calm (0.7 $/lot/s).
- **EDGE_TAKE 0.25 is a fine threshold, but calm takes are held too long.** They earn +0.77 t/lot at 5 s and -0.25
  at 30 s, about 144k of value at risk. The opening part of a take beyond closing (QTY 150 > MAX_POS 100) carries the
  decay.
- **V_EDGE_TAKE does not bind** (0.45 and 0.5 tie). Storm takes realize +1.36. What binds is capacity and the 5 ms
  race: a limit at the best fills only if the best survives the latency.
- **K_INV is a price on capacity.** The score is risk-neutral, so there is no variance term. V_K_INV 0 is
  inconsistent with storm capacity being the most valuable.
- **The flow check double-counts** (it regresses the mid on flow while S_hat and imbalance are already in fair) and
  clips at 1.
- **Overshoot over 100 stays closed.** 900 extra lots earn about 1.9k/s against 10k/s of penalty. Perfect hindsight
  would add +455k to hmm_imm_b2, but the expected net without it is negative for every trigger and hold: [LEVERAGE.md](LEVERAGE.md).
- Proposed A/Bs: one tick of slack on storm takes, close-only calm takes unless the edge is strong, an age skew on
  take-born inventory, a storm passive quote on the reducing side, calibrated flow.

### Probability

The reviewer replayed round3_public plus a twin with known regimes.
- **The fitted generator itself misfits the real tape** (fit error 54.65). The twin's calm placement distances are far
  too spread: P(n = 0) is 0.28 against 0.49 real. Its `cross_mode=drop` also creates a buy/sell imbalance that
  predicts the mid (corr 0.51) where the real tape has none (0.01). Twin results understate the distance evidence and
  overstate staleness.
- **The geometric distance likelihood has the wrong shape.** The real storm/calm log-likelihood ratio for
  n = 0..12 is [-0.68, 0.59, -0.73, 2.09, 0.42, 2.91, 1.89, 3.23, 2.07, 2.15, 2.10, 2.10, 1.48]. It follows a lattice:
  odd n only occurs at even spreads. It peaks at n 5-7 and then flattens. The geometric gives -0.58 + 0.29 n: the
  wrong sign at n = 1 and 2, and growing without bound. That is why H_NMAX 5 beats 20.
- **Orders are strongly correlated.** The variance of the summed calm log-likelihood is 11x the independent value
  at 0.25 s and 25x at 1 s. The geometric works because it under-weights each order about 4x, which acts as hidden
  tempering. b2-like filtering makes about 240 regime switches per hour; a 13-bin pmf tempered by w = 0.04 makes 41
  with the same classification.
- **Inserts seen are 196/s, not 236/s.** H_PQV 3e-5 gives about 170 s of calm (fit 182). H_PVQ 3e-4 gives about
  17 s storms (fit 91). The generator switches as a continuous-time Markov chain, so memoryless switching is the
  right form. The 182 / 91 s come from a 30 s-window HMM that cannot see short storms.
- **Jumps are Pareto above 5 ticks** (alpha 1.82, median 7.3, infinite variance), at 3.3e-5 per order in calm and
  3.6e-4 in storms. With JVAR 25, the jump mode only wins for innovations above about 7 ticks. SIG_V 11 absorbs the
  small jumps plus the post-jump excitation (3.5 |J| e^(-t/3.4 s) ticks^2/s).
- **IOC gaps are exponential** at 8.25 vs 11.1/s, worth about 0.2 nats/s. Sizes carry no regime information.
- **Bug:** under the IMM, H_IOC wrote `self.pv`, which the next order overwrote, so it did nothing. Fixed behind
  H_IOC in hmm_imm_x.py.
- **Censoring:** there is no pile-up at ask-1 / bid+1 on the real tape. Do not treat those orders as censored.
- **The spread is already contained in odd n.** Adding it separately double-counts.

## 8. A/B results of the reviewers' proposals (Lambda, round3_public, base hmm_imm_b2 728,501)

All proposals are switches in `strategies/round3/hmm_imm_x.py`, off by default (off = hmm_imm_b2 exactly).

Sweep e1 (mathematics + economics), one change at a time:

| proposal | public | change |
|---|---|---|
| storm take with one tick of slack (V_EDGE_SLACK 0.75 / 1 / 1.5) | 728,650 / 728,624 / 728,569 | +0.1k, noise |
| age skew on take-born inventory (AGE_SKEW 0.15 / 0.3) | 728,524 / 728,363 | 0 |
| IOC sweep bound on S (IOC_BOUND 2.5 / 1.5) | 728,268 / 724,108 | -0.2k / -4k |
| proper jump mode (J_PROPER; + PJ_OUT 1) | 724,632 / 720,073 | -4k / -8k |
| hard support S above a new bid (SUPPORT) | 724,224 | -4k |
| depth from the market best (MKT_BEST; + KF_LEVELS 1) | 723,803 / 703,037 | -5k / -25k |
| same reading noise per mode (R_SAME) + SIG_V 4.2 / 5 / 11 | 716,388 / 718,154 / 626,309 | -10k / -10k / -102k |
| V_K_INV 0.002 | 712,722 | -16k |
| close-only calm takes (EDGE_TAKE_OPEN 0.4 / 0.6 / 1.0) | 708,084 / 696,379 / 688,097 | -20k to -40k |
| storm passive quote on the reducing side (V_PASS_EDGE 1.25 / 0.75) | 695,283 / 668,350 | -33k / -60k |
| IOC impact into S (IOC_K 0.0258; + K_FLOW 0.7) | 674,213 / 700,335 | -54k / -28k |
| capped-bin tail likelihood (NMAX_TAIL) | 665,442 | -63k |
| flow scale clip 2 (FLOW_CAP) | 625,074 | -103k |

Sweep e2 (probability):

| proposal | public | change |
|---|---|---|
| jump-entry probability per mode (PJ_SPLIT: 3.3e-5 calm, 3.6e-4 storm) | 728,765 | +0.3k, noise |
| post-jump excitation of the storm variance (EXC; SIG_V 11 / 8 / 6 / 4.5) | 727,710 / 715,050 / 707,975 / 665,905 | -1k to -63k |
| IOC-gap evidence in the IMM (H_IOC, weight 0.5 / 0.25) | 724,124 / 725,296 | -4k / -3k |
| storm exit hazard toward the fit (H_PVQ 1e-4 / 5.6e-5) | 724,602 / 725,416 | -4k / -3k |
| SIG_Q 0.9 / 1.3 (toward the fit) | 721,300 / 681,301 | -7k / -47k |
| 13-bin tempered placement pmf (H_CAT; H_W 0.04 / 0.07 / 0.1 / 0.2) | 700,972 / 691,867 / 672,905 / 665,552 | -28k to -63k |
| H_CAT + slower storm exit (H_PVQ 5.6e-5 / 1e-4) | 711,477 / 711,905 | -17k |
| H_CAT + P_ON 0.35 / 0.5 | 678,455 / 670,805 | -50k / -58k |

The better-specified regime model (H_CAT: 41 instead of 240 switches/hour) loses 17-63k. The fast, chattery regime
switch is part of the edge: the volatile policy profits from flipping in and out quickly around short bursts, which
a "correct" slow regime model averages away.

**Lesson: the tuned values are policy knobs, not model errors.** The textbook-correct values are worse every time.
The tuned values compensate for what the clean model leaves out: the 5 ms latency race, the stale book after jumps,
correlated orders, and the decision layer co-tuned to the filter's quirks. Examples:
- SIG_V 11 with the storm noise r_v gives the same Kalman gain as the fitted 4.2 with r_q. But it also sets the mode
  likelihoods, and switching to the clean pair loses 10k.
- The over-weighted capped bin is part of what makes storms trigger at the right moment.

When a new round's tape arrives, re-tune these knobs around the current values. Do not reset them to the generator
fit.
