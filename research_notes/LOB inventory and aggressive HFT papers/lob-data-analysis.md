# Limit Order Book Data Analysis and Microstructure Empirics (computable online from an L3 feed)

Scope: results that can be implemented event-by-event in O(1) / small O(k) from a message feed
(insert / cancel / modify / trade with price, size, side, sometimes owner id), with the reported
magnitudes, and an explicit note on where the result depends on continuous prices, zero latency, or a
reactive (rule-based / strategic) background flow.

Context I kept in mind while writing (from the local repo, not from the literature): the target
environment is a single anonymised instrument, discrete integer ticks, 1-tick spread most of the time,
FIFO price-time priority, 5 ms send/cancel latency, public depth 20 ticks, and a **fixed, exogenous
tape** that never reacts to our orders. `data/sample.csv` is 53,758 tape rows per hour ≈ 15 events/s
(mean inter-event gap ≈ 67 ms), so 5 ms is ≈ 0.07 of an average inter-event gap: at that latency you
usually land *before* the next market event, but not always in busy bursts. Everything below is
filtered through "does this survive that environment?"

---

## 1. Order flow imbalance (OFI): definition, horizons, magnitudes, and whether it survives at short horizons

### Takeaway
OFI is the one microstructure variable with a clean, near-mechanical, large-magnitude relation to
**contemporaneous** mid-price change: average R² ≈ 65% at a 10-second horizon across 50 US stocks,
linear, one parameter, slope ∝ 1/depth. But it is an *accounting* relation, not a forecast: the
out-of-sample predictive R² of OFI-based models at a 1-minute-ahead horizon is **negative** in the
most careful recent study. Anything you can trade at 5 ms latency is the residual, not the 65%.

### Cited Findings

- **Exact OFI definition.** For each event n at the best bid/ask with prices P^b, P^a and queue sizes
  q^b, q^a:
  `e_n = 1{P^b_n ≥ P^b_{n−1}}·q^b_n − 1{P^b_n ≤ P^b_{n−1}}·q^b_{n−1} − 1{P^a_n ≤ P^a_{n−1}}·q^a_n + 1{P^a_n ≥ P^a_{n−1}}·q^a_{n−1}`,
  and `OFI_k = Σ_{n=N(t_{k−1})+1}^{N(t_k)} e_n`. Regression `ΔP_k = α_i + β_i·OFI_k + ε_k` on
  10-second mid-price changes in ticks. — [Cont, Kukanov & Stoikov, arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
- **OFI intentionally confuses a market sell with a cancel of a resting buy**: "this variable treats a
  market sell and a cancel buy of the same size as equivalent, since they have the same effect on the
  size of the bid queue." This is the modelling choice that makes it work; it also means OFI cannot
  distinguish aggressive from non-aggressive liquidity withdrawal. — [arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
- **Magnitudes.** NYSE TAQ, 50 S&P 500 stocks (2010). 273 half-hour subsamples per stock, ~180
  observations each. **Mean R² = 65%**, per-stock averages roughly 47%–79%; statistically significant
  at 95% in ~98% of samples. Adding a quadratic term `OFI·|OFI|` raises the average R² only from 65%
  to 68% and is insignificant in most samples ⇒ the relation is genuinely linear, not concave, at this
  horizon. — [arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
- **Slope.** β is inversely proportional to market depth and shows the same intraday seasonality as
  spreads/depth/volatility; the paper uses this to explain the √-law as a scaling consequence of a
  linear OFI relation. — [arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
- **Trade imbalance is a strictly worse regressor.** They define `TI_k = Σ b_n − Σ s_n` (buyer- minus
  seller-initiated volume) and show that once OFI is in the regression the TI coefficient "must be
  indistinguishable from zero"; also "the relation between price changes and trade volume is found to
  be noisy and less robust than the one based on order flow imbalance". — [arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
- **Multi-level OFI (MLOFI).** 6 Nasdaq stocks, full year 2016, 5 levels. Out-of-sample fit improves
  monotonically with each added level: RMSE improvement ≈ **65–75% for large-tick stocks** and
  ≈ **15–30% for small-tick stocks** relative to level-1 only. OLS on levels 2+ is unstable because net
  order flows at different levels are strongly correlated (multicollinearity); they use Ridge instead,
  and note OLS "over-fits" for some stocks. — [Xu, Gould & Howison, arXiv:1907.06230](https://arxiv.org/abs/1907.06230)
- **"Integrated OFI" (PCA-aggregated multi-level OFI).** Nasdaq ITCH/LOBSTER, contemporaneous
  out-of-sample R² (30-min fit, next 30-min test): best-level OFI **64.6%**, integrated OFI **83.8%**;
  adding cross-asset OFI gives a further ~1.4 pp (66.0% vs 64.6%) at best level. The first principal
  component of the level-wise OFI vector explains **>89%** of total variance ⇒ one scalar can stand in
  for the whole depth profile, which is the O(1) version. — [Cont, Cucuringu & Zhang, arXiv:2112.13213](https://arxiv.org/abs/2112.13213)
- **OFI does not forecast.** Same paper, one-minute-ahead returns, out-of-sample R²: OFI model
  **−0.37**, cross-impact OFI model **−0.10** (negative = worse than the unconditional mean). Lagged
  cross-impact "mainly manifests at short-term horizons and decays rapidly in time". — [arXiv:2112.13213](https://arxiv.org/abs/2112.13213)
- **Horizon stability / short horizons.** The authors claim stability "across time scales and across
  stocks" but the estimation grid in the paper is 10 s (with sub-samples of 10 s to a few minutes).
  I found no verified published estimate of OFI R² at 100 ms–1 s on message-level data. — [arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
- **Deep-learning confirmation of universality.** A large-scale DL model on billions of US equity
  quotes/trades finds a stationary, stock-independent mapping from order-flow history to price moves,
  with stable out-of-sample directional accuracy even on stocks outside the training sample — i.e. the
  OFI-type relation is not an artefact of linear regression. — [Sirignano & Cont, arXiv:1803.06917](https://arxiv.org/abs/1803.06917)
- **Markovian / queueing foundation.** The linear OFI relation is derived as the first-order behaviour
  of a Markovian limit order book with unit order sizes; the price is a functional of the queue
  imbalance at the touch. — [Cont & de Larrard, arXiv:1104.4596](https://arxiv.org/abs/1104.4596) (SIAM J. Financial Mathematics 4(1), 2013)

### Inferences
- OFI is best read as **"how much of the touch queue did the last event(s) remove or add"**, i.e. a
  normalized depth-consumption rate. In a FIFO book with 1-tick spread, `OFI/queue_size` is essentially
  "fraction of the touch consumed", which is exactly the quantity that determines whether the next
  event moves the mid.
- Because the 10 s R² is contemporaneous, a strategy at 5 ms latency is *inside* the window where the
  relation holds mechanically: when an event removes the whole bid queue, the mid moves on that
  event, and you cannot be the one who reacts — you can only predict *which* queue dies next. That is
  a queue-imbalance / hazard-rate problem (Section 3), not an OFI regression problem.
- The negative out-of-sample R² at 1 min is the single most important negative result here: OFI is not
  alpha at horizons a human can act on. Any profitable use must be at the level of "next event" or
  "next few events", monetized via *maker* fills (earn the spread by being in the right queue), not via
  taker direction bets.
- Practical O(1) estimator: keep `e_n` as a running scalar updated per event (4 comparisons, 2 adds),
  plus an EWMA of `|e_n|` for normalization, and `depth = q^b + q^a` as the running slope denominator.
  The integrated (PCA) version needs k levels but the first PC loading vector is stable enough to
  hard-code after an offline fit (they report >89% variance in PC1).

### Gaps
- No verified measurement of OFI R² at sub-second / event-level horizons; the literature jumps from
  event-level "OFI explains individual price changes mechanically" to 10 s aggregates.
- No paper found that measures OFI in a **pure simulation with an exogenous, non-reactive tape** — so
  the "65%" cannot be assumed to transfer; in such a tape OFI has no information about future flow,
  only about the current queue.
- MLOFI results are for 6 stocks in 2016; nothing I verified gives per-level loadings for a 1-tick-spread
  synthetic instrument with 20 visible ticks.

---

## 2. Queue-reactive and self-exciting (Hawkes) models: fitted parameters and their stability

### Takeaway
Queue-reactive models give the empirical *shape* of the intensity functions that a 5 ms strategy
actually needs (probability that the next event is a cancel vs a trade vs a new limit, conditional on
queue size), and they say queue sizes are strongly mean-reverting: cancels rise with queue size, trades
fall exponentially with depth. Hawkes fits say equity order flow is close to critical (branching ratio
n ≈ 0.7–1.0), which is a good *regime* diagnostic but is **not identifiable** from a memoryless
synthetic tape.

### Cited Findings
- **Queue-reactive model.** Split time into periods where the reference price (mid) is constant; inside
  a period the LOB is a Markov queue where limit-order / cancel / market-order intensities depend only
  on the current queue sizes `λ^L_i(n), λ^C_i(n), λ^M_i(n)`, with bid/ask symmetry. Suitable for
  **large-tick assets** (defined as spread almost always 1 tick; "relevant for any asset whose average
  spread is smaller than 2.5 ticks"). — [Huang, Lehalle & Rosenbaum, arXiv:1312.0563](https://arxiv.org/abs/1312.0563)
- **Estimated shapes at the touch (Q±1), France Telecom:**
  limit-order insertion intensity is "approximately a constant function of the queue size, with a
  significantly smaller value at 0" (nobody wants to be alone at the touch);
  cancellation rate is an **increasing concave** function of queue size for q ∈ [0, 25] and then flat /
  slightly decreasing — explicitly *not* the linearly-increasing cancel rate assumed in earlier models;
  market-order intensity **decreases exponentially** with available depth at the touch.
  — [arXiv:1312.0563](https://arxiv.org/abs/1312.0563)
- **At deeper levels (Q±2, Q±3):** limit-order intensity becomes a *decreasing* function of queue size —
  interpreted as "post at the second limit when its queue is small, to seize priority" (a real,
  observable latency/priority strategy). Cancellation rate at Q±3 increases almost linearly for
  q > 3, and is large when q = 1 (cancel fast when alone in the queue). — [arXiv:1312.0563](https://arxiv.org/abs/1312.0563)
- **Reference-price layer.** Two calibrated parameters: reference-price change probability θ = 0.7 and
  LOB reinitialisation probability θ^reinit = 0.85, fitted to 10-minute return volatility and to a
  mean-reversion ratio η = N_c/(2N_a) (N_c = continuations, N_a = alternations of the reference price).
  — [arXiv:1312.0563](https://arxiv.org/abs/1312.0563)
- **Ergodicity/diffusivity of the queue-reactive class.** The Markovian LOB models have a unique
  stationary distribution and diffusive (√t) price behaviour under explicit conditions on the
  intensities — this is what licenses treating a fitted queue-reactive book as a price process.
  — [Huang & Rosenbaum, arXiv:1505.04936](https://arxiv.org/abs/1505.04936)
- **Mutually-exciting Hawkes for microstructure noise.** Trades/quotes modelled as mutually exciting
  point processes; the model reproduces the **signature plot** (realized variance rising then falling as
  sampling coarsens) and the **Epps effect** (correlation → 0 at high frequency), estimated on
  Euro-Bund and Euro-Bobl futures; large-scale Bund/Bobl correlation ρ ≈ 0.77. — [Bacry, Delattre, Hoffmann & Muzy, arXiv:1101.3422](https://arxiv.org/abs/1101.3422)
- **Branching ratio as reflexivity measure.** For a power-law kernel φ(τ) = φ0·τ0^ε/τ^{1+ε}, the
  branching ratio is `n = φ0/ε`; stationary intensity is Λ = μ/(1−n); n→1 is criticality.
  — [Hardiman, Bercot & Bouchaud, arXiv:1302.1405](https://arxiv.org/abs/1302.1405)
- **Hardiman et al. estimate on E-mini S&P mid-price changes, 1998–2011:** power-law kernel with decay
  exponent **≈ −1.15 at short lags (< ~10³ s)** crossing over to **≈ −1.45 for 10³–10⁶ s**; the fitted
  branching ratio "fluctuates around the critical value **n = 1**" for the whole 14 years, and ε ≈ 0.15.
  They explicitly argue that a *sub*-critical n that rises over time (Filimonov–Sornette) is an artefact
  of forcing an exponential kernel onto a power-law kernel. — [arXiv:1302.1405](https://arxiv.org/abs/1302.1405)
- **Filimonov & Sornette (exponential kernel), E-mini 1998–2010:** branching ratio **above 0.6 since
  2002, between 0.7 and 0.8 since 2007**, with spikes to 0.9; ⇒ "more than 70% of the price moves are
  endogenous". They also report that only 70% (1998) → <30% (2007+) of price changes are attributable to
  revealed exogenous information. — [Filimonov & Sornette, arXiv:1201.3572](https://arxiv.org/abs/1201.3572) (Phys. Rev. E 85, 056108)
- **Stability across days/assets:** the two camps agree on the *order of magnitude* (n ≈ 0.7–1) and on
  the presence of long-memory self-excitation, and disagree on the *trend* — a direct, well-documented
  conflict, and a warning that n is kernel-dependent. — [arXiv:1302.1405](https://arxiv.org/abs/1302.1405); [arXiv:1201.3572](https://arxiv.org/abs/1201.3572)
- **Practical estimation without MLE.** Bacry–Mastromatteo–Muzy review non-parametric kernel estimation
  from the autocovariance of the intensity (Wiener–Hopf / contrast functions), which is the only route
  that is cheap enough to run online. — [Bacry, Mastromatteo & Muzy, arXiv:1502.04592](https://arxiv.org/abs/1502.04592) (Market Microstructure and Liquidity 1(1), 2015)

### Inferences
- The queue-reactive intensity *shapes* are the most directly actionable empirical content for a 5 ms
  book: they imply a next-event hazard
  `P(next event = cancel at touch | q) ∝ increasing-concave(q)`,
  `P(next event = trade | q) ∝ exp(−c·q)`.
  That is a 2-parameter, O(1) predictor of whether the queue you are sitting in survives.
- Mean reversion of queue size is the mechanism behind queue-imbalance predictability (Section 3): a
  large bid queue attracts cancels and repels trades, so imbalance decays; the *sign* of the imbalance
  is informative for the next mid move, the *level* is not persistent.
- In the arena environment the tape is documented as Poisson with random cancel timers, so a fitted
  Hawkes branching ratio would come out ≈ 0 (no self-excitation). Hawkes/regime tools (see
  `tools/regime_filter.py`, `tools/id_gap_jump.py` in this repo) therefore measure the *generator's*
  parameters, not a universal market law — useful for round-to-round parameter change detection, not as
  an alpha source per se.
- Criticality (n ≈ 1) implies volatility is nearly non-stationary: small exogenous shocks get amplified
  into long clusters. That is consistent with the observed need for a storm/calm regime filter rather
  than a constant-vol assumption.

### Gaps
- No verified published numbers for the **cross-excitation matrix** between trades and cancels at the
  message level (i.e. "a trade raises the cancel intensity by X%"). The Hawkes literature I could verify
  is on mid-price changes or trade arrivals, not on the trade↔cancel cross terms in an L3 book.
- No verified study of how stable the queue-reactive intensity curves are across days for the *same*
  stock — the paper estimates per stock and reports shapes, not day-to-day stability.
- Nothing verified on queue-reactive parameters for a synthetic/simulated book.

---

## 3. Microprice, weighted mid, queue imbalance: short-horizon fair value and failure modes

### Takeaway
Queue imbalance `I = Q_b/(Q_b+Q_a)` is the strongest *tick-level* directional predictor documented, and
it is essentially free (O(1) from the snapshot). Its edge is large for **large-tick** stocks (spread ≈ 1
tick) and small for small-tick stocks — 50–60% vs 10–30% out-of-sample improvement over a null. This is
the single result in this survey that maps most directly onto a 1-tick-spread, FIFO, 5 ms environment.

### Cited Findings
- **Definitions.** Mid `M = (P_a + P_b)/2`; weighted mid `W = I·P_a + (1−I)·P_b` with `I = Q_b/(Q_b+Q_a)`;
  microprice `P_micro_t := lim_{i→∞} E[M_{τ_i} | F_t]` where τ_i are the successive times the mid
  changes. The microprice is a **martingale by construction**, "the fair price conditional on the
  information in the order book", and "may be expressed as an adjustment to the mid-price that takes
  into account the bid-ask spread and the imbalance". — [Stoikov, DOI:10.1080/14697688.2018.1489139](https://doi.org/10.1080/14697688.2018.1489139); [SSRN 2970694](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694)
- **Empirical claim.** "The micro-price estimated using high-frequency data is empirically a better
  predictor of short-term prices than the mid-price or the weighted mid-price." (Magnitudes are not in
  the abstract and the paper is paywalled; I could not verify a specific R²/MSE number.) — [DOI:10.1080/14697688.2018.1489139](https://doi.org/10.1080/14697688.2018.1489139)
- **Computable form.** The theorem gives `E[M_{τ_i}|F_t] = M_t + Σ_{k=1..i} g_k(I_t, S_t)` and
  `P_micro = M + Σ_k B_k G_1` with `B = (I−Q)^{-1}T`, `G_1 = (I−Q)^{-1}RK`, where Q, T, R are estimated
  transition probabilities of the joint (mid-change, imbalance, spread) chain — i.e. a small Markov
  chain you calibrate offline and then evaluate in O(1). In practice the series "converges rapidly and
  is adequately approximated when we compute the series until τ_6". — [reproduction of Stoikov's theorem, Stanford MS&E448 notes](http://stanford.edu/class/msande448/2021/Final_reports/gr1.pdf)
- **Known failure mode of weighted mid.** Because W changes on *every* imbalance update it is "quite
  noisy for the purpose of high frequency volatility estimation" and is not a martingale, so it has no
  theoretical claim to be a fair price. — [Stoikov (SSRN PDF copy)](https://jamesstuchbery.com.au/assets/pdfs/OBI.pdf)
- **Queue imbalance as a one-tick-ahead predictor (the key magnitude).** 10 liquid Nasdaq stocks.
  Logistic regression of the direction of the next mid move on queue imbalance, formal out-of-sample
  tests: the null of no relation is "strongly rejected" for all 10 stocks; binary-classification
  improvement over the null model is **≈ 50–60% for large-tick stocks** and **≈ 10–30% for small-tick
  stocks**; probabilistic predictions are weaker, **≈ 20–30%** (large tick) and **≈ 2–6%** (small tick).
  — [Gould & Bonart, arXiv:1512.03492](https://arxiv.org/abs/1512.03492)
- **Relation to microprice.** Queue imbalance is the empirically dominant input to the microprice; the
  microprice's extra content is the spread-dependent correction, which matters most when the spread is
  wide relative to the tick. For 1-tick spreads the two collapse onto one another. — [arXiv:1512.03492](https://arxiv.org/abs/1512.03492); [DOI:10.1080/14697688.2018.1489139](https://doi.org/10.1080/14697688.2018.1489139)
- **Survey-level caution.** The empirical LOB literature is large but "many such models poorly resemble
  real LOBs and several well-established empirical facts have yet to be reproduced satisfactorily";
  the survey lists cross-sectional/queue-dynamics regularities as open. — [Gould et al., arXiv:1012.0349](https://arxiv.org/abs/1012.0349)

### Inferences
- In a market where the spread is almost always 1 tick, `I` is a *complete* description of the touch,
  and `P(up | I)` should be monotone, steep, and close to the large-tick regime above. The tradeable
  version is not "predict the mid" but "join the shorter queue": if `I > 1/2` the ask is thin, so the
  next mid move is more likely up, and a resting bid at `P_b` is more likely to be filled *and* more
  likely to be immediately profitable on the next tick.
- The failure modes are all "the book you see is not the book that will be there": (a) displayed depth
  is mostly fleeting (Section 6 — 60% of limit orders cancelled within 10 s market-wide), so `I` is
  measured on liquidity with a half-life of seconds; (b) `I` is trivially manipulable by spoofing at the
  touch; (c) at 5 ms latency the snapshot you condition on can be stale by one event in a burst.
- The safest operational form is `I` **plus** a decay/hazard term: the expected lifetime of the current
  touch queue, estimated from the queue-reactive cancel intensity (Section 2). Imbalance tells you the
  direction; the cancel intensity tells you whether the imbalance will still exist in 200 ms.
- Microprice's martingale property is a statement about the *unconditional* expectation over the
  stationary book distribution; it does not say the microprice is unbiased conditional on a *transient*
  book state (e.g. right after a large IOC sweeps three levels). That transient is where the PnL is.

### Gaps
- No verified numeric comparison of microprice vs weighted mid vs mid (R², MSE, or hit rate) — the
  published abstract is qualitative and the body is paywalled. Anyone quoting "microprice R² = X" for
  this paper should be treated as unverified.
- No verified evidence on the *decay* of queue-imbalance predictive power (how many events/seconds the
  signal lasts), which is the number a 5 ms strategy needs most.
- No verified empirical work on imbalance predictability in a purely synthetic Poisson flow; the
  Gould–Bonart numbers come from real Nasdaq books with strategic, reactive participants.

---

## 4. Price impact and resiliency: square-root law, transient impact, decay kernels

### Takeaway
Aggregate impact is concave, close to √(size), with a prefactor of order 1 in volatility units
(`impact ≈ Y·σ·√(Q/V)`, Y ≈ 1), and it is **not permanent**: the propagator decays as a slow power law,
empirically with an exponent around 0.2–0.5, so a large part of the initial impact mean-reverts over
minutes-to-hours. For a 5 ms strategy this literature is mostly *cost accounting* (what walking the
book costs), not alpha.

### Cited Findings
- **Square-root law.** `impact ≈ Y·σ·√(Q/V)` with **Y of order unity** measured from data; the exponent
  δ ≈ 1/2 for "unit execution" and δ ≈ 2/3 for "ζ-execution"; for execution time much shorter than the
  order lifetime the impact is "strongly concave and, in a first approximation, independent of the
  participation rate", matching data. The mechanism is a latent book whose liquidity vanishes linearly
  near the current price. — [Toth et al., arXiv:1105.1694](https://arxiv.org/abs/1105.1694) (Phys. Rev. X 1, 021006)
- **Direct estimation on real institutional orders (≈700k orders, Citigroup US equities, 2005).**
  Permanent: `I = γ·σ·(X/V)·(Θ/V)^{1/4}`; temporary: `J = I/2 + η·σ·|X/(V·T)|^{3/5}`, with
  **γ = 0.314 ± 0.041** (t = 7.7) and **η = 0.142 ± 0.0062** (t = 23). They explicitly *reject* a pure
  square root in favour of a 3/5 power for temporary impact. — [Almgren, Thum, Hauptmann & Li (2005), PDF](https://www.cis.upenn.edu/~mkearns/finread/costestim.pdf) (Risk 18(7), 57–62)
- **Impact is neither linear in volume nor permanent**; the correlation between signed order flow and
  price change is strong, but a permanent-impact model is only consistent with unpredictable returns
  if signed volume is uncorrelated — which it is not. — [Bouchaud, "Price Impact", arXiv:0903.2428](https://arxiv.org/abs/0903.2428)
- **Decay kernel, exact relation.** If the sign autocorrelation decays as `C(ℓ) ~ ℓ^{−γ}` with
  **γ ≈ 0.5 for stocks**, then market efficiency requires the propagator to decay as `G(ℓ) ~ ℓ^{−β}`
  with **β = (1−γ)/2 ⇒ β ≈ 0.25**. A permanent component G(ℓ→∞) > 0 is possible but must be small.
  — [arXiv:0903.2428](https://arxiv.org/abs/0903.2428)
- **Propagator / transient impact model.** `p_T = p_{−∞} + λ Σ_{n<N} G(N−n)·ε_n·v_n^ψ`: the price is a
  weighted sum of past signed trades with a decaying kernel; "the impact of a single trade" is described
  by G, and the non-trivial shape of G can arise purely from the interplay of market and limit order
  flow even if the "bare" propagators are constant. — [Bouchaud, Farmer & Lillo, arXiv:0809.0822](https://arxiv.org/abs/0809.0822); [Eisler, Bouchaud & Kockelkoren, arXiv:0904.0900](https://arxiv.org/abs/0904.0900)
- **Event-mixture cancellation of memory (important, and quantitative).** Sign autocorrelations of
  market orders are long-memory, *but* incoming limit orders are negatively correlated with cancels and
  market orders, so "the mixture of long-range positive and negative correlations balances such that
  only short-range persistence remains": for ℓ ≲ **100 events (≈ 10–20 s of real time)** returns are
  still autocorrelated; beyond that the memory is largely cancelled. — [arXiv:0904.0900](https://arxiv.org/abs/0904.0900)
- **Beyond the square root.** On a proprietary metaorder dataset, impact is better described as
  **logarithmic** in size/participation over a wide range, with the √-law a good local approximation;
  the paper has a full section on "impact decay and permanent impact" and the role of metaorder
  autocorrelations. — [Zarinelli, Treccani, Farmer & Lillo, arXiv:1412.2152](https://arxiv.org/abs/1412.2152)
- **Resiliency as a queue property.** Because the visible book is thin, "large orders can only be traded
  incrementally", which is *why* order flow is long-memory — impact decay and flow persistence are two
  sides of one constraint. — [arXiv:0809.0822](https://arxiv.org/abs/0809.0822)

### Inferences
- The decay exponent β ≈ 0.25 means the *median* reversion of a trade's impact happens fast while the
  *mean* reversion is slow — i.e. impact decay is extremely fat-tailed. Any "fade the spike" strategy
  is short a fat tail; this is the mechanism behind adverse selection on resting orders right after a
  big trade, and the reason a markout-based fill filter (`tools/markout.py`) is the right instrument.
- For a taker in a discrete-tick book, the √-law reduces to a **staircase**: cost = Σ over consumed
  levels of (level price − touch) × size. The only thing worth estimating online is the *depth profile*
  (how many lots within k ticks), which is exactly what OFI's β ∝ 1/depth already encodes.
- In the arena environment our own trades have **zero** price impact on the tape (the tape is fixed and
  does not react). So every result about impact decay/resiliency applies to *other* participants'
  behaviour, not to our cost. Our cost is purely the mechanical book-walk, and there is no
  "we moved the price and it came back" PnL. This kills a whole class of strategies that only work in
  reactive markets.
- The Eisler et al. "memory is cancelled after ~100 events / 10–20 s" result is a useful horizon marker:
  it says the *maximum* horizon at which a flow-based signal can have any information is tens of
  seconds, and inside that window the dominant term is the immediate queue state.

### Gaps
- No verified numbers on impact decay **in electronic LOB data at the single-trade level** (the decay
  literature is mostly metaorder/aggregate). I could not verify a specific "X% of impact decays within
  T seconds" figure.
- No verified estimate of the propagator exponent on futures/CFD-style instruments comparable to a
  synthetic arena tape.
- Cross-impact magnitudes (how much stock B moves when stock A's OFI moves) are verified to exist
  (Cont–Cucuringu–Zhang) but are irrelevant to a single-instrument arena.

---

## 5. Realized volatility, jump/regime estimation from LOB data

### Takeaway
Realized variance computed from tick data is **not** an estimate of volatility — it diverges linearly in
the number of observations and, scaled by 1/(2n), converges to the variance of the microstructure noise
instead. For liquid instruments the noise only dominates below ~20 s sampling; optimal sampling is
minutes (≈4.5–7.5 min in the classic Bandi–Russell estimates), or use a noise-robust estimator
(two-scale, realized kernel) / bipower variation for jump-robustness. Online, this means: sample the mid
at ~1 s, not per event, and never treat tick-level RV as vol.

### Cited Findings
- **RV diverges under noise.** With iid noise ε, `[Y,Y]^{(all)}_T → 2n·E[ε²]`: "realized volatility
  diverges to infinity linearly in n. Scaled by (2n)^{−1}, it estimates consistently the variance of
  microstructure noise, E ε², rather than ⟨X,X⟩_T." — [Zhang, Mykland & Aït-Sahalia, "A Tale of Two Time Scales", PDF](https://www.stat.uchicago.edu/~mykland/paperlinks/Twoscales-090404.pdf) (JASA 2005)
- **Optimal sampling is finite, with a closed form.** `c* = (T / (12 (E ε²)² ∫σ⁴ dt))^{−1/3}` for the
  sparse-sampling version; the two-scale estimator `⟨X,X⟩_T^{(TS)}` is correctly centred and converges at
  rate **n^{−1/6}**, with variance `n^{−1/6}[8c²(Eε²)² + c⁴(T/3)∫σ⁴]^{1/2}`. — [same PDF](https://www.stat.uchicago.edu/~mykland/paperlinks/Twoscales-090404.pdf)
- **"Sample as often as possible" — if you model the noise.** Aït-Sahalia–Mykland–Zhang show the optimal
  sampling frequency is finite and has a closed form when noise is ignored, but that modelling the noise
  explicitly restores the first-order optimality of using all the data, *even under a misspecified noise
  distribution*. — [Aït-Sahalia, Mykland & Zhang, RFS 18(2) 351–416, DOI:10.1093/rfs/hhi016](https://doi.org/10.1093/rfs/hhi016)
- **Practical magnitudes.** In deep liquid markets "microstructure noise ... does not have a noticeable
  effect on realized volatility for sampling frequencies lower than once every **20 seconds**"; the same
  holds for days with major macro announcements except below ~15 s. — [BIS/FRB "Frequency of Observation and the Estimation of Integrated Volatility", PDF](https://www.bis.org/repofficepubl/apresearchacbcehml080101.pdf)
- **Bandi–Russell optimal frequencies are minutes and shrinking over time:** estimated optimal sampling
  ≈ **4.5 min (SBC)** and **≈7.5 min (XOM)** in their sample, and "optimal frequencies higher than 20
  minutes were not uncommon at the beginning of the sample", declining since 1993 — i.e. as markets got
  faster, the *volatility-relevant* sampling got finer. — [Bandi & Russell, "Separating microstructure noise from volatility", PDF](https://faculty.chicagobooth.edu/-/media/faculty/jeffrey-russell/research/separate.pdf) (J. Financial Economics 2006; ReStud 2008)
- **Jumps.** Bipower variation (Barndorff-Nielsen & Shephard) is the standard jump-robust integrated
  variance estimator, and the RV−BV difference identifies the jump component; this is the canonical
  offline machinery, and it is O(1) per update online (keep three running sums).
  — [Barndorff-Nielsen & Shephard, J. Financial Econometrics 2(1) 1–37, 2004](https://doi.org/10.1093/jjfinec/nbh001)
- **Hawkes-based signature plot / Epps effect.** A mutually exciting point-process model reproduces both
  the realized-variance signature plot and the decay of cross-asset correlation at high frequency; the
  signature plot is thus a *model diagnostic* for self-excitation, not just a noise diagnostic.
  — [arXiv:1101.3422](https://arxiv.org/abs/1101.3422)
- **Criticality as a regime indicator.** n→1 means the process is at the edge of instability, and
  Filimonov–Sornette show n estimated in 10-minute windows varies through the day and spikes before
  stress; they report that the exponential-kernel Hawkes model is rejected for only 0.12% of 10-min
  windows, 1.64% of 20-min windows and 12.64% of 30-min windows — i.e. **the shorter the window, the
  better the fit**, which is the empirical case for short-window regime detection.
  — [arXiv:1201.3572](https://arxiv.org/abs/1201.3572)
- **Volatility clustering / stylized facts.** Heavy tails, volatility clustering, aggregational
  gaussianity and near-zero linear autocorrelation of returns are the canonical stylized facts the LOB
  literature is required to reproduce. — [Cont, "Empirical properties of asset returns", Quantitative Finance 1(2), 2001](https://doi.org/10.1080/713665670)

### Inferences
- The correct online volatility estimator for a 5 ms strategy is **not** per-event RV. Use mid changes
  sampled on a fixed clock (~1 s), which is what `tools/volatility.py`'s `RealizedVol` does; the
  literature says anything finer than ~20 s is noise-dominated and anything coarser than ~5 min is too
  slow for regime detection.
- In a 1-tick-spread discrete book there is an additional, structural source of "noise": the mid can
  only take values on a half-tick grid, so at high sampling frequency RV is dominated by the bid-ask
  bounce (Roll 1984). Realized vol should therefore be computed from **mid** changes (not trade prices)
  to avoid the bounce, and from a *queue-imbalance-conditioned* mid (microprice) if you want the
  "efficient" vol rather than the quoted vol.
- Criticality/branching ratio is a legitimately online-computable regime scalar (it needs only the
  event-time autocovariance over a rolling window, O(1) with an EWMA of the intensity), but it detects
  *self-excitation* in the flow. On a Poisson tape it will read ≈ 0 by construction, so its value here
  is as a **change detector between rounds**, not as a market law.
- Jumps in this environment arrive as *hidden fair-value jumps* (the generator's S), which the LOB only
  reveals through IOC sweeps and id gaps. That is exactly the object `tools/id_gap_jump.py` targets;
  RV/BV jump tests would be far too slow (they need minutes of returns).

### Gaps
- No verified empirical study of volatility signature plots **on LOB mid-price vs microprice** with
  reported numbers, though the qualitative claim (microprice less noisy for vol estimation) is asserted
  in the microprice literature.
- No verified numbers on regime-switching (calm/storm) transition probabilities in intraday LOB data at
  the second scale — the HMM in `tools/regime_filter.py` is a reasonable engineering choice but its
  parameters are not literature-calibrated.
- Nothing verified on realized-variance behaviour under a *discrete tick* with a 1-tick spread
  (all the theory assumes a continuous semimartingale observed with additive noise).

---

## 6. Order book shape and liquidity: depth imbalance, slope, seasonality, and how much of the book is real

### Takeaway
Most of the visible book is not real liquidity. Market-wide, ~37% of limit orders are cancelled within 2
seconds, ~60% within 10 seconds, 98.4% within 10 minutes, and fill rates for posted orders in
NYSE 2010–11 data are **below 3%**. Any "depth imbalance" or "book slope" signal is therefore a signal
about *intentions that will be withdrawn*, and its half-life is seconds, not minutes.

### Cited Findings
- **Fleeting orders, original definition and magnitudes.** "Fleeting" = cancelled within 2 seconds.
  On Island (2001–02): ~25% of limit orders cancelled within 2 s and ~40% within 10 s; fleeting orders
  are **27.7% of all visible orders** (32.5% by share volume); they are "closer substitutes for market
  orders than for traditional limit orders" — i.e. they demand immediacy, they do not supply it.
  — [Hasbrouck & Saar, working paper PDF](https://pages.stern.nyu.edu/jhasbrou/Research/Working%20Papers/HS6.pdf) (J. Financial Markets 12(2), 2009)
- **Time trend.** On INET/Nasdaq 2004, fleeting orders are **36–37% of limit order submissions, twice
  the 1999 rate**; only **6.37% of total limit-order quantity is ever executed**. — [Hasbrouck & Saar (2009), as summarized in a Borsa Istanbul study](https://www.sciencedirect.com/science/article/pii/S2214845015300582)
- **Market-wide lifetimes.** "60 percent of all limit orders in their market-wide sample were cancelled
  within 10 s, and 98.4 percent within 10 min" (Hasbrouck & Saar 2009, quoted). — [Review of Finance 30(2), 2025, "Retail limit orders"](https://academic.oup.com/rof/article/30/2/459/8277182)
- **Fill rates.** Market-wide, "Li, Ye and Zheng (2023) report fill rates of **less than 3 percent** for
  orders on the NYSE during 2010–2011"; retail limit orders achieve ~65%, i.e. an order of magnitude
  better, purely because they are not competing with HFT market makers. — [Review of Finance 30(2), 2025](https://academic.oup.com/rof/article/30/2/459/8277182)
- **Sub-second transients.** In a 2010 Nasdaq sample, **30% of cancelled orders fall in the shortest
  (5 ms) lifetime bin**, and 11.25% of fleeting orders are placed inside the BBO (vs Hasbrouck–Saar's
  finding that most were inside the BBO in 2004) — the placement motive changed from "probe for hidden
  liquidity" to "queue positioning". — [Ye et al., "The Externalities of High Frequency Trading", PDF](https://www.sec.gov/divisions/riskfin/seminar/ye031513.pdf)
- **Cancellation rate responds to volatility and depth.** A one-standard-deviation increase in 5-minute
  return volatility is associated with a 7.2% *decrease* in the probability of a regular limit order, a
  2.9% *increase* in a fleeting limit order, and a 3.7% increase in a market order; a one-SD increase in
  opposite-side depth raises the market-order probability by 4.5%. — [Hasbrouck & Saar, PDF](https://pages.stern.nyu.edu/jhasbrou/Research/Working%20Papers/HS6.pdf)
- **Market-quality cost of fleeting liquidity.** At times when machine-driven post-and-cancel activity
  (orders posted and cancelled within 100 ms, "PC100") is high, quoted and effective spreads increase
  and LOB depth decreases; PC100 is a small subset (<20% of cancellations in March 2013), so it is not
  mechanically the cancel rate. — ["Ultra-fast activity and intraday market quality", J. Banking & Finance 2019](https://www.sciencedirect.com/science/article/abs/pii/S0378426618302735)
- **Intraday seasonality.** The OFI price-impact coefficient has the same intraday U-shape as spreads,
  depth and volatility, and this is explained by depth variation rather than by unobservable information
  asymmetry parameters. — [arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
- **Book slope / depth imbalance, model-free.** The multi-level OFI work shows net order flow at levels
  2–5 is economically significant (65–75% OOS improvement for large-tick stocks) but statistically
  collinear with level 1 — i.e. "the book has a shape, but the shape is close to a one-factor object".
  — [arXiv:1907.06230](https://arxiv.org/abs/1907.06230)

### Inferences
- The correct mental model is **"the book is a queue of intentions with a hazard rate"**, not "the book
  is supply and demand". In a 1-tick-spread FIFO book the actionable quantity is *your position in the
  queue and the queue's expected lifetime*, and the empirical lifetime distribution is heavily
  bimodal: a mass at <100 ms (fleeting / HFT) and a long tail at minutes (genuine / retail).
- Consequently: depth imbalance measured on **raw displayed size** is dominated by orders that will
  vanish. A more robust O(1) variant is a size-weighted imbalance where each order is weighted by a
  survival proxy (e.g. its age, or "not yet cancelled within the modal fleeting window"), or simply the
  imbalance computed on *the part of the queue ahead of us* (which is what actually determines fill
  probability).
- The <3% market-wide fill rate is the single most sobering number for a market-making strategy: in
  real markets, resting at the touch is a lottery against faster cancels. In the arena, the tape is
  Poisson and non-reactive, so our resting orders face a *much* higher and more stable fill rate — the
  documented "fleeting order" problem is largely absent, which is a genuine structural advantage of the
  simulated environment and the main reason not to import real-market fill-rate priors.
- Seasonality: spreads/depth/vol all have U-shapes intraday in real markets. A synthetic tape may or may
  not; if it does not, hard-coded intraday scaling (a common real-market trick) is pure overfitting risk.

### Gaps
- No verified empirical paper with a clean **"book slope"** estimator and magnitudes (regression of
  price impact on cumulative depth / the "book pressure" curve). The closest verified content is
  MLOFI's multi-level loadings; anything stronger would be inference.
- No verified distribution of **order lifetimes conditional on distance from the touch** in a
  large-tick, 1-tick-spread instrument (the Hasbrouck–Saar numbers are for Nasdaq/Island with wider
  ticks and different queue dynamics).
- Nothing verified on "how much of displayed depth is real" measured as *fill probability per unit of
  displayed size at a given distance from the touch* — the papers measure cancellation rates, not the
  mapping from displayed depth to executed depth.

---

## 7. What is documented to NOT be predictable

### Takeaway
Short-horizon returns are close to unforecastable *net of the contemporaneous flow*: the honest
out-of-sample R² of OFI-based forecasting at 1 minute is ≤ 0, trade imbalance is a strictly worse proxy
than OFI because of order splitting, and tick-level returns are negatively autocorrelated (bid-ask
bounce) so short-horizon "momentum" is an artefact. The predictable part is the *next event / next few
events* and the *fill*, not the price path.

### Cited Findings
- **OFI-based forecasting fails out of sample at 1 min:** OS R² = **−0.37** (own-price OFI), **−0.10**
  (cross-impact OFI), on Nasdaq data with strict temporal ordering of fit/test windows. The authors
  explicitly criticise earlier work for reporting in-sample R² or cross-validating in a way that
  "damaged the chronological order of the time-series data". — [arXiv:2112.13213](https://arxiv.org/abs/2112.13213)
- **Trade imbalance is the wrong variable.** Widely used in the literature, but once OFI is included its
  coefficient is indistinguishable from zero; the price–volume relation is "noisy and less robust" than
  the price–OFI relation. — [arXiv:1011.6402](https://arxiv.org/abs/1011.6402)
- **Order splitting contaminates signed-volume signals.** Market-order signs are long-memory
  (`C(ℓ) ~ ℓ^{−γ}`, γ ≈ 0.5) precisely because metaorders are split; a naive signed-volume imbalance
  therefore measures *the same parent order repeatedly*, and the mixture with limit orders and cancels
  is what restores short-range-only predictability (ℓ ≲ 100 events ≈ 10–20 s).
  — [arXiv:0903.2428](https://arxiv.org/abs/0903.2428); [arXiv:0904.0900](https://arxiv.org/abs/0904.0900)
- **Bid-ask bounce ⇒ negative short-horizon autocorrelation.** The Roll (1984) model implies
  `Cov(Δp_t, Δp_{t−1}) = −(s/2)²`, i.e. first-order autocorrelation `ρ_1 = −(s/2)²/(σ² + (s/2)²)`,
  which is strongly negative when the spread dominates the innovation — the situation in a 1-tick market.
  — [Roll, "A Simple Implicit Measure of the Effective Bid-Ask Spread", J. Finance 39(4), 1984](https://doi.org/10.1111/j.1540-6261.1984.tb03897.x)
- **Variance ratios below 1 at short horizons.** Lo–MacKinlay's specification test rejects the random
  walk for US weekly/monthly returns, with variance ratios below one at short horizons (mean reversion)
  — the canonical evidence that "efficient" does not mean "iid at every horizon".
  — [Lo & MacKinlay, "Stock Market Prices Do Not Follow Random Walks", RFS 1(1) 41–66, 1988](https://doi.org/10.1093/rfs/1.1.41)
- **Queue imbalance predictability is regime-dependent, not universal.** The same logistic model that
  buys 50–60% improvement on large-tick stocks buys only 10–30% on small-tick stocks, and only 2–6% on
  the probabilistic version — i.e. in small-tick books the signal is essentially gone.
  — [arXiv:1512.03492](https://arxiv.org/abs/1512.03492)
- **Impact is not permanent, so "follow the big trade" is not a strategy.** Impact decays as a power law
  with β ≈ (1−γ)/2 ≈ 0.25 given γ ≈ 0.5; a large fraction of any trade's price move is transient.
  — [arXiv:0903.2428](https://arxiv.org/abs/0903.2428)
- **Models poorly reproduce the book.** The survey conclusion is blunt: "many such models poorly
  resemble real LOBs and ... several well-established empirical facts have yet to be reproduced
  satisfactorily" — a warning against assuming any fitted LOB model generalises.
  — [arXiv:1012.0349](https://arxiv.org/abs/1012.0349)
- **Latency erodes microstructure alpha specifically.** Lehalle & Mounjid model limit-order placement
  with adverse selection and show the optimal placement depends on latency; a slow trader's "optimal"
  quote is systematically different (wider / less aggressive) than a fast one's — the alpha in
  short-horizon signals is a decreasing function of reaction time.
  — [Lehalle & Mounjid, arXiv:1610.00261](https://arxiv.org/abs/1610.00261)

### Inferences
- The consistent pattern across all these results: **contemporaneous explanatory power is huge,
  predictive power is ~zero, and the residual predictability lives at the event scale (≤ 10–20 s)**.
  A 5 ms strategy is on the right side of that boundary *only if* its edge is "I know the next event's
  effect on the queue better than the queue itself does" — i.e. queue position and fill probability —
  not "I know where the price will be in a minute".
- Negative first-order autocorrelation at the tick scale means a naive market-making strategy that
  quotes symmetrically and gets filled on both sides *earns* the bounce rather than paying it; and a
  naive taker that chases a 1-tick move *pays* it. This is the mechanical source of maker edge in a
  1-tick market, and it is robust across the literature.
- Order splitting means the *number* of trades is not the *number* of decisions. Any signal built on
  trade counts (e.g. "3 buys in a row") is measuring one parent order; signals must be built on
  *net queue depletion* (OFI-like) instead.
- The Lo–MacKinlay / Roll results imply that a variance-ratio test computed online (VR(q) for q = 2, 5,
  10 events) is a cheap, literature-grounded **no-trade detector**: when VR ≈ 1 at the event scale, your
  edge must come from the spread, not from direction.

### Gaps
- No verified modern (post-2015) variance-ratio study at the **event/tick** scale for LOB data; the
  classical results are daily/weekly. The tick-scale claim rests on Roll-type bounce theory.
- No verified paper that measures how much of the queue-imbalance edge survives **5 ms of latency**
  explicitly; Lehalle–Mounjid give the qualitative result and the modelling framework, not a measured
  decay curve.
- I found no verified evidence either way on whether trade-sign autocorrelation / order splitting exists
  in a *synthetic Poisson* tape (by construction it should not), which matters because it means a whole
  class of "flow persistence" signals would be pure noise there.

---

## Appendix: citation verification log

Checked against the arXiv API (title, authors, ID, journal ref, DOI) or the publisher/PDF page.
IDs below are confirmed to resolve to the stated paper. Where I could not confirm a journal volume, the
link is to arXiv only.

| Short name | Verified identifier | Notes |
|---|---|---|
| Cont, Kukanov, Stoikov (2014) | arXiv:1011.6402; J. Financial Econometrics 12(1) 47–88; DOI 10.1093/jjfinec/nbt003 | journal ref + DOI from arXiv metadata; full text read via ar5iv |
| Eisler, Bouchaud, Kockelkoren | arXiv:0904.0900 | no journal ref in metadata |
| Huang, Lehalle, Rosenbaum | arXiv:1312.0563 | published in JASA (2015); metadata has no journal ref |
| Huang & Rosenbaum | arXiv:1505.04936 | |
| Cont & de Larrard | arXiv:1104.4596; SIAM J. Financial Math. 4(1) 1–25 (2013) | journal ref from metadata |
| Bacry, Delattre, Hoffmann, Muzy | arXiv:1101.3422 | Quantitative Finance 13(1), 2013 (metadata has no journal ref) |
| Bacry, Mastromatteo, Muzy | arXiv:1502.04592 | review; Market Microstructure & Liquidity 1(1), 2015 |
| Hardiman, Bercot, Bouchaud | arXiv:1302.1405 | |
| Filimonov & Sornette | arXiv:1201.3572; Phys. Rev. E 85, 056108; DOI 10.1103/PhysRevE.85.056108 | |
| Stoikov (2018) | DOI 10.1080/14697688.2018.1489139; Quantitative Finance 18(12) 1959–1966; SSRN 2970694 | abstract verified; body paywalled |
| Gould & Bonart | arXiv:1512.03492 | |
| Xu, Gould, Howison | arXiv:1907.06230 | |
| Cont, Cucuringu, Zhang | arXiv:2112.13213 | v4, May 2023 |
| Bouchaud, Farmer, Lillo | arXiv:0809.0822 | |
| Bouchaud, "Price Impact" | arXiv:0903.2428 | Encyclopedia of Quantitative Finance chapter |
| Toth et al. | arXiv:1105.1694; Phys. Rev. X 1, 021006; DOI 10.1103/PhysRevX.1.021006 | |
| Zarinelli, Treccani, Farmer, Lillo | arXiv:1412.2152 | |
| Almgren, Thum, Hauptmann, Li | Risk 18(7) 57–62 (2005); PDF at cis.upenn.edu | coefficients read directly from the PDF |
| Gould et al., LOB survey | arXiv:1012.0349 | |
| Sirignano & Cont | arXiv:1803.06917 | |
| Lehalle & Mounjid | arXiv:1610.00261 | |
| Aït-Sahalia, Mykland, Zhang | RFS 18(2) 351–416; DOI 10.1093/rfs/hhi016 | |
| Zhang, Mykland, Aït-Sahalia | JASA 100(469) 2005; PDF at stat.uchicago.edu | two-scale estimator |
| Bandi & Russell | J. Financial Economics 2006 / ReStud 2008; PDF at chicagobooth.edu | optimal sampling 4.5 / 7.5 min |
| Barndorff-Nielsen & Shephard | J. Financial Econometrics 2(1) 1–37 (2004); DOI 10.1093/jjfinec/nbh001 | bipower variation |
| Hasbrouck & Saar | J. Financial Markets 12(2) 253–283 (2009); working-paper PDF at pages.stern.nyu.edu | fleeting orders |
| Roll | J. Finance 39(4) 1127–1139 (1984); DOI 10.1111/j.1540-6261.1984.tb03897.x | bid-ask bounce |
| Lo & MacKinlay | RFS 1(1) 41–66 (1988); DOI 10.1093/rfs/1.1.41 | variance ratio |
| Cont, stylized facts | Quantitative Finance 1(2) 223–236 (2001); DOI 10.1080/713665670 | |

**Corrections made during verification** (candidate IDs that turned out to belong to unrelated papers,
worth recording so they are not reused): 1105.3423 (Xiao–Wu, autocovariance inference), 1206.4811
(max-entropy Monte Carlo, physics), 2008.07996 (quasi-clique graph mining), 1103.5544 (cohomology over
finite fields), 1201.4772 (t-tbar cross section), 1304.0169 (double quantum dots), 1711.03562 (ErMnO₃
thermal conductivity), 1812.10766 (SMPL 3D pose). The correct IDs for those works are the ones in the
table above (1101.3422, 0904.0900, 1201.3572, 1302.1405, 2112.13213, 1907.06230, 1505.04936, 1104.4596,
1512.03492).
