# Machine learning for limit order books, RL for trading, and how to validate it honestly

Scope: 2015–2026 plus the older RL reference points. All numbers below were read from the primary
papers (PDF text) or the publisher record during this session; every reference was checked against
arXiv/publisher metadata. Where I could not verify a claim I say so in **Gaps** rather than assert it.

Reader context (matters for every "so what" below): we must ship **one Python file**, numpy/pandas/
scipy/scikit-learn only, no saved state, and we have **no training data beyond the current tape**.

---

## 1. DeepLOB and successors: what accuracy is actually reported, at what horizon, under what protocol

### Takeaway
The headline numbers everyone quotes (≈78–84% accuracy / F1 0.78–0.83 on FI-2010) are **not** a
statement about tradable direction prediction. They are a 3-class classification score on a
2010 Nordic dataset with *smoothed* labels and a 0.2% threshold, and the same model on the authors'
own LSE data reports **61–70% accuracy at 20–100 event horizons**. Every credible re-evaluation on
modern NASDAQ data lands far lower, and the strongest ones report that the profit-relevant metric
collapses to ≈6–15%.

### Cited Findings

**The original results, exactly as published**

- DeepLOB (Zhang, Zohren, Roberts), *IEEE Trans. Signal Processing* 67(11), 2019, DOI
  10.1109/TSP.2019.2907260, arXiv:1808.03668. Architecture: CNN + Inception + LSTM, **60k parameters,
  0.253 ms forward pass**, 100×40 input (100 events × 10 levels × 4). On FI-2010 with the dataset's
  own *anchored forward* CV ("Setup 1"): **k=10 acc 78.91% / F1 77.66; k=50 75.01 / 74.96;
  k=100 76.66 / 76.58**. With more training data per fold ("Setup 2"): **k=10 acc 84.47 / F1 83.40;
  k=20 74.85 / 72.82; k=50 80.51 / 80.35**. Note the non-monotonicity between k=10 and k=20 —
  a first hint that folds, not signal, dominate the differences. [arXiv:1808.03668](https://arxiv.org/abs/1808.03668)
- Same paper, **out-of-sample on London Stock Exchange**, one year of quotes, trained on 5 stocks
  (LLOY, BARC, TSCO, BT, VOD), tested on a disjoint 3 months: **k=20 acc 70.17% / F1 70.15;
  k=50 63.93 / 63.49; k=100 61.52 / 60.65**. Transfer to 5 unseen stocks (GLEN, HSBC, CNA, BP, ITV):
  **68.62 / 68.48, 63.44 / 62.84, 61.46 / 60.77**. The authors describe this as "remarkably stable"
  and evidence of universal features. [arXiv:1808.03668](https://arxiv.org/abs/1808.03668)
- Baselines on FI-2010 from the same table for scale: ridge regression 48.0% acc / F1 41.0;
  the previous SOTA C(TABL) 84.70 / 77.63 at k=10; B(TABL) 78.91 / 69.20. So DeepLOB's k=10 edge over
  the previous best *attention-bilinear* network was **F1 77.66 vs 77.63 in Setup 1** — i.e. the
  celebrated improvement is ~0.03 F1 points at the shortest horizon and only becomes large in the
  setting where the comparison models were trained on less data. [arXiv:1808.03668](https://arxiv.org/abs/1808.03668)

**The benchmark itself**

- FI-2010 (Ntakaris, Magris, Kanniainen, Gabbouj, Iosifidis), *Journal of Forecasting* 37(8),
  DOI 10.1002/for.2543, arXiv:1705.03233: 5 NASDAQ-Nordic stocks, **10 consecutive days**, ~4M
  time-series samples, already filtered/normalised/labelled, with a day-based anchored forward CV
  protocol. [arXiv:1705.03233](https://arxiv.org/abs/1705.03233)
- The dataset is downsampled to every 10th LOB update → ≈394k events, and it "comes in an already
  pre-processed (filtered, normalized, and labeled) format so that the original LOB cannot be
  backtracked"; it is "too simplistic, leaving ample space for models' overfitting". [Briola,
  Bartolucci, Aste 2025, §3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)

**Re-evaluations**

- Briola, Turiel, Aste, "Deep Learning modeling of Limit Order Book: a comparative perspective",
  arXiv:2007.07319: random models, logistic regression, LSTM, attention-LSTM, CNN-LSTM and MLP all
  compared on the **same task, feature space and dataset**. Result: "the **Multilayer Perceptron
  performs comparably to or better than state-of-the-art CNN-LSTM architectures**, indicating that
  dynamic spatial and temporal dimensions are a good approximation of the LOB's dynamics, but not
  necessarily the true underlying dimensions." [arXiv:2007.07319](https://arxiv.org/abs/2007.07319)
- Prata et al., "LOB-based deep learning models for stock price trend prediction: a benchmark study",
  *Artificial Intelligence Review* (2024), DOI 10.1007/s10462-024-10715-4, arXiv:2308.01915. They
  built LOBCAST and tested **15 state-of-the-art DL models**: "**all models exhibit a significant
  performance drop when exposed to new data**, thereby raising questions about their real-world
  market applicability." [Springer](https://link.springer.com/article/10.1007/s10462-024-10715-4);
  [arXiv:2308.01915](https://arxiv.org/abs/2308.01915)
- Briola, Bartolucci, Aste, "Deep limit order book forecasting: a microstructural guide",
  *Quantitative Finance* 25(7):1101–1131 (2025), DOI 10.1080/14697688.2025.2522911. 15 NASDAQ stocks,
  LOBSTER L2, **2017–2019**, 45 train / 5 validation / 10 test days per year, **5-day rolling
  feature-wise z-score** normalisation, and — critically — **tick-level labels**: sign of
  (m_{t+Δτ} − m_t) with |move| ≥ **1 tick**, horizons 10/50/100 LOB updates. DeepLOB results:
  **F1 > 0.45 without thresholding and > 0.7 with probability thresholds; accuracy > 0.7 without
  thresholds, > 0.9 with thresholds** — but the article's own framing is that these numbers are
  misleading, see §2. Cost: **135 experiments, 959 GPU-hours** on 6 GPU types.
  [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)
- Same paper, the profit-relevant metric: they define **p_T = P(executing a correct complete
  transaction)** from the chronological forecast sequence (open a position on the first
  Up/Down signal, close on the next opposite signal). Results: **small-tick stocks p_T ≈ 0.06
  (CHTR/GOOG/GS) to 0.12 (IBM/MCD/NVDA) at H10; medium-tick ≈ 0.14; large-tick ≈ 0.10 at H10,
  ≈0.15 at H50 and H100**. At probability thresholds > 0.5, p_T ≈ 0 for small- and medium-tick
  stocks — i.e. **filtering for confidence destroys the ability to complete any trade**, even as
  MCC/F1 rise. [PMC full text, §7.2](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)
- HLOB (Briola, Bartolucci, Aste, arXiv:2405.18938) — a graph/homological CNN over an
  information-filtering network, benchmarked against 9 SOTA models on 3 real NASDAQ datasets
  (15 stocks each). Its contribution is explicitly about "the **degradation** of information ...
  over increasing prediction horizons" and characterising "the scenarios where HLOB outperforms".
  [arXiv:2405.18938](https://arxiv.org/abs/2405.18938)

### Inferences
- A useful calibration for reading the field: **3-class accuracy on smoothed labels ≈ 0.78–0.84;
  on raw tick labels ≈ 0.45–0.70 depending on horizon and tick size; profit-relevant completion
  probability ≈ 0.06–0.15.** The first number is what gets cited, the third is what a trader has.
- The cleaned-up benchmark ordering (MLP ≈ CNN-LSTM) implies the "architecture" contribution over a
  decent feature set is small; the data/label/tick-size contribution is large. Anyone selling an
  architecture is probably selling the dataset.
- The transfer-learning claim (test stocks unseen) is the one genuinely interesting result, and it is
  consistent with Sirignano–Cont (§3): the *map* from order flow to next-move direction is more
  stationary across assets than across time-regimes.

### Gaps
- DeepLOB never reports net-of-cost PnL; it reports a "simple simulation ... not a fully developed,
  stand-alone trading strategy" (their words). No cited paper gives a like-for-like DeepLOB Sharpe
  after fees and latency on equities.
- No source found that quantifies how much of DeepLOB's 84.47% is attributable to the *class
  balancing* (probability thresholds) versus the labels; the two effects are reported separately but
  never decomposed.

---

## 2. How much of that accuracy is a label-construction artefact

### Takeaway
A large part. The label is the sign of a **forward moving average** over the next k events against a
**0.2% threshold**; that mechanically produces autocorrelated, low-frequency targets that are far
easier to classify than the next tick move. The DeepLOB authors themselves say this label form is
"detrimental for designing trading algorithms" and abandon it on their own data. Overlapping windows
then leak across any random split.

### Cited Findings
- FI-2010's label (Eq. 8 of the dataset paper): ℓ_i = ( (1/k)Σ_{j=i+1}^{i+k} m_j − m_i ) / m_i with
  **k ∈ {1,2,3,5,10} events** and a threshold of **0.002 (0.2%)**: ≥ +0.2% → up, ≤ −0.2% → down,
  in between → stationary. [arXiv:1705.03233](https://arxiv.org/abs/1705.03233)
- DeepLOB restates both label conventions and rejects FI-2010's: with m⁻ = mean of previous k mids,
  m⁺ = mean of next k mids, the two conventions are
  **ℓ = (m⁺ − p_t)/p_t** (FI-2010) and **ℓ = (m⁺ − m⁻)/m⁻** (the alternative they adopt).
  Their words: FI-2010's method "produced labels [that] are less consistent ... because this method
  fits closer to real prices as smoothing is only applied to future prices. **This is essentially
  detrimental for designing trading algorithms as signals are not consistent here leading to many
  redundant trading actions thus incurring larger transaction costs.**" They also report that on LSE
  data "the resulting labels are rather stochastic" and so they switched. [arXiv:1808.03668, §III](https://arxiv.org/abs/1808.03668)
- They also state the motivation for smoothing at all: "if we simply compare p_t and p_{t+k} to
  decide the price movement, the resulting label set will be noisy." I.e. the smoothing exists to
  make the classification task *easier than trading*. [arXiv:1808.03668, §III](https://arxiv.org/abs/1808.03668)
- On the LSE experiments they add that "the classes are roughly balanced" — achieved by choosing
  horizon/threshold, which is itself a free parameter tuned per dataset. [arXiv:1808.03668, §V-C](https://arxiv.org/abs/1808.03668)
- Independent evidence that the artefact is load-bearing: switching to *tick-level* labels
  (|Δm| ≥ 1 tick) on real NASDAQ data drops reported F1 to 0.45–0.7 (before thresholding) and the
  tradable completion probability to 0.06–0.15. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)
- Class imbalance is systematic in these labels: at H10 large-tick stocks are dominated by
  "stationary" (e.g. BAC 4.59e5 stationary vs 1.24e4 down / 1.23e4 up per day), while at H50/H100
  the imbalance flips. Papers that report plain accuracy on such data are reporting the base rate.
  [PMC, Table 3–5](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)
- Overlapping windows: with stride-1 sampling and horizon k, consecutive samples share k−1 future
  observations, so a random train/test split is leakage by construction. The dataset's own defence is
  the **day-based anchored forward** protocol; the DeepLOB "Setup 1" is exactly that (growing
  historical training window, test on the next day), which is why its numbers (78.91) are lower than
  the random-ish "Setup 2" (84.47). [arXiv:1705.03233](https://arxiv.org/abs/1705.03233); [arXiv:1808.03668](https://arxiv.org/abs/1808.03668)
- Global normalisation is a second leak: most of the literature z-scores with statistics of the whole
  training set (or worse, the whole file); Briola et al. use a **5-day rolling, feature-wise** z-score
  explicitly because global statistics are invalid "in an evolving and strongly non-stationary
  environment". [PMC, §4](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)

### Inferences
- Rule of thumb for reading a reported LOB accuracy: subtract the majority-class rate of the label
  distribution *at your horizon*, then ask whether what remains is bigger than the spread you can
  cross. On FI-2010-style labels the majority class is often 40–70%.
- The forward-average label is effectively a **low-pass filter on the price**, so models learn
  volatility/drift state, not the next move. That is why a 2-layer MLP matches a CNN-LSTM: the task
  is mostly "is the market currently trending", which needs little spatial structure.
- Any comparison that changes the label definition (FI-2010 vs DeepLOB-LSE vs Briola tick labels)
  is not a comparison of models. This is the single largest source of inflated claims in the field.

### Gaps
- No paper found that reports the **same architecture with the same features under all four label
  definitions** (raw next-tick, forward-mean, both-side-mean, thresholded-at-1-tick). Briola et al.
  2025 is the closest (tick-level) and Prata et al. the closest on model breadth, but neither does the
  full 4×N label ablation.
- I found no study quantifying label autocorrelation directly (e.g. effective sample size after
  smoothing), which would settle the "how much is artefact" question numerically.

---

## 3. Feature engineering that survives: hand-crafted microstructure vs learned representations

### Takeaway
Two things survive contact with data: (a) **order-flow imbalance at the touch and near levels**,
which is a low-dimensional, O(1)-computable, near-universal feature; and (b) **the claim that the
order-flow → next-move map is approximately universal across stocks** (Sirignano–Cont), which is the
strongest scientific result in the area and also the one with the least PnL attached to it. Deep
representations buy little over these features, and cost gigabytes and GPU-hours.

### Cited Findings
- Sirignano & Cont, "Universal features of price formation in financial markets: perspectives from
  deep learning", *Quantitative Finance* 19(9):1449–1459 (2019), DOI 10.1080/14697688.2019.1622295,
  arXiv:1803.06917. Data: NASDAQ, **~500 stocks**, **17 months training / 3 months out-of-sample**
  (test ≈15% of the data, "hundreds of millions of observations"); training distributed over
  **500 GPU nodes** for stock-specific models, **25 nodes with asynchronous SGD** for the universal
  model. [arXiv:1803.06917](https://arxiv.org/abs/1803.06917); [author PDF](http://rama.cont.perso.math.cnrs.fr/pdf/SirignanoCont2019.pdf)
- Their headline quantitative claims: deep nets beat stock-specific linear VAR+probit models by
  **5–10 percentage points of accuracy for most stocks**; "**an increase of even 1% in accuracy is
  considered a significant improvement in high-frequency modeling**"; the *universal* model (trained
  on all stocks pooled, no per-stock normalisation) matches or beats stock-specific models and
  transfers to stocks never trained on; accuracy is "remarkably stable ... even a year out of sample";
  and accuracy keeps improving as you feed **longer order-flow history** ("path-dependence and
  long-range dependence"), e.g. an LSTM on a 5000-step sequence beats one on 100 steps.
  [author PDF](http://rama.cont.perso.math.cnrs.fr/pdf/SirignanoCont2019.pdf)
- The same paper's own disclaimer: "Determining whether this predictability leads to profitable
  trading strategies requires a careful analysis of transaction costs, taking into account factors
  such as latency, a detailed study of which is **beyond the scope of this paper**."
  [author PDF](http://rama.cont.perso.math.cnrs.fr/pdf/SirignanoCont2019.pdf)
- The mechanism behind the universality is a *queueing* relation: Cont & De Larrard showed a
  model-free relation between bid depth, ask depth and the probability of the next price move being
  down, under symmetric flow; Sirignano–Cont confirm it empirically across assets. This is why depth
  imbalance features generalize. [author PDF, §3.1](http://rama.cont.perso.math.cnrs.fr/pdf/SirignanoCont2019.pdf)
- On which levels matter: DeepLOB's sensitivity analysis and its cited evidence say **L1 dominates
  price discovery**, with all deeper levels contributing "as little as 20%", and >90% of orders end in
  cancellation — hence their choice to convolve/summarise deeper levels rather than feed them raw.
  [arXiv:1808.03668, §IV-B](https://arxiv.org/abs/1808.03668)
- The modern, better-specified version of the hand-crafted route: Kolm, Turiel, Westray, "Deep order
  flow imbalance: Extracting alpha at multiple horizons from the limit order book", *Mathematical
  Finance* 33(4):1044–1081 (2023), DOI 10.1111/mafi.12413. They generalise OFI to multiple levels and
  a cross-section, and report that predictive R² **decays with horizon** and that gains come mostly
  from the cross-sectional/multi-level aggregation rather than depth of network.
  [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/mafi.12413); [SSRN 3900141](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141)
- Predictability is a property of the *stock*, not the model: papers converge on tick size, liquidity,
  spread/depth distribution and "information richness" as the determinants — large-tick stocks are
  predictable-but-unprofitable (1-tick spread), small-tick stocks are profitable-but-unpredictable.
  [PMC §6–7](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)

### Inferences
- For a one-file, O(1)-per-event strategy the defensible feature set is: signed volume at the touch
  and 1–3 levels away, queue imbalance, microprice deviation, event-rate/volatility state, and
  recent signed IOC flow — all of which are the human-readable core of what the deep models
  rediscover. Nothing in the literature suggests a learned representation beats these by more than a
  couple of accuracy points on *tick-level* labels.
- The Sirignano–Cont universality result is directly usable without training: it justifies fitting a
  *small* parametric map (a few coefficients on depth imbalance, flow, history) rather than an
  architecture, and it justifies **pooling across time/regimes with slow adaptation** rather than
  refitting per day.

### Gaps
- Sirignano–Cont do not publish the numeric accuracy level of the universal model in the text (only
  the relative improvement figures and histograms), so "55% vs 50%" style numbers cannot be quoted
  from the paper; I did not find them in a secondary source either.
- No head-to-head study found that reports hand-crafted OFI features vs a deep model **on the same
  tape with the same cost model**. Kolm et al. is the closest but the comparison is internal.

---

## 4. RL for market making and execution: what it achieves, and against what baseline

### Takeaway
RL in this domain is a **numerical method for a model you already wrote down**, not a way to discover
alpha. Where it is evaluated honestly (Guéant–Manziuk) it is benchmarked against the analytic solution
of its own model and wins on *dimensionality*, not on realism. Where it is evaluated against market
data (Nevmyvaka et al.) the win is against a weak execution heuristic and pre-cost. The only systematic
attempt to close the simulator gap is adversarial training against a perturbed simulator, and it
reports robustness, not profit.

### Cited Findings

**Execution (the reference point that still holds)**
- Nevmyvaka, Feng, Kearns, "Reinforcement Learning for Optimized Trade Execution", ICML 2006.
  First large-scale empirical RL for execution: state = (time remaining, inventory remaining, order
  book features such as spread, bid/ask volume imbalance, signed transaction volume); action =
  market/limit at various prices; reward = **trading cost in basis points versus the arrival mid**;
  learned over millions of NASDAQ order-book events. Reported: improvements over the S&L baseline of
  **27.16%–35.50%** depending on state resolution, and "**50% or more** over S&L policies" in the
  aggregated comparison; the paper's own abstract-level claim is "can improve relative performance by
  as much as 50%". Their feature ablation gives the ranking (spread 7.97%, immediate market-order cost
  4.26%, signed volume 2.81%, spread+immediate cost 8.69%, all three 12.85% of improvement).
  [author PDF](https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf)
- Reading note: the baseline is a heuristic ("S&L"), the cost model is the paper's own, latency is
  not modelled at the microsecond scale, and the 50% figure is relative to a policy no desk would run.

**Market making in simulation**
- Spooner, Fearnley, Savani, Koukorinis, "Market Making via Reinforcement Learning", AAMAS 2018,
  arXiv:1804.04216. "High-fidelity simulation of limit order book markets"; **linear tile-coding**
  value function (not deep); "custom reward function that controls inventory risk"; claims to
  outperform "both simple benchmark strategies and a recent online learning approach".
  [arXiv:1804.04216](https://arxiv.org/abs/1804.04216)
- Spooner & Savani, "Robust Market Making via Adversarial Reinforcement Learning", arXiv:2003.01820 —
  the explicit answer to the simulator gap: train the maker against an adversary that perturbs market
  dynamics, and report robustness of the learned policy across simulated regimes.
  [IJCAI 2020 paper citing it](https://ijcai.org/proceedings/2020/0633.pdf)
- Gašperov & Kostanjčar, "Market making with signals through deep reinforcement learning" (IEEE
  Access, 2021): adversarial RL + neuroevolution, evaluated "on historical data", i.e. one of the few
  that leaves the simulator. [IEEE](https://ieeexplore.ieee.org/iel7/6287639/9312710/09410223.pdf)

**Market making as a PDE solver**
- Guéant & Manziuk, "Deep reinforcement learning for market making in corporate bonds: beating the
  curse of dimensionality", *Applied Mathematical Finance*, DOI 10.1080/1350486X.2020.1714455,
  arXiv:1910.13205. The honest framing: classical finite-difference solutions of the
  Avellaneda–Stoikov-type HJB **cannot be used** for a universe of hundreds of bonds, so they use a
  **model-based actor-critic with deep networks** to approximate the optimal quotes. The paper is
  explicit that it approximates the optimum *of a model they specify*. Value: scalability and
  agreement with reference solutions, not evidence about real markets.
  [arXiv:1910.13205](https://arxiv.org/abs/1910.13205)
- Historical reference point: Chan & Shelton, "An Electronic Market-Maker" (MIT AI Lab, 2001) — the
  first RL market maker; still cited as the origin of the inventory-penalty reward design. I did not
  verify a stable public link this session (see Gaps).

**Reward design that recurs across all of the above**
- Inventory penalty λ·q² (or λ·|q|), terminal liquidation/mark-to-market at the end of the episode,
  and often a drawdown or variance penalty; the quote skew that RL "discovers" is qualitatively the
  Avellaneda–Stoikov reservation-price shift. Reported gains over analytic baselines come mostly from
  the RL policy learning a *state-dependent* λ, i.e. quoting wider when inventory is large and
  volatility high. [Guéant–Manziuk](https://arxiv.org/abs/1910.13205); [Spooner et al.](https://arxiv.org/abs/1804.04216)

### Inferences
- The RL literature's real finding is a **negative** one for our purposes: whenever a strong analytic
  baseline exists (single asset, Poisson fills, no adverse selection), RL roughly rediscovers it; the
  gains appear only where the state space is too large for the analytic solution (many assets, many
  book levels). We have neither the state space nor the training budget.
- "Beats the analytic baseline" in these papers almost always means "beats a *mis-specified* baseline"
  (AS with constant λ, no adverse selection, no queue position). That is a statement about the
  baseline, not about alpha.
- The simulator gap is not closable by more RL. Adversarial training changes the *worst case* the
  policy is robust to; it cannot import the parts of reality absent from the simulator (queue
  priority races, latency jitter, other agents reacting to you).

### Gaps
- I found no peer-reviewed RL market-making result reporting **net-of-fees, net-of-latency PnL on
  real exchange data with a live or paper-trading record**. All the credible ones are simulator-only
  or one-tape backtests with an endogenous fill model.
- Chan & Shelton (2001) not link-verified here; treat as a historical citation.

---

## 5. LOB simulators and synthetic tapes: what they reproduce, and how they fail

### Takeaway
Three families exist: **agent-based discrete-event** (ABIDES), **queue-reactive stochastic** (Huang–
Lehalle–Rosenbaum), and **learned world models** (Coletta et al.). All three reproduce the
*unconditional* stylized facts (heavy tails, volatility clustering, concave impact, spread/depth
distributions) reasonably well. None is validated on *conditional* dynamics — how the book responds to
**your** order — which is exactly what a strategy exploits. Calibration is also non-identifiable:
many parameter sets match the same facts.

### Cited Findings
- ABIDES (Byrd, Hybinette, Balch), "ABIDES: Towards High-Fidelity Market Simulation for AI Research",
  arXiv:1904.12066 (2019); ACM SIGSIM PADS 2020 version. Discrete-event, agent-based, **tens of
  thousands of trading agents** against an exchange agent, with explicit message latency and
  configurable exchange rules; designed so an RL agent can act inside it.
  [arXiv:1904.12066](https://arxiv.org/abs/1904.12066)
- Calibration of ABIDES is *ill-posed*: Bai et al., "Calibrating Non-Identifiable High-Dimensional
  Simulation Models" (ACM, 2025) — "for ABIDES calibration, the approach identifies **configurations**
  matching real data according to stylized facts describing limit order book" — plural configurations,
  i.e. the map from facts to parameters is not one-to-one.
  [ACM](https://dl.acm.org/doi/full/10.1145/3742427)
- Queue-reactive model: Huang, Lehalle, Rosenbaum, "Simulating and Analyzing Order Book Data: The
  Queue-Reactive Model", *JASA* 110(509):107–122 (2015), arXiv:1312.0563. Limit/cancel/market
  intensities are functions of the **queue sizes** at each level; parameters estimated from data; the
  paper establishes limiting behaviour and argues it is "a very relevant market simulator, notably in
  view of the analysis of complex trading tactics".
  [arXiv:1312.0563](https://arxiv.org/abs/1312.0563)
- Its structural limitation, stated by its own authors' follow-up work: the QRM is stationary around a
  reference price, so reproducing realistic price diffusion requires **localising** the model
  (recalibrating per regime) or adding an exogenous price — hence the 2024 "Localising the Queue
  Reactive Model" project and a stream of extensions ("A Novel Approach to Queue-Reactive Models",
  arXiv:2405.18594). [CMAP project page](http://www.cmap.polytechnique.fr/~charles-albert.lehalle/projects/2024QR/);
  [arXiv:2405.18594](https://arxiv.org/html/2405.18594v1)
- Learned simulators: Coletta et al., "Learning to simulate realistic limit order book markets from
  data as a World Agent", ICAIF 2022, arXiv:2210.09897 — "a world model simulator that accurately
  emulates a limit order book market — it requires **no agent calibration** but rather learns the
  [dynamics]"; it reproduces stylized facts and market impact, and is explicitly designed to make RL
  agents face a *reactive* market. Follow-up: conditional generators for LOB environments,
  arXiv:2306.12806. [arXiv:2210.09897](https://arxiv.org/abs/2210.09897);
  [arXiv:2306.12806](https://arxiv.org/abs/2306.12806)
- The classic agent-based reference in this lineage is Chiarella & Iori's double-auction model
  (*Quantitative Finance*, 2002), which produces realistic spread/depth/clustering from heterogeneous
  zero-intelligence-plus agents; it is the standard "stylized facts emerge from agent heterogeneity"
  argument, not a calibrated generator. (Not re-verified by DOI this session — see Gaps.)
- The failure that matters commercially: Briola et al. enumerate exactly the assumptions that make a
  DL backtest on a tape meaningless — "(i) technical and infrastructural potential to record and
  process live data, produce forecasts and execute in due time; (ii) being always executed; (iii)
  having **zero market-impact**; (iv) having **zero transaction costs**" — and conclude that
  "backtesting a trading strategy based on the outputs of a deep learning model by using historical
  data-only is **not possible**". [PMC §7.2](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)

### Inferences
- For our setting the useful role of a synthetic tape is **not realism** but **ground truth**: a
  generator with a known latent price S lets you measure detection delay, false positives, and the PnL
  attributable to a mechanism, none of which a single real tape can tell you. That is a different
  (and much stronger) use than "train a model on synthetic data and hope it transfers".
- Judge any generator by conditional, not unconditional, statistics: response of the book to a
  one-lot IOC at the touch, fill probability vs queue position, and the autocorrelation of signed
  order flow at 10 ms–1 s. None of the papers above report all three.
- A generator that makes price mean-revert (as stationary QRMs do) will make *any* inventory-reducing
  market maker look profitable. This is the most common way simulator-based MM results are wrong.

### Gaps
- No numeric scorecard found ("fraction of stylized facts matched") for ABIDES on a specific tape;
  claims are qualitative. Bai et al. 2025 is the closest to a systematic calibration study.
- Chiarella–Iori 2002 DOI not verified in this session.

---

## 6. Evaluation methodology: walk-forward, purged CV, PBO, deflated Sharpe, minimum backtest length

### Takeaway
The honest protocol has four independent defences, and published LOB papers typically use at most one:
**(1) time-ordered splits with purging+embargo, (2) many paths rather than one backtest, (3) an
explicit multiple-testing correction tied to the number of configurations you tried, (4) a metric that
reflects the trading decision rather than classification accuracy.** The concrete thresholds below are
the ones with citable definitions.

### Cited Findings
- **Purged k-fold + embargo, and combinatorial purged CV (CPCV)**: López de Prado, *Advances in
  Financial Machine Learning*, Wiley 2018, ch. 7. Procedure: split into K folds; **purge** from the
  training set any observation whose label window overlaps a test observation's window; **embargo** a
  fraction of the observations immediately *after* each test fold (the book's default is 1% of T) to
  kill serial-correlation bleed; CPCV then recombines the K folds into many train/test paths, giving a
  *distribution* of backtest outcomes instead of one. [Method summary](https://quantmemo.com/concepts/purged-embargoed-cv);
  [implementation](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html);
  [author's list of innovations](https://www.quantresearch.org/Innovations.htm)
- **Why walk-forward alone is insufficient**: it produces a single path, so the observed Sharpe is one
  draw; you cannot separate skill from path luck, and selecting the best walk-forward configuration is
  itself a multiple test. CPCV exists precisely to produce the distribution that PBO needs.
  [author's innovations page](https://www.quantresearch.org/Innovations.htm)
- **Probability of backtest overfitting (PBO)**: Bailey, Borwein, López de Prado, Zhu, "The Probability
  of Backtest Overfitting", *Journal of Computational Finance* 20(4), 2017. Definition: split the
  performance matrix (N configurations × T observations) into S disjoint submatrices; for each
  combination, pick the configuration with the best in-sample statistic and record its **out-of-sample
  rank**; PBO is the probability that this rank is below the median (logit ≤ 0). PBO = 0.5 means the
  selection procedure is a coin flip. Their simulations show PBO far above 0.5 for realistic numbers of
  trials even on data with no signal. [author PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- **Pseudo-mathematics / minimum backtest length (MinBTL)**: Bailey, Borwein, López de Prado, Zhu,
  "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample
  Performance", *Notices of the AMS* 61(5), 2014. The argument: with N independent trials, the
  expected maximum in-sample Sharpe grows like √(2 ln N / T), so the amount of history required to
  keep the expected best-of-N Sharpe at a target level grows with ln N; hence "most discoveries in
  finance are likely false" when N is large relative to T. [CARMA copy](https://carmamaths.org/jon/backtest.pdf)
- **Deflated Sharpe Ratio (DSR)**: Bailey & López de Prado, "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality", *Journal of Portfolio Management* 40(5),
  2014. DSR = PSR evaluated against a benchmark SR* equal to the **expected maximum Sharpe under N
  trials**, and it also corrects for skewness/kurtosis and track length. Operational threshold in the
  paper: a strategy is credible when **DSR > 0.95**, i.e. the probability that the true Sharpe exceeds
  the selection-adjusted benchmark is at least 95%. [author PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- **What "getting it right" looks like in the LOB literature**: Sirignano–Cont — 17 months train, 3
  months out-of-sample, ~500 stocks, per-stock holdout and pooled model compared, and **no PnL claim**.
  Briola et al. — 45/5/10-day splits per year over 3 years, rolling normalisation, tick-level labels,
  MCC/F1/accuracy *and* a decision-level metric (p_T), and an explicit refusal to publish a backtest
  PnL. Prata et al. — a fixed open-source harness (LOBCAST) with out-of-distribution tests across 15
  models. Bailey/LdP — selection-adjusted statistics. The common feature is that **none of them reports
  a Sharpe without also reporting the search that produced it.** [author PDF](http://rama.cont.perso.math.cnrs.fr/pdf/SirignanoCont2019.pdf);
  [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/); [arXiv:2308.01915](https://arxiv.org/abs/2308.01915)

### Inferences
- Minimum viable protocol for a one-tape project like ours: (a) never split randomly; (b) hold out
  whole time chunks *and* embargo at least the strategy's decision horizon after each chunk; (c) count
  every configuration you evaluated and deflate; (d) report a decision-level metric (fills, round trips,
  per-trade markout) alongside PnL; (e) treat a ±5% PnL difference between variants as noise unless the
  standard error says otherwise.
- Concretely for our numbers: a 500k PnL with ≈26k standard error (measured on our own round-2 runs)
  means **anything under ~50k of improvement is not evidence**. That is the same arithmetic as
  MinBTL, applied at the scale of a single day's tape. [local: round2 audit](../round2-deeper-strategy-research/round2-audit-and-inference.md)
- PBO with our search intensity (dozens of strategy variants per round) will be high by construction;
  the practical substitute is **out-of-sample tapes** (unseen synthetic seeds, mirrored market,
  reshuffled session order), which is what `scripts/robustness.py` does.

### Gaps
- The embargo "1% of T" default is from the book, not from an LOB-specific study; no citation found
  that tunes embargo length for event-time LOB data.
- No paper found that applies PBO/DSR to a *single-tape intraday* strategy; all applications are
  multi-year, multi-asset return series, so the mapping to our setting is an inference.

---

## 7. Non-stationarity and regime shift: how fast do the relationships rot

### Takeaway
Two findings coexist and must be reported together: at the **event horizon**, the order-flow → next-move
direction map is surprisingly stable across stocks and up to a year (Sirignano–Cont); at the
**trading horizon**, the *conditional* predictability and the usable signal decay over weeks and differ
by regime, so any fixed threshold or normalisation goes stale (Briola, Prata). Reconciliation: the
*shape* of the map is stable, its *scale* (volatility, activity, spread) is not — which is exactly an
argument for online estimation of scale and offline-fixed shape.

### Cited Findings
- Stability side: universal model trained on pooled data keeps "remarkably stable out-of-sample
  accuracy ... even a year out of sample"; accuracy keeps rising with more order-flow history;
  transfers to unseen stocks. [author PDF](http://rama.cont.perso.math.cnrs.fr/pdf/SirignanoCont2019.pdf)
- Decay side: DeepLOB's own LSE accuracy falls monotonically with horizon (70.17 → 63.93 → 61.52 at
  k=20/50/100) and its per-day boxplots show large day-to-day dispersion; Briola et al. report
  year-by-year MCC differences and find it necessary to use **5-day rolling** standardisation rather
  than global statistics "in an evolving and strongly non-stationary environment".
  [arXiv:1808.03668](https://arxiv.org/abs/1808.03668); [PMC §4](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)
- All 15 models in Prata et al. degrade "when exposed to new data" — the generalisation failure is
  temporal, not architectural. [arXiv:2308.01915](https://arxiv.org/abs/2308.01915)
- Predictability is regime-dependent: Prata et al. and Briola et al. both tie forecast performance to
  **volatility clustering** and to activity/liquidity state; the class distribution itself flips between
  H10 and H100 (stationary-dominant for large-tick at H10, up/down-dominant at H50+), which means a
  model calibrated on one horizon's base rate is mis-calibrated at another.
  [PMC Tables 3–5](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)
- Volatility clustering is the one stylized fact that all cited microstructure work agrees is robust and
  dominant in high-frequency data, and it is the reason any fixed-horizon label mixes regimes.
  [Cont, "Volatility Clustering in Financial Markets"](http://rama.cont.perso.math.cnrs.fr/pdf/clustering.pdf)

### Inferences
- The adaptive toolkit that is defensible with numpy only: **forgetting-factor RLS** for the linear
  part of the map, a **scalar Kalman filter** for the latent level, **exponentially-weighted** scale
  estimators, and a small **regime filter** (HMM forward pass on event counts) that only rescales
  parameters. This matches the evidence: shape stationary, scale not.
- Sample-size arithmetic kills per-regime fitting: at "six jumps/hour" or "a dozen storms/day" you have
  tens of observations per regime per tape; anything with more than ~3 parameters per regime is fitting
  noise. Use regimes to *modulate* a pooled model, not to train separate ones.
- Change-point detection is worth having as a *diagnostic* (it tells you when your residuals stop being
  exchangeable) but the literature does not support it as an alpha source by itself (see the local
  round-2 finding that BOCPD-style reset did not beat a simpler covariance inflation).

### Gaps
- No study found that measures the **half-life of LOB feature→return relationships** across days on a
  single venue with a consistent methodology; the "5-day rolling window" in Briola et al. is a design
  choice, not an estimated quantity.
- No verified source quantifies how much of the cross-year degradation is regime change versus simple
  parameter drift.

---

## 8. Realistic expectations, compute/data requirements, and the base rate of "ML for trading works"

### Takeaway
The base rate is bad in a specific, predictable way: **classification metrics replicate, net-of-cost
PnL does not.** Every credible paper in this literature either (a) reports accuracy without costs, or
(b) reports a decision-level metric that is far worse than the accuracy suggests (p_T ≈ 0.06–0.15), or
(c) explicitly declines to backtest. The compute and data requirements of the credible work (hundreds
of GPU-hours, hundreds of millions of events, 500 stocks) are three to five orders of magnitude beyond
what a single-file, one-tape strategy can support.

### Cited Findings
- **Cost realism, stated by the field's own authors**: backtesting DL signals on a tape requires
  assuming "being always executed", "zero market-impact", "zero transaction costs"; the combination
  "ruins any attempt to produce a reliable backtest". [PMC §7.2](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)
- **The decision-level number**: p_T = 0.06–0.15 across stock classes and horizons; at probability
  thresholds above 0.5, p_T ≈ 0 for small/medium-tick stocks, i.e. higher-confidence forecasts cannot
  complete transactions. Note this metric is *before* costs — it is about the chronological placement of
  errors, not fees. [PMC §7.2, Tables 8–10](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/)
- **Compute/data scale of the credible results**: DeepLOB — 1 year LSE quotes, 10 stocks, GPU training,
  60k params; FI-2010 — 10 days × 5 stocks. Briola et al. — 135 runs / **959 GPU-hours** / 15 stocks ×
  3 years. Sirignano–Cont — **~500 stocks, 17 months, hundreds of millions of observations, 500 GPU
  nodes**. Prata et al. — 15 models × out-of-distribution tests in a dedicated harness. There is no
  published deep-LOB result at "one day, one instrument, laptop".
  [arXiv:1808.03668](https://arxiv.org/abs/1808.03668); [PMC §5](https://pmc.ncbi.nlm.nih.gov/articles/PMC12315853/);
  [author PDF](http://rama.cont.perso.math.cnrs.fr/pdf/SirignanoCont2019.pdf)
- **The base rate in the adjacent, easier literature** (monthly equity factors, decades of data, no
  latency): Jensen, Kelly, Pedersen, "Is There a Replication Crisis in Finance?", *Journal of Finance*
  78(5):2465–2518 (2023) — a **82.4% replication rate** once you use a Bayesian hierarchical model and
  CAPM-alpha tests instead of naive t-stats; i.e. the widely-claimed "most factors are false" is too
  pessimistic *for that literature*. This is the *best case*: slow signals, huge samples, decades of
  data, no queue position. [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249);
  [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3774514)
- **The multiple-testing counterweight**: PBO and DSR (§6) say the same thing from the other side —
  selection among many configurations mechanically produces high in-sample Sharpe with no signal, and
  the correction is a function of how many configurations you tried, not of how good the best one looks.
  [PBO PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf); [DSR PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)

### Inferences
- **Do not invest in deep learning for this project.** Not because the papers are wrong, but because
  the required inputs — hundreds of millions of labelled events, a GPU budget, and a multi-year
  out-of-sample span — are structurally absent, and the literature's own decision-level metrics
  (p_T ≈ 0.1) say the signal, even when it exists, is smaller than the spread on most instruments.
- What the evidence *does* support, at our scale: (i) O(1) online estimators with 3–10 parameters fitted
  online (RLS/Kalman/decayed regressions), (ii) microstructure features that are already known to be
  universal (depth imbalance, signed touch flow, microprice), (iii) regime *modulation* rather than
  regime *classification*, (iv) validation by ablation on synthetic tapes with known S plus mirrored and
  reshuffled variants of the real tape, (v) explicit noise floors (standard errors, PnL-per-trade
  distributions) before believing any improvement.
- If someone insists on ML here, the only defensible version is **scikit-learn with < 20 features and
  < 10^3 parameters, refit online or on a rolling window, validated with purged time splits and a
  decision-level metric.** That is a different project from "deep learning on LOB", and the literature
  offers no evidence it beats a well-tuned linear model with the same features (Briola 2020: MLP ≈
  CNN-LSTM; Sirignano–Cont: the *only* robust win of depth is over a linear VAR+probit, by 5–10 pp on a
  task with no cost model).

### Gaps
- I found **no** study reporting the fraction of published ML-for-HFT signals that survive realistic
  transaction costs and latency. The closest quantitative statements are Briola's p_T and the general
  refusal of the credible papers to publish net PnL. This is a genuine hole in the literature, not a
  search failure — treat any "X% of signals survive costs" claim as unsourced.
- No source found that reports a DeepLOB-class model's Sharpe net of exchange fees and 5 ms-scale
  latency on any venue.
- The Jensen–Kelly–Pedersen 82.4% figure is for monthly equity factors; I found no equivalent
  replication study for intraday/HFT strategies.

---

## Appendix: citation index (all verified this session)

| Reference | ID / venue | Used for |
|---|---|---|
| Zhang, Zohren, Roberts, DeepLOB | IEEE TSP 67(11) 2019, DOI 10.1109/TSP.2019.2907260, arXiv:1808.03668 | acc/F1 tables, LSE results, label critique |
| Ntakaris et al., FI-2010 benchmark | J. Forecasting 37(8), DOI 10.1002/for.2543, arXiv:1705.03233 | dataset + label definition (0.2%, k∈{1,2,3,5,10}) |
| Briola, Turiel, Aste, comparative perspective | arXiv:2007.07319 | MLP ≈ CNN-LSTM |
| Prata et al., LOBCAST benchmark | AI Review 2024, DOI 10.1007/s10462-024-10715-4, arXiv:2308.01915 | all 15 models degrade OOD |
| Briola, Bartolucci, Aste, microstructural guide | Quant. Finance 25(7):1101–1131, DOI 10.1080/14697688.2025.2522911, arXiv:2403.09267 | tick labels, MCC/F1, p_T, LOBFrame, 959 GPU-h |
| Briola et al., HLOB | arXiv:2405.18938 | information degradation with horizon |
| Sirignano & Cont, universal features | Quant. Finance 19(9):1449–1459, DOI 10.1080/14697688.2019.1622295, arXiv:1803.06917 | 500 stocks/17mo/500 GPUs, +5–10 pp over linear, no PnL |
| Kolm, Turiel, Westray, deep OFI | Math. Finance 33(4):1044–1081, DOI 10.1111/mafi.12413 | multi-level OFI, R² decay with horizon |
| Nevmyvaka, Feng, Kearns | ICML 2006 | RL execution, 27–35% / "50% or more" vs S&L |
| Spooner et al., MM via RL | AAMAS 2018, arXiv:1804.04216 | tile coding, inventory-risk reward |
| Spooner & Savani, adversarial MM | arXiv:2003.01820 | simulator-gap robustness |
| Guéant & Manziuk, DRL for corporate bonds | Appl. Math. Finance, DOI 10.1080/1350486X.2020.1714455, arXiv:1910.13205 | RL as PDE solver, 100s of assets |
| Gašperov & Kostanjčar, MM with signals | IEEE Access 2021 | adversarial + neuroevolution on historical data |
| Byrd, Hybinette, Balch, ABIDES | arXiv:1904.12066; SIGSIM PADS 2020 | agent-based discrete-event sim |
| Bai et al., ABIDES calibration non-identifiability | ACM 2025, DOI 10.1145/3742427 | many configs match same stylized facts |
| Huang, Lehalle, Rosenbaum, queue-reactive | JASA 110(509):107–122, arXiv:1312.0563 | queue-dependent intensities; stationarity limit |
| Coletta et al., world agent | ICAIF 2022, arXiv:2210.09897; arXiv:2306.12806 | learned LOB simulator, reactive to agent |
| Bailey, Borwein, LdP, Zhu, PBO | J. Computational Finance 20(4) 2017 | PBO definition |
| Bailey et al., pseudo-mathematics / MinBTL | Notices AMS 61(5) 2014 | trials vs history trade-off |
| Bailey & LdP, deflated Sharpe | JPM 40(5) 2014 | DSR > 0.95 threshold |
| López de Prado, purged CV / CPCV | *Advances in Financial ML*, Wiley 2018, ch. 7 | purge + embargo (1% default) |
| Jensen, Kelly, Pedersen | J. Finance 78(5):2465–2518, DOI 10.1111/jofi.13249 | 82.4% replication (best-case base rate) |
| Cont, volatility clustering | author PDF | dominant HF stylized fact |
