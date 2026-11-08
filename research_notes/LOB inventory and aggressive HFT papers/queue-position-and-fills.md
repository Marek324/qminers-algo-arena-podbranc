# Queue Position, Limit-Order Fill Probability and the Economics of Order Placement

Scope note for the report writer: this note is written for a discrete-event simulator with integer ticks, strict price-time priority, a one-tick spread, ~900 lots resting at the touch, 5 ms send/cancel/modify latency and a hard |I| ≤ 100 inventory cap. Every formula below is stated so that it can be maintained online (counters, EWMAs, small fixed-size tables). Where a published result assumes a continuous price grid, an infinitesimal order size, or an infinitesimally short queue, that assumption is flagged.

---

## 1. Queue position and the value of priority: how is being k-th in the queue quantified?

### Takeaway

The canonical decomposition is **order value = fill probability × (liquidity premium − adverse selection)**, where all three terms are functions of queue position (Moallemi & Yuan 2016). In large-tick US equities the resulting value of a front-of-queue position is **0.03–0.41 ticks**, i.e. the same order of magnitude as the half-spread, and it decays steeply over the first few thousand shares of queue ahead, while adverse selection *rises* with queue position.

### Cited Findings

**The valuation identity.** Moallemi & Yuan derive, by optional stopping on the latent efficient price P_t,
`V_t = α_t (δ_t − AS_t)`, with `α_t = P(FILL | F_t)`, `δ_t = P_A − P_t` (the "liquidity premium", equal to the half-spread if P_t is the mid), and `AS_t = E[P_{τ*} − P_t | F_t, FILL]` (adverse selection) — [Moallemi & Yuan, *A Model for Queue Position Valuation in a Limit Order Book* (2016)](https://moallemi.com/ciamac/papers/queue-value-2016.pdf). They describe this as "an accounting identity that breaks down the expected profitability of liquidity provision at the level of an individual order: order value = fill probability × (liquidity spread premium − adverse selection cost)" — same source. An earlier SSRN version is [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2996221); a conference version of the same material is [the Paris market-microstructure slides](http://market-microstructure.institutlouisbachelier.org/uploads/91_7%20MOALLEMI%202014-12-paris-mm-queue-value.pdf), which state "value of queue position can be very large. For some stocks, it has same order of magnitude as the half-spread! Must account for queue value in making [placement decisions]".

**Closed form (no cancellations, exponential trade sizes).** For two-sided exponential trade size density `f(u) = (θ/2) e^{−θ|u|}`, no cancellations, Poisson trades (rate μ), Poisson price jumps (rate γ, P(up)=p⁺_J), the value function is `V(q,δ) = α(q)δ − β(q)` with
`α(q) = p⁺_J + [μ(1−p⁺_J)/(μ+2γ)] e^{−bq}`, `b ≡ (γ+ζ)θ/(μ/2+γ)`, `ζ = μ+γ+η⁺`,
and β(q) given by a similar exponential plus a `q e^{−bq}` term — [Moallemi & Yuan (2016), Theorem 3](https://moallemi.com/ciamac/papers/queue-value-2016.pdf). With no cancellations (η⁺=0, so ζ=μ+γ) this collapses to `b = 2θ = 2/E|u|`: **the e-folding length of fill probability in queue position is half the mean taker order size**. General statement: `α_t = α(q)` and `AS_t = β(q)/α(q)`; α is non-increasing in q; β/α > 0 (adverse selection always positive); with no cancellations `lim_{q→∞} α(q) = p⁺_J` and `lim β(q) = J̄⁺` (a maximally back-of-queue order is only filled by a price jump) — same source, Theorem 2.

**Marginal value of one position.** Because `V(q) = α(q)δ − β(q)`, the value of moving one lot forward is `V(q−1) − V(q) = [α(q−1) − α(q)]δ − [β(q−1) − β(q)]`, i.e. **the gain in fill probability times the premium, plus the reduction in adverse selection**. Moallemi & Yuan note adverse selection is *increasing* in queue position (orders deep in a queue are only reached by large, informed trades; "a good queue position acts as a filter on the population of contra-side market orders"), attributing the first statement of this to Glosten (1994) — same source.

**Empirical magnitudes (NASDAQ ITCH, large-tick US equities, Aug 2013).** For BAC on 8/9/2013 vs 8/20/2013, order value in ticks by queue position (0 → 8×10⁴ shares ahead): 0.41/0.27 → 0.23/0.12 → 0.19/0.09 → 0.17/0.07 → 0.16/0.06 → 0.15/0.05 → 0.14/0.04 → 0.13/0.04 → 0.12/0.03 — [Moallemi & Yuan (2016), Figure 3(a)](https://moallemi.com/ciamac/papers/queue-value-2016.pdf). The same figure's fill probability falls 0.78/0.71 (front) to 0.605/0.575 (8×10⁴ shares back), Figure 3(b); adverse selection *rises* 0.42/0.28 ticks → 0.745/0.595 ticks, Figure 3(c) — same source. Cross-sectional Table 3 (model vs simulation): BAC order value 0.14 / 0.14 ticks, fill prob 0.62, AS 0.57 ticks, "touch value" 0.36 ticks; CSCO 0.08, GE 0.08, F 0.13, INTC 0.11, PFE 0.12, PBR −0.03, EMM 0.07, EFA 0.03 — same source. Calibration used δ₀ = half-spread + 0.3-tick NASDAQ rebate = 0.8 ticks — same source.

**Independent corroboration of the magnitude.** Guo, Ruan & Zhu summarise: "Recent empirical studies by Moallemi and Yuan show that values of order positions (if appropriately defined) have the same order of magnitude of a half spread" — [Guo, Ruan & Zhu, *Dynamics of Order Positions and Related Queues in a Limit Order Book*, arXiv:1505.04810](https://arxiv.org/html/1505.04810v2).

**Queue position as a filter is also a stated industry/regulatory concern.** Moallemi & Yuan cite the WSJ controversy over "exotic order types" that "allow high-speed trading firms to trade ahead of less-sophisticated investors" as evidence that priority itself is priced — [Moallemi & Yuan (2016), §1](https://moallemi.com/ciamac/papers/queue-value-2016.pdf).

**Theory: end-of-queue orders lose money.** In an equilibrium model where limit orders submitted near-simultaneously are randomly sequenced, "front-of-queue limit orders expect higher profit than end-of-queue orders, which, in fact, lose money", so equilibrium depth "always overshoots" the zero-marginal-profit level; with two symmetric makers the overshoot factor is `ŷ = ȳ/(1−(1−β)β)` and at maximum queuing uncertainty `ŷ = (4/3)ȳ` — [Yueshen, *Queuing Uncertainty of Limit Orders*, Management Science 72(6), 2025/26](https://pubsonline.informs.org/doi/10.1287/mnsc.2023.03371).

### Inferences

- **Discrete map to our setting.** Take q = lots ahead of us at the touch (0–900), δ = 1 tick (one-tick spread, no rebate), and AS in ticks. Then `V(q) = α(q)·1 − β(q)` ticks per lot, and the *decision-relevant* number is `V(0)−V(q_ours)`: what we give up by joining at the back instead of being first. On the BAC numbers, going from q=0 to q≈10⁴ shares costs ≈0.18 ticks/order on 8/9 — i.e. **roughly 20% of the spread per order**, and the next 10⁴ shares cost only ~0.04 ticks. The value curve is strongly convex at the front.
- **Because decay is exponential with length `1/b ≈ E|u|/2`, the marginal value of one position scales like `b·V_front·e^{−bq}`** — it shrinks by ~63% every `E|u|/2` lots of queue ahead. With a 900-lot touch this means the *whole* economically interesting part of the queue is the first few hundred lots; beyond that, position is nearly worthless and the order is effectively an option on a price jump.
- **Adverse selection partially offsets position value.** Since β(q) is increasing, the *net* marginal value of moving up is larger than the fill-probability gain alone. In our simulator, where we can measure our own markout, β′(q) can be estimated directly rather than modelled.
- **Warning about the "constant fill probability" shortcut.** Guo, Ruan & Zhu explicitly note that in Cont–de Larrard and in Guo–de Larrard–Ruan the fill probability is "assumed to be a constant", and only Cont–Stoikov–Talreja / Hult–Kiessling compute it — [arXiv:1505.04810](https://arxiv.org/html/1505.04810v2). Any placement model imported from that literature therefore *cannot* tell you whether to post at the touch or step back; it is blind to the variable we care about.

### Gaps

- Moallemi & Yuan assume the queue position is **continuous** and the order **infinitesimal**, and that "over the time scale of the model the bid and ask prices do not change" with spread exactly one tick — [as stated in §2 of the paper](https://moallemi.com/ciamac/papers/queue-value-2016.pdf). Our book is discrete (integer lots) and prices move; the closed form is a good local approximation only.
- I found no source that reports the *value of one queue position* in a ~900-lot, one-tick, 5 ms-latency book. The 0.03–0.41-tick numbers are US large-tick equities with 10⁴–10⁵ share queues.
- No verified source gives a closed-form marginal value when the marginal order is *large* relative to the queue (our order can be 100 lots against 900).

---

## 2. Fill probability estimation: classical, survival, and ML approaches; what functional form fits

### Takeaway

Empirically the dominant drivers of a top-of-book fill probability are **liquidity ahead (queue position), the opposite-side queue size, and imbalance** — a 3-parameter linear model reaches R² ≈ 0.95 on a signal-bias-free sample. The classic survival literature fits an **accelerated-failure-time / generalized-gamma** model in which execution time is extremely sensitive to limit price (distance to mid) and to depth ahead, and shows that first-passage-time (GBM) models fit badly. Reported one-tick-spread fill probabilities range from **<2% to ~50%** depending only on whether orders are allowed to be cancelled — a factor-25 difference that is a measurement artefact, not a market fact.

### Cited Findings

**Signal bias is the first-order issue.** "Given that >>95% of submitted maker orders are cancelled", historical estimates of fill probability are "implicitly conditioned on the trader's decision *not* to cancel"; Lokin & Yu report "a fill probability of less than 2% for orders placed by other market participants when the spread is one tick wide and bid and ask queues are balanced, compared to the 50% fill probability we find in the same situation" — [Albers, Cucuringu, Howison & Shestopaloff, *The Market Maker's Dilemma*, arXiv:2502.18625](https://arxiv.org/html/2502.18625v2). Their experiment: 232,897 minimum-sized maker orders posted live on Binance BTC-perp over one week, 127,051 filled, 105,846 cancelled unfilled (54.6% fill rate), with cancellation only when the price moved away — same source.

**Fill probability surface and its regression.** Fill probability "decreases with the near-side queue size Q^near and increases with the opposite-side queue size Q^opp"; it ranges from **as low as 30%** (large near queue, small opposite queue) to **above 90%** (near queue very small, opposite queue very large) — [arXiv:2502.18625, §4](https://arxiv.org/html/2502.18625v2). OLS on normalised queue sizes and imbalance `imb = (Q^near−Q^opp)/(Q^near+Q^opp)`:
`z = 0.5649 + 0.0159·Q^near + 0.1013·Q^opp − 0.3166·imb`, **R² = 0.946**, with the authors naming near-side queue size and imbalance as the dominant drivers (p = 0.000) — same source. (Caveat in Gaps: the reported β₁ sign is small and positive, which sits awkwardly with the stated monotonicity; the imbalance term dominates.)

**Fill probability at intermediate times is an upper bound.** "A basic 'zero-intelligence' approach, ignoring LB_t (liquidity behind)… might suggest that the fill probability of o at time t is the same as that of another order o′ at its initial time if they have the same liquidity ahead… This estimate is thus an upper bound on the order's fill probability at the intermediate time", because a build-up of liquidity behind reduces near-side taker arrivals and cancellations and raises opposite-side cancellation — [arXiv:2502.18625, §4](https://arxiv.org/html/2502.18625v2).

**Larger orders.** The probability that an order of size q fills fully, conditional on liquidity-ahead LA, equals the fill probability of a minimum-size order with liquidity-ahead `LA + q` — [arXiv:2502.18625, §4](https://arxiv.org/html/2502.18625v2). This is the single most useful practical trick in the paper: no extra estimation is needed to price size.

**Classical survival approach.** Lo, MacKinlay & Zhang model limit-order execution times with survival analysis on limit orders for the 100 largest S&P 500 stocks (Aug 1994–Aug 1995, electronic brokerage): "a generalized gamma model with an accelerated failure time specification fits the data quite well"; execution times are "very sensitive to some explanatory variables, e.g., limit price, but not to others"; partial fills "do not change the time priority"; and "hypothetical limit-order execution" models "constructed theoretically from the first-passage times of geometric Brownian motion… perform very poorly when confronted with actual limit-order data" — [Lo, MacKinlay & Zhang, *Econometric models of limit-order executions*, JFE 2002](https://web.mit.edu/Alo/www/Papers/JFE2002_Pub.pdf); NBER working-paper version [w6257](https://www.nber.org/system/files/working_papers/w6257/w6257.pdf); journal record [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0304405X02001344). Their regressors are explicitly queue-aware: `MKD1 = (1 + P_b − P_l) × log S_b` measures "the minimum number of shares that have higher priority for execution scaled by the distance the limit-buy price is below the bid price" — same source.

**State-dependent queueing (semi-analytic).** Lokin & Yu derive "semi-analytical expressions for… the probability of a mid-price change, the fill probabilities of orders placed at the best quotes, and those of orders placed deeper in the book before the opposite best quote moves", modelling the book as interacting queues with state-dependent intensities, validated on FX spot; they find deeper-level fill probabilities "typically negligible" — [Lokin & Yu, arXiv:2403.02572](https://arxiv.org/abs/2403.02572).

**ML approaches.** A deep-learning model predicts fill probabilities and conditional expected fill times for synthetic top-of-book orders inserted into NASDAQ data and "outperforms benchmarks" — [*A Deep Learning Approach to Estimating Fill Probabilities in a Limit Order Book* (Columbia working paper)](https://business.columbia.edu/sites/default/files-efs/citation_file_upload/deep-lob-2021.pdf). Albers et al. counter that the simple 3-variable model "call[s] into question the need for black-box machinery lacking interpretability and ease of replicability to solve what turns out to be a relatively simple problem" — [arXiv:2502.18625](https://arxiv.org/html/2502.18625v2).

### Inferences (estimators, all incremental)

1. **Queue-based hazard estimator (recommended primary).** For a resting order with A_t lots ahead and a taker arrival process whose size distribution has survival `S̄(v)`, the instantaneous fill hazard is `λ_fill(t) ≈ ν_t · S̄(A_t)`, where ν_t is the taker arrival rate on our side. Under exponential trade sizes `S̄(v)=e^{−v/ū}`, so `λ_fill ≈ ν_t e^{−A_t/ū}` and `P(fill within h) ≈ 1 − exp(−h ν_t e^{−A_t/ū})`. Maintain: EWMA of `ū` (mean taker size on our side) and EWMA of `ν` (taker events per second) — both O(1) per event. This is exactly the Moallemi–Yuan α(q) shape with `b = 1/ū`, and matches the empirical monotonicity in Q^near.
2. **Discrete-time conditional fill table (robust, non-parametric).** Bucket by (ticks from mid at submission, A_t quartile) and maintain two counters per bucket: `n_fill`, `n_terminal` (fill + cancel + price-move-away). Then `p̂ = n_fill/n_terminal` is a proper conditional fill probability, updatable in O(1) and immune to the survival-model specification. With, say, 3 price levels × 4 queue buckets = 12 buckets, this converges within minutes of simulated time.
3. **Add the opposite side.** Fill probability is strongly increasing in Q^opp (0.1013 per normalised unit) and decreasing in imbalance (−0.3166). A multiplicative model `p̂ = clip(β₀ + β₁ Q^near + β₂ Q^opp + β₃ imb)` with β from an online RLS/EWMA regression is cheap and captures the documented surface.
4. **Size adjustment for free:** to price an order of size q, evaluate the same estimator at `A_t + q`. This is exactly the published equivalence and needs no new parameters.
5. **Do not use first-passage/GBM fill models.** The published verdict is that they "perform very poorly"; in a one-tick-spread regime the price cannot diffuse continuously, so they are doubly inappropriate.

### Gaps

- **No verified source gives an explicit distance-to-mid decay law with fitted parameters for both large-tick and small-tick stocks.** The exponential-in-distance form is standard in the market-making literature (see §4) but I could not verify a paper that tabulates the decay constant separately for large-tick vs small-tick names. Flagged as a genuine gap.
- The 2%-vs-50% contrast is a *cancellation-policy* contrast; there is no published estimate for a regime with 5 ms latency and forced queueing behind 900 lots.
- The Albers et al. regression's β₁ sign (small positive coefficient on near-side queue) is inconsistent with the paper's own prose; I report it verbatim and do not rely on it. The surface figure and the imbalance coefficient are the load-bearing evidence.

---

## 3. Queue-reactive and zero-intelligence models: which give closed forms a strategy can use?

### Takeaway

Only **queue-reactive** models give state-dependent (hence usable) fill probabilities, and they do so numerically, not in closed form: the book is a finite-state Markov chain on (bid queue, ask queue) whose stationary distribution and absorption probabilities are computable but not elementary. Zero-intelligence models and constant-fill-probability "queueing" market-making models are *not* usable for placement decisions because they omit the very state dependence that matters.

### Cited Findings

**Queue-reactive model.** Huang, Lehalle & Rosenbaum split time into periods where the reference (mid) price is constant, and within each period treat the LOB as "a Markov queuing system… the intensities of the order flows only depend on the current state of the order book"; they "establish the limiting behavior of this model and estimate its parameters from market data", then add a stochastic mechanism to switch between constant-price periods; the framework is proposed as "a market simulator or… a tool for the transaction cost analysis of complex trading algorithms" — [Huang, Lehalle & Rosenbaum, *Simulating and analyzing order book data: the queue-reactive model*, arXiv:1312.0563](https://arxiv.org/abs/1312.0563), JASA 110(509):107–122, 2015, [journal record](https://www.tandfonline.com/doi/abs/10.1080/01621459.2014.982278). Because intensities are state-dependent, the model reproduces the empirical fact that a long queue depresses further limit-order arrivals and cancellations in front of it — which is the mechanism the Albers et al. "upper bound" argument also describes ([arXiv:2502.18625](https://arxiv.org/html/2502.18625v2)).

**Queue dynamics with explicit execution-time formulas.** Guo, Ruan & Zhu derive a fluid limit in which "the rate of the order position approaching zero is proportional to the mean of order arrival intensities and to the summation of the average size of market orders and the 'modified' average size of cancellation orders in the queue", plus explicit expressions for the average time to execution, the average queue-depletion time, first-hitting-time distributions, and Gaussian fluctuations of order position around its fluid limit with mean reversion whose "speed… is proportional to the order arrival intensity and the rate of cancellations" — [arXiv:1505.04810](https://arxiv.org/html/1505.04810v2). They also explicitly note that most of the algorithmic-trading literature "ignored order positions", with fill probability "either assumed to be a constant" or computed numerically from a full-book Markov chain — same source.

**Zero-intelligence baseline.** Albers et al. label the ignore-liquidity-behind approximation a "zero-intelligence" approach and show it is an upper bound — [arXiv:2502.18625](https://arxiv.org/html/2502.18625v2).

**Price-impact side (usable and linear).** Prices over short intervals are driven by order-flow imbalance with a *linear* relation whose slope is inversely proportional to depth: `ΔP_k = β·OFI_k + ε_k`, average **R² ≈ 65%** (falling to 35–60% when price-changing events are excluded from OFI), with `β ≈ 1/(2D)` and estimated `λ ≈ 1` in `β = c/D^λ` — [Cont, Kukanov & Stoikov, *The price impact of order book events*, arXiv:1011.6402](https://arxiv.org/html/1011.6402v3). Their OFI is a cumulative sum of signed queue-size changes across limit orders, market orders and cancellations at the touch — same source. This is the cheapest available *predictor of the next mid move*, which is the variable that determines both our fill and our markout.

### Inferences

- **What is actually implementable at 5 ms / ~10⁴ events per hour:** not the queue-reactive Markov chain (state space = full book), but its *local* content — the state-dependent intensity idea — in the form of the λ_fill ≈ ν·S̄(A) estimator in §2. The queue-reactive literature is best read as justification that (i) a stationary distribution of queue length exists and (ii) intensities must be conditioned on queue size, which is exactly what the two-counter table does.
- **The fluid-limit result is directly quotable as a strategy prior:** expected time to be filled ≈ (lots ahead) / (market-order volume rate + front-of-queue cancellation rate). In our book, with ~900 lots and a taker flow of a few lots per second, the naive fluid answer is *minutes*, which tells us that at the touch we are buying a lottery ticket on a price move, not a fill.
- **OFI as the fill/adverse-selection covariate.** Because `ΔP ≈ OFI/(2D)`, a large positive OFI with no price change "suggests that prices are about to go up" ([arXiv:1011.6402](https://arxiv.org/html/1011.6402v3)). In a one-tick book, D ≈ 900 lots, so one lot of OFI moves the expected mid by ≈1/(2·900) ticks: small per event, but cumulative and cheap to maintain (a single running sum over a fixed 100–200 ms window).
- **Cont–Bouchaud caveat:** the well-known Cont & Bouchaud paper is a herding model of *aggregate returns*, not a limit-order-queue model; it cannot supply fill probabilities. (See Gaps.)

### Gaps

- I did not verify a specific citation for Smith, Farmer, Gillemot & Krishnamurthy's zero-intelligence continuous-double-auction model, nor for Cont & Bouchaud (2000). The brief's "Cont-Bouchaud" almost certainly refers to the herding/returns paper, which is not a queue model — this should not be cited as a fill-probability source.
- Huang–Lehalle–Rosenbaum do not provide a closed-form fill probability; anyone quoting a "closed form" from that paper is over-reading it. The tractable closed form in this literature is Moallemi–Yuan's (exponential trade sizes, no cancellations).

---

## 4. Order placement optimisation: posting at the touch vs stepping back

### Takeaway

The market-making literature's answer is a **fill-intensity model `λ(δ) = A e^{−κδ}` in distance-to-mid δ plus an inventory/risk term**, which produces a spread floor of the form `(2/γ)·ln(1 + γ/κ)`: the optimal quote is at a finite distance from mid, not at the touch, whenever the adverse-selection/risk term exceeds the queue-position value. All of these models are **queue-blind** (constant fill intensity given distance), so in a 900-lot one-tick book the queue-position term from §1 must be added by hand — and it is large enough to flip the sign of the decision.

### Cited Findings

**GLFT.** Guéant, Lehalle & Fernandez-Tapia solve the market-making problem with inventory risk and a fill intensity that decays with distance from the mid; the solution is a closed-form (or asymptotically closed-form) pair of quotes with an inventory skew — [Guéant, Lehalle & Fernandez-Tapia, *Dealing with the Inventory Risk*, arXiv:1105.3115](https://arxiv.org/abs/1105.3115). The adverse-selection/risk component of the optimal spread is `(2/γ)ln(1 + γ/κ)`, which acts as "the minimum spread even when there is no inventory risk" — [Samani, *Markouts: Measuring Adverse Selection*](https://uditsamani.com/markouts/) (secondary exposition that also derives the same term from a markout model).

**Calibrated decay constants (secondary source, BTC-perp).** Fitting `log λ = log A − κ·(ticks from mid)` on a crypto LOB gives `A = 0.843, κ = 0.0170` over the full range and `A = 2.99, κ = 0.0424` when restricted to ≤70 ticks, i.e. an e-folding distance of ~24 ticks in the shallow region — [hftbacktest, *GLFT Market Making Model and Grid Trading* tutorial](https://hftbacktest.readthedocs.io/en/py-v2.1.0/tutorials/GLFT%20Market%20Making%20Model%20and%20Grid%20Trading.html). Treat as an order-of-magnitude calibration, not a peer-reviewed estimate.

**Queue-blindness is acknowledged.** In Cont–de Larrard and Guo–de Larrard–Ruan the fill probability is "assumed to be a constant" — [Guo, Ruan & Zhu, arXiv:1505.04810](https://arxiv.org/html/1505.04810v2). Albers et al. make the same point more bluntly about Avellaneda–Stoikov and Cartea-type models: they assume "maker orders fill randomly at Poisson rates, with no impact on the subsequent mid-price evolution", and conclude that "such market making strategies are a quick path to bankruptcy due to negative price drift" — with the naive two-sided minimum-size strategy losing "almost 60%" over ~3.3 days, Sharpe **−109**, average holding time 11 s — [arXiv:2502.18625](https://arxiv.org/html/2502.18625v2).

**Empirical guidance on where to post.** (i) Posting in the direction of imbalance means posting at the *back* of the longer queue — the worst combination of low fill probability and bad markout. (ii) A high fill probability requires posting at the *front of the short queue*, i.e. counter to the imbalance, which "entails trading against the expected return". (iii) The only configuration with both decent fill odds and positive markout is **front-of-queue in a large near-side queue with a small opposite-side queue** (average markout −0.058 bp) — [arXiv:2502.18625](https://arxiv.org/html/2502.18625v2). (iv) To get there one must "post… when the queue is still small but becomes large shortly thereafter", which usually means posting when imbalance is *unfavourable* — same source.

### Inferences (decision rule for our simulator)

Let q = lots ahead, δ = 1 tick, and define per-lot expected profit at the touch as
`π_touch(q) = α(q)·δ − β(q)`, `π_back(k) = α(∞)·δ − β(∞)` for a level k ticks away (where α(∞)=P(price jump) is the only fill route at a step-back level).
- **Post at the touch iff `α(q)·(δ − AS(q))` exceeds the best step-back alternative plus the option value of the position.** With one-tick spread, δ−AS is often ≈ 0.4–0.5 ticks at the front and *negative* at the back (BAC AS 0.57–0.85 ticks vs premium 0.8) — i.e. **at the back of a 900-lot one-tick queue the expected value of the position can be negative**, which is the theoretical version of "queue position is a filter".
- **Queue-length threshold.** Since α decays like `e^{−q/ū}`, the touch is worth it only while `q ≲ ū·ln(A/κ...)`; in practice, compute `π_touch(q)` from the two fitted EWMAs (ū, ν) and compare against the step-back value. Because the book is one tick wide, stepping back is not "one tick worse" — it is a *different asset* (a fill only on a price jump), so the comparison must use the jump-rate p⁺_J, not a continuous δ.
- **Multi-level ladder.** In a one-tick-spread, 900-lot book the second level is one tick away but fills only when the touch is consumed; the literature gives no queue-aware multi-level optimiser that I could verify. The practical ladder is: full size at the touch when q is small; reduce or step back when q/ū is large; never post at the back of a long queue hoping for a favourable markout, because the data say back-of-queue fills have the *worst* markouts (−0.775 to −1.157 bp) — [arXiv:2502.18625, Table 1](https://arxiv.org/html/2502.18625v2).
- **GLFT is not directly usable here** because with a one-tick spread the "optimal distance" is degenerate: δ ∈ {0, 1, 2, …} ticks and the model's continuous δ collapses to "at the touch or not". Use GLFT only for its qualitative content (spread floor `(2/γ)ln(1+γ/κ)`, inventory skew) and for the `A, κ` calibration procedure.

### Gaps

- No verified paper optimises placement with *both* queue position and a discrete one-tick grid. This is the central theoretical gap for our regime.
- I verified the GLFT paper and its structure but not the exact closed-form quote formula; only the `(2/γ)ln(1+γ/κ)` spread-floor term is corroborated by a second source.
- The `A, κ` numbers come from a library tutorial, not a journal; their transferability to our tape is unverified.

---

## 5. Cancellation economics, order lifetime, and queue hysteresis

### Takeaway

Order lifetimes are extremely short and cancellations are the dominant terminal state (≈93% of limit orders cancelled on INET; roughly one-third cancelled within 2 s). The cancellation hazard is **decreasing in expected profitability and specifically sensitive to queue position**: front-of-queue depth and back-of-queue depth enter with materially different coefficients, which is direct evidence that traders price their position. The cost of cancel-and-repost is exactly the queue value `V(q)` from §1.

### Cited Findings

**Lifetime and cancellation rates.** Hasbrouck & Saar document that on NASDAQ's INET book "93% of limit orders submitted through NASDAQ INET are canceled" and that fleeting orders (canceled within ~2 s) are a large share; in the earlier Island ECN sample "over one quarter of the limit orders… are canceled (unexecuted) within two seconds or less", and "the extensive use of these 'fleeting' orders" is interpreted as a dynamic strategy of chasing queue/price — [Hasbrouck & Saar, *Limit Orders and Volatility in a Hybrid Market: The Island ECN*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=310940); [Hasbrouck & Saar, *Technology and Liquidity Provision: The Blurring of Traditional Definitions*](https://pages.stern.nyu.edu/jhasbrou/Research/Working%20Papers/HS6.pdf); [Hasbrouck & Saar, *Low-latency trading*](https://pages.stern.nyu.edu/~jhasbrou/Research/lowLatencyTrading/lowLatencyTradingHasbrouckSaarJFM.pdf).

**Queue position drives the cancellation hazard.** Using order-level data with explicit front-of-queue and back-of-queue depth decompositions, "the hazard rate of cancellation is decreasing with proxies of the expected limit order profitability"; "if the time priority of an order had no value, we should expect equal coefficients on changes in Back-of-Queue Depth and Front-of-Queue Depth", but the front-of-queue coefficient "is substantially less negative"; and quantitatively "a one-standard-deviation increase in [back-of-queue depth] prolongs the expected limit order lifetime by about nine seconds, or 35% of the average time to order cancellation", while the corresponding statistic for front-of-queue depth "is merely 6%" — [Dahlström, *The determinants of limit order cancellations*, Financial Review 2024](https://onlinelibrary.wiley.com/doi/full/10.1111/fire.12363). The paper concludes cancellations are "primarily due to market making… rather than to liquidity-demanding strategies" and that queue position is "the most influential" variable — same source.

**Fleeting liquidity as equilibrium behaviour, not manipulation.** "Joining Baruch and Glosten (2013), this paper argues that at least some of such fleeting, short-lived liquidity is a natural equilibrium manifestation of liquidity suppliers' competition"; the model predicts cancellations are *negatively* correlated with order-flow toxicity, that "larger orders tend to be faster and more likely to concentrate in the front of the queue", and that depth "initially overshoots but then quickly reverts" — [Yueshen, Management Science](https://pubsonline.informs.org/doi/10.1287/mnsc.2023.03371).

**Cancel is not free, but with 5 ms latency it is nearly free in queue terms.** Yueshen's mechanism is that a supplier who learns ex post that they are at the end of the queue will "(partially) cancel it"; with resolution of queuing uncertainty the fastest supplier effectively takes the front — same source.

### Inferences (cancellation rule)

- **Cost of cancel = V(q_cancelled) − V(q_new).** Cancelling and reposting at the back destroys `V(q) − V(0_future)`; reposting also costs 5 ms of exposure during which other orders may join. Using the exponential form, `lost value ≈ [α(q) − α(0)]δ − [β(q) − β(0)]`, which vanishes when q ≫ 1/b. So: **cancel freely when deep in the queue, and almost never cancel a front-of-queue order.** That is the precise version of "queue hysteresis".
- **Hazard-ratio rule.** Maintain two counters for our own resting orders: fills vs cancels, bucketed by (A_t/ū, Q^opp). Cancel when the conditional fill probability in our bucket falls below the level at which holding the position is better than paying the spread to re-enter later. Dahlström's evidence (hazard falls when back-of-queue depth rises, i.e. when others queue behind us) supports *holding* when liquidity behind us grows — because it improves our markout, even though it worsens our fill odds. Those two effects have opposite signs and both are economically significant.
- **Do not cancel on a fixed timer.** The empirical hazard is driven by state (queue position and depth), not by elapsed time; a time-based cancel rule throws away the position value documented in §1.

### Gaps

- No source quantifies the cost of cancel-and-repost in *ticks of queue value* for a specific book; the inference above is mine, built on Moallemi–Yuan's V(q).
- I did not find a paper measuring queue hysteresis (path dependence in cancel decisions) directly; Dahlström's hazard coefficients are the closest evidence.
- The Hasbrouck–Saar "one third cancelled within 2 s" figure is attributed differently across secondary sources (some say one third, some say 36.69% of *fleeting* orders are priced ahead of the BBO); the primary PDFs should be checked before quoting a precise number.

---

## 6. Adverse selection of fills: markout methodology and magnitudes

### Takeaway

Markout (post-fill mid drift, signed by side) is the standard measure, and the reported magnitudes for passive fills at 1–5 s are **−0.3 to −1.2 bp**, systematically *worse* for back-of-queue fills and worse when the near-side queue is large relative to the opposite side. At 10 ms–1 s horizons the effect is dominated mechanically by the fill itself (the taker order that filled you moved the mid against you), so the sign at short horizons is essentially predetermined.

### Cited Findings

**Mechanical sign at the shortest horizon.** "If the snapshot just after an order's terminal time (fill or cancel) shows a mid-price move that is zero or against the order, the order must have filled; if… a favorable mid-price movement the order must have been cancelled. That is, the fill probability of a maker order, conditioned on the immediate mid-price change, is 0 when that change is positive, and 1 when it is zero or negative" — [arXiv:2502.18625, §6](https://arxiv.org/html/2502.18625v2). This is the discrete version of "you only get filled when the price is about to move against you" and is exact in a one-tick book.

**Magnitudes at 1 s, by queue position and queue sizes** (markout in bp, mid 1 s after fill, excluding rebate) — [arXiv:2502.18625, Table 1](https://arxiv.org/html/2502.18625v2):

| Near / opposite queue | QP 0–0.1 | 0.1–0.4 | 0.4–0.75 | 0.75–1 |
|---|---|---|---|---|
| large near, small opp. | −0.058 | −0.586 | −0.743 | −0.775 |
| large near, large opp. | −0.296 | −0.882 | −0.967 | −1.157 |
| small near, small opp. | −0.562 | −0.711 | −0.622 | −0.677 |
| small near, large opp. | −0.539 | −0.645 | −0.686 | −0.763 |

with standard deviations 0.8–2.2 bp. Note the interaction: **when the near-side queue is small, queue position "does not matter much"**, because front and back are nearly the same — same source.

**Fill probability vs. return is a strict trade-off.** Plotting the fill rate against the 5-s post-terminal return shows "orders with negative subsequent five-second returns are highly likely to fill, whereas those with positive returns are much less likely to fill"; the unconditional expected return of the maker orders is "approximately −0.8 bp, or −0.3 bp net of rebate" — [arXiv:2502.18625, §6](https://arxiv.org/html/2502.18625v2). Strategy-level: naive two-sided maker −0.43 bp/trade (12,623 trades, 22.6 s holding), imbalance-maker without cancel −0.47 bp, with cancel −0.49 bp, imbalance-taker −1.96 bp — [Table 2, same source](https://arxiv.org/html/2502.18625v2).

**Markout as a methodology.** Markout is defined as the signed difference between the post-trade mid and the trade price over a horizon grid, and "a robust analysis normally reports a markout curve across several horizons rather than selecting a single point that would favour a particular conclusion" — [KX glossary](https://kx.com/glossary/what-is-markout-and-how-is-it-calculated/); implementations typically compute `avg(((bid+ask)/2 − price)/price·1e4)` per horizon step for passive vs aggressive fills — [QuestDB markout cookbook](https://questdb.com/docs/cookbook/sql/finance/markout/); Databento notes liquidity providers "use markout curves to determine the toxicity and adverse selection from price takers" — [Databento](https://databento.com/microstructure/markout).

**Flow-toxicity measures.** VPIN (volume-synchronised probability of informed trading) was proposed as a real-time toxicity metric correlated with future short-term volatility and reaching highs before the flash crash — summarised in [Andersen & Bondarenko, *VPIN and the flash crash*, JFM 17:1–46, 2014](https://ideas.repec.org/a/eee/finmar/v17y2014icp1-46.html). The same paper's own empirical finding is that "VPIN… is a poor predictor of short run volatility, that it did not reach an all-time high prior [to], but rather after, the flash crash, and that its predictive content is due primarily to a mechanical relation with the underlying trading intensity" — [Kellogg summary of Andersen & Bondarenko](https://www.kellogg.northwestern.edu/academics-research/research/detail/2014/vpin-and-the-flash-crash/). Easley, López de Prado & O'Hara replied in a rejoinder — [JFM 17:47–52](https://ideas.repec.org/a/eee/finmar/v17y2014icp47-52.html). For our purposes the honest reading is: **signed order flow carries information, but VPIN specifically is contested**; a plain signed-flow / OFI EWMA is the better-supported choice, and OFI has R² ≈ 65% for contemporaneous price change — [Cont, Kukanov & Stoikov, arXiv:1011.6402](https://arxiv.org/html/1011.6402v3).

### Inferences (markout estimator)

- Maintain per-fill a small record `(side, p_fill, mid_0, t_fill, q_at_fill, A_at_fill)` and update, at each of a fixed horizon grid h ∈ {0.1, 0.5, 1, 5, 30 s}, an EWMA of `markout_h = side·(mid_{t+h} − p_fill)/tick`. Cost: one ring buffer of length = number of fills within 30 s (tens), so O(1) amortised per event.
- **Sign convention for our simulator:** because a passive fill in a one-tick book mechanically moves the mid by up to half a tick against us, the 0–10 ms markout is ≈ −0.5 tick by construction; it is information-free. Use 1 s and 5 s horizons for decisions, exactly as the empirical table does.
- **Condition on queue position at fill, not at post.** The published result is that markout depends on QP at the moment of execution (the paper had to reconstruct it from Binance's per-fill sequence IDs) — [arXiv:2502.18625, §5](https://arxiv.org/html/2502.18625v2). In a simulator we get this for free from our own position counter, so we should record `A_at_fill` and bucket the markout by it.
- **The trade-off is the objective function.** Since fill probability and markout are negatively correlated, optimising either alone is a documented way to lose money (−0.43 bp/trade for the naive strategy). The implementable objective is `E[Δpnl per post] = p̂_fill·(δ + markout_1s)`, both terms estimated online in the same bucket.

### Gaps

- No verified source reports markout magnitudes at the 10–100 ms band separately from 1 s; the shortest-horizon claim above (mechanical sign) is derived from the paper's own logic, not from a published table.
- The −0.3 to −1.2 bp numbers are for BTC-perp minimum-size orders on Binance with a −0.5 bp maker rebate and a ~0.03 bp tick; the *economics* (spread vs. adverse selection) do not transfer directly to a 1-tick-spread equity-like tape where the half-spread is far larger.
- I found no source that measures how markout varies with queue position *within* a 900-lot one-tick queue; the published bucketing is by decile of queue position, which is the closest available proxy.

---

## 7. Large-tick vs small-tick regimes, and latency × queue

### Takeaway

In a large-tick regime the spread is pinned at one tick, price competition is impossible, liquidity concentrates at the touch, and the *only* competitive margin is queue position; widening the tick demonstrably lengthens queues and raises liquidity-provider profits. Latency matters because it randomises your queue position among everyone who acts in the same window: with 5 ms latency and a busy tape, the position you get is a draw from the orders submitted within ~5 ms, not the position you computed.

### Cited Findings

**Definition and mechanics of the large-tick regime.** Large-tick assets are those where "the bid-ask spread rarely exceeds the minimum tick size" (citing Eisler et al. 2012), they "tend to have large queues in the limit order book", and "instead of competing through price, market participants tend to form queues"; conversely "for small-tick assets… investors can obtain priority in the order book by competing on price and the importance of queueing (and hence the importance of valuing queue position) is diminished" — [Moallemi & Yuan (2016), §2](https://moallemi.com/ciamac/papers/queue-value-2016.pdf). Their Figure 2 (courtesy Robert Almgren, futures, Jul–Aug 2013) plots "% time bid-ask spread > 1 tick" against "displayed liquidity / average trade size", showing queueing dominates exactly where the spread is pinned — same source.

**Tick-size pilot evidence.** The US Tick Size Pilot widened the minimum increment from $0.01 to $0.05; a central premise was that wider ticks "should create longer queue lines at the best [bid or offer]", and "to the extent wider tick sizes result in longer queue lines, the migration of order flow away [from lit venues follows]" — [McCrary and co-authors, *Evidence from the Tick Size Pilot Study*](https://eml.berkeley.edu/~jmccrary/FINAL_SSRN.pdf). The same study finds "market maker profits increased by approximately 40% in transactions involving securities that have been treated with nickel tick sizes" — same source. The SEC's own assessment covers liquidity, trading and market quality under the pilot — [SEC, *Assessment of the Plan to Implement a Tick Size Pilot Program*](https://www.sec.gov/files/TICK%20PILOT%20ASSESSMENT%20FINAL%20Aug%202.pdf).

**Queuing uncertainty is the formal statement of latency × queue.** "Limit orders submitted around the same time are subject to random latencies and will be queued accordingly"; "at the time of order submission, there is no telling of the exact queue positions, only queuing uncertainty"; "the profitability of the same limit order will be high (low, even negative) if it is the first (last) in queue" — [Yueshen, Management Science](https://pubsonline.informs.org/doi/10.1287/mnsc.2023.03371). Equilibrium consequences: end-of-queue orders lose money, depth overshoots, and a faster supplier "will play more aggressively (i.e. submitting larger orders)" — same source. The paper also notes that speed bumps / latency levelling "essentially increase limit orders' queuing uncertainty, intensify competition, and consequently might induce too much inefficiency" — same source.

**The cost of latency, quantified.** Moallemi & Sağlam build "a theoretical model for the quantitative valuation of latency" measuring "the trading frictions created by the presence of [latency]" — [Moallemi & Sağlam, *The Cost of Latency in High-Frequency Trading*, SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1571935); published as [Operations Research (2013)](https://pubsonline.informs.org/doi/10.1287/opre.2013.1165).

**Latency matters more for takers.** Albers et al. explicitly set latency aside for maker orders but cite their own companion work on taker execution, and warn that ignoring latency "potentially lead[s] to an overstatement of results" for taker strategies — [arXiv:2502.18625, §6 fn.14](https://arxiv.org/html/2502.18625v2).

### Inferences (concrete for 5 ms and 900 lots)

- **Expected position penalty from latency.** If our side's touch receives new orders at rate ν_join (lots/s), our realised position is `A_post + ν_join·L` on average with L = 5 ms, and the *variance* is Poisson-like. At ν_join = 200 lots/s, that is ~1 lot — negligible; at ν_join = 20,000 lots/s (a genuine race after a price change), it is ~100 lots, which is a large fraction of a 900-lot queue and destroys most of the front-of-queue value. **So latency is second-order in the steady state and first-order only in the post-price-change race.** This is the single most actionable implication of this section, and it is an inference, not a published number.
- **The "cancel arrives after a fill" case.** With 5 ms on cancels, any cancel triggered by an observation is exposed to 5 ms of fills. The relevant guard is the fill hazard during that window, `1 − exp(−0.005·λ_fill)`; with λ_fill ≈ 0.05/s this is 0.025%, with λ_fill ≈ 1/s (front of a fast queue) it is 0.5%. Only the front-of-queue case matters, and there the position is worth holding anyway (§5).
- **Why the one-tick regime favours queue-aware logic over speed.** Widening the tick raises LP profits ~40% and lengthens queues ([McCrary](https://eml.berkeley.edu/~jmccrary/FINAL_SSRN.pdf)); in our book the spread is already at the floor, so there is no price-improvement escape hatch, and no amount of latency improvement buys priority except in the instantaneous post-move race. The binding constraint is *where in the 900 lots we sit*, i.e. §1–§2.
- **Small-tick contrast to flag in the report:** every queue-value result here is conditional on the spread being pinned. If a future round's tape has a multi-tick spread, the model should switch to price-priority logic (step inside by a tick) and the queue value collapses.

### Gaps

- Moallemi & Sağlam's *quantitative* latency cost (in bp, for a given latency) could not be extracted — the SSRN/OR pages were not retrievable in full. The report should either cite it qualitatively or find the working-paper PDF.
- No source measures realised queue position versus intended position at 5 ms in a one-tick market; the `ν_join·L` formula is my construction from Yueshen's queuing-uncertainty premise.
- The tick-pilot literature is about *changing* the tick, not about the value of position within an already-large-tick book; the transfer is directional, not quantitative.

---

## Cross-cutting summary of the load-bearing citations (for the report writer)

| Claim | Source |
|---|---|
| Value = fill prob × (premium − adverse selection); V(q)=α(q)δ−β(q) | [Moallemi & Yuan 2016](https://moallemi.com/ciamac/papers/queue-value-2016.pdf) |
| Queue value ≈ half-spread in large-tick stocks; 0.03–0.41 ticks | [Moallemi & Yuan 2016](https://moallemi.com/ciamac/papers/queue-value-2016.pdf) |
| Exponential fill prob in queue position, length ≈ ½ mean trade size | [Moallemi & Yuan 2016, Thm 3](https://moallemi.com/ciamac/papers/queue-value-2016.pdf) |
| Fill prob surface driven by near queue, opposite queue, imbalance; R²=0.946 | [Albers et al., arXiv:2502.18625](https://arxiv.org/html/2502.18625v2) |
| 2% vs 50% fill prob for one-tick balanced queues = cancellation artefact | [arXiv:2502.18625](https://arxiv.org/html/2502.18625v2); [Lokin & Yu, arXiv:2403.02572](https://arxiv.org/abs/2403.02572) |
| Markout −0.06 to −1.16 bp at 1 s, monotone in queue position | [arXiv:2502.18625, Table 1](https://arxiv.org/html/2502.18625v2) |
| Generalized-gamma AFT survival model; GBM first-passage models fail | [Lo, MacKinlay & Zhang 2002](https://web.mit.edu/Alo/www/Papers/JFE2002_Pub.pdf) |
| Queue-reactive state-dependent intensities, no closed form | [Huang, Lehalle & Rosenbaum, arXiv:1312.0563](https://arxiv.org/abs/1312.0563) |
| Fluid limit: position decays at (market order + cancel) volume rate | [Guo, Ruan & Zhu, arXiv:1505.04810](https://arxiv.org/html/1505.04810v2) |
| Linear impact on OFI, R²≈65%, slope ≈ 1/(2D) | [Cont, Kukanov & Stoikov, arXiv:1011.6402](https://arxiv.org/html/1011.6402v3) |
| GLFT exponential fill intensity + inventory skew; spread floor (2/γ)ln(1+γ/κ) | [Guéant, Lehalle & Fernandez-Tapia, arXiv:1105.3115](https://arxiv.org/abs/1105.3115) |
| 93% of INET limit orders cancelled; ~¼ of Island orders cancelled within 2 s | [Hasbrouck & Saar](https://pages.stern.nyu.edu/jhasbrou/Research/Working%20Papers/HS6.pdf); [Island ECN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=310940) |
| Cancel hazard falls with queue profitability; front vs back depth coefficients differ; +9 s from 1 s.d. back-depth | [Dahlström 2024](https://onlinelibrary.wiley.com/doi/full/10.1111/fire.12363) |
| Queuing uncertainty: end-of-queue loses money, depth overshoots 4/3 ȳ | [Yueshen, Mgmt Sci](https://pubsonline.informs.org/doi/10.1287/mnsc.2023.03371) |
| Wider tick → longer queues, LP profits +40% | [McCrary et al.](https://eml.berkeley.edu/~jmccrary/FINAL_SSRN.pdf); [SEC](https://www.sec.gov/files/TICK%20PILOT%20ASSESSMENT%20FINAL%20Aug%202.pdf) |
| VPIN contested as a toxicity predictor | [Andersen & Bondarenko 2014](https://ideas.repec.org/a/eee/finmar/v17y2014icp1-46.html) |
