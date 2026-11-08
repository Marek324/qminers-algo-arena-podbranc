# Inventory management and market-making theory: stochastic control, optimal quotes, and what survives a discrete-tick book

Scope note: everything below is the *canonical model literature*. Symbols are kept consistent
across sections so formulas can be ported. The running example is the arena setting
(|I| ≤ 100 lots hard cap, $10,000 per second outside it, orders rejected if |I + s(R+q)| > 1000,
5 ms send/cancel latency, integer ticks).

Global symbol table used throughout:

| symbol | meaning |
|---|---|
| `S_t` (or `s`) | reference / mid price (ticks). Arithmetic Brownian motion `dS = μ dt + σ dW` unless stated |
| `σ` | volatility, ticks·s^(-1/2) |
| `q` (or `I`) | signed inventory, lots |
| `Δ` | lot size of one quote (constant), lots |
| `δ^b, δ^a` | quote *distance* from reference: bid `S^b = S − δ^b`, ask `S^a = S + δ^a`, ticks |
| `λ^b(δ), λ^a(δ)` | fill intensities, s^(-1) |
| `A` | intensity scale, s^(-1): `λ(δ) = A e^(−kδ)` |
| `k` | intensity decay, tick^(-1) (larger = fills die faster with distance) |
| `γ` | absolute (CARA) risk aversion, 1/currency (or 1/tick if P&L measured in ticks) |
| `Q` | hard inventory bound, lots (`|q| ≤ Q`) |
| `ψ` | quoted bid-ask spread = `δ^a + δ^b` |
| `r` | reservation (indifference) price |
| `T`, `τ = T − t` | horizon and time-to-go |
| `φ` | running inventory-penalty coefficient (currency per lot² per second) |
| `ℓ(·)`, `ϑ` | terminal liquidation penalty (convex in `|q_T|`); `ϑ` its quadratic coefficient |

---

## Avellaneda & Stoikov (2008): reservation price, optimal spread, and why the result is fragile

### Takeaway
A&S reduce market making to two separable decisions — a *reservation price* that shifts linearly in
inventory with slope `γσ²(T−t)`, and a *spread* that is (to first order) independent of inventory and
set by risk aversion plus fill-intensity curvature. The closed form is a small-inventory Taylor
expansion of an exact HJB solution that was never proved admissible; it assumes Poisson fills at a
continuous distance, no queue, no latency, and no informed flow.

### Cited Findings
- Model: `dS_t = σ dW_t`; fills are Poisson with intensities `λ^b(δ^b) = A e^(−k δ^b)` and
  `λ^a(δ^a) = A e^(−k δ^a)`; objective is `sup E[−exp(−γ(X_T + q_T S_T))]` over predictable quote
  distances bounded below — [Avellaneda & Stoikov 2008, *Quantitative Finance* 8(3):217–224](https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf), DOI [10.1080/14697680701381228](https://doi.org/10.1080/14697680701381228).
- The two-step structure the paper advertises: "First, the dealer computes a personal indifference
  valuation for the stock, given his current inventory. Second, he calibrates his bid and ask quotes
  to the limit order book, by considering the probability with which his quotes will be executed as a
  function of their distance from the mid-price." — [A&S 2008](https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf).
- Closed-form quotes (as reproduced and re-derived by GLFT, `Δ = 1`):
  `δ^(b*) ≈ (1/γ)·ln(1 + γ/k) + ((1 + 2q)/2)·γσ²(T − t)`
  `δ^(a*) ≈ (1/γ)·ln(1 + γ/k) + ((1 − 2q)/2)·γσ²(T − t)`
  — [Guéant, Lehalle & Fernandez-Tapia, arXiv:1105.3115 §4](https://arxiv.org/abs/1105.3115).
- Implied reservation price and spread. Adding and subtracting the two lines above:
  - **reservation price** `r(s,q,t) = s − q·γσ²(T − t)` (the mid of the two quotes),
  - **quoted spread** `ψ = 2/γ·ln(1 + γ/k) + γσ²(T − t)`, *independent of q*,
  - so `S^b = r − ψ/2`, `S^a = r + ψ/2`. The same `r` appears in the secondary literature as
    `r_b = s − (δ^a + δ^b)/2 − γσ²(T−t)q` — [Quant Atlas summary of A&S](https://rzeng0812.github.io/quant-atlas/70-Papers/Avellaneda-Stoikov-2008-High-Frequency-Trading-Limit-Order-Book).
- A&S's own numerical experiment: they simulate price paths and compare the inventory-based strategy
  to a "naive" symmetric-around-mid strategy and report that the inventory strategy "yields P&L
  profiles and final inventories that have significantly less variance" — [A&S 2008](https://people.orie.cornell.edu/sfs33/LimitOrderBook.pdf).
  Note this is a *variance* claim on simulated paths, not an out-of-sample P&L claim.
- Known mathematical weakness, stated by the authors who fixed it: "in the absence of inventory
  limits, no proof of optimality is available for the quotes claimed to be optimal in [3] and their
  admissibility appears to be an open problem" — [GLFT, arXiv:1105.3115](https://arxiv.org/abs/1105.3115).
  (Admissibility = the candidate optimal quote may not be integrable/feasible for all `q`, `t`.)
- Known modelling weaknesses, stated by Guéant: models à la A&S "can hardly be applied to most stock
  markets for at least two reasons: (i) the discrete nature of prices (especially … large tick size),
  and (ii) the fact that the very nature of the limit order books, which are queuing systems with
  priorities and volumes, is not taken into account" — [Guéant 2017, arXiv:1605.01862 §1](https://arxiv.org/abs/1605.01862).
- The `q`-expansion is only valid near `T`: GLFT show A&S's formulas "are nothing but the Taylor
  expansions of the optimal quotes for `t` close to `T`", and that the true optimal quotes are
  "almost independent of `t` as soon as `t` is far from the terminal time `T`" — [GLFT §4](https://arxiv.org/abs/1105.3115).

### Inferences
- The A&S formula is not a quoting rule for a long session; it is a terminal-liquidation rule. Using
  it with a literal `T − t` makes quotes converge to symmetric-asymptotic as `T − t → 0` (spread
  collapses to `2/γ ln(1+γ/k)`, skew collapses to zero) which is the *opposite* of what a
  terminal-penalty model wants. For a continuously running maker, use the ergodic (GLFT) version
  with a constant skew coefficient (§2) or a rolling `T`.
- Because `ψ` is independent of `q` in this approximation, A&S predict *pure mid skew, constant
  width*. That is precisely the pattern documented empirically by Hendershott & Menkveld (§6), which
  is the strongest argument for implementing skew as a shift of both quotes rather than widening one side.
- Sensitivity: `∂ψ/∂σ² = (T−t) + (2/γ)·∂/∂σ² ln(1+γ/k) > 0` and skew slope `∂r/∂q = −γσ²(T−t)`
  grows linearly in both `γ` and variance. With `σ = 0.3` tick·s^(-1/2), `γ = 0.01` tick^(-1),
  `T−t = 600 s`, the slope is `0.01·0.09·600 = 0.54` ticks per lot — i.e. at `q = 100` the
  reservation price is 54 ticks away from mid. That is far outside any plausible book, which is the
  practical statement that A&S's coefficient is unusable at large `|q|` without the bounded-inventory
  correction.

### Gaps
- I did not obtain a verified statement of the specific `γ` values A&S used in their numerical
  section (commonly quoted as 0.01/0.1/0.5 in 1/$), so no magnitude claim is made for that paper.
- No source found that tests A&S out-of-sample on real LOB data with queueing; all published
  evaluations I found are simulations or the authors' own backtests (§2, §6).

---

## Guéant–Lehalle–Fernandez-Tapia (2013) and Guéant (2016/2017): exact solution, the formula used in practice, and the hard inventory bound

### Takeaway
GLFT solve the A&S control problem *exactly* (including a hard bound `|q| ≤ Q`) by a change of
variables that turns the HJB equation into a tridiagonal linear ODE system; the practical output is
the "GLFT formula", which is structurally A&S but with `γσ²(T−t)` replaced by a *time-independent*
coefficient `g = √(σ²γ/(2kA)·(1+γ/k)^(1+k/γ))`. Under the hard bound the policy is trivially
implementable: **never post the quote that would increase `|q|` past `Q`** — the binding side simply
shuts off, and the optimal distance on the *reducing* side stays finite but the whole quote pair is
computed from a value function that blows up at the boundary.

### Cited Findings
- Exact optimal quotes (Theorem 1), `Δ = 1`, `q ≠ Q` and `q ≠ −Q`:
  `δ^(b*)(t,q) = (1/k)·ln( v_q(t) / v_(q+1)(t) ) + (1/γ)·ln(1 + γ/k)`
  `δ^(a*)(t,q) = (1/k)·ln( v_q(t) / v_(q−1)(t) ) + (1/γ)·ln(1 + γ/k)`
  `ψ*(t,q) = −(1/k)·ln( v_(q+1)(t)·v_(q−1)(t) / v_q(t)² ) + (2/γ)·ln(1 + γ/k)`
  where `v(t) = exp(−M(T−t))·1`, `1 = (1,…,1)ᵀ`, and `M` is tridiagonal with diagonal `α q²` and
  off-diagonal `−η`, with `α = (k/2)γσ²` and `η = A(1 + γ/k)^(−(1 + k/γ))` — [GLFT, Theorem 1 & Prop. 1–2](https://arxiv.org/abs/1105.3115).
- Boundary conditions that encode the hard cap: for `q = Q` the bid term is *dropped* from the HJB
  equation, and for `q = −Q` the ask term is dropped ("a market maker with inventory `Q` … will never
  set a bid quote and symmetrically … will never set an ask quote") — [GLFT §2–3](https://arxiv.org/abs/1105.3115).
  Equivalently, in Guéant's general framework the intensities carry indicators
  `λ^b = Λ^b(δ^b)·1_{q_− < Q}`, `λ^a = Λ^a(δ^a)·1_{q_− > −Q}` — [Guéant 2017, eq. (2.3)](https://arxiv.org/abs/1605.01862).
- Ergodic limit (the version to actually implement): with `f⁰` the eigenvector of `M` for its
  smallest eigenvalue,
  `δ_∞^(b*)(q) = (1/γ)ln(1+γ/k) + (1/k)·ln( f⁰_q / f⁰_(q+1) )`,
  `δ_∞^(a*)(q) = (1/γ)ln(1+γ/k) + (1/k)·ln( f⁰_q / f⁰_(q−1) )`,
  and `f⁰` minimizes `Σ α q² f_q² + η Σ (f_(q+1) − f_q)² + η f_Q² + η f_(−Q)²` subject to `‖f‖₂ = 1`
  — [GLFT, Theorem 2](https://arxiv.org/abs/1105.3115). The two `η` boundary terms are exactly where
  `Q` enters; drop them and you recover the unbounded problem.
- Closed-form approximation (the "GLFT formula", the one used in production):
  `δ_∞^(b*)(q) ≈ (1/γ)ln(1+γ/k) + ((2q+1)/2)·g`
  `δ_∞^(a*)(q) ≈ (1/γ)ln(1+γ/k) − ((2q−1)/2)·g`
  `ψ_∞*(q) ≈ (2/γ)ln(1+γ/k) + g`, with
  **`g = √( σ²γ/(2kA) · (1 + γ/k)^(1 + k/γ) )`** — [GLFT §4](https://arxiv.org/abs/1105.3115).
  Read as a policy: `reservation price = s − q·g`, `half-spread = (1/γ)ln(1+γ/k) + g/2`.
- Accuracy caveat from the authors: the approximation is "satisfactory in most cases and … always
  very good for small values of the inventory `q`"; it degrades for large `|q|` "because we are
  approximating expressions that depend on ratios of the form `f⁰_q / f⁰_(q+1)`" — [GLFT §4](https://arxiv.org/abs/1105.3115).
  This is the single most important caveat for a hard-cap implementation.
- Drift/alpha extension: with `dS = μ dt + σ dW`, the approximation becomes
  `δ_∞^(b*)(q) ≈ (1/γ)ln(1+γ/k) + [ −μ/(γσ²) + (2q+1)/2 ]·g` and symmetrically for the ask
  — [GLFT §5.1](https://arxiv.org/abs/1105.3115). Note the drift enters as a *shift of effective
  inventory*: `q_eff = q − μ/(γσ²)`.
- Adverse-selection / self-impact extension: with `dS = σ dW + ξ dN^a − ξ dN^b` (a fill on your bid
  pushes the reference price down by `ξ`), the formulas gain `+ξ/2` on each quote and the skew term
  is multiplied by `e^(kξ/4)`, spread gains `+ξ` — [GLFT §5.2](https://arxiv.org/abs/1105.3115). The
  authors interpret this literally as adverse selection and show the two effects separately:
  `ξ/2` = "adverse selection" (pure markup), `e^(kξ/4)` = "side-effect" (deeper quotes ⇒ higher risk
  of being stuck with inventory).
- Comparative statics (verified numerically in the paper): `∂ψ/∂σ² > 0`, `∂ψ/∂A < 0`,
  `∂ψ/∂k < 0`; **the effect of `γ` is ambiguous** because higher `γ` simultaneously (i) narrows quotes
  to reduce execution randomness and (ii) widens them to reduce price risk — [GLFT §6](https://arxiv.org/abs/1105.3115).
  Sign of `∂δ^b/∂σ²` flips with the sign of `q`: for `q > 0`, `∂δ^b/∂σ² > 0` and `∂δ^a/∂σ² < 0`
  (a long maker widens the bid and cuts the ask), which is the correct qualitative skew behaviour.
- Their own backtest protocol (worth copying): quotes rounded to the nearest tick; an order of size
  ATS is sent and "not canceled nor modified for a given period of time `Δt`, unless a trade occurs";
  `σ, A, k` calibrated on trade-by-trade LOB data; `γ` "chosen … arbitrary value for which the
  inventory stayed between −10 and 10 during the day". Fill assumption: "we assumed that our orders
  were entirely filled when a trade occurred at or above the ask price quoted by the agent" —
  [GLFT §7](https://arxiv.org/abs/1105.3115). Test asset: France Telecom, 15 March 2012, ATS = 1105 shares.
- Rigorous generalization (arbitrary decreasing intensities `Λ`, general trade size `Δ`, terminal
  penalty `ℓ`, multi-asset): value function `u(t,x,q,S) = ∓exp(−γ(x+qS+θ(t,q)))` (CARA) or
  `x + qS + θ(t,q)` (risk-neutral + running penalty), with `θ` solving a tridiagonal ODE system, and
  the optimal quote given implicitly by
  `δ^(b*) = (δ̃^b)_ξ( [θ(t,q) − θ(t,q+Δ)]/Δ )`, `p = δ̃ − (1/(ξΔ))·log(1 − ξΔ·Λ^b(δ̃)/Λ^(b)′(δ̃))`,
  `ξ = γ` for CARA and `ξ = 0` for the risk-neutral/running-penalty model — [Guéant 2017, eqs. (3.9)–(3.14)](https://arxiv.org/abs/1605.01862).
  For exponential intensities this collapses to the closed form `H_ξ(p) = (AΔ/k)·C_ξ·e^(−kp)` with
  `C_ξ = (1 + ξΔ/k)^(−k/(ξΔ) − 1)` (`C_0 = e^(−1)`), and the ODE system is *linear* — [Guéant 2017, eq. (3.13)](https://arxiv.org/abs/1605.01862).
  A useful degenerate check: at `ξ = 0` the exponential-intensity rule is `δ^b = p + 1/k`, i.e. the
  pure monopoly markup `argmax_δ δΛ(δ)` plus the shadow value of inventory `p`.
- Book treatment: Guéant's monograph devotes Part IV to market making ("Market Making: From
  Avellaneda–Stoikov to Guéant–Lehalle, and Beyond", "The Avellaneda–Stoikov Model", "Generalization
  of the Avellaneda–Stoikov Model") and Part II to optimal liquidation in the same Hamiltonian
  framework — [Guéant, *The Financial Mathematics of Market Liquidity*, CRC Press 2016](https://doi.org/10.1201/b21350).

### Inferences
- The hard cap changes the *shape* of the skew, not just its magnitude. Near the boundary the value
  function difference `θ(t,q) − θ(t,q+Δ)` grows without bound (the eigenvector `f⁰_q` decays like
  `exp(−½√(α/η) q²)`, so `ln(f⁰_q/f⁰_(q+1)) ≈ √(α/η)·(q + ½)`, i.e. *linear* only in the interior).
  Consequence for implementation: use the linear rule `s − q·g` in the interior and taper it with a
  convex, boundary-blowing function as `|q| → Q`; a purely linear skew will under-price the last
  10–20 lots.
- Because the *only* thing the bound does to the control is switch off one side, a hard cap is
  cheap to enforce exactly: at `q = +Q` send no bid, at `q = −Q` send no ask. Everything else is
  optional refinement. This is a much stronger guarantee than a penalty, and with a $10,000/s
  outside-the-band penalty it is the correct architecture (§9).
- `g` scales as `√γ` (for small `γ/k`) and as `1/√(kA)`: skew is *cheap* to get wrong by a factor of
  2 in `γ` but *expensive* to get wrong in `k` or `A`, because those also set the spread. Practical
  consequence: estimate `k` (fill-decay) and `A` (fill scale) online with more care than `γ`; `γ` can
  be set as a tuning knob to hit a target inventory distribution (which is exactly what GLFT did in
  their backtest: "γ … for which the inventory stayed between −10 and 10").
- Worked numbers from GLFT's own parameter set (`σ = 0.3` tick·s^(-1/2), `A = 0.9 s⁻¹`,
  `k = 0.3` tick⁻¹, `T = 600 s`). My arithmetic from the cited formula:
  `γ = 0.01` ⇒ `g ≈ 0.068` ticks/lot, base half-spread `≈ 3.28` ticks, spread `≈ 6.63` ticks;
  `γ = 0.1` ⇒ `g ≈ 0.23` ticks/lot, base half-spread `≈ 2.88` ticks, spread `≈ 5.98` ticks.
  So raising risk aversion 10× *narrows* the spread 12% while *tripling* the skew slope — a clean
  demonstration of the `γ` ambiguity, and a warning that "more risk aversion" is not the same as
  "wider quotes" in this family of models.
- The `q_eff = q − μ/(γσ²)` form is the safest way to add an alpha/drift view: it keeps the
  spread formula intact and degrades gracefully (a small `γ` makes you lean hard into the signal,
  a large `γ` makes you ignore it). Note the model's own logic: for `γ = 0.01`, `σ = 0.3`,
  `μ = 0.01` tick/s, the offset is `0.01/(0.01·0.09) ≈ 11` lots — i.e. a persistent drift of
  1 tick per 100 s is worth about 11 lots of inventory. That is the right order of magnitude for
  deciding whether an alpha signal is worth acting on at all.

### Gaps
- I did not verify a page-level derivation of the *bounded-inventory* extension in Guéant's book
  (chapter numbering / exact theorem). The verified primary source for the bound is GLFT Theorem 1
  plus Guéant (2017) eq. (2.3)/(3.14); the book is cited only as the consolidated treatment.
- No published study found that reports the *cost* of using the linear approximation instead of the
  exact eigen-solution near the bound. This matters for us (Q = 100 is deep in the nonlinear region)
  and is cheap to measure locally.

---

## Cartea–Jaimungal–Penalva: running inventory penalty, terminal liquidation penalty, and alpha

### Takeaway
CJP replace the exponential utility with *expected P&L minus penalties*, which is what practitioners
actually specify. Two knobs: a running penalty `φ q²` (mean-reverts inventory continuously, with a
term linear in `q` and in time-to-go) and a terminal liquidation penalty `ℓ(|q_T|)` (forces unwinding
near `T`, aggressively, because the marginal cost of carrying inventory to `T` is explicit). Their
alpha extension skews the *half-spreads* by `±E∫α`, not just the mid.

### Cited Findings
- CJP objective (market making at the touch, `Δ` = tick, `ℓ^± ∈ {0,1}` = posted/not posted):
  `max E[ X_T + q_T(S_T − (Δ/2 + ϑ q_T)) − φ ∫_0^T q_u² du ]` — terminal penalty with slope `Δ/2`
  plus quadratic `ϑ`, running penalty `φ`; the terminal condition of the DPE is
  `h(T, α, q) = −q(Δ/2 + ϑ q)` — [CJP book §10.4.2, worked notebook by Jaimungal](https://gist.github.com/sebjai/c6ff3850dea37d28d3fa7d3aef59722b).
- Value function form and the inventory channel: with `H(t,x,S,q,α) = x + qS + h(t,α,q)`,
  `0 = (∂_t − ζα∂_α + ½η²∂_αα)h + αq − φq² + λ^+ max{…} + λ^- max{…}`, and the posting rule is
  `ℓ^{+,*} = 1{ Δ/2 + E[h(t,α+ε^+,q−1) − h(t,α+ε^+,q)] > 0 } ∩ {q > q̲}` — i.e. the quote is posted iff
  the half-spread exceeds the marginal value of the inventory it creates — [CJP §10.4.2](https://gist.github.com/sebjai/c6ff3850dea37d28d3fa7d3aef59722b).
- Alpha enters as an additive drift in the mid, `S_t = S_0 + ∫(υ + α_u)du + σW_t`, with `α` a
  mean-reverting jump process (`dα = −ζα dt + η dW^α + ε dM^+ − ε dM^-`) — [CJP §10.4.2](https://gist.github.com/sebjai/c6ff3850dea37d28d3fa7d3aef59722b).
- Effect of alpha on quotes (stated in words and as a corollary): "if short-term-alpha is positive,
  the HFT's sell halfspread is increased to avoid being picked off, and at the same time the buy
  halfspread decreases to take advantage of the [trend]" — [Cartea, Jaimungal & Penalva, "Algorithmic Trading, Stochastic Control, and Mutually-Exciting Processes"](https://oxford-man.ox.ac.uk/wp-content/uploads/2020/05/Algorithmic-Trading-Stochastic-Control-and-Mutually-Exciting-Processes.pdf).
  First-order expansion: `δ^{±*} = δ⁰_± + B(δ⁰_±; κ^±) ± S^±_λ·E∫_t^T α_u du + φ·S^±_λ·[b_φ + (1 ∓ 2q)(T − t)]`
  — same source, Corollary 4. Structural reading: **the alpha term is `±E∫α` (a spread skew, sign
  follows the signal), the inventory term is `φ·(1 ∓ 2q)(T − t)` (linear in `q` and in time-to-go,
  scaled by `φ`)**.
- Effect of the running penalty on behaviour: "the fourth term is an inventory management component
  that introduces asymmetry in the postings so that the HFT does not build large long or short
  inventories. This component of the halfspread is proportional to the penalization parameter `φ > 0`
  which induces mean reversion to the optimal inventory position" — [CJ&P, SIAM Review / Oxford-Man PDF](https://oxford-man.ox.ac.uk/wp-content/uploads/2020/05/Algorithmic-Trading-Stochastic-Control-and-Mutually-Exciting-Processes.pdf).
- Terminal-penalty behaviour (the "aggressive unwinding" result, in the authors' words): "as the
  dynamic trading strategy approaches the terminal date `T`, orders are posted nearer to the midquote
  to induce mean-reversion to zero in inventories, which avoids having to post collateral overnight
  and bearing inventory risks until the market opens the following day" — [CJ&P](https://oxford-man.ox.ac.uk/wp-content/uploads/2020/05/Algorithmic-Trading-Stochastic-Control-and-Mutually-Exciting-Processes.pdf).
- The two objectives are the *same* problem up to a parameter map. Guéant writes CJP's criterion as
  `E[X_T + q_T S_T − ℓ(|q_T|) − ½γσ²∫q_t² dt]` (his "Model B") and shows that Model A (CARA with
  risk aversion `γ`) and Model B satisfy the same ODE family, differing only by `ξ = γ` vs `ξ = 0`
  — [Guéant 2017, §2.2 and §3.1](https://arxiv.org/abs/1605.01862). **Therefore a running penalty
  `φ q²` with `φ = ½γσ²` is *exactly* CARA risk aversion `γ` in this model class.** That is the
  cleanest way to translate between the two literatures, and the cleanest way to pick `φ` from a
  risk-appetite statement rather than fitting it.
- CJP also supply risk metrics for exactly this tuning exercise (max expected terminal wealth subject
  to capital and inventory constraints, with risk metrics trading off inventory risk against expected
  profit) — [Cartea & Jaimungal, *Mathematical Finance* 25(3):576–611](https://doi.org/10.1111/mafi.12023).
  Their model there includes midprice jumps from market-order impact that create momentum — i.e. the
  "momentum" the maker is paid to lean against is endogenous to order flow, not exogenous alpha.

### Inferences
- For our setting the running penalty is the *wrong* tool and the terminal penalty is nearly
  irrelevant: we have no fixed `T` (the round is a continuous session) and the real cost is a
  step-function at `|I| = 100`. A running penalty is a soft surrogate for a hard constraint and, per
  the map `φ = ½γσ²`, it is equivalent to choosing `γ`; the *level* of `φ` therefore only sets how
  often you touch the band, not whether you can breach it. Use the hard constraint (§2) plus a
  running penalty as a *buffer-shaping* device (keep `|I|` small in normal times so the band is
  reachable only via jumps).
- The `φ(1 ∓ 2q)(T−t)` structure says the penalty term *vanishes* as `t → T`. For a session with no
  `T`, that means naive implementations of the CJP formula converge to no skew at all. Fix by using
  a rolling horizon: replace `(T−t)` with a constant `τ_eff` (the inventory half-life you want), which
  is what practitioners do, and what the "horizon folded into the gains" implementations do in
  practice (see e.g. [the `cj_reservation_shift(alpha·h − φ·q_norm·h)` formulation with `h = 1.0`](https://vike.io/docs/trader/market-making/short-term-alpha)).
- The alpha result `±E∫α` is a *spread* skew, not a mid shift, in CJP; the GLFT drift result is a
  *mid* shift (`q_eff = q − μ/(γσ²)`). They are different policies with different risk: CJP's
  version refuses to be picked off on the wrong side (widens the losing side) and does not chase;
  GLFT's version leans in (shifts both quotes). For a 5 ms-latency venue where you can be picked off,
  CJP's asymmetric widening is the safer of the two.
- Sign convention check for implementation: with `q > 0` (long) you want to *sell*, so the ask must
  get closer (`δ^a` smaller) and the bid further (`δ^b` larger). Both A&S and GLFT satisfy this
  (`∂δ^a/∂q < 0`, `∂δ^b/∂q > 0` with slopes `∓γσ²(T−t)` / `∓g`). A sign error here is the single
  most common bug in ported implementations and it is *profitable-looking* on trending tapes.

### Gaps
- I could not obtain the exact closed-form half-spread of CJP's book for the *non-asymptotic* case
  (the `B(δ⁰;κ)` correction term) — it requires the book's §10.2.2. The structural result (linear in
  `q`, linear in `T−t`, scaled by `φ`) is verified above; the constant is not.
- No source found quantifying the welfare/PNL difference between CARA and running-penalty objectives
  in a simulation with a hard cap. Guéant asserts they are "part of the same family" but does not
  rank them.

---

## Optimal execution as the dual problem: Almgren–Chriss, Obizhaeva–Wang, and liquidation at the horizon

### Takeaway
Market making and optimal execution are the same stochastic control problem with the sign of the
inventory target flipped: execution minimizes impact plus variance for a required `q → 0`, market
making maximizes spread minus variance at `q ≈ 0`. Almgren–Chriss gives the canonical *schedule*
(exponential decay with rate `κ = √(λσ²/η)`), Obizhaeva–Wang shows that with transient impact the
optimum is a *block plus TWAP*, and both tell you what to do when a hard cap or a deadline forces
unwinding: front-load, and cross the spread when the shadow cost of carrying inventory exceeds the
half-spread.

### Cited Findings
- Almgren–Chriss: minimize `E[C] + λ·Var[C]` where `C` = implementation cost with linear permanent
  impact (`γ̃`) and linear temporary impact (`η`); the efficient frontier of schedules is constructed
  explicitly and a quadratic utility (or VaR / "liquidity-adjusted VaR") selects a point on it —
  [Almgren & Chriss, *Journal of Risk* 3(2):5–39](https://doi.org/10.21314/JOR.2001.041) (DOI [10.21314/jor.2001.041](https://doi.org/10.21314/jor.2001.041)).
  The optimal holdings path is `x(t) = X·sinh(κ(T−t))/sinh(κT)` with `κ = √(λσ²/η)`; `κ → 0`
  recovers TWAP, large `κ` front-loads. (Standard result of that paper; the `κ` parameterization is
  reproduced in [Guéant's chapter 3](https://www.gbv.de/dms/zbw/845604678.pdf).)
- Obizhaeva–Wang: with *transient* (resilient, exponentially decaying) impact, "the overall price
  impact of a sequence of trades depends only on their total size and is independent of their
  distribution over time" is false, and the optimal strategy is not a continuous schedule: it uses
  discrete blocks at the start and end with continuous trading in between —
  [Obizhaeva & Wang, *Journal of Financial Markets* 16(1):1–32](https://doi.org/10.1016/j.finmar.2012.09.001),
  PDF [web.mit.edu/wangj/www/pap/ObizhaevaWang13.pdf](https://web.mit.edu/wangj/www/pap/ObizhaevaWang13.pdf).
- The formal link between the two problems is the shared intensity/cost model: GLFT state that
  "Bayraktar and Ludkovski and Guéant, Lehalle, Fernandez-Tapia used … a similar model to optimally
  liquidate a portfolio" — [GLFT §1](https://arxiv.org/abs/1105.3115); Guéant's book puts optimal
  liquidation (Part II) and market making (Part IV) in one Hamiltonian framework, and treats
  "participation constraints" and portfolio liquidation in the same language as quoting —
  [Guéant 2016](https://doi.org/10.1201/b21350), TOC [here](https://www.gbv.de/dms/zbw/845604678.pdf).
- Terminal-penalty ⇒ aggressive unwinding: CJP's terminal condition `h(T,α,q) = −q(Δ/2 + ϑq)` prices
  liquidation at the touch as `Δ/2` per lot plus a quadratic term, and the optimal postings move
  toward the midquote as `t → T` — [CJP §10.4.2](https://gist.github.com/sebjai/c6ff3850dea37d28d3fa7d3aef59722b),
  [CJ&P](https://oxford-man.ox.ac.uk/wp-content/uploads/2020/05/Algorithmic-Trading-Stochastic-Control-and-Mutually-Exciting-Processes.pdf).
  GLFT's `ℓ(|q_T|)` generalization makes the same point rigorously: a convex liquidation premium at
  `T` is admissible and leaves "mutatis mutandis" the same solution — [GLFT §2, footnote 7](https://arxiv.org/abs/1105.3115).

### Inferences
- The duality gives a *pricing* rule for the endgame that our setting needs: the marginal cost of
  carrying one more lot to a point where it must be dumped is `Δ/2 + 2ϑq` (half-spread to cross plus
  penalty slope). A maker should therefore cross the spread *voluntarily* whenever
  `(expected adverse move) + (inventory shadow cost) > half-spread`. With a $10,000/s step penalty,
  the shadow cost of the last few lots before the cap is effectively infinite, so the endgame policy
  is: **cross, don't quote**, once `|I|` exceeds a buffer threshold. This is the "aggressive" half of
  the aggressive/passive trade-off and it is absent from pure A&S/GLFT (which have no market orders).
- Obizhaeva–Wang's block-plus-TWAP result maps to a concrete microstructure insight for a resilient
  book: if the book refills within ~1 s, waiting to unwind is *cheaper* than paying the spread now,
  because impact decays; if the book does not refill (thin tape), front-load. The discriminator is
  the estimated resilience/decay rate, i.e. something you must estimate online (our repo already has
  `RealizedVol`, `RegimeFilter`; a resilience estimator is the missing piece).
- `κ = √(λσ²/η)`: the ratio `σ²/η` (variance per unit of impact) is the only thing that matters for
  the *shape* of an unwinding schedule. In a 5 ms-latency venue `η` is not a constant — it is the
  depth at the touch, which is exactly what a queue-aware model would estimate.

### Gaps
- I did not verify a page-level statement that Almgren–Chriss's closed-form `sinh` schedule appears in
  the original paper with exactly that notation (it is the standard result and is reproduced in
  Guéant's chapter 3, which I cite). No numbers for `λ` (risk-aversion) magnitudes were extracted
  from Almgren–Chriss.
- No source found that solves optimal execution *with a hard position cap and a step penalty*; the
  closest are participation constraints (Guéant ch. 5.2) and the terminal-penalty formulations above.

---

## Robust / ambiguity-averse market making, adverse selection, and informed traders

### Takeaway
Two literatures matter here. (i) Robust/ambiguity-averse MM replaces the single reference model with
a worst case over a KL/Wasserstein ball, and the *qualitative* result is uniform: ambiguity about
drift or fill rates makes the maker mean-revert inventory *faster* and increase *total* depth (to
recover adverse-selection losses), while ambiguity about the execution price makes it quote *tighter*
(fear of missing business). (ii) The classical informed-trader models (Glosten–Milgrom,
Copeland–Galai, Kyle) say the spread has an adverse-selection component that scales with the
probability of informed flow and with value uncertainty — which in practice means: the fill you get
is not a random draw, it is selected.

### Cited Findings
- Cartea, Jaimungal et al., "Algorithmic Trading with Model Uncertainty" (*SIAM J. Financial
  Mathematics*): ambiguity over (i) market-order arrival rate, (ii) limit-order fill probability,
  (iii) midprice drift; entropy penalty with separate aversion coefficients `φ_α, φ_λ, φ_κ`;
  analytical robust optimal strategies with verification. Results: ambiguity on drift or arrival
  rate ⇒ "reversion to the optimal level of inventory is quicker" and "total depth (buy depth plus
  sell depth) increases"; ambiguity on the *execution price* ⇒ "smaller depth on both sides";
  simulation with a realistic true model (jump alpha, mutually-exciting arrivals) shows "there are
  ranges where increasing ambiguity aversion produces a significant increase in the Sharpe ratio"
  — [DOI 10.1137/16M106282X](https://doi.org/10.1137/16M106282X).
  Quantitatively they illustrate with `φ_α = 20` (drift-ambiguity coefficient) in their Figure 5.
- Ambiguity-averse MM behaves like a risk-averse MM with an inventory-dependent drift: "the MM picks a
  model where the midprice drifts at `η*(q)(φ_α) = α − φ_α σ² q`" — [same paper](https://doi.org/10.1137/16M106282X).
  Note this is *exactly* the GLFT `q_eff` structure: an adversarial drift proportional to `q`.
- Adversarial-RL formulation of robustness (MM vs. an adversary controlling the environment, zero-sum,
  built on the A&S model): ARL "leads to the emergence of risk-averse behaviour without constraints or
  domain-specific penalties" and improved robustness to model uncertainty vs. single-agent RL —
  [arXiv:2003.01820](https://arxiv.org/abs/2003.01820).
- Wasserstein-DRO market making with entropy-regularized stochastic policies; the robust problem
  "can be reformulated as a convex optimization question", with a method for choosing the
  Wasserstein-ball radius — [arXiv:2503.04072](https://arxiv.org/html/2503.04072).
- Informed-trader background: Glosten–Milgrom and Kyle are the canonical "information-based" models
  of the spread, as distinct from inventory-based models; latency arbitrage is a *third* source,
  from symmetrically-observed public signals — [Aquilina, Budish & O'Neill, *QJE* 137(1):493–564](https://doi.org/10.1093/qje/qjab032);
  Kyle (1985), *Econometrica* 53(6):1315, DOI [10.2307/1913210](https://doi.org/10.2307/1913210).
- The classic inventory-vs-information decomposition, and its empirical resolution, is stated crisply
  by Hendershott & Menkveld: the spread "captures the cost of processing orders (or monopoly rents)",
  whereas *midquote* changes "capture the dynamic trade off the intermediary does between staying one
  more period on a costly suboptimal position or discounting prices to mean-revert that position" —
  [Hendershott & Menkveld, *JFE* 114(3):405–423](https://doi.org/10.1016/j.jfineco.2014.08.001), appendix
  [PDF](https://albertjmenkveld.com/text/hendershottmenkveld1_ia.pdf).
- Reduced-form adverse selection is what GLFT's `ξ` is: `dS = σdW + ξ dN^a − ξ dN^b`, giving `+ξ/2`
  per quote and an `e^(kξ/4)` multiplier on the inventory skew — [GLFT §5.2](https://arxiv.org/abs/1105.3115).
  CJP instead use an explicit state-dependent "buffer to cover adverse selection costs" driven by
  order-flow information — [Cartea & Jaimungal, *Math. Finance* 25(3)](https://doi.org/10.1111/mafi.12023).
- Recent (2025) work re-opens exactly the two channels that matter for a quoting algorithm: adverse
  selection from informed flow, and "price reading" — the MM's own quotes reveal inventory direction —
  [arXiv:2508.20225, "Optimal Quoting under Adverse Selection and Price Reading"](https://ideas.repec.org/p/arx/papers/2508.20225.html).

### Inferences
- Every robust result points the same way for a hard-cap implementation: **increase depth when
  uncertain about fills (so you are not stuck), decrease depth when uncertain about the price you
  will get (so you are not picked off), and increase the inventory-mean-reversion gain when uncertain
  about drift.** In our setting the dominant uncertainty is the *tape's* generator (parameters change
  between rounds), so the drift/arrival-ambiguity branch applies: bias toward faster mean reversion
  and wider depth, not tighter quotes.
- The `η*(q) = α − φ_α σ² q` result is the theoretically-correct form of the ad-hoc "inventory
  penalty" and coincides with GLFT's `q_eff = q − μ/(γσ²)` once you set `φ_α = 1/(γσ²)·(something)`.
  Implementation-wise: **one number** (the coefficient multiplying `q` inside the alpha term) buys
  both the robustness story and the risk-aversion story. Do not implement them separately.
- "Price reading" (your own quotes leak inventory) is a real effect in a simulated market with
  strategic counterparties, and it is not in any of the canonical models. It implies you should not
  let the *observable* quote asymmetry be a deterministic function of `q` — add noise or re-randomize
  the skew slightly. This is a cheap robustness measure with essentially zero model risk.
- `ξ` (adverse selection) enters the spread *linearly* (`+ξ`) and the skew *multiplicatively*
  (`e^(kξ/4)`); both are the wrong shape for a jump-driven book (see §9), where the pick-off loss is
  a jump size, not a diffusion drift. Estimate `ξ` as "mean adverse mid move in the 100 ms after a
  fill", not as a fitted impact coefficient.

### Gaps
- No source found that solves a market-making problem with an *indicator-type* running penalty (a
  fixed $ per second outside a band). All robust/penalty results assume smooth (quadratic or
  entropy) costs. This is a genuine modelling gap relative to the arena's rule.
- Magnitudes: the only concrete ambiguity-aversion number found is `φ_α = 20` in Cartea et al.'s
  Figure 5; no mapping from `φ_α` to a P&L or Sharpe number is given in the text I could read.

---

## Multi-asset / cross-hedging inventory, borrowing and position limits, and how real makers behave

### Takeaway
Multi-asset MM replaces the scalar `q` in the risk term with the vector `Σq`, so the skew of asset `i`
depends on its *hedged* inventory: correlated positions net out and the optimal quotes of correlated
assets move together. Empirically, real intermediaries do exactly what the single-asset theory says
(shifts in the midquote proportional to inventory) and *not* what naive theory says about width
(the spread is nearly inventory-invariant) — and inventory mean-reverts with a half-life of order a
day, i.e. real makers are far more patient than a 600-second model implies.

### Cited Findings
- Multi-asset formulation: the quadratic risk term becomes `½γ qᵀΣq` (Σ = price covariance), so the
  inventory state that drives quotes is `Σq`, and closed-form approximations for the optimal quotes of
  a multi-asset maker are obtained for general intensities — [Guéant 2017, §5](https://arxiv.org/abs/1605.01862).
  Guéant motivates it directly: "for a market maker in charge of several correlated assets, applying
  an independent market making strategy to each asset is suboptimal in terms of risk management …
  especially in the case of corporate bonds where there are often dozens of bonds issued by the same
  company" — [Guéant 2017, §1](https://arxiv.org/abs/1605.01862).
- Closed-form multi-asset proxies (value-function proxies usable directly as a greedy quoting rule,
  or as an initial value function for RL), including drift, client tiering, several request sizes and
  fixed transaction costs — [Bergault, Evangelista, Guéant & Vieira, arXiv:1810.04383](https://arxiv.org/abs/1810.04383).
  A factor method to reduce dimensionality for many correlated assets is due to Bergault & Guéant
  (cited there as [5]).
- Large-universe solution: model-based deep actor-critic over a universe of corporate bonds, because
  "classical finite difference methods … cannot be used" in high dimension —
  [Guéant & Manziuk, *Applied Mathematical Finance* 26(5):387–452](https://doi.org/10.1080/1350486X.2020.1714455), [arXiv:1910.13205](https://arxiv.org/abs/1910.13205).
- Empirical inventory behaviour (NYSE specialists, 12 years of daily position data):
  - price pressure is large and inventory-driven: "$100,000 inventory shock causes price pressure of
    1.01% for the small-capitalization stocks and 0.02% for the large-cap stocks"; average transitory
    volatility 0.49% with half-life 0.92 days — [Hendershott & Menkveld 2014](https://doi.org/10.1016/j.jfineco.2014.08.001).
  - the *midquote* is what moves: the intermediary "lowers the midquote on long positions and raises
    it on short positions", and "relative to midquote changes the bid-ask spread appears constant
    across states. It is only slightly higher than the spread that a risk-neutral intermediary would
    charge: `2(2θ)^(-1) = 0.2`" — [H&M online appendix](https://albertjmenkveld.com/text/hendershottmenkveld1_ia.pdf).
  - the inventory policy is close to linear in inventory ("price pressure is linear in inventory and
    does not depend on wealth"), which validates the `r = s − q·g` form — same source.
  - earlier literature is mixed on whether inventory affects the spread at all: "Zabel (1981) and
    Mildenstein and Schleef (1983) find that the spread is independent of inventory. Ho and Stoll
    (1981) find inventory has a very small effect on the spread" — same source.
- Position limits as the practical constraint: Guéant's framework makes `Q` an explicit risk limit
  ("`Q` … depending in practice on risk limits") and the model's answer is to stop quoting the
  increasing side — [Guéant 2017, §2.1](https://arxiv.org/abs/1605.01862). GLFT make the same
  assumption — [GLFT §2](https://arxiv.org/abs/1105.3115).
- Competitive/multi-agent effects are *not* in these models: "the competition between market makers is
  not explicitly modeled" — [Guéant 2017, §1](https://arxiv.org/abs/1605.01862); a mean-field-game
  treatment with strategic market takers exists — [arXiv:2203.13053](https://doi.org/10.48550/arxiv.2203.13053).

### Inferences
- Cross-hedging is the one multi-asset result with immediate payoff even in a single-asset arena:
  if we ever quote two correlated instruments, the correct state variable is the *portfolio* inventory
  projected on the covariance, `Σq`, not `q_i`. With one asset it degenerates, so the practical
  takeaway is architectural: keep the skew a function of a vector `Σq` so a second instrument can be
  added without rewriting the rule.
- The empirical "spread constant, mid moves" result is strong evidence *against* implementing
  inventory control by widening the losing side. Widening the bid when long reduces fill probability
  on the side you want filled last; moving both quotes down keeps you just as likely to trade and
  collects the price improvement. Do the latter.
- The 0.92-day half-life is a warning about horizons: real makers take a day to work off inventory,
  while GLFT's ergodic skew assumes infinitely patient mean reversion and A&S assumes a 600 s
  liquidation. Our round is hours, so the ergodic (GLFT) form is the closest, but the *target*
  inventory distribution should be much tighter than "0.92-day half-life" implies, because the
  penalty for breach is a step function, not a financing cost.
- Borrowing/locate costs and short-selling constraints appear nowhere in this literature; the only
  hard limits treated are symmetric `|q| ≤ Q`. Treat asymmetric limits (e.g. harder to be short) by
  using an asymmetric bound `Q^+ ≠ Q^-` in the same ODE system — the derivation only uses the two
  boundary indicators, so it generalizes trivially (my inference from Guéant eq. (2.3)).

### Gaps
- I found no paper that estimates `γ` (risk aversion) for a *high-frequency* maker from LOB data;
  Hendershott–Menkveld identify risk aversion from daily NYSE specialist data but do not report a
  point estimate in the material I could read.
- No verified source on multi-asset MM *with a joint hard constraint* (e.g. a risk limit on `Σq` or on
  gross exposure). Guéant's multi-asset treatment has box constraints per asset only.
- Borrowing/position-limit literature (securities lending, locate fees) was not covered by any of the
  sources found; the request for it is noted as a genuine gap.

---

## Reinforcement learning for market making: what it learns, and what it does not beat

### Takeaway
RL reproduces the *qualitative* analytic policy (inventory mean reversion via asymmetric quotes, and
leaning with a drift) but has not been shown to beat a well-tuned analytic policy on risk-adjusted
terms in a realistic LOB; the credible gains are in (a) high-dimensional/multi-asset settings where
the PDE cannot be solved, and (b) reward-shaping insights (dampening the directional component of P&L
is what produces inventory-neutral behaviour). Reported numbers are noisy and mostly in-sample-ish.

### Cited Findings
- Chan & Shelton (2001), the first model-free RL market maker: SARSA / actor-critic in a
  Glosten–Milgrom-style simulated market with informed and uninformed traders; in the single-price
  model "we are able to determine the optimum strategies analytically and show that reinforcement
  algorithms successfully converge to these strategies"; in the two-price model "the actor-critic
  algorithm generates stochastic policies that correctly adjust bid/ask prices with respect to order
  imbalance" — but "the algorithm does not always converge to a single strategy, primarily due to the
  partial observability of the problem" — [Chan & Shelton, MIT AI Memo 2001-005](https://www.cs.ucr.edu/~cshelton/papers/docs/AIM-2001-005.pdf).
- Chakraborty & Kearns, "Market Making and Mean Reversion" (ACM EC 2011, pp. 307–314): RL/online
  learning for MM in a model with mean-reverting fundamentals — cited as the reference for
  mean-reversion-aware MM by [Spooner et al., ref. [11]](https://www.ifaamas.org/Proceedings/aamas2018/pdfs/p434.pdf).
- Spooner, Fearnley, Savani & Koukorinis (AAMAS 2018): high-fidelity LOB simulation built from
  historical data, SARSA(λ) with linear tile-coding, custom reward. Headline results on HSBA.L
  out-of-sample (normalised daily PnL in units of 10⁴, mean ± sd; MAP = mean absolute position):
  fixed symmetric strategy at 5 ticks: `2.80 ± 10.30`, MAP `2678 ± 1981`; consolidated RL agent:
  `15.43 ± 13.01`, MAP `104 ± 179`. Across ten securities the consolidated agent is "superior
  risk-adjusted" overall but *negative* mean PnL on ING.AS (`−3.21 ± 29.05`) and NOK1V.HE
  (`−5.28 ± 33.42`) and zero on GASI.MI — [Spooner et al. 2018, Tables 3 & 6](https://www.ifaamas.org/Proceedings/aamas2018/pdfs/p434.pdf), [arXiv:1804.04216](https://arxiv.org/abs/1804.04216).
  Key mechanism: *asymmetric dampening* of the reward — "gains from a profitable directional
  inventory position count for less, while losses from such exposure still count in full" — with
  `η ≈ 0.1` the point at which "the agent begins to converge on fundamentally different policies"
  (near-zero inventory). Without it, plain Q-learning/SARSA "were unable to obtain useful results"
  — [arXiv:1804.04216](https://arxiv.org/abs/1804.04216), [alphaXiv summary](https://www.alphaxiv.org/abs/1804.04216).
- Guéant & Manziuk: deep *model-based* actor-critic (not model-free RL) to approximate the
  A&S-family optimal quotes over a large universe of corporate bonds, i.e. as a numerical method for
  a high-dimensional PDE, benchmarked against the analytic solution where available —
  [AMF 26(5):387–452](https://doi.org/10.1080/1350486X.2020.1714455), [arXiv:1910.13205](https://arxiv.org/abs/1910.13205).
  Their framing is explicitly "beating the curse of dimensionality", not "beating the analytic policy".
- Multi-agent dealer market (Ganesh, Vadori, Xu, Zheng, Reddy & Veloso): RL maker learns "to manage
  inventory by smartly selecting asymmetric prices on the buy and sell sides (skewing), and
  maintaining a positive (or negative) inventory depending on whether the market price drift is
  positive (or negative)" — i.e. it rediscovers both the inventory skew and the alpha lean —
  [arXiv:1911.05892](https://arxiv.org/abs/1911.05892).
- Robustness via adversarial RL on top of A&S (see §5) — [arXiv:2003.01820](https://arxiv.org/abs/2003.01820).
- A candid negative datapoint from a textbook treatment of RL market making benchmarked against the
  A&S reservation-price model: the learned policy "demonstrates economically interpretable
  inventory-aware quote skewing … However … the current implementation shows higher wealth
  dispersion and larger terminal inventory than analytical baselines, making it defensible as a
  demonstration of adaptive quote control rather than evidence of superiority" — [ML for Trading, ch. 21](https://ml4trading.io/third-edition/chapters/21_rl_execution_hedging/).
- A Hawkes-process-based LOB simulator + DRL controller is reported to beat "multiple market making
  benchmarks … with respect to various risk-reward metrics, even under significant transaction costs",
  with the explicit critique that A&S-style models produce "phantom gains under backtesting" because
  price moves are independent of order arrivals — [Gašperov & Kostanjčar, arXiv:2207.09951](https://arxiv.org/html/2207.09951v1).

### Inferences
- RL's transferable lesson for a hand-written strategy is the *reward*, not the network: damping the
  directional (mark-to-market) component of P&L asymmetrically is equivalent to imposing the risk
  aversion that the analytic models impose via `γ`, and it is the single change that flips behaviour
  from "speculative" to "market making". In analytic terms: `reward = spread capture + (1−η)·MtM`,
  `η ≲ 0.1` — which is just a softer version of `γ`.
- All credible RL results that beat a baseline use a *better simulator* (Hawkes arrivals, queue
  dynamics, multi-agent competition), not a better optimiser. That is a statement about model error,
  not about RL: A&S's independence of price moves and order arrivals is the actual defect, and it is
  fixable analytically (GLFT's `ξ`, Hawkes intensities, our `IdGapJump`).
- For our problem (one asset, one hard cap, cheap closed form) RL is not worth the sample
  complexity: the analytic policy is available in closed form and is *provably* constraint-respecting
  at the boundary. RL would need to learn the hard cap from a $10,000/s penalty signal, which is a
  sparse, high-variance reward — a bad fit.

### Gaps
- **The title "Guéant & Manziuk (2019), *Deep reinforcement learning for market making in a
  multi-agent limit order book*" could not be verified.** Searches surface (a) Guéant & Manziuk,
  "Deep reinforcement learning for market making in corporate bonds: beating the curse of
  dimensionality" (arXiv:1910.13205) and (b) Ganesh et al., "Reinforcement learning for market making
  in a multi-agent dealer market" (arXiv:1911.05892). The supplied title appears to conflate the two;
  no paper with that exact title by those authors was found. Flagging rather than guessing.
- No head-to-head study found that benchmarks a *correctly implemented, hard-constrained* GLFT policy
  against DRL on the same LOB simulator. All RL comparisons I found use a fixed-distance or A&S
  baseline, not the GLFT-with-bound policy — so the "RL beats analytic" claim is weaker than it reads.

---

## Orders of magnitude and sensitivity of `γ`, `φ`, `k`, `A`

### Takeaway
Published parameter sets cluster around `γ ∈ [0.01, 0.1]` per unit of price (i.e. risk aversion on the
order of "1% of wealth per unit risk"), `φ` (running penalty) around `½γσ²` by the exact mapping, and
`k` around `0.1–1` per tick. The policy is *robust* to `γ` (spread moves ~10–20% for a 10× change) and
*very sensitive* to `k` and `A` (both spread and skew scale as `1/√(kA)`), which is the opposite of
where most implementations spend their tuning effort.

### Cited Findings
- GLFT's reference parameter set (used for all their figures and comparative statics):
  `σ = 0.3` tick·s^(-1/2), `A = 0.9 s⁻¹`, `k = 0.3` tick⁻¹, `γ = 0.01` tick⁻¹, `T = 600 s`;
  a second set for the wide-spread case: `σ = 0.4–1.0`, `A = 0.2–0.9`, `k = 0.3–0.9`, `γ = 0.01`
  — [GLFT §4, Figures 1–5 captions](https://arxiv.org/abs/1105.3115).
- GLFT's calibration/selection protocol: `σ, A, k` are calibrated on trade-by-trade LOB data; `γ`
  "has to be chosen" and in their backtest they "assign `γ` an arbitrary value for which the inventory
  stayed between −10 and 10 during the day (the unit being the ATS)" — [GLFT §7](https://arxiv.org/abs/1105.3115).
  This is the practitioner's answer to "what `γ`": **pick `γ` to hit the inventory distribution you
  want**, do not interpret it as a utility parameter.
- They also warn that `A` and `k` "have to depend at least on the actual market bid-ask spread … in
  practice, the value of `A` and `k` are function of the market bid-ask spread" — [GLFT §7](https://arxiv.org/abs/1105.3115).
  This is an explicit acknowledgement that the exponential-intensity fit is not scale-free.
- CJP worked example (market making at the touch): `Δ = 0.01` (price units), terminal liquidation
  penalty `ϑ = 0.01`, running penalty `φ = 0` (deliberately zero to isolate alpha), `σ = 0.005`,
  intensities `λ^± = 0.8333 s⁻¹`, inventory grid `|q| ≤ 20`, `T = 60 s`, alpha parameters
  `ζ = 0.5 s⁻¹`, `η = 0.001`, `ε = 0.002` — [CJP §10.4.2 notebook](https://gist.github.com/sebjai/c6ff3850dea37d28d3fa7d3aef59722b).
- Cartea et al.'s robustness example uses `φ_α = 20` as the drift-ambiguity coefficient and reports
  that Sharpe improvements appear "in ranges" rather than monotonically — [SIAM 10.1137/16M106282X](https://doi.org/10.1137/16M106282X).
- Sensitivity, exactly stated by GLFT: `∂ψ*/∂σ² > 0`, `∂ψ*/∂A < 0`, `∂ψ*/∂k < 0`, and the `γ`
  effect is sign-ambiguous; empirically "the 'no-volatility' effect dominated for the values of the
  inventory under consideration" (i.e. `ψ` decreases in `k`) — [GLFT §6](https://arxiv.org/abs/1105.3115).
- My arithmetic on GLFT's reference set: 10× increase in `γ` (0.01 → 0.1) moves the spread from
  `≈ 6.63` to `≈ 5.98` ticks (−10%) while the skew slope `g` moves from `≈ 0.068` to `≈ 0.23`
  ticks/lot (+240%). Same set with `k` halved (0.3 → 0.15) and `A` fixed: `g` grows by `√2` and the
  base half-spread `(1/γ)ln(1+γ/k)` grows from `3.28` to `6.62` ticks (slightly more than 2×, because
  the log term dominates at small `γ/k`). Both computations use the GLFT §4 formulas verbatim.

### Inferences
- Set `γ` last, as the knob that fixes the *distribution* of inventory; set `k` and `A` first and
  re-estimate them per regime (calm/storm), because they control both the spread level and the skew
  scale and a 2× error in `k` is a 40% error in `g` and a 2× error in spread.
- For our cap: the relevant dimensionless number is "skew slope × cap / spread", i.e. how many
  spreads of price concession you are willing to give to shed 100 lots. With GLFT's numbers that is
  `0.068 × 100 / 6.63 ≈ 1.0` spread — reasonable. If an implementation produces a ratio ≫ 3, the
  skew is too aggressive and will cross the book (converting a maker into a taker at exactly the
  wrong moment).
- The `φ = ½γσ²` mapping means there is exactly *one* free risk parameter, not two. Any code that
  exposes both `γ` and `φ` should constrain them by this identity or it will double-count risk
  aversion (a common and silent over-skew bug).

### Gaps
- No published, data-fitted `γ` for an intraday equity/FX maker; no published `k`/`A` estimates from a
  tick-level book at 1-lot granularity, which is our granularity. These must be estimated locally.
- No source gives a rule for choosing `γ` under a *step* penalty (as opposed to a utility or a
  quadratic penalty); GLFT's "pick `γ` so inventory stays in a range" is a heuristic, not a derivation.

---

## Where the fill and latency assumptions break in a discrete-tick book with 5 ms latency and price-time priority

### Takeaway
Four assumptions do the work in every model above, and all four fail here: (i) fill intensity is a
smooth, memoryless function of distance `δ`; (ii) quotes execute in full, immediately, when touched;
(iii) queue position is irrelevant; (iv) the price process is exogenous and independent of order
arrivals. In a tick-constrained book with 5 ms delay, fills are *queue-share*-limited at the touch,
the maker can be picked off in flight, and the adverse-selection term is a jump, not a drift.

### Cited Findings
- GLFT state the discreteness problem explicitly and give their fix: "the model is continuous in both
  time and space while the real control problem is intrinsically discrete in space, because of the tick
  size, and in time, because orders have a certain priority and changing position too often reduces
  the actual chance to be reached by a market order. Hence, the model has to be reinterpreted in a
  discrete way. In terms of prices, quotes must not be between two ticks and we decided to round the
  optimal quotes to the nearest tick. In terms of time, an order of size ATS is sent to the market and
  is not canceled nor modified for a given period of time `Δt`, unless a trade occurs" — [GLFT §7](https://arxiv.org/abs/1105.3115).
  Two operational consequences: (a) round `δ` to ticks, (b) impose a minimum re-quote interval to
  protect queue priority.
- Their backtest fill rule is the "optimistic" convention: "our orders were entirely filled when a
  trade occurred at or above the ask price quoted by the agent" — [GLFT §7](https://arxiv.org/abs/1105.3115).
  That rule ignores queue position entirely and therefore overstates fill rates at the touch
  (and, in a real book, overstates P&L by exactly the adverse-selection cost).
- Guéant's own assessment of applicability: A&S-family models "can hardly be applied to most stock
  markets for at least two reasons: (i) the discrete nature of prices (especially in the case of
  stocks with a large tick size), and (ii) the fact that the very nature of the limit order books,
  which are queuing systems with priorities and volumes, is not taken into account"; only Guilbaud &
  Pham's model is "really well suited to stocks" — [Guéant 2017, §1](https://arxiv.org/abs/1605.01862).
- The A&S/GLFT independence assumption is the source of "phantom gains": in the original A&S model
  "price movements are assumed to be completely independent of the arrivals of market orders and the
  LOB dynamics", which produces "preposterous events, such as price decreases after a large buy market
  order" in backtests — [Gašperov & Kostanjčar, arXiv:2207.09951](https://arxiv.org/html/2207.09951v1).
- Latency is a first-order cost, not a rounding error: latency-arbitrage races are "about one per
  minute per symbol for FTSE 100 stocks", the modal race lasts 5–10 **microseconds**, races account
  for ~20% of volume, and "the liquidity provider's realized spread in races is significantly
  negative, whereas it is modestly positive in nonrace liquidity provision" — [Aquilina, Budish & O'Neill, *QJE* 2022](https://doi.org/10.1093/qje/qjab032).
  Also: "about 90% of races are won by an aggressive order as opposed to a cancel attempt",
  and the races tax is ~0.5 bp of volume, ~17% of the cost of liquidity.
  Note the scale: races are won in *microseconds*, so a 5 ms delay is ~3 orders of magnitude longer
  than a competitive reaction time.
- A queue-aware alternative to Poisson fills exists: Cont, Stoikov & Talreja's stochastic LOB model
  gives closed-form transient probabilities of the book (including queue dynamics) —
  [*Operations Research* 58(3):549–563](https://people.orie.cornell.edu/sfs33/publications.html).
  Not a quoting rule, but the right source for a fill model with queues.

### Inferences
- **The 5 ms delay is the dominant model error, and it is asymmetric.** In the model, a quote at
  distance `δ` fills with intensity `Ae^(−kδ)` and the maker can re-quote instantly. Here: (a) an
  order sent at `t` can first be hit at `t + 5 ms`; (b) a cancel at `t` stops protecting you only at
  `t + 5 ms`, so *in-flight* inventory can exceed the intended level. Consequence: the effective
  bound for a controller is not 100 but `100 − (max lots that can fill in 5 ms)`. If a single market
  order can sweep multiple lots, that buffer should be several lots; if the tape trades 1 lot at a
  time, 1–2 lots. **Set the control bound `Q_ctl = 100 − buffer` and the hard rule at `|I| ≥ 100`
  must be "cancel the increasing side and cross", never "quote and hope".**
- Diffusion-based pick-off loss is negligible; jump-based is not. Expected adverse move over 5 ms at
  `σ = 0.3` tick·s^(-1/2) is `σ√(0.005) ≈ 0.02` ticks (invisible), but a jump of 2–5 ticks inside
  5 ms is a full half-spread or more. Therefore the correct protection is *not* a wider `ξ` in the
  GLFT spread (which would cost ~1 tick permanently) but a **state-conditional gate**: detect the
  jump onset (id-gap / clamped-inside-touch, i.e. our `IdGapJump`), pull quotes for ~20–50 ms, and
  only then re-enter. This is the same conclusion as Cartea et al.'s "buffer to cover adverse
  selection" but triggered by an observable event rather than priced continuously.
- Because fills at the touch depend on queue rank, `A` and `k` are *not* structural constants and the
  exponential form must be fitted locally over `δ ∈ {1, 2, 3}` ticks at most. In a wide-tick book
  (spread ≥ 2 ticks) the useful action set is effectively `{join, improve by 1 tick, sit 2+ ticks
  back, cross}`, and the fill model should be a 4-point lookup indexed by (side, distance, queue-ahead)
  rather than a smooth curve. Anything smoother is fitting noise.
- The "no cancel churn" rule from GLFT (`Δt` minimum re-quote interval) is doubly important here: at
  5 ms latency, churn both loses priority and exposes you to being filled on a stale quote you meant
  to cancel. A minimum quote lifetime of order 100 ms–1 s is implied by their argument, and by the
  race data (races resolve in µs; you cannot win them, so do not enter them).
- Under a *step* penalty (our $10,000/s), the theoretically right formulation is a **state-constrained**
  problem, not a penalised one: keep the state in the safe set with probability 1 given the latency
  and fill model, and treat the boundary as absorbing-with-huge-cost. GLFT's hard bound is exactly
  this, and it is the only formulation in this literature that *guarantees* feasibility rather than
  trading it off.

### Portable formula sheet (minimum viable implementation)

Given estimates `σ` (ticks/√s), `A` (fills/s at zero distance), `k` (1/ticks), and a chosen `γ`
(1/ticks), with `Δ` = 1 lot and control bound `Q_ctl`:

1. Inventory-risk slope: `g = sqrt( σ²·γ / (2·k·A) · (1 + γ/k)^(1 + k/γ) )`   [GLFT §4]
2. Base half-spread: `h = (1/γ)·ln(1 + γ/k)`   [GLFT §4; equals A&S's `(1/γ)ln(1+γ/k)`]
3. Reservation price: `r = s − q·g` (optionally `q → q − μ/(γσ²)` for an alpha view)   [GLFT §4, §5.1]
4. Quotes: `bid = r − (h + g/2)`, `ask = r + (h + g/2)`; total spread `ψ = 2h + g`
5. Hard-cap guard: if `q ≥ Q_ctl` post no bid; if `q ≤ −Q_ctl` post no ask   [GLFT Thm 1 / Guéant eq. (2.3)]
6. Boundary taper (refinement, since the linear rule is only valid for small `|q|`): replace step 3 by
   `r = s − g·h(q)` where `h(q) = q·(1 + (|q|/Q_ctl)^p)` with `p ≈ 2`; this reproduces the convex
   blow-up of the exact eigen-solution near the bound (my construction, motivated by
   `f⁰_q ∝ exp(−½√(α/η)q²)` in [GLFT Thm 2](https://arxiv.org/abs/1105.3115))
7. Alpha (spread-skew variant, safer under latency): `δ^ask += E[α]·τ_eff`, `δ^bid −= E[α]·τ_eff`
   [CJ&P Corollary 4]
8. Adverse selection: add `ξ/2` to each half-spread and multiply `g` by `exp(kξ/4)` where `ξ` = mean
   adverse mid move (ticks) in the 100 ms after a fill   [GLFT §5.2]

Quantities that must be estimated online: `σ` (per regime), `A` and `k` (per distance bucket, and
they drift with the market's own spread), `ξ` (per side, if adverse selection is asymmetric),
`μ`/`α` (any drift/view; must be shrunk hard), and the fill-vs-queue relationship if you quote at the
touch. `γ` is a tuning knob, not an estimate.
