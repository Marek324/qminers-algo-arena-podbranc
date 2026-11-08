# Aggressive High-Frequency Trading: Literature Survey (speed budget = 5 ms)

**Framing used throughout.** Every finding is tagged with a speed bucket:

- **(A) Out of reach** — economics depend on winning a race decided in microseconds, on colocation, on cross-venue simultaneity, or on exchange order types (pegged, hidden, IOC-at-multiple-venues, cancel-on-disconnect).
- **(B) Usable at 5–50 ms** — the edge comes from *information content* that decays over ≥100 ms to minutes, not from being first. A 5 ms order round-trip is enough to act on a signal you can already compute.
- **(C) Illegal, or void in a closed single-venue simulator** — spoofing, layering, momentum ignition, pinging-as-a-tactic.

The distinction matters because the empirical literature is unambiguous that (A) is a tournament with a handful of winners, and that the *losers* of (A) are precisely the resting liquidity providers — which is the role a 5 ms participant is forced into.

---

## Latency arbitrage: Budish–Cramton–Shim, and the Aquilina–Budish–O'Neill measurement

### Takeaway
Latency arbitrage is real, frequent and measurable, but the prize per event is tiny (about half a price tick) and the race is decided 5–10 microseconds apart, with >80% of wins going to six firms. The aggregate take is ~0.4 bp of volume — roughly $5bn/year globally. At 5 ms you cannot be the sniper; you can only be the sniped. What *is* transferable is the mechanism: a public signal that mechanically makes a resting quote stale.

### Cited Findings

- Mechanism: continuous-time serial processing means that even *symmetrically observed public information* creates arbitrage rents, because the first message to arrive at each venue wins; the rents harm liquidity provision and induce a never-ending arms race — [Budish, Cramton & Shim, QJE 130(4), 1547–1621, 2015](https://academic.oup.com/qje/article/130/4/1547/1916146), [DOI](https://doi.org/10.1093/qje/qjv027).
- Their empirical anchor is one specific trade — S&P 500 futures vs SPY ETF — and that single pair is worth "approximately $75 million per year" — [as summarised by Aquilina, Budish & O'Neill, QJE 2022, §1](https://academic.oup.com/qje/article/137/1/493/6368348).
- Policy proposal: replace the continuous limit order book with frequent batch auctions (uniform-price double auctions, e.g. every 100 ms), which converts speed competition into price competition — [BCS 2015](https://academic.oup.com/qje/article/130/4/1547/1916146).
- Frequency: the average FTSE 100 symbol has **537 latency-arbitrage races per day**, "about one race per minute per symbol" — [ABO, QJE 2022](https://academic.oup.com/qje/article/137/1/493/6368348).
- Speed: "in the modal race, the winner beats the first loser by just **5–10 microseconds**"; the reported **mean race duration is 78.65 µs** (FTSE 100) — [ABO QJE, Table I / appendix](https://www.fca.org.uk/publication/occasional-papers/occasional-paper-50-appendix.pdf).
- Volume: races account for **about 20% of total daily FTSE 350 trading volume** — [Chicago Booth Fama-Miller summary of ABO](https://www.chicagobooth.edu/research/fama-miller/finance-research/working-papers/2020/quantifying-the-high-frequency-trading-arms-race).
- Concentration: "the top **6 firms** account for over **80%** of all race wins and losses" — [same](https://www.chicagobooth.edu/research/fama-miller/finance-research/working-papers/2020/quantifying-the-high-frequency-trading-arms-race).
- Per-event size: "the average race is worth just a small amount (**about half a price tick**)" — [same](https://www.chicagobooth.edu/research/fama-miller/finance-research/working-papers/2020/quantifying-the-high-frequency-trading-arms-race).
- Aggregate magnitude: a **0.42 bp "latency arbitrage tax"** on volume, ≈ **£60m/year in the UK**, **$4.8bn/year globally in 2018** (range $2.3–8.4bn; **$7bn in 2020**, range $3.1–11.4bn) — [ABO, QJE 2022](https://academic.oup.com/qje/article/137/1/493/6368348), [BIS WP 955](https://www.bis.org/publ/work955.pdf).
- Decomposition of the cost of liquidity: races are **33% of the effective spread** and **31% of price impact**; eliminating latency arbitrage would cut the cost of liquidity by **17%** — [BIS WP 955 abstract](https://www.bis.org/publ/work955.pdf).
- Methodology point that generalises: you need **message data**, not LOB data, because LOB data hides the losers of each race; you cannot see races from the book alone — [ABO NBER WP 29011](https://www.nber.org/papers/w29011).
- Sensitivity: estimates are computed over race windows of **50 µs to 3 ms**, and the aggregate barely moves — [ABO NBER WP, footnote 12](https://www.nber.org/system/files/working_papers/w29011/w29011.pdf). (This is the closest thing in the literature to an answer to "how fast must you be": the *aggregate* is insensitive to the window, but the *identity of the winner* is not.)
- Related FCA work on stale reference prices in UK dark pools puts that narrower arbitrage at "approximately £4.2 million per year" — [cited in ABO QJE 2022, §1](https://academic.oup.com/qje/article/137/1/493/6368348).
- Physical speed floor, for calibration: the fastest commercial microwave path CME (Aurora) → Equinix NY4 is **3.96 ms one-way** in 2020 (4.00 ms in 2013), vs **6.55 ms** on the best fibre; the network "operates within 1% of c-latency end-to-end" — [Bhattacherjee et al., ACM IMC 2020](https://users.cs.duke.edu/~bmm/assets/pubs/BhattacherjeeALMS20.pdf).
- The 2010–2012 step change: a **3 ms** reduction in one-way Chicago→New York latency (7.25–7.95 ms fibre → ~6.65 ms with Spread Networks' new fibre, then microwave), with industry-wide spend "to exceed $500 million" — [arXiv:1302.5966](https://arxiv.org/pdf/1302.5966).

### Inferences
- **Minimum viable speed, empirically:** to be a *systematic winner* you must be within ~10 µs of the fastest participant — i.e. colocation, kernel bypass, FPGA. Being inside 1 ms of the leader is already not enough. This is bucket (A), unambiguously.
- A 5 ms participant is ~500× slower than the modal winning margin and ~3 orders of magnitude slower than the participants who populate 20% of FTSE volume. In a fragmented market, 5 ms is roughly the *one-way inter-venue distance* on the Chicago–NY corridor (3.96 ms) — i.e. you arrive after the race is over.
- Because races are ~half a tick and are won by cancelling/repricing ahead of the stale quote, the *transfer* is from resting liquidity providers to snipers. In a single-venue simulator with 5 ms cancel latency, **we are structurally the liquidity provider in this trade**. The defensive implication (widen/withdraw when a public signal makes our quote stale) is worth more than any offensive latency play.
- ABO's 33%-of-effective-spread number is the single best "how much is at stake" statistic for a market maker deciding how much to spend on speed — and it is also the upper bound on how much adverse selection a slow maker should expect to eat.

### Gaps
- No paper measures latency arbitrage *within a single venue with no competing venues*; the entire measured phenomenon is cross-venue or cross-instrument. There is no published estimate of a "latency tax" for a single-venue continuous book, which is our setting.
- I did not find a peer-reviewed estimate of the **cost side** of the arms race (total industry spend on latency) that is comparable to the $5bn prize; BCS argue the race is socially wasteful but the dollar figure for spend is not in the papers I verified.
- Menkveld & Zoican's exchange-latency model ("Need for Speed? Exchange Latency and Liquidity", RFS 2017) and Foucault, Kozhan & Tham ("Toxic Arbitrage", RFS 2017) are the standard theory references for the *liquidity-supply* side of this trade-off; I did not verify their bibliographic details or numbers in this session and they are not cited above.

---

## Order anticipation, pinging and liquidity detection

### Takeaway
The best available regulatory evidence (FCA, 2016, on UK message-level data) finds **no** systematic anticipation of near-simultaneous cross-venue orders at the millisecond scale — but does find patterns consistent with anticipation at **seconds to tens of seconds**. That is a direct empirical endorsement of bucket (B) and a direct refutation of bucket (A) mattering for order anticipation. Pinging is a *legal* liquidity search that becomes abusive only when the information is used to trade ahead; no study quantifies its P&L.

### Cited Findings
- Definition (regulator's own words): "A 'pinging' order is an immediate-or-cancel order that can be used to search for and access all types of undisplayed liquidity, including dark pools and undisplayed order types at exchanges and ECNs" — SEC 2010 Concept Release on Equity Market Structure, quoted via [SEC 2010 comment file / CRS report](https://www.congress.gov/crs_external_products/R/HTML/R43608.html).
- The legal line: the SEC distinguishes "using tools such as pinging orders as part of a normal search for liquidity with which to trade" from "using such tools to detect and trade in front of large trading interest as part of an 'order anticipation' trading strategy" — [SEC Concept Release, quoted in Goldstein et al., Financial Review 49, 2014](https://fusionmx.babson.edu/__SP_T4/computerized-and-high-frequency-trading.pdf); SEC's own taxonomy of order anticipation vs momentum ignition is in its [Equity Market Structure Literature Review Part II, 2014](https://www.sec.gov/files/marketstructure/research/hft_lit_review_march_2014.pdf).
- Technique as described by a buy-side commenter: sending small test orders at several price levels, cancelling anything not instantly filled, then positioning ahead of the detected large order — "akin to using sonar to locate a whale" — [Better Markets comment letter to the CFTC, 2013](https://comments.cftc.gov/Handlers/PdfHandler.ashx?id=25234).
- **The key empirical result:** "We do not find evidence that HFTs systematically anticipate near-simultaneous marketable orders sent to different trading venues by pure non-HFTs… When moving from very short periods (milliseconds) to longer durations (seconds or tens of seconds), we find patterns consistent with HFTs anticipating the order flow" — [FCA Occasional Paper 16, Aquilina & Ysusi, 2016](https://www.fca.org.uk/publication/occasional-papers/occasional-paper-16.pdf).
- Their explanation for the null at ms scale is structural: all UK venues are physically close, so "it takes microseconds for messages to go from one venue to the other", and the UK routing set-up makes it "more difficult to predict where orders will be routed, compared to the US market" — [FCA OP16 landing page](https://www.fca.org.uk/publications/occasional-papers/occasional-paper-no-16-are-high-frequency-traders-anticipating-order).
- FCA explicitly cautions it cannot distinguish "HFTs reacting more rapidly to new information" from "order-flow anticipation" — same source.
- The earlier literature they replicate: Brogaard (2010) and Hirschey (2013) "find patterns consistent with HFTs anticipating the order flow", with the caveat that "the relative reaction of HFTs seems to be higher in Hirschey's study than in our data… it is possible that HFTs in the US can anticipate the order flow more easily than in the UK" — [FCA OP16](https://www.fca.org.uk/publication/occasional-papers/occasional-paper-16.pdf).
- Regulatory status: order anticipation from public information is not per se manipulation; it becomes actionable when it involves a duty to the counterparty, misappropriation, or deception — [WilmerHale summary of the SEC Concept Release](https://www.wilmerhale.com/-/media/files/shared_content/files/editorial/publication/periodicals/securities_alert_1_20_10.pdf).
- Both sides of the political economy are represented in the comment record: industry argues the practice is a normal search for liquidity, and that "there are some indications that may at least in part challenge this characterization of 'predatory' liquidity detection" — [CRS, HFT: Background, Concerns and Regulatory Developments, 2014](https://www.congress.gov/crs_external_products/R/HTML/R43608.html).

### Inferences
- **Detection statistic for pinging / probing (bucket C as a tactic, but a useful signal as data):** the observable signature is a burst of small IOC orders on one side that all cancel unfilled (or fill trivially), followed within Δt by a marketable order on the *opposite* side from the same or a correlated participant. Computable statistics: (i) share of IOC volume that is never filled, (ii) P(large opposite-side aggressive order within Δt | probe burst), (iii) the ratio of probe size to typical trade size. The FCA result implies the informative Δt is **≥1 s, not milliseconds** — which is why this is bucket (B) at best.
- **A 5–50 ms participant cannot "ping" profitably in a fragmented market**, because the reaction must be faster than the *next* child order of the parent algorithm, and FCA finds that reaction does not systematically happen at ms scale even for colocated firms.
- In our simulator, the honest analogue is not *doing* this but *reading* it: the tape contains `F` (IOC) rows. A cluster of IOC rows that do not trade is the closest thing to a probe, and the fill/no-fill outcome of IOC flow is a legitimate, causal-free signal about the tape's impatience.
- Pinging *as a tactic* is void here: there is no hidden liquidity (the book shows every resting order within `public_depth`), no dark pool, and the tape does not react to our orders, so a probe cannot elicit information.

### Gaps
- **No published P&L estimate for order anticipation.** Neither the SEC nor the FCA quantifies profits; the FCA explicitly declines to conclude on detriment. Anyone quoting a number for "how much order anticipation makes" is extrapolating.
- The US evidence (Brogaard 2010, Hirschey 2013) is directional but I did not verify those papers' specifications or magnitudes; FCA OP16 reports that the US reaction is *larger*, but this is second-hand.
- MiFID II's concrete anti-probing tool is the **message-to-trade ratio** limit under RTS 6 and the MAR Annex II list of manipulation indicators (including "orders that are cancelled within X ms"); I did not verify the exact thresholds in this session.

---

## Momentum ignition and spoofing: mechanism, detection, and measured harm

### Takeaway
Spoofing works through a *measurable causal channel*: displayed depth imbalance moves price. That is why the best detection methods are structural rather than pattern-matching — compare book imbalance immediately before and after a market order, or test whether observed depth behaviour is consistent with the optimal spoofing strategy given the calibrated imbalance→price map. Enforcement has extracted ~$1bn-scale penalties (JPM, ~$920m, 2020), which is the only reliable magnitude estimate.

### Cited Findings
- Definition: "Spoofing is an illegal act of artificially modifying the supply to drive temporarily prices in a given direction for profit" — [Tao, Day, Ling & Drapeau, arXiv:2009.14818](https://arxiv.org/abs/2009.14818).
- Model: a multilevel imbalance statistic is introduced, the spoofer's optimal placement is solved as a non-local optimisation, and conditions are given under which a book *admits* spoofing as a function of market characteristics; the model is calibrated on real **Level 2 TMX data** — [arXiv:2009.14818](https://arxiv.org/abs/2009.14818); published version: [Quantitative Finance 22(8), 1405–1425, 2022](https://doi.org/10.1080/14697688.2022.2059390).
- **Detection statistic (the useful part):** "a successful spoofing strategy requires the execution of a market order. We therefore compare the imbalance ι⁻(t) *before* a market order at time t with the imbalance ι⁺(t) *after* this market order. If the market order is legitimate, the behavior of these two imbal[ances differs]" — i.e. a legitimate taker *consumes* imbalance; a spoofer's taker order follows a book whose imbalance was *manufactured*. They implement real-time monitoring with a **Wasserstein distance** between the empirical distributions of pre/post imbalance — [arXiv:2009.14818](https://arxiv.org/abs/2009.14818).
- A secondary empirical regularity they rely on: spoofing "is more likely to happen at a reasonable distance from the top of the limit order book" — i.e. the depth ladder at levels 2–5 is where the fake size lives — [arXiv:2009.14818](https://arxiv.org/abs/2009.14818).
- Scale of the enforcement problem: "In 2020, JP Morgan settles spoofing lawsuit alleging fraud for about **$920 million**" — [Tao et al., QF 2022, fn. 2](https://doi.org/10.1080/14697688.2022.2059390).
- The canonical fact pattern (the 2010 "flash crash" spoofing case): placing thousands of E-mini orders in one day and modifying/moving them "more than 20 million times before they were cancelled" — [Tao et al., arXiv:2009.14818, §1](https://arxiv.org/abs/2009.14818).
- Taxonomy: the SEC classifies **momentum ignition** as "initiating a series of orders and trades in an attempt to ignite a rapid price move up or down", and **order anticipation** as ascertaining large buyers/sellers and trading ahead of them; both are flagged as directional strategies that "may present serious problems in today's market structure" — [SEC Equity Market Structure Literature Review Part II, 2014](https://www.sec.gov/files/marketstructure/research/hft_lit_review_march_2014.pdf).
- Manipulation law: MAR Article 12 and Commission Delegated Regulation (EU) 2016/522 Annex II list "placing orders… with the intention of cancelling before execution" and layering among the indicators; the US analogue is CEA §4c(a)(5)(C) (Dodd-Frank §747) — [Delegated Regulation (EU) 2016/522, EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0522).

### Inferences
- **Observable signatures in LOB/message data**, in decreasing order of evidential strength:
  1. *Pre/post-imbalance asymmetry* on the aggressor's own trades (Tao et al.). Strongest, because it is model-consistent rather than heuristic.
  2. *Order-to-trade ratio* on one side combined with *fill rate ≈ 0* on that side, at levels ≥2.
  3. *Cancel latency conditional on an opposite-side trade*: depth at levels 2–5 disappearing within τ ms of a print that moved price in the spoofed direction.
  4. *Depth that is never at the touch*: layering specifically avoids queue position risk.
- Magnitude of the price effect being manufactured: the spoofer's profit is bounded by the calibrated imbalance→price elasticity times the size they can execute. The Tao et al. calibration exists precisely because that elasticity is the whole business model — so the *defensive* statistic for us is the same elasticity: how much does multi-level imbalance predict the next mid move?
- Momentum ignition is only profitable if the ignited move attracts *other* participants' momentum algos. It is therefore a strategy about *other people's reactions*, which makes it (C) both legally and mechanically in a closed simulator: our orders do not move the tape, so no ignition is possible. (The tape's own bursts of aggressive flow are still informative as a *signal*, which is a legitimate bucket (B) use of the same statistic.)

### Gaps
- No peer-reviewed estimate of aggregate spoofing profits or of the aggregate price distortion; only case-level enforcement penalties.
- I found no verified study that measures the *cross-sectional* predictive power of multi-level LOB imbalance at the 100 ms–1 s horizon in a single-venue continuous book (which is what would let us size a "defensive" reaction). This is a genuine gap and, for our purposes, the most valuable missing number.
- Cartea, Jaimungal & Wang, "Spoofing and price manipulation in order-driven markets", Applied Mathematical Finance (2020), is cited by Tao et al. as the theory reference for spoofing in an order-driven market; I did not verify it directly.

---

## Stop-loss and liquidation cascades: what a cascade looks like in the flow

### Takeaway
The empirically documented cascade anatomy is: a large *unconditional* sell program → liquidity providers absorb, then withdraw → depth collapses (to <1% of normal) → HFTs flip from absorbing to selling with the trend → "hot potato" churn at near-zero net position → prices rebound when fundamental buyers arrive. The single most useful, published detection statistic is Kirilenko et al.'s *hot-potato ratio*: |Δnet position| / volume on a 5-second moving average.

### Cited Findings
- Trigger: at 2:32 p.m. on 6 May 2010 a "large Fundamental Seller (a mutual fund complex) initiated a program to sell a total of **75,000 E-Mini** contracts", against a backdrop of "negative market sentiment and thinning liquidity" — [CFTC–SEC Staff Report, Findings Regarding the Market Events of May 6, 2010](https://www.cftc.gov/sites/default/files/idc/groups/public/%40otherif/documents/ifdocs/staff-findings050610.pdf).
- Depth: liquidity fell to "less than **1%** of its depth from that morning's level" as the E-mini dropped — [same report](https://www.cftc.gov/sites/default/files/idc/groups/public/%40otherif/documents/ifdocs/staff-findings050610.pdf).
- The HFT response: HFTs "began to quickly buy and then resell contracts to each other – generating a 'hot-potato' volume effect", and net HFT holdings "fluctuated around zero so rapidly that they rarely held more than **3,000 contracts** long or short on that day" — [same report](https://www.cftc.gov/sites/default/files/idc/groups/public/%40otherif/documents/ifdocs/staff-findings050610.pdf).
- Peak churn: "between 2:45:13 and 2:45:27, HFTs traded over **27,000 contracts**, which accounted for about **49 percent** of the total trading volume" — a 14-second window — [same report](https://www.cftc.gov/sites/default/files/idc/groups/public/%40otherif/documents/ifdocs/staff-findings050610.pdf).
- Direction: HFTs "traded with the price trend on May 6 and, on both an absolute and net basis, **removed significant buy liquidity** from the public quoting markets during the downturn" — [same report](https://www.cftc.gov/sites/default/files/idc/groups/public/%40otherif/documents/ifdocs/staff-findings050610.pdf).
- Timing (minute-by-minute): 13:32 program starts; **13:42 HFTs reverse direction and start selling**; 13:45 hot potato / lack of buyers; 13:45:28–13:45:33 a 5-second trading pause; 13:46 fundamental buyers lift prices; **14:08 prices back at the 13:32 level** — i.e. a ~36-minute round trip — [CFTC presentation deck, Kirilenko/Kyle/Samadi/Tuzun](https://cap.columbia.edu/sites/cap.columbia.edu/files/content/Documents/11-1910/The%20Flash%20Crash-%20The%20Impact%20of%20High%20Frequency%20Trading%20on%20an%20Electronic%20Market.pdf).
- Peer-reviewed version, and its careful conclusion: HFTs "did not trigger the Flash Crash, but their responses to the unusually large selling pressure on that day exacerbated market volatility"; their trading pattern "did not change when prices fell" (small inventory, high volume, taking more liquidity than providing) — [Kirilenko, Kyle, Samadi & Tuzun, Journal of Finance 72(3), 967–998, 2017](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12498); [working paper with the inventory dynamics](https://www.cftc.gov/sites/default/files/idc/groups/public/@economicanalysis/documents/file/oce_flashcrash0314.pdf).
- Concentration: 17 HFT firms accounted for "between **40% and 50%** of total dollar volume" on 6 May, roughly in line with prior days — [SEC staff statement on the May 6 flash crash](https://www.sec.gov/newsroom/speeches-statements/spch101310geb-market-participants-may-6-flash-crash).
- **What it looks like ex post in UK equities** (the most directly usable study for us): during "flash events", "by the time price starts to revert, the **depth of the order book has dropped significantly**"; HFTs "take longer than hybrid firms to restore the liquidity they provide at the top of the order book"; and "**HFTs on average make small profits** out of these events… unlike hybrid firms that make a loss", attributed to speed allowing them to "better manage their risks and avoid adverse selection" — [FCA Research Note, How do participants behave during flash events?, 2018](https://www.fca.org.uk/publication/research/research-note-participants-behave-flash-events-evidence.pdf).
- The slow, non-cascading counterpart: deviations of price from the efficient price due to intermediary inventory ("price pressures") average **0.49%** with a **half-life of 0.92 days**, implying a social cost of ~10% of the cost of immediacy — [Hendershott & Menkveld, JFE 114(3), 405–423, 2014](https://doi.org/10.1016/j.jfineco.2014.08.001).

### Inferences
- **Cascade signature, in order of appearance** (all computable from our snapshot+trades feed): (1) aggressive same-side volume / resting depth on that side rises above its rolling quantile; (2) best-quote depth falls in absolute terms while spread widens; (3) a *churn* statistic — |Δ signed inventory of the *market's* takers| relative to volume — goes flat while volume spikes ("hot potato"); (4) the move is largely retraced within minutes.
- Kirilenko's hot-potato ratio is directly transferable: `|net taker position change| / taker volume`, 5-second moving average. Near zero with high volume = intermediaries passing risk, not absorbing it. High ratio = genuine absorption, i.e. the cascade has found a buyer.
- **Cascades are bucket (B), and arguably the single best bucket-B opportunity.** The event lasts tens of seconds to tens of minutes (13:32 → 14:08), so 5 ms latency is irrelevant to capturing it; what matters is recognising the regime within the first second. FCA's finding that HFTs make small *profits* in flash events while hybrids lose is evidence that a fast-but-not-fastest participant with a liquidity-withdrawal rule does fine.
- The FCA result also contains the warning: those HFT profits came from *avoiding adverse selection and managing risk*, not from picking the bottom. The corresponding failure mode for us is buying into a cascade with inventory we cannot mark out.
- The 0.92-day half-life of price pressure means the *slow* mean-reversion trade is not a microstructure trade at all; it is a multi-day inventory-risk trade requiring balance sheet, and is out of scope for a 5 ms bot even though it is "slow".

### Gaps
- The mechanistic stop-loss-cascade models the brief asks about — Farmer & Joshi (2002), "The price dynamics of common trading strategies", JEBO, and the Thurner/Farmer/Kirilenko-style "endogenous shocks in complex systems" line — I could not verify online in this session, and I believe the brief conflates at least two different papers (Kirilenko et al. is the *empirical* flash-crash paper, not a cascade model). Treat any claim about those models' specific results as unverified.
- No study I verified gives a *conditional* estimate: "given the first 200 ms of a cascade, what is the expected P&L of buying at the touch?" That is exactly the number needed to size a bucket-B cascade strategy.
- Circuit-breaker / trading-halt interaction (LULD bands, the 5-second CME pause) is documented in the CFTC–SEC report but I found no study measuring the profitability of trading *into* a halt.

---

## Cross-market lead–lag and ETF arbitrage: the cleanest aggressive signal that survives 5 ms

### Takeaway
This is the one family where the literature gives both a mechanism and a decay speed that sits inside a 5–50 ms budget. The pure BCS ES-vs-SPY trade decays in single-digit milliseconds (bucket A), but the *general* phenomenon — one instrument's price move predicting another's, with decay of tens of ms to seconds in less liquid names — is bucket B. The documented slow counterpart (price pressure, 0.92-day half-life) shows the same mechanism operating at a completely different timescale.

### Cited Findings
- The canonical example and its measurement: correlations between ES futures and SPY "completely break down" at millisecond horizons, creating "obvious mechanical arbitrage opportunities" that are "built into the market design" — [BCS 2015, abstract and §I](https://academic.oup.com/qje/article/130/4/1547/1916146); the duration of ES–SPY arbitrage opportunities is plotted over time in their Figure IV — [author PDF](https://ericbudish.org/files/high_frequency_trading_arms_race.pdf).
- Size of that one trade: ~$75m/year — [as reported by ABO 2022](https://academic.oup.com/qje/article/137/1/493/6368348).
- Cross-venue arbitrage is one of the four strategy families the SEC identifies, and it is explicitly a speed game ("arbitrage") — [SEC Equity Market Structure Literature Review Part II](https://www.sec.gov/files/marketstructure/research/hft_lit_review_march_2014.pdf).
- Cross-venue reaction times are bounded by physics, not software: the CME→NY4 corridor is 3.96 ms one-way, so *any* Chicago→New Jersey lead-lag signal decays over ≥4 ms before it can be acted on, and simultaneously-observed signals across that corridor cannot be arbitraged by a single-site participant at all — [Bhattacherjee et al., IMC 2020](https://users.cs.duke.edu/~bmm/assets/pubs/BhattacherjeeALMS20.pdf).
- HFT cross-market activity is documented at the firm level: the single HFT studied by Menkveld "employs a cross-market strategy as half of its trades materialize on a large incumbent market and the other half on a small, high-growth entrant market", with participation rates of **8.1%** and **64.4%** respectively — [Menkveld, JFM 16(4), 712–740, 2013](https://doi.org/10.1016/j.finmar.2013.06.006).
- Lead-lag at the *slow* end of the same mechanism: intermediary inventory shocks produce price pressure of **0.28% per $100,000** of inventory shock, average transitory deviation **0.49%**, half-life **0.92 days** — [Hendershott & Menkveld 2014](https://doi.org/10.1016/j.jfineco.2014.08.001); [NBER/SSRN version with the $100k figure](https://conference.nber.org/confer/2009/MMf09/Hendershott_Menkveld.pdf).
- HFT order flow in one instrument is correlated with "public information, such as macro news announcements, market-wide price movements, and limit order book imbalances", and "the direction of HFTs' trading predicts price changes over short horizons measured in **seconds**" — [Brogaard, Hendershott & Riordan, RFS 27(8), 2267–2306, 2014](https://academic.oup.com/rfs/article-abstract/27/8/2267/1582754) (see also [author PDF](https://faculty.haas.berkeley.edu/hender/HFT-PD.pdf)).

### Inferences
- **Empirical decay ordering** (from the sources above): ES↔SPY mechanical arb ≈ 1–10 ms (A); intraday cross-instrument/react-to-imbalance signal "measured in seconds" (B, comfortably); inventory price pressure ≈ 1 day (not a fast trade at all).
- The bucket-B version of this trade is: a signal in a *fast/liquid* instrument (or in aggregate order flow) predicting a move in a *slower/less liquid* one, where the slower instrument's quotes take ≫5 ms to adjust. Our single-instrument setting has no second instrument, so the literal trade is unavailable; the *substitute* is the same statistic applied within one book — signed aggressive flow and multi-level imbalance predicting the next mid move over 100 ms–2 s.
- BHR's result is the load-bearing one for us: HFTs' **liquidity-demanding** orders move price in the direction of the permanent change and predict returns "over short horizons measured in seconds", while their **liquidity-supplying** orders are adversely selected. That is the academic statement of "take liquidity when you have a seconds-scale signal; do not expect to profit from resting at the touch."
- Implication for our 5 ms budget: 5 ms is *far* inside a seconds-scale signal's life. Cross-market/lead-lag-style aggressive trading is the one place where our latency is genuinely not the binding constraint.

### Gaps
- I found no verified study that measures the *cross-sectional* distribution of lead-lag decay half-lives by liquidity bucket (the "how fast does information decay in a mid-cap" number). The BCS and ABO papers give the liquid-index case only.
- Aquilina, Foley, O'Neill & Ruf, "Asymmetric arbitrage" (cross-venue arbitrage in UK equities) would be the best source for the medium-liquidity case; not verified here.
- Nothing in the verified set measures lead-lag *within* a single venue between order-flow and price at 10–100 ms. That is the exact measurement we would need to justify an aggressive intra-book signal, and it is a gap worth filling locally (our tape is a controlled experiment for it).

---

## When to cross the spread: optimal taking, and whether takers actually make money

### Takeaway
Theory (Cartea–Jaimungal–Penalva) frames taking as a stopping problem driven by the alpha decay rate versus the half-spread plus impact. Empirics resolve the sign: **aggressive HFT orders are informed and profitable at seconds horizons; passive HFT orders are adversely selected.** The average *trade* is close to a coin flip — Virtu's own CEO put ~51–52% of trades profitable — so the money is in the sizing of the winners and in avoiding the adverse fills, not in hit rate.

### Cited Findings
- Framework: taking decisions are an optimal-stopping problem in which the agent compares expected alpha over the remaining horizon against the half-spread plus market impact, with the solution characterised by an alpha-decay-dependent reservation price — Cartea, Jaimungal & Penalva, *Algorithmic and High-Frequency Trading*, CUP 2015 (standard reference; **bibliographic details not re-verified in this session**).
- **The central empirical result on taker profitability:** "Overall HFTs facilitate price efficiency by trading in the direction of permanent price changes and in the opposite direction of transitory pricing errors… This is done through their **liquidity demanding orders**. In contrast, HFTs' **liquidity supplying orders are adversely selected**. The direction of HFTs' trading predicts price changes over short horizons measured in seconds." — [Brogaard, Hendershott & Riordan, RFS 2014](https://academic.oup.com/rfs/article-abstract/27/8/2267/1582754).
- The mirror image for non-HFTs: "non-HFTs' trade in the opposite direction of permanent price changes and in the direction of transitory pricing errors" — [ECB WP 1602 version](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1602.pdf).
- Hit rate reality: "high-frequency trading firm Virtu's CEO Doug Cifu indicated that around **51–52% of their trades are profitable**" — [ABO, NBER WP 29011, §1](https://www.nber.org/system/files/working_papers/w29011/w29011.pdf).
- Firm-level decomposition, the cleanest available: the HFT market maker "earned a net **€1.55** on the spread but lost **€0.68** on its position" per trade; and the position loss decomposes into "a **€0.45 profit** on positions of less than five seconds, but a **loss of €1.13** on longer duration positions"; 80% of its trades were passive — [Menkveld, JFM 2013](https://doi.org/10.1016/j.finmar.2013.06.006); [Tinbergen TI 11-076 version, abstract](https://papers.tinbergen.nl/11076.pdf); [2012 version with the <5 s / >5 s split](https://c.mql5.com/forextsd/forum/102/high_frequency_trading_and_the_new_market_makers.pdf).
- Interpretation of that decomposition: spread capture per trade (€1.55) dominates the position loss (€0.68) — the firm is a market maker whose *inventory* is a cost centre, and whose very-short-horizon positions (<5 s) are actually profitable, i.e. it has short-horizon alpha on top of the spread.
- Taking more liquidity than providing is the *normal* state for the E-mini HFTs, on normal days as well as on 6 May 2010 — [CFTC–SEC report](https://www.cftc.gov/sites/default/files/idc/groups/public/%40otherif/documents/ifdocs/staff-findings050610.pdf).

### Inferences
- **Does the average taker pay the spread?** Yes, in expectation, absent signal — which is why the hit rate is ~51%. The profitable takers are those whose signal has positive expected drift over the holding period exceeding half-spread + impact. This is exactly the condition we can test offline on the tape.
- **Empirical form of the take rule** (bucket B): take when `E[Δmid over h] > half_spread + expected_impact`, where `h` is the alpha half-life. BHR implies `h` is seconds, not milliseconds — so a 5 ms decision loop is over-engineered for the signal but adequate for the execution.
- Menkveld's <5 s vs >5 s split is the sharpest available calibration for a markout-scored setting like ours: **positions held for seconds have positive expected alpha; positions held for minutes do not.** Our score marks out leftover inventory at the far touch, so the practical rule is "hold for the alpha, then be flat", which is the same economics.
- **Adverse selection is the tax on being slow.** BHR's "supplying orders are adversely selected" is the same phenomenon as ABO's latency tax and as Kirilenko's "removed buy liquidity": the resting order is filled precisely when it is wrong. For us the operative control is a take/quote gate conditioned on flow toxicity and on recent own-fill markouts.

### Gaps
- No verified study isolates the profitability of *pure* taker strategies across a broad cross-section; the evidence is either firm-level (Virtu's disclosure, Menkveld's single firm) or HFT-vs-non-HFT aggregate.
- The exact Cartea–Jaimungal–Penalva chapter/result on optimal taking was not verified; the citation above is the standard one but should be checked before publication.
- I found no source quantifying the *cost* of a 5 ms delay specifically (i.e. slippage from signal to fill at 5 ms in a liquid book). This is measurable on our own data and is the most decision-relevant missing number.

---

## Order-flow toxicity: VPIN and its critique

### Takeaway
The attractive idea — a real-time toxicity meter that tells a market maker when to widen — is empirically weak in its original form. VPIN is mechanically correlated with volume and volatility by construction, did not reach an extreme before the 2010 flash crash (it peaked *after*), and loses its predictive content once you control for current volume and volatility. The *construct* (signed-volume imbalance as a stress indicator) survives; the *specific metric* does not.

### Cited Findings
- The claim being tested: VPIN is "a real-time indicator of order flow toxicity" that "may help signal impending market turmoil, exemplified by historical high readings of the metric prior to the flash crash" — Easley, López de Prado & O'Hara, as summarised in [Andersen & Bondarenko, "VPIN and the Flash Crash", JFM 17, 1–46, 2014](https://doi.org/10.1016/j.finmar.2013.05.005).
- The critique, in their own words: "our empirical investigation of VPIN documents that it is a **poor predictor of short run volatility**, that it did **not reach an all-time high prior, but rather after, the flash crash**" — [A&B 2014, abstract](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1881731).
- The core econometric objection: "the VPIN metric… **by construction** will be correlated with trading volume and return volatility (innovations). Whether VPIN is more strongly correlated with volume or volatility depends on the exact implementation." After controlling for current volume and volatility: "we find **no evidence of incremental predictive power** of VPIN for future volatility" — [A&B, "Reflecting on the VPIN dispute", JFM 17, 53–64, 2014](https://www.sciencedirect.com/science/article/abs/pii/S1386418113000475).
- A sharper identification result: VPIN built with **perfect trade classification** "leads to diametrically opposite results relative to ELO", i.e. the bulk-volume-classification step is doing the work — [same](https://www.sciencedirect.com/science/article/abs/pii/S1386418113000475).
- On the day itself: "both the transaction-VPIN and fixed volume bin VPIN measures rise prior to the crash, but **do not reach extreme values** around the crash" — [A&B 2014](https://doi.org/10.1016/j.finmar.2013.05.005).
- The authors' reply, for balance: A&B "dispute our empirical findings and argue that VPIN essentially does not work. This is incorrect, and is refuted by results in AB and by independent research"; they also argue A&B "attack a methodology we do not advocate" — [Easley, López de Prado & O'Hara, "VPIN and the Flash Crash: A rejoinder", JFM 17, 47–52, 2014](https://www.sciencedirect.com/science/article/abs/pii/S1386418113000293).
- A conciliatory note from the critics: "signed order flow indicators **may** contain useful information for gauging real-time market stress indicators. Nonetheless, we caution against the adoption of any specific metric until its performance has been compared thoroughly to suitable benchmarks" — [A&B 2014, abstract](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1881731).

### Inferences
- **Do not implement VPIN as published** (volume-bucketed, BVC-classified). If a toxicity gate is wanted, the defensible version is a *plain signed-volume imbalance* over a short window, benchmarked against current volume and volatility — which is exactly what A&B say is the residual signal, and is also what our `TakeGate` already approximates via innovation z².
- **Detection statistic that survives the critique:** the *residual* of toxicity on contemporaneous volume and volatility (a regression residual, not a level). A level-based threshold will fire in every busy period and never in a quiet crash.
- The meta-lesson is worth stating for the report: in this literature, almost every "predictive microstructure metric" turns out to be a monotone transform of volume × volatility. Any signal we build should be tested as an increment over that baseline.

### Gaps
- The original Easley–López de Prado–O'Hara VPIN papers (RFS 2012 "Flow Toxicity and Liquidity in a High-Frequency World"; JPM 2011 "The Microstructure of the Flash Crash"; "The Exchange of Flow Toxicity") are cited here only through A&B's and ELO's summaries; I did not verify their bibliographic details.
- No verified source establishes a *better* toxicity metric; the literature is mostly destructive. This is a real gap and the BreachGate/TakeGate design is on its own.

---

## The empirical profitability of HFT: how much, from what, and how concentrated

### Takeaway
HFT is extraordinarily profitable on a risk-adjusted basis but the profits are concentrated in a few firms and are ~entirely compensation for (i) speed rank and (ii) inventory-risk management, not for directional views. Median annualised Sharpe ratios of 4.3–9.35 are reported; per-trade edge is cents; the top decile earns ~4× the median per contract.

### Cited Findings
- Median HFT firm in E-mini S&P 500 over two years: annualised **Sharpe ratio 4.3** and four-factor annualised **alpha 22.02%**; revenues "persistently and disproportionally accumulate to top performing HFTs, consistent with winner-takes-all industry structure" — [Baron, Brogaard, Hagströmer & Kirilenko, JFQA 54(3), 993–1024, 2019](https://doi.org/10.1017/S0022109018001096); [CFTC working-paper version](https://www.cftc.gov/sites/default/files/idc/groups/public/@economicanalysis/documents/file/oce_riskandreturn0414.pdf).
- Dispersion: "profit per contract in the top decile of firms is **$2.01**, compared with **$0.46** for the median" — [CFTC version, 2014](https://www.cftc.gov/sites/default/files/idc/groups/public/@economicanalysis/documents/file/oce_riskandreturn0414.pdf).
- By style (2014 version): Passive HFTs Sharpe **5.85** (lowest annualised return, 23.13%); Mixed 5.26; Aggressive 4.29 — same source. (An earlier draft reports much higher numbers, 8.5–10.5, so treat style-level Sharpe numbers as unstable across drafts.)
- The single-firm benchmark: an annualised gross **Sharpe ratio of 9.35**, average daily excess return **5.86 bp** with **9.69 bp** standard deviation, on maximum capital of **€2.052m per stock** — [Menkveld, JFM 2013 / Tinbergen version](https://papers.tinbergen.nl/11076.pdf); Menkveld stresses "performance is very sensitive to cost of capital assumptions" — [abstract](https://doi.org/10.1016/j.finmar.2013.06.006).
- **Speed causes the performance, not just correlates with it:** "differences in *relative* latency account for large differences in HFTs' trading performance. HFT firms that improve their latency rank due to colocation upgrades see improved trading performance", through both a "short-lived information channel" and a "risk management channel" — [Baron, Brogaard, Hagströmer & Kirilenko, JFQA 2019 abstract](https://doi.org/10.1017/S0022109018001096).
- Price-discovery contribution: HFT demanding orders trade with permanent price changes and against transitory errors; supplying orders are adversely selected — [BHR, RFS 2014](https://academic.oup.com/rfs/article-abstract/27/8/2267/1582754).
- Concentration at the extreme: in the UK races, the top **6 firms** take >80% of wins and losses — [ABO / Chicago Booth summary](https://www.chicagobooth.edu/research/fama-miller/finance-research/working-papers/2020/quantifying-the-high-frequency-trading-arms-race).
- **PnL decomposition (the only clean one I found):** spread component **+€1.55/trade**, positioning component **−€0.68/trade**; 80% of trades passive; and the positioning loss flips sign by holding period (+€0.45 if held <5 s, −€1.13 if longer) — [Menkveld 2013](https://papers.tinbergen.nl/11076.pdf), [2012 version](https://c.mql5.com/forextsd/forum/102/high_frequency_trading_and_the_new_market_makers.pdf).
- Rebates are a real but bounded component of the decomposition: US maker-taker exchanges charge ~30 cps to take and pay ~29 cps to make (NASDAQ/ARCA), with inverted venues paying the taker (e.g. −15 cps fee / +18 cps rebate on BATS Y) — [indicative fee table reproduced in a peer-reviewed study of maker-taker subsidies](https://www.sciencedirect.com/science/article/pii/S0378426624002449).

### Inferences
- **The decomposition that matters for us:** Menkveld's firm earns the spread and *loses* on inventory. Our scoring function (`cash + markout(I) − penalty`) is exactly a spread-plus-inventory decomposition, so the mapping is direct: our equivalent of "spread capture" is passive fills at the touch, and our equivalent of "positioning loss" is the markout on leftover inventory.
- Sharpe 9.35 is *gross* and capital-assumption-sensitive. Any claim that "HFT earns Sharpe 9" should be paired with the caveat, or it misleads.
- **Winner-takes-all + speed-causes-performance ⇒ the industry-average number is not achievable by a marginal entrant.** The correct read of Baron et al. is that the *rank* of your latency determines your revenue, which is precisely why a 5 ms participant should not plan to compete in the strategies whose revenue is latency-ranked (top-of-book market making, sniping).
- Rebates are worth ~29 cps/share = $0.00029/share. On a $50 stock that is 0.058 bp per side. That is roughly an order of magnitude smaller than a typical quoted spread, i.e. **rebates cannot rescue a strategy whose spread capture is negative** — they are a modifier, not a business model, at our level.

### Gaps
- No verified study decomposes HFT PnL into spread / directional / rebate components across a broad sample; Menkveld's single firm is the only clean decomposition, and it is one firm in 2006–2007 European equities.
- Post-2015 profitability of HFT (the period when the latency tax was measured at 0.42 bp) is not covered by any of the verified papers; Baron et al. and Menkveld are both pre-2015 samples.
- No verified evidence on how HFT profitability changed as tick sizes and access fees were reformed (2016–2025).

---

## Market-structure constraints: Reg NMS, MiFID II, tick regimes, and the maker/taker trade-off

### Takeaway
Fee schedules and tick sizes are the parameters that decide whether aggressive or passive is the profitable side — and they are being actively changed. The 2024 US reform cut the access-fee cap from 30 to 10 mils and shrank tick sizes; MiFID II's double volume cap was replaced by a single volume cap in 2025; the tick-size pilot showed that *widening* ticks raised spreads, raised volatility, and lowered price efficiency. For our simulator (no fees, one venue, integer ticks) the entire fee debate collapses, but the tick-size results do not: they are about the discreteness of the price grid, which we have.

### Cited Findings
- US access fees: Rule 610(c) as adopted in 2005 capped fees at **30 mils ($0.003/share)** for protected quotes ≥$1; the Commission **adopted a 10 mil cap** and 0.1% for sub-$1 stocks in the October 2024 final rule — [Reg NMS final rule, Release 34-101070, 89 FR (8 Oct 2024)](https://www.govinfo.gov/content/pkg/FR-2024-10-08/pdf/2024-21867.pdf), [SEC PDF](https://www.sec.gov/files/rules/final/2024/34-101070.pdf).
- The economics of the cap: "the predominant exchange fee structure is maker-taker, in which an exchange charges a fee to liquidity takers and pays a rebate to liquidity providers, and the rebate is typically funded through the access fee"; lowering the cap "would likely reduce rebates" and liquidity makers "would in turn raise the price of liquidity" — [same, and the DC Circuit's description of the SEC's reasoning](https://www.govinfo.gov/content/pkg/USCOURTS-caDC-24-01350/pdf/USCOURTS-caDC-24-01350-0.pdf).
- Scale: access fees "totaled over half a billion dollars in 2022 with other estimates as high as $2 billion annually" — [peer-reviewed study of maker-taker subsidies](https://www.sciencedirect.com/science/article/pii/S0378426624002449).
- Inverted (taker-maker) venues exist and price the *opposite* way: EDGA −2 cps take fee / +5 cps make fee; BATS Y −15 / +18; BX −17 / +19 — [same table](https://www.sciencedirect.com/science/article/pii/S0378426624002449). So "aggressive is subsidised" is a real, venue-specific regime, not a curiosity.
- Rule 611 (Order Protection Rule) is the reason cross-venue sniping exists at all: it forbids executing at a worse price than a protected quote displayed elsewhere, which mechanically links venues and creates the stale-quote race — [SEC/EMSAC description, and the 2024 final rule's discussion of Rule 611 and 610 together](https://www.govinfo.gov/content/pkg/FR-2024-10-08/pdf/2024-21867.pdf).
- **Tick-size pilot (US, Oct 2016 – Sep 2018), the cleanest natural experiment:** tick from $0.01 to $0.05 for ~1,200 randomly chosen small caps in three test groups vs a ~1,200-stock control. Result: "market quality deteriorates for stocks in the Test Groups… an increase in spreads and volatility and a decrease in price efficiency"; displayed depth at the best quotes rises, but "this increase in depth does not appear to lower the transaction costs of large orders, which also increase"; effects are larger for stocks whose pre-pilot spread was below $0.05 — [Hu, Hughes, Ritter, Vegella & Zhang, SEC DERA white paper, 2018](https://www.sec.gov/files/marketstructure/research/dera_wp_tick_size-market_quality.pdf).
- Price-level consequence: a stock-price decrease "between **1.75% and 3.2%** for small spread stocks affected by the larger tick size relative to a control group" — [Albuquerque, Song & Yao, "The price effects of liquidity shocks: A study of the SEC's tick size experiment", JFE, 2021](https://www.sciencedirect.com/science/article/pii/S0304405X20301884).
- But the announcement itself moved nothing: "the announcement of the assignment of stocks to the Test Groups and the Control Group did not generate significant abnormal returns" — [Pachare & Rainer, SEC DERA, 2018](https://www.sec.gov/files/marketstructure/research/dera_wp_does_the_tick_size_affect_stock_prices.pdf).
- MiFID II dark-trading limits: the double volume cap restricted use of the reference-price and negotiated-transaction waivers, with suspension triggered when an instrument exceeded **4% on a single venue or 8% across venues** (operational thresholds **3.75% / 7.75%**), calculated on a rolling 12-month basis — [ESMA DVC page](https://www.esma.europa.eu/double-volume-cap-mechanism); a worked example of a suspension round (337 instruments suspended as of 8 June 2020) is in [ESMA's DVC press release](https://www.esma.europa.eu/press-news/esma-news/mifid-ii-esma-issues-latest-double-volume-cap-data-23).
- The DVC is now history: "the MiFIR review replaces the double volume cap (DVC) with a single volume cap (SVC), only relating to the reference price waiver", applied from 18 months after entry into force, first publication **9 October 2025** — [ESMA](https://www.esma.europa.eu/double-volume-cap-mechanism); the review is Regulation (EU) 2024/791 — [ESMA public statement](https://www.esma.europa.eu/sites/default/files/2024-03/ESMA74-2134169708-7163_Public_statement_on_specific_revised_MiFIR_provisions.pdf).
- Evidence that the caps 'worked' only weakly: "neither a recent ban on dark pool trading, nor its suspension, significantly affected the transaction costs of investors in the UK equity market", though investors *can* reduce execution costs by choosing less pre-trade-transparent venues — [FCA Occasional Paper 60, Neumeier, Gozluklu, Hoffmann, O'Neill & Suntheim, 2021](https://www.fca.org.uk/publications/occasional-papers-fca-research/occasional-paper-no-60-banning-dark-pools-venue-selection-and-investor-trading-costs).

### Inferences
- **The maker/taker trade-off is a pure function of the fee schedule.** With a 30 cps take fee and a 29 cps rebate, the effective spread a taker pays is ~59 cps wider than in a zero-fee world; on a one-cent-spread stock that is the entire economics. On an inverted venue the sign flips. Therefore any claim of the form "aggressive trading is/isn't profitable" is meaningless without stating the fee regime — and in *our* regime (no fees, spread only) the aggressive trade needs to clear exactly one half-spread, which is the minimum possible hurdle.
- **Tick discreteness is a first-order effect, and it is the one market-structure finding that transfers to us.** Wider ticks ⇒ wider spreads, more depth, less price efficiency. Our tape has integer ticks and a spread of a few ticks, which is the "wide tick" regime; the literature says expect depth to sit at fewer price points and expect less informative prices per level. That is consistent with our observation that the tape's book has coarse, lumpy levels.
- The US 2024 reform direction (smaller ticks + lower fees) is the opposite of the tick pilot, and the literature implies it should compress spreads and *reduce* the value of queue position — which further weakens bucket-A strategies and strengthens the case that edge must come from information, not from queue rent.

### Gaps
- MiFID II's **tick-size regime (RTS 11)** — the A/B/C liquidity-band table with the double-step for large-tick stocks — I did not verify; it is the European analogue of the tick pilot and would be the natural next reference.
- No verified study on how the 2024 US access-fee cut (30 → 10 mils) has affected aggressive-vs-passive profitability; the rule is too recent for peer-reviewed evidence.
- MiFID II RTS 6's message-to-trade-ratio limits and the "algo testing/change-control" obligations are relevant to any HFT strategy but were not verified here.

---

## Speed budget: the explicit (A) / (B) / (C) classification

### Takeaway
Only two families in this literature have a documented edge that (i) survives 5–50 ms and (ii) is not a speed tournament: **reacting to multi-second order-flow/imbalance information** (BHR's "predicts price changes over short horizons measured in seconds") and **trading the cascade/regime-transition** (Kirilenko's 36-minute event, FCA's flash-event profits). Everything whose revenue is *ranked by latency* is out of reach, and everything whose revenue depends on *deceiving other participants* is both illegal and void here.

### Cited Findings

| Strategy | Bucket | Required speed (empirical) | Reported magnitude | Detection statistic |
|---|---|---|---|---|
| Latency arbitrage / sniping | **A** | winner beats first loser by **5–10 µs**; >80% of wins to 6 firms | 0.42 bp tax; $5bn/yr global; ~half a tick per race | time between winner and first loser in a race; % of volume in races (from **message** data, not LOB) — [ABO](https://academic.oup.com/qje/article/137/1/493/6368348) |
| ES–SPY / index arb | **A** | sub-ms; 3.96 ms is the physics floor CME→NY4 | ~$75m/yr for that one pair | duration of violation; correlation breakdown at ms horizon — [BCS](https://academic.oup.com/qje/article/130/4/1547/1916146), [IMC 2020](https://users.cs.duke.edu/~bmm/assets/pubs/BhattacherjeeALMS20.pdf) |
| Top-of-book market making / queue rent | **A** | colocation; latency rank determines revenue | spread +€1.55/trade, position −€0.68 | relative latency rank vs revenue — [Baron et al.](https://doi.org/10.1017/S0022109018001096), [Menkveld](https://doi.org/10.1016/j.finmar.2013.06.006) |
| Order anticipation at ms scale | **A (and empirically absent)** | FCA finds **no** systematic ms-scale anticipation | none published | P(opposite-side aggressive order within Δt \| probe burst), Δt in ms — [FCA OP16](https://www.fca.org.uk/publication/occasional-papers/occasional-paper-16.pdf) |
| Order anticipation at seconds scale | **B** | seconds–tens of seconds | not quantified | same statistic at Δt ≥ 1 s — [FCA OP16](https://www.fca.org.uk/publication/occasional-papers/occasional-paper-16.pdf) |
| Aggressive taking on short-horizon signal | **B** | signal horizon **seconds** | HFT demanding orders profitable; supplying adversely selected; 51–52% hit rate | PnL (markout) of aggressive vs passive fills — [BHR](https://academic.oup.com/rfs/article-abstract/27/8/2267/1582754), [ABO](https://www.nber.org/system/files/working_papers/w29011/w29011.pdf) |
| Cascade / flash-event participation | **B** | event lasts ~36 min; onset recognition ~1 s | HFTs make small profits, hybrids lose | aggressive volume/depth ratio; depth collapse; **hot-potato ratio** \|Δnet\|/volume on 5 s MA — [CFTC–SEC](https://www.cftc.gov/sites/default/files/idc/groups/public/%40otherif/documents/ifdocs/staff-findings050610.pdf), [FCA 2018](https://www.fca.org.uk/publication/research/research-note-participants-behave-flash-events-evidence.pdf) |
| Cross-instrument lead–lag, less-liquid leg | **B** | decay tens of ms – seconds | not quantified per instrument | cross-correlation / Hayashi–Yoshida at lag τ; decay half-life — [BHR](https://academic.oup.com/rfs/article-abstract/27/8/2267/1582754) |
| Spoofing / layering | **C** | ms (must cancel before execution) | JPM $920m penalty | pre- vs post-trade imbalance (Wasserstein distance) — [Tao et al.](https://arxiv.org/abs/2009.14818) |
| Momentum ignition | **C** | sub-second bursts | none published | aggressive burst + reversal fraction — [SEC lit review](https://www.sec.gov/files/marketstructure/research/hft_lit_review_march_2014.pdf) |
| Pinging as a tactic | **C** (legal as liquidity search) | ms, cross-venue | none published | unfilled-IOC share + follow-on opposite order — [SEC 2010](https://www.congress.gov/crs_external_products/R/HTML/R43608.html) |

Additional cited facts supporting the table:
- The physics budget: a *single* CME→NY4 hop is 3.96 ms one-way, and the industry operates ~20 parallel microwave networks in that corridor — [IMC 2020](https://users.cs.duke.edu/~bmm/assets/pubs/BhattacherjeeALMS20.pdf). So a 5 ms participant in a fragmented US setting has, at best, one inter-venue hop of budget.
- The information-horizon measurement technique: ABO compute the horizon over which races occur from the distribution of time between an outbound partial fill and the subsequent outbound message, finding the spike start at the microsecond level (histogram truncated at ±500 µs) — [FCA OP50 Internet Appendix](https://www.fca.org.uk/publication/occasional-papers/occasional-paper-50-appendix.pdf).

### Inferences
- **The boundary between A and B is roughly 1 ms, and it is set by the winner–loser gap, not by the event duration.** ABO's own sensitivity analysis (race windows 50 µs–3 ms) shows the *aggregate* estimate is stable — but that is because the window only changes which events you count, not who wins them. Winning requires being inside ~10 µs.
- Because ~20% of volume is decided in races, and >80% of race wins belong to 6 firms, the residual for a slow participant is: be liquidity in the non-race 80%, and be *aggressive* only where the signal horizon is seconds. That is a coherent, literature-supported strategy posture, and it is the one our tooling (TakeGate, BreachGate, FlowReversion, IdGapJump) already assumes.
- In a **closed single-venue simulator**: bucket (A) ceases to exist in its pure form (no second venue to race to), and bucket (C) ceases to have any causal effect (the tape does not react). What remains is a single-venue version of (B): predict the tape's next move from its own order flow, and manage the adverse selection that our 5 ms cancel latency imposes. The 5 ms cancel delay should be treated as a *structural* adverse-selection cost — the exact analogue of the "cancel loses the race" phenomenon ABO measure — and budgeted for, not fought.
- **Illegality is not the binding constraint here; physics is.** The tactics in (C) are unavailable to us because the tape is exogenous and our orders do not move it. But the same *statistics* that detect them in real data (pre/post-imbalance asymmetry, order-to-trade ratio, burst-then-reversal) are legitimate and potentially useful as *signals about the tape*.

### Gaps
- No verified source gives the profitability of a *deliberately slow* (5–50 ms) aggressive strategy. The closest is FCA's flash-event result (HFTs profit, hybrids lose) and BHR's seconds-horizon predictability; neither is a direct measurement at our latency.
- The literature has essentially no evidence on single-venue, zero-fee, integer-tick markets (i.e. our exact setting). Every magnitude above is from a fragmented, fee-bearing, multi-venue market. The magnitudes should be treated as upper bounds on what is *contested*, not as forecasts of what is *available*.
- I did not systematically survey arXiv q-fin.TR beyond the two papers verified here (2009.14818, 1302.5966); a broader sweep might surface recent ML-based spoofing/cascade detectors that postdate the journals covered.

---

## Cross-cutting notes for the report writer

1. **The single best "size of the prize" number** for aggressive speed games is ABO's 0.42 bp / $5bn-per-year, with 33% of the effective spread attributable to races. Use it instead of the frequently-misquoted BCS figure: BCS's own quantified estimate is one instrument pair at ~$75m/year, and ABO is explicit that this was the limitation BCS could not overcome.
2. **The single best "speed is not the edge" number** is FCA OP16: no systematic ms-scale order anticipation, but patterns at seconds. This is a regulator, using message data, finding the opposite of the popular narrative, and it directly supports a 5–50 ms design.
3. **The single best "what a fast-but-not-fastest participant earns" evidence** is the FCA 2018 flash-event note: HFTs make small profits during flash events while hybrid firms lose, because they manage risk and avoid adverse selection better — not because they are first.
4. **Every profitability number in this literature is gross of the cost of speed** (colocation, microwave, FPGA, engineering). Menkveld's Sharpe 9.35 is explicitly sensitive to the capital assumption. Present these as gross returns to a *rank*, not as achievable returns to an entrant.
5. **Attribution caution:** the brief's "Thurner/Farmer/Kirilenko, Endogenous shocks in complex systems" appears to conflate the empirical flash-crash paper (Kirilenko, Kyle, Samadi & Tuzun, JF 2017) with a separate complex-systems modelling line (Farmer & Joshi 2002 on stop-loss strategies; Thurner/Farmer/Gehl on endogenous shocks). I verified only the former. Do not cite the model papers' results without checking them.
