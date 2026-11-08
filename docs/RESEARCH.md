# Research shortlist

Papers that are relevant to our market (see [MARKET.md](MARKET.md)), ranked by how well they fit it. Collected 2026-11-07 from arXiv and related sources. Most entries are based on abstracts and summaries, not full reads, so open the paper before relying on a number.

## What our market rewards

The constraints that decide which papers matter:

- **No fees.** A passive (maker) fill earns the spread and an aggressive (taker) fill pays it. Market making is the natural strategy here.
- **Full order-by-order book with an `owner` field.** We can compute our exact queue position, and we can see individual market orders (sizes, ids, how long they live). Most papers have to estimate these.
- **5 ms latency, and cancels can lose the race.** Stale quotes get picked off, so adverse selection is the main risk.
- **±100 inventory limit with a $10k/s penalty, and leftover inventory is sold at markout.** Inventory has to be skewed back toward flat, which is exactly what the classic market-making models do.
- **Pure Python, called on every event, no saved files.** Linear and logistic models with hard-coded coefficients are fine. Deep nets and RL agents are not practical to run per event, and we have nowhere to load trained weights from.
- **One instrument.** Cross-asset signals don't apply.
- **The real rounds look nothing like `sample.csv`.** Prefer robust, low-parameter ideas over fitted ones. `round1` comes from the same engine as `sample_v2`, with a hidden fair price that the mid chases, a spread of 1 tick 93% of the time, and pure-noise IOC sides. Generator and parameters: [DATA_ANALYSIS.md](DATA_ANALYSIS.md).

## Tier 1: build on these

### 1. Order-book imbalance and microprice: the fair-value signal
- **Gould & Bonart, "Queue Imbalance as a One-Tick-Ahead Price Predictor in a Limit Order Book"** ([arXiv:1512.03492](https://arxiv.org/abs/1512.03492)). A logistic regression of the next mid move on `(bid_qty − ask_qty)/(bid_qty + ask_qty)` at the touch. The effect is strongest for **large-tick** instruments, where the spread is usually 1 tick, which looks like our case.
- **Stoikov, "The Micro-Price"** ([SSRN 2970694](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694)). Fair price = mid + an adjustment that depends on imbalance and spread, estimated from historical top-of-book transitions. A drop-in replacement for mid as the price we quote around.
- **Blakely, "High resolution microprice estimates from LOB data…"** ([arXiv:2411.13594](https://arxiv.org/abs/2411.13594), 2024). Adds a correction to the microprice from deeper levels, using L3 data like ours.
- **Rola, "Boltzmann Price"** ([arXiv:2507.09734](https://arxiv.org/abs/2507.09734), 2025). A maximum-entropy alternative to the microprice built from top-of-book imbalance. Cheap to compute.

**For us:** compute imbalance on every event (O(1) if we maintain it incrementally), then quote around the microprice instead of the mid.

### 2. Order flow imbalance (OFI): a short-horizon direction signal
- **Cont, Kukanov & Stoikov, "The Price Impact of Order Book Events"** ([arXiv:1011.6402](https://arxiv.org/abs/1011.6402)). Over short intervals, price changes are roughly linear in OFI (net change in bid-queue size minus net change in ask-queue size at the touch), with a slope of about 1/depth.
- **Cont, Cucuringu & Zhang, "Cross-Impact of Order Flow Imbalance in Equity Markets"** ([arXiv:2112.13213](https://arxiv.org/abs/2112.13213)). Multi-level OFI (combined across levels with PCA) explains more than touch-only OFI. Ignore the cross-asset part.
- **Kolm, Turiel & Westray, "Deep order flow imbalance"** (Mathematical Finance 2023). Models fed *order-flow* features beat models fed raw book levels, and the useful horizon is about 2 price changes. Takeaway: features should be flows, not snapshots. The deep-net part isn't usable for us.

**For us:** OFI over the last N events or the last few hundred ms, at levels 1–3, sets how far we skew our quotes. This is a better version of what `trend_follower.py` does with raw trades.

### 3. Inventory-aware quoting: where to put the bid and ask
- **Avellaneda & Stoikov, "High-frequency trading in a limit order book"** (Quantitative Finance 2008). Reservation price = fair − γ·σ²·q·(T−t), and quotes go symmetrically around it.
- **Guéant, Lehalle & Fernandez-Tapia, "Dealing with the Inventory Risk"** ([arXiv:1105.3115](https://arxiv.org/abs/1105.3115)). Closed-form approximations of the optimal quotes under a hard inventory cap (the "GLFT formulas"). They don't need a time horizon, which fits a replay of unknown length.

**For us:** skew both quotes against inventory (long → lower both quotes so we sell more easily), and stop quoting the side that would push past a cap well inside ±100. Because tick sizes are integers and the spread is often 1 tick, in practice this means "which side do we quote at the touch, how much, or step back one tick".

### 4. Adverse selection and queue position: when to quote and when to pull
- **Albers, Cucuringu, Howison & Shestopaloff, "The Market Maker's Dilemma: Navigating the Fill Probability vs. Post-Fill Returns Trade-Off"** ([arXiv:2502.18625](https://arxiv.org/abs/2502.18625), 2025). The orders most likely to fill have the worst returns after the fill. A maker order **at the front of the queue on the side the imbalance favours** fills often and makes money, and the edge fades further back in the queue. Taking liquidity in the imbalance direction lost money after the spread. Also: "reversals", where imbalance points the wrong way, are profitable for makers.
- **Moallemi & Yuan, "A Model for Queue Position Valuation in a Limit Order Book"** (Columbia Business School working paper, 2016; not on arXiv). A place in the queue is worth about as much as the spread. Orders further back face more adverse selection.
- **Garriott, van Kervel & Zoican, "Queuing and inventories in limit order markets"** (Journal of Financial Markets 2025). Queue position drives both adverse selection and how easily you can unload inventory.

**For us:** this is our edge. The book's `owner` and per-order data tell us our exact queue position. Rules to try: (a) keep orders that are early in the queue rather than cancelling and re-posting, since `modify_order` shrinks an order without losing its place; (b) pull or don't post on the side that imbalance/OFI says is about to be run over; (c) avoid crossing the spread unless we are flattening inventory.

### 5. Latency: surviving 5 ms
- **"Optimal Market Making in the Presence of Latency"** ([arXiv:1806.05849](https://arxiv.org/abs/1806.05849)). The market maker's losses grow with latency when conditions turn against them. Quotes should be wider or more cautious when expected volatility over the latency window is high.
- **"Integrating Tick-level Data and Periodical Signal for High-frequency Market Making"** ([arXiv:2306.17179](https://arxiv.org/abs/2306.17179)). Models cancel delay explicitly: a cancel only works if no fill happens first, which is our rule.
- **arXiv:2505.12465** (2025). An RL market-making paper that argues ignoring latency causes unintended cancels and fills. Worth reading for the latency modelling, not the RL.

**For us:** decide using a forecast of the book *5 ms ahead*, not the current book. Assume any resting quote may fill before its cancel lands, and size quotes so that the worst case still stays inside the inventory cap.

## Tier 2: worth an experiment

- **ClusterLOB** (Zhang, Cucuringu, Shestopaloff, Zohren, [arXiv:2504.20349](https://arxiv.org/abs/2504.20349), 2025). Clusters order-by-order events into directional, opportunistic and market-making participants, and uses each cluster's OFI as a separate signal. We have order ids, sizes and lifetimes, so a hand-made version is doable, e.g. splitting OFI by order size or by how long orders live. Counterpoint: **"Order-Flow Filtration and Directional Association with Short-Horizon Returns"** ([arXiv:2507.22712](https://arxiv.org/abs/2507.22712), 2025) found that filtering orders by lifetime or modification did **not** consistently improve the imbalance signal. Treat this as a test to run, not an assumption.
- **Zhang, Lim & Zohren, "Deep Learning for Market by Order Data"** ([arXiv:2102.08811](https://arxiv.org/abs/2102.08811)). Order-by-order data adds information on top of aggregated book levels. This supports using order-level features. The model itself is too heavy for us.
- **Hawkes-process order flow.** Kumar, "Deep Hawkes Process for High-Frequency Market Making" ([arXiv:2109.15110](https://arxiv.org/abs/2109.15110)), and Jain et al., impulse-control market making in a Hawkes book ([arXiv:2510.26438](https://arxiv.org/abs/2510.26438), 2025). Main idea: trades cluster, so a burst of aggressive orders on one side means more are coming. Widen or pull quotes on that side. A cheap version is an exponentially decayed count of recent aggressive volume per side.

## Tier 3: read for ideas, don't build

- **Deep LOB price models.** TLOB ([arXiv:2502.15757](https://arxiv.org/abs/2502.15757), 2025) and the LOBCAST benchmark ([arXiv:2308.01915](https://arxiv.org/abs/2308.01915)). Too heavy to run per event in Python, they need trained weights we can't load, and LOBCAST itself reports that results transfer poorly to new data.
- **RL market making.** Spooner et al. ([arXiv:1804.04216](https://arxiv.org/abs/1804.04216)), Zimmer & Valle Costa ([arXiv:2509.12456](https://arxiv.org/abs/2509.12456), 2025) and Cheridito & Weiss, multi-level market making ([arXiv:2608.18195](https://arxiv.org/abs/2608.18195)). The learned policies need training on data like the real rounds, which we don't have. The reward designs (inventory penalty plus spread capture) are still useful for choosing our own parameters.

## Findings so far (sample.csv only; it is synthetic and unlike the real rounds)

Signal quality on any tape now comes from the `analysis` package (`signals` section); the script used here has been removed.

- **Touch imbalance is weak on the sample.** P(next mid move is up) only goes from 0.47 to 0.55 across deciles.
- **Decayed OFI (500 ms) is strong.** P(up) goes from 0.16 to 0.87 across deciles.
- **A strong signal does not help passive quotes.** An earlier pure market-making version (since removed; the maker leg of `adaptive_mm.py` replaces it) was allowed to improve inside the touch on a strong signal. Fills went from about 70 to over 2,000 and it lost up to −33k. The improved quote is filled first by the aggressive orders that then move the price against it, which is the Market Maker's Dilemma in practice. Used only to back quotes off, every setting landed between −1.5k and +0.3k, all within noise. That is why the maker leg never quotes inside the touch.
- **The pull filter almost never triggered** at `PULL_RATIO = 1.0`.
- **Takeaway:** on this tape the edge comes from *taking* in the direction of the predicted move (the trend follower makes +1,878), not from passive quoting. Re-check all of this on real round data before deciding.
- **`strategies/round1/adaptive_mm.py`** (first called merged.py) was tuned with `scripts/sweep.py` on the two halves of the sample (with a split script, since removed). Results:
  - Taker leg alone: about +10k at a 5-lot target. Maker leg alone: about 0. The maker adds nothing on top of the taker.
  - PnL grows roughly linearly with the target position: +12k at 5 lots, +40k at 20, +115k at 50, +175k at 80. That is what you'd expect from a trending synthetic tape, and it is **not** evidence for real rounds.
  - The imbalance term hurts as a taker signal (`IMB_WEIGHT` 3 → losses), so it is off.
  - A 1 s OFI decay beats 0.1 s, 0.25 s, 2 s and 5 s.
  - With the defaults (50-lot target, 50-lot IOCs, 1 s decay) the full sample makes +117k with no penalty, while holding about ±47 lots half the time.

## Overfitting check for adaptive_mm.py (2026-11-07)

**Ceiling.** A perfect-foresight script (since removed; `analysis` has a `max_pnl` section that also covers making) computed a perfect-foresight taker that buys and sells only at the touch, with no latency, keeping |inventory| ≤ P.

| Tape | Taker ceiling P=1 | Taker ceiling P=50 | adaptive_mm.py | trend_follower |
|---|---|---|---|---|
| sample (full) | 5,470 | 256,056 | 117,324 (46% of P=50 ceiling) | 1,878 (34% of P=1 ceiling) |
| sample half 1 / half 2 | 2,686 / 2,781 | 125,541 / 130,324 | 53,640 / 63,572 | — |

At 1 lot, adaptive_mm.py makes 2,190 (40% of its ceiling), close to the trend follower. The big PnL comes from holding 50 lots of the same edge, not from finding something the trend follower misses. The two halves agree, and the parameter sweeps show a wide plateau, so the parameters are **not** knife-edge fits to the sample.

**Control tapes.** `scripts/make_null_tape.py` builds tapes shaped like the sample: same spread, similar volatility (sum |Δmid| ≈ 7,200 vs 9,700), aggressive orders every few seconds, but **no momentum**. On them adaptive_mm.py loses **about 860 per lot per hour**: −1.0k at 1 lot, −10k at 10 lots, −43k at 50 lots on every seed. A tape where each aggressive order moves the mid one extra tick (`aggr_follow=0.7`) still loses −42k.

**Verdict.** The parameters are not overfit, but the **strategy is entirely a bet on strong, multi-second momentum** after order flow, which is how the sample was generated. If real rounds lack that momentum, the current 50-lot sizing loses tens of thousands per hour. Before submitting, the strategy needs to check during the run whether the momentum is actually there, and size by it.

**Momentum check (added).** adaptive_mm.py now paper-trades a 1-lot shadow copy of its taker, with the 5 ms latency, buying at the ask and selling at the bid. It scales the real target by that shadow's PnL over roughly the last 5 minutes (`EDGE_FULL = 60` ticks per lot gives the full 50 lots). The maker leg is now off by default because it lost on every control tape.

| Tape | Before | After |
|---|---|---|
| sample half 1 / half 2 | 53,640 / 63,572 | 51,699 / 61,679 |
| sample (full) | 117,324 | 115,511 |
| null tapes ×3 | about −43k each | −8 / 0 / 0 |
| follow-through tapes (0.3 / 0.7 / 1.0) | about −42k | about −13 / 0 / −13 |
| trend half → null half | — | 48,346 |
| null half → trend half | — | 46,227 |

The check settings sit on a flat region: `EDGE_FULL` 20 to 150 and a decay of 5 to 15 minutes all keep 85–100% of the sample PnL. A 1-minute decay is too jumpy. **Remaining risk:** real rounds with intermittent or weaker momentum are untested, and the first few minutes of each run are spent mostly measuring.

(Note: the provided `trend_follower.py` never clears `_order_in_flight` on CANCELED/REJECTED, so it stops trading after its first missed IOC. The deck's version has the fix. Its control-tape numbers aren't meaningful for that reason.)

## sample_v2 (realistic tape, 2026-11-07)

2M rows over 5,750 s, a dense book (touch about 820 lots per side), mean spread 1.58 ticks, and range-bound (9,965–10,023, ends +4).

- **The signals predict direction but not enough size to take.** Touch imbalance is now the strongest signal (P(up) from 0.20 to 0.81 across deciles); OFI is weaker (0.30 to 0.70). But the mean move 1–5 s ahead in the extreme deciles is only about ±0.25–0.3 ticks, against a round-trip cost of about 1.6 ticks.
- **Taker ceiling** (perfect-foresight script, since removed): only 705 per lot over the whole tape, so 35k at ±50. Even perfect timing barely pays for crossing the spread.

| Config on v2 (half 1 / half 2) | PnL |
|---|---|
| adaptive_mm.py defaults (maker off) | 0 total: the momentum check never sizes up |
| taker without momentum check | −19.5k (full tape) |
| trend_follower | −531 (full tape) |
| maker only, 5-lot quotes | 13.0k / 13.2k |
| maker only, 15-lot quotes | 31.5k / 31.8k |
| taker + maker, 15-lot quotes (after the size-0 fix) | 31.5k / 31.8k, and still 51k / 61k on sample halves, +313 on a null tape |

**Bug fixed:** with the momentum check at size 0, the taker still crossed the spread to flatten maker inventory, which cut maker PnL by more than half. Now the taker does nothing at size 0.

**Maker experiments (per v2 half, both halves agree):**
- **Imbalance guard** (pull the quote that imbalance says is about to be run over, or keep it but don't re-place it): worse at every threshold. 31.5k → 27.8k (|imb| > 0.5), 29.5k (0.7), 30.7k (0.85), in both modes. The fills it skips are profitable on net. Removed.
- **Quote size × inventory skew:** quote size is the big lever. At skew 0.1: 15 lots → 27k, 40 → 53k, **60 → 63k**, 80 → 65k (levels off, more inventory). Lower skew earns more on v2 (60 lots at 0.02 → 76k) but loses more on the follow-through control tape (−5.2k vs −2.1k).
- **New defaults:** `QUOTE_SIZE = 60`, `SKEW_PER_LOT = 0.1`. Full v2: **125,470** (±2,207), up from 63,310. Inventory median 30, 99.7% within ±64, no penalty or rejects. Sample 113.7k, controls −2.1k to +0.4k.

**Lessons:** on realistic data the edge is **passive spread capture**, the reverse of the synthetic sample. The momentum check did its job (it stopped a −19.5k loss), but the default config earned nothing because the maker was off.

## Suggested build order

1. **Baseline maker.** Join the best bid and ask with small size, skew against inventory, hard cap at about ±50 including resting and in-flight orders, and flatten near the end of the run. Measure PnL and `breach_penalty`.
2. **Add a fair value.** Imbalance/microprice decides which side to quote and when to step back a tick (Gould–Bonart, Stoikov). On these tapes the strongest fair value is the generator's hidden price S. It can be estimated from new limit-order prices with a 100–200ms half-life, and it predicts the mid 0.5s ahead with R² of about 30%. See [DATA_ANALYSIS.md](DATA_ANALYSIS.md).
3. **Add OFI and recent-trade-intensity filters** to pull the side about to be run over (Cont–Kukanov–Stoikov, Hawkes idea).
4. **Queue-aware rules.** Keep early queue positions and use `modify_order` instead of cancel and re-post (Market Maker's Dilemma, Moallemi–Yuan).
5. **Only then** try taking liquidity, and only when the predicted move covers the spread.

Check each step with a backtest (no unit tests, see AGENTS.md), and compare on more than `sample.csv` as soon as real round data is available.

## Maker improvements from the sub-second findings (full sample_v2)

Baseline: 125,470 (60-lot quotes, combined forecast and inventory skew with a 0.3-tick threshold).

| Idea | Best result | Verdict |
|---|---|---|
| Re-quote delay (keep the old quote when the touch moves, since the bid gives back about half of each move), 50 ms – 1 s, two modes | 119k at 50 ms down to 60k at 1 s | worse: fewer fills |
| Microprice term in the quote signal, weight −1 … +2 | 121k – 128k | no effect beyond noise (±2.2k); not adopted |
| Quote one tick inside the spread when it is 2 ticks wide | 30k – 40k, with 2.3× the fills | badly picked off |
| Inside the spread only when it is ≥ 3 ticks wide | ≈ baseline | too rare to matter |
| Looser forecast step-back (one shared threshold 0.3 → 2.0 → none) | 140k → 188k | better, but inventory then sits at the cap |
| **Separate forecast step-back (1.0 tick) and inventory step-back (free 30 lots, then 1 tick per 30)** | **168,093** (±3,248) | **adopted** |

- **PnL tracks fill count**, at about +26 per passive fill. On this tape, rules that skip fills lose money even when the skipped fills look risky.
- **The old shared threshold was far too tight.** One 60-lot fill pushed that side 6 ticks off the touch.
- **Inventory free zone** (forecast step-back at 1.0, 30-lot steps), with the follow-through control tape as a risk check:

| `INV_FREE` | v2 | `follow_1` | Note |
|---|---|---|---|
| 0 | 126k | −1.9k | |
| 15 | 138k | −3.2k | |
| **30** | **168k** | **−5.2k** | chosen |
| 60 | 183k | −8.5k | |

The free zone of 30 lots is a deliberate compromise between v2 PnL and loss on trending data. Cancels on v2 fell from 23k to 9k because the quotes move far less.

## Round 1 (data/round1.csv, 2026-11-07)

Tape statistics come from the existing `analysis` package (`uv run analysis.py data/round1.csv` → `logs/analysis/round1.txt`).

**The tape:** 4.4M rows over 12,000 s (3.3 h), 368 rows/s.
- **Spread:** 1 tick 93% of the time (mean 1.07; sample_v2: 1.58).
- **Volatility:** about 0.92 ticks per √s, roughly twice sample_v2.
- **Price:** a random walk from 1 s up (variance ratio 60 s / 1 s = 1.07, reversal share 46.3%, drift not significant).
- **Imbalance:** touch imbalance is the best 1 s predictor (Spearman +0.26, hit rate 63%). A linear mix of book features reaches out-of-sample R² 0.11.

**Ceilings** (perfect foresight, |inventory| ≤ 100, flat at the end):

| | PnL |
|---|---|
| Theoretical max | 1,389,093 |
| Making only | 1,350,883 |
| Taking only | 483,450 |

**adaptive_mm.py (defaults, untouched): +202,677** (±6,075).
- That is 15% of the making-only ceiling. All of it comes from the maker; the momentum check kept the taker off.
- PnL rises steadily: +27k to +33k every half hour, with a maximum drawdown of only 668.
- It ends at +90 lots, the cap.

| Variant | PnL |
|---|---|
| taker without momentum check | −157,341 |
| trend_follower | −410 |
| QUOTE_SIZE 30 / 60 (INV_FREE 15–30) | 131k–133k / 203k–207k |
| QUOTE_SIZE 90, BACKOFF 2.0, INV_FREE 30 | 251,304 (not adopted yet: untested on sample_v2) |

**Market maker markout** (all passive volume in the tape, ticks per lot, after 1 s):
- at the touch: +0.03
- one level deeper, filled only when an IOC sweeps through: **+0.33**
- when the spread is 2 ticks: **+0.41**
- fills of 46+ lots: about 0; fills under 46 lots: +0.23 to +0.26

Large fills from big IOCs carry the adverse selection. That makes a second quote one tick behind the touch a candidate worth testing.

### Round 1: more aggressive maker (tuned on round 1 only from here on)

| Change | Round 1 PnL |
|---|---|
| Before (60-lot quotes, step-back 1.0, free 30 lots, cap 90) | 202,677 |
| 120-lot quotes (in practice: as much as the cap allows) | 232k – 264k, depending on the rest |
| Forecast step-back 1.0 → 2.0 → off | +5–8% (it rarely triggers at 2.0) |
| Position cap 90 → 99 (still below the 100-lot penalty line) | +1–6% |
| Inventory free zone 15 / 30 / **60** / 80 / 99 lots | 238k / 262k / **270k** / 267k / 262k |
| Inventory step 30 vs 60 lots, pull filter on/off | within ±2k |

**New defaults:** `QUOTE_SIZE = 120`, `BACKOFF_THRESHOLD = 2.0`, `INV_FREE = 60`, `MAX_POS = 99`.

On round 1 that gives **270,439** (±8,469), up 33%. Full-log run: no penalty and no position-limit rejects. Inventory median is 47 and reaches 99 at least 5% of the time. Maximum drawdown is 796, and the run ends at +67 lots. Turning the forecast step-back off entirely gives 273.7k, within noise.

## IMC Prosperity winners' techniques vs our exchange (2026-11-07)

Sources:
- [Frankfurt Hedgehogs, Prosperity 3 (2nd)](https://github.com/HendrikMS/imc-prosperity-3)
- [Linear Utility, Prosperity 2 (2nd)](https://github.com/ericcccsliu/imc-prosperity-2)
- [jmerle, Prosperity 2 (9th)](https://github.com/jmerle/imc-prosperity-2)

Their single-product market making:

1. **Fair value from the big passive participant.** Linear Utility: the "market maker mid" (a bot quoting 20–30 lots), which ignores a small 1–5 lot bot. Frankfurt: the "wall mid" (deep levels). jmerle: the price level with the most volume.
2. **Take first, then penny-jump.** Take anything priced through fair, then post one tick better than the best quotes while keeping positive edge.
3. **Clear inventory at zero edge.** Trade at fair to free position-limit capacity (Linear Utility: +3%), or force liquidation near the limit (jmerle).
4. **Follow informed flow.** Frankfurt tracked one bot ("Olivia") that bought at daily lows and sold at highs, identified by its 15-lot size, then by trader ID.

**Why they don't transfer one to one:** Prosperity runs in discrete snapshots with no latency and wipes all orders every step ("speed and order cancellation were irrelevant"). Ours is continuous, with 5 ms latency and price-time queues of about 900 lots at the touch (round 1 median; median taker order 60 lots).

**Tested on round 1:**

| Technique | Round 1 PnL |
|---|---|
| baseline | 270,439 |
| penny-jump the side the forecast favours when the spread is ≥ 2 | 179,228 |
| penny-jump both sides | 50,124 |

Wide spreads last only about 23 ms (median), so a jump usually lands late or is picked off. The same was true on sample_v2.

**Reading of the round 1 report:** the edge is in **queue position**.
- At the back of a 900-lot queue, our quote is filled only by large sweeping orders, and fills of 46+ lots keep about 0 ticks after 1 s.
- Fills one level behind the touch keep +0.33. Fills under 46 lots keep +0.23 to +0.26.
- The making-only ceiling (1.35M) assumes a perfect queue position: first at new levels, or joining just after a level trades.

**Next experiments, in priority order:**
1. **Queue ladder.** Rest orders 1–2 ticks behind the touch and leave them there, so when the touch moves onto them they are near the front.
2. **Zero-edge inventory clearing.** When the forecast already points toward the unwind side, cross to unwind; this frees the position cap, which binds at least 5% of the time.
3. **Fair value from a size-filtered "wall mid"** instead of OFI alone. Needs a size breakdown of who is informed, ideally as a new `analysis/` section.

The 700k target is about 50% of a perfect-foresight ceiling. Treat it as a stretch goal; the queue ladder is the one idea with a mechanism that could close a large part of the gap.

### Queue ladder (round 1)

Instead of one quote that is cancelled whenever the touch moves, keep resting orders on up to `LADDER_LEVELS` price levels per side, from the quote price outward. An order inside that band is never moved. When the touch steps a tick away, the order stays one level behind, and when the touch comes back it is near the front of the queue.

| Variant | Round 1 PnL |
|---|---|
| one level (same as before; check) | 270,016 |
| **2 levels, 120 lots each (capped by risk room)** | **353,645** (±7,781) |
| 2 levels, top 120, second 40 / 80 | 342,464 / 345,278 |
| 2 levels, top 60 / 90 | 304k – 311k / 336k – 347k |
| 3 levels (sizes 20–120) | 266k – 292k |
| 4 levels | 207k – 245k |
| deeper levels get risk room first | 3–15% worse |
| never send fresh orders to the top level (only keep ones the touch moved onto) | 69k – 124k |
| 2 levels with inventory free zone 40 / 60 / 80 | 335.5k / 353.6k / 358.7k (80 within noise of 60; kept 60) |

- **Result:** +31% over the single quote. More than two levels spreads the 99-lot position room too thin.
- **Full-log run:** no penalty and no position-limit rejects. Maximum drawdown 529. PnL per half hour: 49k – 55k.

### Zero-edge inventory clearing (round 1): rejected

The idea comes from Linear Utility (Prosperity 2, +3%). Once inventory is more than `CLEAR_FROM` lots from the target, cross to unwind (sell at the bid or buy at the ask) whenever the microprice is within `CLEAR_EDGE` ticks of that price. One IOC is in flight at a time.

| `CLEAR_FROM` | edge 0.1 | edge 0.25 | edge 0.4 |
|---|---|---|---|
| 30 | 340k | 304k – 314k | 269k – 284k |
| 60 | 338k – 340k | 327k – 328k | 306k – 313k |
| 90 | 349k | 346k | 341k |

Every setting does worse than no clearing (353,645), and the more it clears, the worse the result. With a 1-tick spread, crossing costs more than the freed position room earns in extra passive fills. Linear Utility had wide spreads and a 20-lot limit, so the trade-off went the other way for them. Code removed.

### Iterating on the opening research (round 1, 2026-11-07)

Starting point: the 2-level ladder at 353,645.

| Idea (source) | Result | Kept? |
|---|---|---|
| **Imbalance guard** (Gould & Bonart; Market Maker's Dilemma): when the touch imbalance is beyond ±0.7–0.85, step the quote it points against back a tick, unless it unwinds | 366k – 368k (0.5: 355k, 0.95: 361k). "No fresh order there" variant: 356k – 360k | **yes** (0.7) |
| Join only short queues (queue position, Moallemi & Yuan): fresh top order only if the market queue is ≤ 300 / 600 / 1000 / 2000 lots | 318k / 342k / 357k / 372k | no |
| Pull the side being swept by aggressive flow (Hawkes clustering) | off 367k, 0.5: 352k, 1.0: 368k, 2.0: 370k | no, removed |
| Forecast (OFI) step-back threshold 1.0 / 1.5 / 2.0 / 3.0 | 347k / 362k / 368k / 372k | kept at 2.0 (safety; rarely triggers) |
| **Second level only keeps orders the touch moved away from** (no fresh order there; frees room for the top) | 378.8k | **yes** |
| Keep orders 2–3 ticks behind | 263k – 317k | no |
| Quote 200 lots per level | 341k – 372k | no |
| **Inventory free zone 80, step 15** | 383k (60: 372k – 380k, 99: 378k – 380k) | **yes** |
| Looser position cap 110 – 200, with an emergency IOC above 100 | 86k – 207k after penalties of 166k – 330k; worse even before the penalty | no |

**Final round 1 result: 385,561** (±7,856). Full log: no penalty and no position-limit rejects. Maximum drawdown 834. PnL per half hour 54k – 59k throughout. Ends at +86 lots.

**Overfitting checks:**
- **Neighbourhood**, one setting at a time: imbalance guard 0.6 / 0.8, free zone 70 / 90, step 10 / 20, quote 100 / 150, forecast step-back 1.5 / 3.0. All between 375k and 392k (−3% to +2%), so it is not a knife-edge fit. The ladder itself matters: 1 level gives 312k, 3 levels 332k.
- **Time:** every half hour contributes 54k – 59k, so the result is not driven by one episode.
- **Another tape:** sample_v2 was never used for these choices.

| | before the ladder | ladder | final |
|---|---|---|---|
| sample_v2 | 209.4k | 203.1k | 209.1k |
| follow-through control | −7.6k | −7.6k | −8.1k |

So today's +43% on round 1 does **not** carry over to sample_v2. The gains exploit round 1's own microstructure (1-tick spread, how the touch reverts). They will hold if the next rounds come from the same kind of tape, and will mostly be neutral, not harmful, if they don't.

**Realistic numbers:** round 1 now sits at 385k, which is 28% of the 1.35M making-only ceiling. The tweaks above have plateaued around 380k – 392k. 400k+ needs a new mechanism (for example, better queue-position estimates per order, or a fair value that predicts which fills are adverse), not more tuning.

## Learning from the colleague's imbalance_mm.py (615k on round 1)

`strategies/round1/imbalance_mm.py` (JPod101010) makes 615,583 on round 1 against our 385,561. Its basis is in `docs/DATA_ANALYSIS.md`:
- The generator places limit orders around a **hidden fair price S** that random-walks, and the visible mid lags S by about 0.5 s.
- An estimate Ŝ, the average of the bid-side and ask-side running means of newly arrived limit-order prices, explains 32% of the mid move 0.5 s ahead.

**Their components, each switched off in turn** (round 1, base 615.6k):

| Switched off | PnL | Change |
|---|---|---|
| hidden-price term `K_S` | 449.9k | −166k |
| aggressive-flow term `K_FLOW` | 505.2k | −110k |
| size 100 instead of 200 | 535.3k | −80k |
| crossing when fair is through the best price | 553.2k | −62k |
| imbalance term | 600.7k | −15k |
| keeping orders a tick behind | 616.1k | 0 |

Even without Ŝ their fair-value-plus-edge quoting (450k) beats our rule-based step-backs (386k).

**Our maker is now rebuilt on that design** (fair value plus edge-based target prices). It reproduces 615,583 exactly. Tuning on top:

| Change | Round 1 |
|---|---|
| passive edge 0.4 → 0.3 | 629.2k |
| flow weight 0.5 → 0.8, fade 200 → 100 ms | **650.8k** |
| imbalance 0 / 0.3, inventory weight 0.004 | 600k / 530k – 582k / 285k – 465k (worse) |
| hidden-price averaging speed 0.05 / 0.2, weight 0.9 / 1.0 | all worse (380k – 592k) |
| ladder 1 / 2 / 3 levels | 650.4k / 650.8k / 650.4k (no longer matters) |

**Final: 650,844** (±10,070).
- Full log: no penalty and no position-limit rejects. Maximum drawdown 552. PnL per half hour 96k – 101k throughout.
- Inventory median 82, at the 100 cap at least 5% of the time, and it ends at 100.
- **Neighbourhood check**, each setting nudged down and up (edge, take edge, flow weight and fade, hidden-price weight and speed, imbalance weight): 626k – 655k, i.e. −4% to +1%.
- **Other tapes:**

| | sample_v2 | null_1 | follow_1 |
|---|---|---|---|
| ours | 242.2k (previous version 209.1k) | −84.8k | −95.2k |
| colleague's | 261.1k | −83.4k | −93.6k |

The control-tape losses come from the hidden-price term (without it: −3k / −28k). On tapes where new orders are not placed around a hidden price, Ŝ misleads the quotes. An online check of whether Ŝ actually predicts the mid, scaling `K_S` toward 0 when it does not (like the taker's momentum check), would cap that risk.

### Hidden-price check

The hidden-price term is now weighted online:
- Every 50 ms a sample (δ = Ŝ − mid, mid) is stored.
- After 0.5 s it is paired with the mid move since then.
- The correlation over decayed sums (memory `HP_TAU_US`) scales `K_S` linearly from 0 (correlation ≤ 0, or fewer than 100 samples) up to full at `HP_CORR_FULL`.

| | round 1 | sample_v2 | null_1 | null_2 | follow_1 |
|---|---|---|---|---|---|
| no check | 650,844 | 242,211 | −84,782 | −83,769 | −95,168 |
| check, full weight at correlation 0.3, 2 min memory | 650,558 | 242,079 | −13,300 | −8,205 | −28,013 |
| **check, full weight at correlation 0.3, 5 min memory** (default) | **650,611** | — | −11,107 | −5,998 | −24,222 |
| check, full weight at correlation 0.45 / 0.6 | 641k – 644k / 597k – 600k | | −5k – −9k | −6k – −7k | −25k – −27k |

The check costs nothing on round 1 and removes most of the control-tape loss. What is left there (−6k to −24k) is about what the strategy loses without the hidden-price term at all (−3k / −28k), so it comes from the other parts.

## Fill-level diagnostics of adaptive_mm.py on round 1

Every fill was recorded with its context and then marked against the mid 0 / 0.1 / 1 / 5 / 30 s later (40k fills, 2.27M lots, +0.29 ticks per lot after 5 s overall).

- **No losing class of fill.** Every bucket (order type, inventory, hidden-price alignment, edge, age, size, imbalance) is positive after 5 s.
- **Crossing the spread pays best:** −0.51 ticks per lot at the trade, +0.53 after 5 s, +182k in total.
- **Hidden-price alignment matters:** fills on the side δ points to make +0.45 ticks per lot; fills against it about +0.1.
- **Older orders fill better:** orders that waited over 1 s make +0.40 ticks per lot, 200 ms – 1 s +0.25, 50 – 200 ms +0.15. The front of the queue pays.
- **Inventory is at the cap a lot:** median |inventory| 82, at 100 about a third of the time. 39% of volume is filled at |inventory| ≥ 90. Fills that add inventory make more (+0.35) than fills that reduce it (+0.28).
- **The end of the run costs nothing measurable:** the last 100 s earn +5.2k, a typical 100 s.

**Tweaks to get off the inventory cap (all rejected):**

| Setting | Values tested | Round 1 PnL |
|---|---|---|
| Smaller edge on the side that reduces inventory | 0.1 / 0.2 | 592k / 621k |
| Lower cross threshold when crossing reduces inventory | 0.1 | 419k – 511k |
| Extra inventory lean on passive quotes only | 0.002 / 0.005 / 0.01 | 590k / 470k / 357k |

Sitting at the cap isn't the problem it looks like. The extra fills these settings bring are worse than the ones they displace.

## Queue hysteresis (round 1)

**Why we cancel** (current strategy, round 1, 71k cancels):

| Reason | Share |
|---|---|
| free room for the target order (the deepest kept order) | **58%** |
| edge still positive but under 0.3 | 36% |
| the market left our price, so we'd be alone at a stale price | 4% |
| edge negative, or the order is too deep | 1% |

**Hysteresis:** an order priced better than the target is kept, and counts as the target order, while it still has `KEEP_EDGE` of edge. Joining still needs `EDGE` = 0.3. The first version gave at most +2k, because the room-freeing step cancelled the kept order straight away. Counting it as the target order fixed that.

| `KEEP_EDGE` | round 1 | sample_v2 | null_1 | follow_1 |
|---|---|---|---|---|
| off (0.3) | 650.6k – 653.7k | 238.5k – 242.1k | −11.1k – −13.9k | −24.2k – −28.0k |
| **0.25** (default) | **662.4k** | 240.4k | −14.8k | −39.3k |
| 0.2 | 661.0k | 238.9k | −34.5k | −46.5k |
| 0.15 / 0.1 / 0.0 | 658.2k / 647.9k / 624.7k | | | |

- Requiring a short queue ahead (300 / 800 lots) for keeping made no difference.
- Switching off room-freeing costs about 140k (510k – 533k): position room has to follow the target.
- **Verdict:** +12k on round 1, about one standard error, but every value from 0.15 to 0.25 is above the baseline. The cost is a larger loss on the trending control tape. Adopted at 0.25 as the moderate setting.

### Synthetic round-1 twins (data/synthetic/synth_r1_s{1,2,3}.csv)

These are generated by `scripts/gen_tape.py` from the fitted round 1 model, 12,000 s each; unzip with `gunzip -k`. The model is slightly harsher on market makers than round 1 (see `docs/DATA_ANALYSIS.md`).

| Strategy | s1 | s2 | s3 | round 1 |
|---|---|---|---|---|
| imbalance_mm.py (colleague) | 582,195 | 609,543 | 594,397 | 615,583 |
| adaptive_mm.py without queue hysteresis | 607,310 | 604,565 | 601,232 | 650,611 |
| **adaptive_mm.py with queue hysteresis (`KEEP_EDGE` 0.25)** | **615,807** | **623,335** | **609,351** | **662,359** |

Hysteresis helps on all three twins (+8.5k / +18.8k / +8.1k), which confirms the round 1 gain on data it was not tuned on.

## Preparing for a harder round 1 test tape (stress set)

The scored data could be more volatile or otherwise harder than the public round 1 file. So `scripts/gen_tape.py` (seed 11, 3,000 s each) made round 1 twins with one thing changed, in `data/stress/` (git-ignored):

| Scenario | Change vs round 1 fit | adaptive_mm (edge 0.3) | edge 0.4 | imbalance_mm |
|---|---|---|---|---|
| base | none | 149.2k | 154.6k | 145.8k |
| vol150 / vol200 | hidden-price volatility ×1.5 / ×2 | 143.7k / 136.3k | 146.8k / 140.4k | 138.4k / 132.3k |
| push133 | IOC price push ×1.33 (κ 0.01) | 109.0k | 112.1k | 107.5k |
| volpush | volatility ×1.5 and push ×1.33 | 104.1k | 106.6k | 107.7k |
| wide | new orders 0.3 ticks further from S | 212.9k | 218.0k | 213.3k |
| busyioc | IOC rate 8 → 12 per s | 206.6k | 217.1k | 206.1k |
| fastcancel | cancel rate 2.1 → 3.0 per s | 184.1k | 193.1k | 190.1k |

- **Profitable in every scenario.** The main risk is a stronger IOC push (−27%), the adverse selection from large informed-like orders. Doubling volatility costs only 9%.
- **A stronger flow term (`K_FLOW` 1.1)** helps only the push scenarios (+5k to +9k) and costs elsewhere, so it stays at 0.8.

**Edge / keep edge on the full-length tapes:**

| `EDGE` / `KEEP_EDGE` | round 1 | twin s1 | twin s2 | twin s3 |
|---|---|---|---|---|
| 0.3 / 0.25 | 662.4k | 615.8k | 623.3k | 609.4k |
| 0.35 / 0.25 | 665.0k | 628.2k | 645.3k | 629.9k |
| **0.4 / 0.25** | **667.9k** | **636.2k** | **655.1k** | **638.5k** |
| 0.45 / 0.2 – 0.25 | 655.6k – 656.1k | 640.6k – 645.0k | 654.2k – 657.1k | 642.8k – 643.2k |
| 0.5 / 0.2 – 0.25 | 648.8k – 654.2k | 643.7k – 644.8k | 657.7k – 658.4k | 638.9k – 639.0k |

- Setting the keep edge equal to the join edge (no hysteresis gap) is clearly worse at every edge.
- **Adopted `EDGE` 0.4 with `KEEP_EDGE` 0.25.** It is best or tied on round 1, all three twins and 7 of 8 stress scenarios.
- Controls are unchanged: sample_v2 242.2k, null_1 −13.8k, follow_1 −39.0k.

## Second research pass: Prosperity 4 and queue-management papers (2026-11-07)

**Why this direction:** on generated round-1 sessions, `imbalance_mm.py` fed the *true* hidden price (re-tuned) scored
188k per hour against 194k as-is (see [round1.md](round1.md)). A better price estimate does not raise the score, so the
remaining room is in quote and queue management, not in estimating S.

### IMC Prosperity 4 (2026): techniques new to us

Sources: [Durpie-Git, 4th globally / 1st Europe](https://github.com/Durpie-Git/imc-prosperity-4),
[Alpha Search team](https://github.com/fabianbaiertum/IMC-Prosperity-4),
[nathanw3456, 57th](https://github.com/nathanw3456/Prosperity_4_Writeup). No overall winner has published a write-up.

| Technique | Fits round 1? |
|---|---|
| **Inventory thresholds from dynamic programming** (Durpie): a value function over inventory −80…+80 gives bid and ask edge thresholds per inventory level, instead of a linear skew | **Yes, untried.** `imbalance_mm.py` uses a linear skew (`K_INV`). The synthetic sessions can supply the data for per-inventory thresholds. |
| **Choose settings by E[PnL] − 0.25·SD across simulations**, not by the best single run (Durpie); rank changes by how they hold up across days (nathanw3456) | **Yes, untried.** Possible now with the synthetic sessions. |
| **Split PnL into directional carry and spread capture** (Alpha Search): carry = position × Δmid, capture = Σ fills × (mid at fill − fill price) | **Yes, a missing diagnostic.** It shows whether the score comes from spread capture or from inventory drifting the right way. |
| Hold a position in a dead zone instead of cycling to zero | Already done: the inventory free zone in `imbalance_mm.py`. |
| Empty-book takers, recurring takers, "100-jump" reversion, bidding zero | **No.** They exploit IMC's bots. Our book never empties and IOC sides are coin flips. |

### Papers not covered above

1. **Lehalle & Mounjid, "Limit Order Strategic Placement with Adverse Selection Risk and the Role of Latency"**
   ([arXiv:1610.00261](https://arxiv.org/abs/1610.00261), Market Microstructure and Liquidity 2017). Stochastic control
   of a single limit order on a large-tick asset: keep it or cancel it, driven by queue imbalance, with latency eroding
   the value of acting. This is the decision our market maker makes on every event.
2. **Queue-reactive model**: Huang, Lehalle & Rosenbaum, "Simulating and analyzing order book data: the queue-reactive
   model" (JASA 2015), and the extension with order sizes ([arXiv:2405.18594](https://arxiv.org/abs/2405.18594)).
   Arrival and cancel rates depend on queue size, which gives a fill-probability model. Relevant because round-1 limit
   sizes run from 1 to 124k lots.
3. **hftbacktest, ["Queue-Based Market Making in Large Tick Size Assets"](https://hftbacktest.readthedocs.io/en/latest/tutorials/Queue-Based%20Market%20Making%20in%20Large%20Tick%20Size%20Assets.html).**
   A worked strategy for an asset whose spread is usually one tick. Its best version drops fair-value pricing: join the
   best price, except step back a tick when the queue there is thin **and** inventory says not to add on that side.
   We tested "join only short queues" and the imbalance guard, not this inventory-conditioned version.
4. **Cartea & Wang, "Market Making with Alpha Signals"** (IJTAF 2020). A signal that jumps with each aggressive order
   and decays between them; the market maker uses it to avoid adverse fills and to cross the spread opportunistically,
   more so as inventory tolerance rises. This is the design of `K_FLOW` + `EDGE_TAKE`, with theory for scaling the take
   threshold.
5. **Capponi, Figueroa-López & Yu, "Market Making with Stochastic Liquidity Demand"**
   ([arXiv:2101.03086](https://arxiv.org/abs/2101.03086)). Fills depend linearly on the distance between the quote and
   the fundamental price, and forecasts of fundamental drift enter the quotes. In our market this holds literally: IOCs
   are priced at round(S ± 1), so an ask at price a can only be hit when S ≥ a − 1.5.

Checked and not useful here: Barzykin, "Optimal Quoting under Adverse Selection and Price Reading"
([arXiv:2508.20225](https://arxiv.org/abs/2508.20225)); there "price reading" means other traders inferring the
dealer's inventory from its quotes, which a fixed tape can't do. Jane Street ETC write-ups are self-reported and mostly
simple fair-value bond trading.

### Two ideas that follow from the fitted market model

- **Queues shrink at a known rate.** Every limit order has its own exponential cancel timer (round 1: mean 0.47 s),
  whatever its size. With no IOC, the lots ahead of a resting order therefore shrink by a factor of e every 0.47 s on
  average, even with a 124k-lot order in front. A long queue is less of a problem than it looks, and keeping an order
  is worth much more than re-posting it. This supports the queue ladder (+31%), and the value of keeping an order can
  be computed instead of tuned.
- **Which fills are possible depends on S.** With IOCs at round(S ± 1), a quote one tick behind the best price is only
  filled once S has moved through the touch. With the push per IOC in round 1, that is the bad case. The model can
  give the expected value of each quote position for a given S estimate, which turns "where to rest" into a table.

### Suggested order of experiments (synthetic round-1 train sessions, then validation, then the real tape once)

1. PnL split into carry and spread capture for `imbalance_mm.py`: cheap, and it shows where the 615k comes from.
2. Inventory-conditioned thin-queue step-back (hftbacktest version 3) on top of `imbalance_mm.py`.
3. Keep-or-cancel value from the queue shrink rate (Lehalle–Mounjid style).
4. Per-inventory thresholds from dynamic programming (Durpie).
5. Pick finalists by E[PnL] − 0.25·SD on the validation sessions, then confirm once on `data/round1.csv`.

### Market regimes we know (summary)

- **Between tapes:** `sample` trends (makers lose −9 ticks/lot at 5 s); `sample_v2` is calm with a 2-tick spread 58%
  of the time and no impact; `round1` has a 1-tick spread 93% of the time, 1.85× the volatility and a permanent push
  per IOC.
- **Within round 1:** no regime changes. Rates are flat across 10-minute blocks, per-minute volatility barely varies
  (coefficient of variation 0.10, autocorrelation +0.01), and `imbalance_mm.py` makes 74–83k in every 25-minute slice.
- **Micro-states within round 1** (what a strategy reacts to): 1- vs 2-tick spread (2-tick spreads last about 23 ms);
  the book lagging S (half-life about 200 ms); the second after a big IOC (S just pushed, about +0.8 ticks for 1,000
  lots); strong queue imbalance; inventory near ±100.

## Private round 1 run vs public (AdaptiveMarketMakerPlus, colleague's shrink version, EDGE 0.3)

| | public round1.csv | private scored tape |
|---|---|---|
| PnL | 678,057 (±10,421) | 667,357 (±10,365) |
| our fills / lots traded | 29,488 / 2.245M | 29,123 / 2.213M |
| PnL per lot traded | 0.302 | 0.3015 |
| volatility (ticks/√s) | 1.15 | 1.146 |
| market taker events / lots | 96,012 / 13.72M | 95,143 / 13.58M |
| taker size mean / median | 143 / 59 | 143 / 59 |
| net price drift | +29 | −61 |
| inventory median, max drawdown | 84, 547 | 85, 592 |

- **The private tape is the same generator** with a different random session: same volatility, same taker size distribution, and about 1% less taker volume.
- **The −10.7k gap is about one standard error.** Per lot traded the strategy earned the same (0.302 vs 0.3015). About 1.4% fewer lots traded explains roughly −9.5k.
- **The loss is spread evenly** across the half hours (−4.4k to +2.9k each), with no single bad episode and no penalty.
- Expect about ±10k to ±20k between sessions. The three synthetic twins spread 636k – 655k for adaptive_mm.

## Closing the gap to 800k: experiments on adaptive_mm_plus (what was submitted)

**Benchmark** (round 1 / three twins / total):

| Strategy | Round 1 | Twins | Total |
|---|---|---|---|
| adaptive_mm (edge 0.4) | 667.9k | 636k – 655k | 2,597.7k |
| adaptive_mm_plus | 678.1k | 660k – 673k | 2,674.8k |
| queue_mm | 605.4k | 651k – 672k | 2,584.0k |
| penalty_mm | −1.30M | −1.33M to −1.39M | −5.39M (8.0M in penalties) |

**Permanent IOC push in Ŝ** (shift the hidden-price estimate by κ · size^0.65 per aggressive order): rejected.
- κ 0.004 gives +17k on round 1 but −30k to −50k on every twin; 0.009 and 0.014 are much worse.
- Ŝ already absorbs the push through new orders placed around the moved S, so adding it double-counts.
- A textbook case of fitting the tape rather than the generator.

**Edge 0.4 on adaptive_mm_plus:** adopted.

| | Round 1 | Twins (each) | Stress set (8 tapes) | Controls (sample_v2 / null / follow) |
|---|---|---|---|---|
| edge 0.3 | 678.1k | 660k – 673k | 1,325.9k | 249.6k / −11.8k / −25.3k |
| **edge 0.4** | 672.9k | **679k – 691k** | **1,374.6k** (better in all 8) | 252.0k / −12.1k / −25.0k |

**Other settings at edge 0.4** (all rejected):
- **Quote size 150:** worse on round 1 (−25k) and the twins.
- **Flow weight 1.0:** +7k on round 1, −11k to −16k per twin.
- **Take edge 0.31:** +9k to +14k per twin but −17k on round 1. About +4k on the 4-session average, so 0.25 is kept.

**Shrink fix:** a shrink that comes back REJECTED keeps the order tracked even with a cancel in flight. Before, it was forgotten while possibly still live, which could put inventory out of sync. No PnL effect; that drift was never seen in testing.

**Lesson:** tune on the generator (many twins plus the public tape), not on the public tape alone. The private round 1 tape was a different session from the same generator: profit per lot was identical (0.302 vs 0.3015).

## Round 2: where adaptive_mm_vol's PnL comes from (full-logs run, pnl 131,725)

Our fills marked out against later market-vs-market trade prices (ticks per lot):

| our fills | lots | 1 s | 5 s | 30 s |
|---|---|---|---|---|
| crossing LIMIT (fair EDGE_TAKE through the opposite best) | 257k | +0.75 | +0.87 | +0.57 |
| passive, closing inventory | 223k | -0.22 | -0.35 | +0.21 |
| passive, opening inventory (incl. from flat) | 76k | -0.51 | -0.13 | -0.37 |
| taker-leg IOC | 8k | -0.72 | -2.90 | +1.33 |

- The crosses (following Ŝ and the flow term) earn the money; passive opening fills and the OFI IOC leg lose it.
- Inventory is not stuck long: |inventory| >= 90 for 56% of the time, split 30% long / 27% short. 200-lot passive quotes
  flip it from one cap to the other in a single fill, which leaves no room for the profitable crosses.
- Passive fills by the size of the IOC that hit us: IOCs up to 150 lots +0.3 to +0.5 at 5 s, 400+ lots -0.5 to -0.8.
- `strategies/round2/adaptive_mm_r2.py` acts on this: IOC leg off, passive quotes only close inventory (PASSIVE_OPEN), excess
  passive size is trimmed, separate passive inventory skew (K_INV_QUOTE). Untested.

**"Buy the dip" does not fit this tape.** Variance ratio 60 s/1 s is 1.27 (trending), 1 s returns autocorrelate +0.03,
the next mid change reverses only 38% of the time, and the market's passive fills are still -0.49 ticks/lot after 60 s.
In the fitted generator each IOC moves the hidden price S for good, so a dip caused by a big sell does not come back.
Buying dips is what a passive bid already does, and it is the losing part of the book above.

**What the literature says can be predicted:** the jump itself, from what we already see.
- Zheng, Moulines & Abergel, "Price Jump Prediction in Limit Order Book" ([arXiv:1204.1381](https://arxiv.org/abs/1204.1381)):
  a jump = a market order that executes beyond the best price; trade sign, market order size and the liquidity at the
  best quotes predict it. For us: a thin touch on one side plus Ŝ leaning that way means the next big IOC sweeps it.
- Donier, Bonart, Mastromatteo & Bouchaud, transient vs permanent impact ([arXiv:1412.0141](https://arxiv.org/abs/1412.0141)),
  and Taranto et al. ([arXiv:1602.02735](https://arxiv.org/abs/1602.02735)): in real markets part of the impact
  decays, which is what makes dip-buying pay there. Our generator's push is permanent, so there is no decay to harvest.
- Jump-aware market making (Heston-Bates style HJB quotes, [arXiv:2405.03496](https://arxiv.org/abs/2405.03496)):
  widen or pull when jump intensity is high; no closed form we can drop in.
- Competitions: Optiver Ready Trader Go has no published winning write-up; the 2021 winners describe heuristics plus "a
  model capturing short-term market dynamics". IMC Prosperity notes are above. None faced a jumpy, permanent-impact tape.

**Lead to try next:** the market's passive fills made right after a sweep lose most (spread 2 before the fill: -1.11 at
1 s, spread 3+: -1.29, against -0.25 at a 1-tick spread). Pulling passive quotes on the side a sweep is heading to, for a
few hundred ms after a large IOC or a widened spread, targets exactly the fills that hurt.

### adaptive_mm_r2 tuning on round 2 (one run at a time, full tape)

| setting | pnl |
|---|---|
| adaptive_mm_vol.py | 131,725 |
| adaptive_mm_r2, old settings (IOC leg off, passive only unwinds) | 174,377 |
| EDGE_TAKE 0.12 (0.0: 39k, 0.05: 190k, 0.1: 207k, 0.15: 199k, 0.3: 156k) | 209,944 |
| S_ALPHA 0.25 (0.05: 208k, 0.2: 226k, 0.3: 227k, 0.45: 215k), EDGE_TAKE 0.13 | 231,426 |

- Crossing is now the whole edge (about +1 tick/lot at 5 s), so what moves PnL is when to cross and how fast Ŝ
  follows S. Faster Ŝ (S_ALPHA) was the second-largest gain.
- Unwinding faster loses: K_INV_QUOTE 0.004: 194k, 0.008: 55k, 0.015: -257k. Passive closes lose at 5 s but gain at 30 s.
- No effect: passive EDGE 0.4–1.0, CROSS_DEEP (cross several levels), the post-sweep guard (pulling passive quotes
  after a 100–1000 lot sweep). K_FLOW 0.8 beats 0.5 and 1.2. K_S 0.5–1.0 is flat.
- **Next: a Kalman filter for S** (a continuous-state HMM). Observations: new limit prices (buy ≈ S − X, sell ≈ S + X)
  and IOC prices (round(S ± 1)); process noise grows with elapsed time and jumps after big IOCs. That makes the
  S_ALPHA speed adaptive: fast right after a jump, slow when calm. A discrete regime HMM (calm/jumpy) is less promising:
  per-minute volatility has lag-1 autocorrelation of only 0.17, so jumpy minutes do not persist, and drift is not
  significant (t = −0.85).

### Kalman filter for S on round 2 (`tools/kalman_price.py`, `KF` in adaptive_mm_r2)

- 1-D Kalman filter: new limit orders near the touch are noisy readings of S (buy ≈ S − m, sell ≈ S + m, with m and
  the noise learned online); variance grows with time (KF_Q × vol scale²) and after every aggressive order
  (KF_JUMP × lots^1.3); an order one tick from the opposite best is treated as a bound only (censored); swept trades
  bound S (IOC price = round(S ± 1)).
- **Prediction:** on the first 1,100 s, R² of (Ŝ − mid) for the next mid move: 0.1 s 0.107 vs 0.071 (running
  averages, S_ALPHA 0.25), 0.5 s 0.090 vs 0.062, 1 s 0.044 vs 0.029. Slope at 0.5 s is 0.91, so Ŝ is about calibrated.
- **PnL:** 235,976 (EDGE_TAKE 0.2, K_S 0.65) vs 231,426 with the averages: within noise (pnl_se ≈ 19k).
- **What matters inside it:** jump inflation. KF_JUMP 0: 206k, 2e-4: 230k, 1e-3: 232k, 3e-3: 226k, 1e-2: 219k.
  KF_Q 0.05–0.4 flat. Mean push (KF_PUSH) 0.005 same, 0.015 worse. Censoring off: −4k.
- **Reading:** a 50% better short-horizon forecast added little PnL, and |inventory| median rose to 80. The crosses
  are now limited by the ±100 position cap more than by the estimate. Next levers are about capacity: exit a position
  once Ŝ − mid has closed (signal-based exit by crossing at small edge), or size crosses by forecast strength so the
  cap is spent on the strongest signals.

### Freeing position capacity on round 2: rejected

Baseline adaptive_mm_r2 with KF: 235,976 (|inventory| median 80).

| change | pnl |
|---|---|
| CROSS_FULL 1.5 / 2 / 3 (cross only up to MAX_POS × edge / (EDGE_TAKE × CROSS_FULL)) | 233k / 224k / 208k |
| EXIT_MULT 0.7 / 0.4 / 0 (flatten by crossing at a lower edge) | 224k / 218k / 193k |
| EXIT_MULT 1.5 / 2.5 (exits need more edge) | 233k / 204k |
| K_INV 0 / 0.004 (inventory skew of the fair used for crossing; 0.0015 now) | 229k / 164k |

Together with K_INV_QUOTE (stronger passive unwinding: 55k at 0.008), every way of getting flat sooner or holding less
loses. The signal persists (crosses still mark out +0.6 to +1.5 ticks/lot at 30 s), so a full position is worth
keeping until the opposite signal crosses it out. Sitting near ±100 is a symptom of a long-lived edge, not a leak.
Remaining room is in entering earlier or at better prices, not in capacity.

### Three bigger changes on round 2, all rejected (built as `strategies/adaptive_mm_r3.py`, since deleted)

Ideas from the "more drastic changes" list, built behind flags on top of adaptive_mm_r2 (235,976) and ablated.
EDGE_TAKE 0.2 throughout (pnl, fills):

| fitted fair | entries | queue | pnl | fills |
|---|---|---|---|---|
| off | off | off | 235,976 | 5,774 |
| on | on | on | 210,978 | 7,665 |
| on | on | on, EDGE_TAKE 0.1 | 139,844 | 10,623 |
| off | on | on | 210,732 | 8,423 |
| on | off | on | 207,796 | 3,057 |
| on | on | off | 196,013 | 10,481 |
| on | off | off | 198,944 | 3,180 |
| on (W_FLOW 1.9) | off | off | 209,336 | 7,088 |
| off | on | off | 210,395 | 11,610 |
| off | on (ENTRY_EDGE 0.3) | off | 216,926 | 7,270 |
| off | off | on | 226,301 | 5,191 |
| off | off | on (JOIN 300, KEEP 600) | 231,556 | 5,454 |

Settings: entries joined the best price with full size once fair was ENTRY_EDGE (0.15 × vol scale^1.2) beyond it;
the queue rule posted only where at most QUEUE_JOIN (150) market lots were queued and cancelled orders with more than
QUEUE_KEEP (300) ahead.

| on | pnl |
|---|---|
| none (= adaptive_mm_r2) | 235,976 |
| all three | 210,978 |
| fitted fair only | 198,944 (209,336 with the flow weight set back to 1.9) |
| resting entries only | 210,395 (216,926 with a stricter entry edge) |
| queue-front only | 226,301 (231,556 with looser limits) |

- **Fitted fair:** OLS of the 1 s mid move on Ŝ − mid (0.55), imbalance (0.0), flow (0.49) and last trade sign (0.09)
  forecasts slightly better (R² 0.034 vs 0.029) but trades worse. The hand-tuned flow weight (about 2) is far above
  its forecast weight: crossing right after a big IOC pays more than its average 1 s move says.
- **Resting entries:** joining the touch ahead of the cross doubles the fills and loses 20k–26k. Passive touch fills are
  adversely selected on round 2 even on the side the signal favours, so paying the spread on a strong signal wins.
- **Queue-front quoting:** never joining long queues and cancelling deep orders loses 5k–10k; it mostly blocks the
  passive unwinds.
- Every knob moved toward "off" as it was loosened, so these are not tuning misses. Round 2 rewards taking on a strong
  signal and holding; making is a cost centre here.

## regime_mm on round 2: where the remaining PnL is (2026-11-07)

regime_mm.py (team): 448,818 on round2_public (269k/h); validation 1,032,909 over 12,000 s (310k/h), so it generalises.
`strategies/round2/regime_mm_kf.py` adds the Kalman filter in the volatile regime: 469,844. Tuning its other knobs (take
edges, S_ALPHA, passive or earlier exits, max hold) only lost.

**PnL by regime** (full-logs run, regime switches recorded): quiet 79% of the time 228k (passive 752k lots at +0.11/lot
after 5 s, crosses 159k lots at +0.78), volatile 21% of the time 221k (IOC snipes 171k lots at +1.37). Against the
hindsight max per minute: quiet minutes 104k of 288k, mixed 213k of 808k, mostly-volatile 132k of 608k. PnL in the
5–30 s around regime switches is positive on both sides, so switch timing is not a leak.

**The estimate of S is the bottleneck.** regime_mm with S replaced by the real mid some time ahead (local cheat,
`ORACLE_MID2` built from book_deltas; scratch only, never submittable):

| S used | public PnL |
|---|---|
| current mid | 79k |
| our Kalman / running averages | 470k |
| mid 100 ms ahead | 888k |
| mid 300 ms ahead | 1,021k (500 ms: 993k) |
| mid 300 ms ahead, volatile regime only | 735k |

**Correction (same day):** this overstates the value of estimating S. The lookahead curve (20 ms: 440k, 50 ms: 750k)
shows our estimate is already worth about as much as knowing the mid 20 ms ahead; the rest comes from events inside
the lookahead window (the next IOC sweep, new orders) that no estimate of S(t) can know. See the next section.

## Estimating S on round 2: what was tried and what is left (2026-11-07)

**Grid filter** (`strategies/round2/regime_mm_grid.py`; the `tools/grid_price.py` copy was deleted on 2026-11-08): exact Bayesian filter over (regime, S) with
order offsets fitted per regime from IOC prices (quiet X gamma(0.854, loc 0.025, θ 0.655), mean 0.59; volatile
gamma(0.384, loc 2.594, θ 2.903) from a two-part mixture, since 62% of the orders regime_mm labels volatile look quiet).
It lost: 262k with the grid in both regimes, 324k in the volatile regime only, against 470k for regime_mm_kf. Its
P(volatile) stayed about 0.10 inside volatile spells, so the volatile part of the model is wrong.

**Each estimate against the mid move** (corr², 100 ms grid, regime from regime_mm):

| estimate | quiet, 300 ms | volatile, 100 ms |
|---|---|---|
| s_hat (running means, top 2 levels) | 0.134 | 0.002 |
| S2 (clamp-aware running mean, 8 levels) | 0.204 | **0.104** |
| Kalman | **0.231** | 0.057 |
| grid | 0.215 | 0.033 |

**Oracle tests** (regime_mm_kf with the estimate replaced; local scratch copies only):

| S used | public PnL |
|---|---|
| real mid now / +20 / +50 / +100 / +300 ms | 79k / 440k / 750k / 888k / 1,021k |
| mid +300 ms only in the 2.5% of time around the 22 one-second moves of 20+ ticks (baseline 310k) | 395k |
| mid +300 ms everywhere except those windows | 927k |
| exact IOC prices (round(S ± 1)) fed to the Kalman as they arrive, no lookahead | 248k–377k (worse) |
| the same, corrected for each IOC's own push | 155k–372k (worse) |

**Information channels the simulator exposes:**
1. New limit-order prices (used by every estimator). Clamped orders (one tick from the opposite best) are bounds.
2. Trades with `taker_order_id`: per IOC, the deepest price swept bounds S (round(S ± 1) ≥ that price).
3. Order ids come from one counter, so an id that never shows up (not a visible new order, not a trading IOC) is an
   order placed beyond the 20 visible ticks or an IOC that did not trade. On the tape only 82 limit orders landed
   beyond 20 ticks, 80 of them within [−1 s, +2 s] of a 20+ tick move: a near-perfect jump flag, but the first one
   came anywhere from 17 ms before to 1.3 s after the mid started moving.
4. IOC prices themselves are not visible. Even known exactly they predict the mid poorly (corr² 0.02–0.04 against
   the Kalman's 0.12–0.24 on the same samples): S sits within about half a tick of the mid and the price is rounded
   to a whole tick.
5. Sizes, sides, cancel times and arrival times are independent of S in the generator.

**Literature checked:** Bibinger, Hautsch & Ristig, "Jump detection in high-frequency order prices"
([arXiv:2403.00819](https://arxiv.org/abs/2403.00819)): local maxima of bids and minima of asks reveal a jump sooner
than the mid; in our tape that is the clamped orders piling up one tick from the opposite best. Robert & Rosenbaum's
uncertainty-zones model (J. Financial Econometrics 2011) estimates the efficient price from when the traded price
changes; our generator places orders around S directly, so the order-placement likelihood is the sharper source.
Particle filters for latent order-book factors (Takada; Kyoto RIMS Kôkyûroku 1933) and IMM / regime-switching Kalman
filters (Blom & Bar-Shalom; [arXiv:2402.08051](https://arxiv.org/pdf/2402.08051)) are the general forms of what the grid
filter and regime_mm already do.

**Reading:**
- In the quiet regime (79% of the time) S is probably not worth estimating better: it behaves like round 1, where
  imbalance_mm fed the true S on generated sessions scored no better than with Ŝ.
- In the volatile regime every estimate is weak (corr² ≤ 0.10). S2 is the best and the Kalman runs the strategy best,
  so the estimate that predicts best is not the one that trades best there either.
- Most of the future-mid oracle's value is from the next few events, which cannot be estimated. A perfect estimate
  of S(t) is worth something between 0 and the jump-window oracle's +85k plus part of the volatile oracle's gain; it
  cannot be measured on the public tape because the true S is not in it.

**Ranked options:**
1. **Round 2 twin with the true S** (extend scripts/gen_tape.py with the two regimes, the volatile order offset and
   jumps, fit to round2_public). The only way to measure what a perfect S is worth before building more estimators,
   and it gives many sessions to tune on instead of one tape with ±23k noise. Largest decision value.
2. **Fit the volatile regime properly**: joint MLE of the two-regime model (switch rates, X per regime, σ per regime)
   with the forward filter, the two-regime version of scripts/x_fit_slice.py. Then the grid filter's regime
   probability should work; the current fit came from a mislabelled mixture.
3. **Jump onset detector** from clamped-order bursts (Bibinger-style bid maxima / ask minima) plus the
   invisible-order id gap, to sweep the stale book earlier. Ceiling about +85k (jump-window oracle).
4. Not worth more time: IOC-price reconstruction from sweeps (weak information), quiet-regime estimators (round 1
   evidence), deep or ML price models (too slow per event, and nothing to train on).

## Team round 2 strategies on round2_public, split by regime (2026-11-07)

Regime labels from regime_mm's own detector; pnl.csv per second.

| strategy | total | quiet | volatile |
|---|---|---|---|
| hmm_mm | **502,258** | **242,763** | **259,495** |
| regime_mm | 448,818 | 228,115 | 220,703 |
| adaptive_mm_r2 | 235,976 | 43,214 | 192,762 |
| jump_mm | 220,583 | 165,875 | 54,708 |
| lag_taker | −42,592 | 7,713 | −50,305 |

hmm_mm is best in both regimes, so a hybrid of the best engine per regime is hmm_mm itself. The round 2 twin was
dropped: one 6,000 s tape is too little data to fit regimes and jumps (an offline two-state EM on order offsets gave
volatile spells of about 1 s, against about 3 s from regime_mm's detector).

### Jump detector on hmm_mm: rejected (built as `strategies/hmm_mm_jump.py`, since deleted)

| change on hmm_mm (502,258) | pnl |
|---|---|
| CENSOR: orders one tick from the opposite best are bounds only | 423,633 |
| BOUND: a market IOC that swept to P proves S ≥ P − 1.5 | 505,747 (noise) |
| BOUND + GAP_JUMP: id gap + one-sided clamp burst → S jumps 15 ticks past the touch | 415,135 |

- At a 1-tick spread every buy joining the bid is "one tick from the ask", so censoring discards most information.
- Id gaps online: 2,162 events (0.36/s), almost all IOCs that never traded, not the 82 far limit orders. Gap clusters
  (≥ 3 in 500 ms) catch all 22 moves of 20+ ticks but raise 192 alarms; a one-sided clamp burst at a gap predicts
  only +1.5 / −2.2 ticks over the next second. Offline the far orders are a clean jump flag; online they cannot be
  told apart from non-trading IOCs.
- With the jump-window oracle capped at about +85k and no clean online signal, jump detection is closed.

## hmm_mm run on round2_public: where it earns and where it leaks (2026-11-07)

Full-logs run `logs/r2-sample` (502,258), regime from hmm_mm's own P(volatile) ≥ 0.5 (16% of the time).

| regime / fills | lots | 1 s | 5 s | 30 s | PnL at 5 s |
|---|---|---|---|---|---|
| volatile crosses | 176k | +1.33 | +1.41 | +1.48 | +250k |
| quiet crosses | 210k | +0.54 | +0.69 | +0.24 | +144k |
| quiet passive | 813k | +0.16 | +0.09 | +0.28 | +75k |
| volatile passive | 2.5k | | | | +5k |

- **Quiet passive by the size of the IOC that hit us:** up to 400 lots +181k; 400–1,000 lots −15k; over 1,000 lots
  −88k (−1.12 ticks/lot, about the push 0.0159 · size^0.63 of such an order). The big sweeps are the main leak.
  KEEP_LEVELS 0 (no resting one tick behind): 501,834; QTY 100: 474,571. Neither fixes it.
- **Inventory:** |inv| ≥ 90 for 47% of quiet time and 87% of volatile time.
- **Jumps net +30k.** Over the 22 moves of 20+ ticks, PnL from 1 s before to 5 s after is −4k when the jump ran
  against our position and +34k when it ran with it. The two visible drops in the explorer (5,398 s and 5,569 s, about
  −9k each) are up-jumps while short, recovered by the volatile sniping afterwards.
- **Very steady:** every 300 s window is positive, the worst 30 s window is −11.5k, max drawdown 12.5k; PnL per 1,000 s
  66k / 56k / 112k / 129k / 68k / 71k.
- V_EDGE_TAKE 0.4 / 0.7: 501,110 / 491,321 (0.55 kept).

### Fixing the big-sweep leak on hmm_mm: rejected (built as `strategies/hmm_mm_q.py`, since deleted)

Quiet passive fills by market lots queued ahead of our order at the fill (5 s markout): 0 ahead 345k lots +0.18;
1–50: 219k lots +0.20; 51–150: 132k lots +0.09; 151–300: 59k lots −0.26; 301–600: 35k lots −0.02; 601–1,000:
12k lots −1.86; 1,000+: 5k lots −0.42. Up to 150 ahead: +117k; beyond: −41k.

| change on hmm_mm (502,258) | pnl |
|---|---|
| cancel quiet passive orders with > 150 / > 600 market lots ahead | 393,176 / 458,220 |
| don't join a price with > 400 market lots queued | 463,646 (with the 600 cancel: 462,176) |
| KEEP_LEVELS 2 / 3 (rest further behind the touch) | 502,204 / 502,208 |
| quiet EDGE 0.2 / 0.4 | 499,237 / 482,889 |

A deep order moves up as the orders ahead of it are cancelled, and its later front-of-queue fills pay for the
occasional big sweep. The leak is the price of queue position; hmm_mm's quoting is at a local optimum.

### Filtering volatile crosses by the recent flow: rejected (built as `strategies/hmm_mm_xf.py` and `strategies/hmm_mm_v2_xf.py`, since deleted)

Rule: a volatile-regime cross needs the recent aggressive flow on its side (sign · flow/depth > 0.1). In hmm_mm's
crosses, those against the flow lost −2.2 ticks/lot after 5 s (−42k on 19k lots). On round2_public it added +24k to
hmm_mm (526,287) and +16.5k to hmm_mm_v2 (532,026), better in 5 of 6 time chunks, mirrored market equal. On arena
validation it did not hold up: hmm_mm_xf 1,288k against hmm_mm_v2's 1,291,224, and by the team's validation run the
v2 version did not beat v2 either. A gain of about one public standard error did not survive the longer tape; hmm_mm_v2 stays the strategy.

## Validation twin and tuning hmm_mm_v2 on it (2026-11-07)

**Note:** the numbers below come from a first version of the twin (`scripts/gen_val_twin.py`, my own `s_path` code in
analysis/generator.py). It was dropped when merging with the team's `scripts/path_twin.py` (commit 9c3593a), which
does the same job with a better calibration: hmm_mm scores 0.92 of real on it (1,236,914 vs 1,290,784) against 1.13
on the first version. Rerun the comparisons below on path_twin tapes before relying on them.

**Twin (first version):** `scripts/gen_val_twin.py data/validation_r2/pnl.csv` turned the SimpleProbeMM run on the 12,000 s validation
session (+10 lots held from t = 1 s, so mark = 10,001 + pnl / 10) into that session's 1 s price path and drives the
generator's S through it (`generate(..., s_path=...)` in analysis/generator.py): the path's 1 s moves of 5+ ticks as
jumps inside their second, a Brownian bridge in between, storms where the 30 s RMS of 1 s moves exceeds 3 ticks (43%
of the time; the path has exactly the 51 moves of 20+ ticks the PnL analysis found), IOC pushes fading over 30 s.
Order flow from `round2_drop_v2` (fitted on round2_public, kept on the team server in logs/generator/).

| | twin seed 1 | twin seed 2 | real validation |
|---|---|---|---|
| SimpleProbeMM | 4,690 | | 4,800 |
| hmm_mm_v2 | 1,449,169 | 1,473,686 | 1,291,224 |
| hmm_mm | 1,465,549 | | 1,290,784 |
| regime_mm | 962,104 | 906,828 | 1,032,909 |

The price path is right (probe within 2%); hmm-family strategies come out about 13% high (the same optimism as
synthetic public: +15%), regime_mm about 9% low.

**Screen of hmm_mm_v2 on twin seed 1** (28 one-at-a-time variants): moves beyond ±12k only for K_FLOW 0.5 (+83k;
1.2: −154k), K_INV 0.001 (+49k; 0.004: −249k), EDGE_TAKE 0.35 (+29k; 0.18: −52k), SIG_V 15 (+25k; 6: −91k).

| change | twin seed 2 | real round2_public | generated public-like tape |
|---|---|---|---|
| K_FLOW 0.5 | +78k | −34k | +53k |
| K_INV 0.001 | +53k | −13k | +19k |
| EDGE_TAKE 0.35 | +23k | −6k | |
| SIG_V 15 | +37k | −41k | +3k |
| all four | +135k | −83k | +55k |

**Reading:** the quiet-regime gains appear on any generated tape, path or not, and reverse on the real one. The
generator's order flow does not reproduce how the real market responds to aggressive flow and inventory, so flow and
inventory settings cannot be tuned on generated tapes. The twin is trustworthy for its price path (jump timing, storm
share), not its order flow. No change to hmm_mm_v2. Next step for the twin: make the generated flow → price response
match the real one (on round2_public a quiet flow weight of 0.8 beats 0.5), then repeat the screen.

**Validation result of the twin-tuned test version** (`strategies/round2/hmm_mm_v2_twin.py`: K_FLOW 0.5, K_INV 0.001,
EDGE_TAKE 0.35, SIG_V 15): **1,293,574** on arena validation against hmm_mm_v2's 1,291,224, a tie. The first twin
predicted about +135k, round2_public about −16% (−83k on 515k); the real effect was +2k. Neither guide transferred:
the twin overstated the gain and the half-length public tape overstated the loss. hmm_mm_v2 stays; settings that
move order-flow and inventory behaviour need a real tape (or a twin whose flow response is checked against one).

## Deeper research pass: Round 2 audit and next experiments (2026-11-08)

Full synthesis: [round2-deeper-strategy-research.md](../../reports/round2-deeper-strategy-research.md).

**No, not every researched technique has been tried on Round 2.** Several ideas were tested only on Round 1/`sample_v2`, and others arrived only inside combined strategies. The current reference is **`hmm_mm_v2` at 515,547**, versus `hmm_mm` at 502,258; that difference is within the approximately 26k SE, so use v2 as the baseline and `hmm_mm` as the clean secondary anchor without calling the gain significant. `regime_mm_kf` / `regime_mm` scored 469,844 / 448,818. The exact grid is a settled failure (262k both regimes, 324k volatile only, versus about 470k KF), exact IOC-price oracles also lost (248k–377k; push-corrected 155k–372k), and the ID-gap/clamp detector scored 415,135. Most future-mid oracle value came from orders and IOCs inside the lookahead window, not better present-S information. `RegimeFilter` has only the approximately 172k combined `storm_mm` result, which also changed the far trigger, momentum guard, quotes, cap and storm response; its isolated effect is unknown. Static queue-ahead thresholds and recent-flow cross filters also failed to produce robust gains. Original IOC request prices and invisible IOCs are not observable, so a two-channel estimator requiring them is not a valid proposal. Executed sweep price/volume remain observable auxiliary bounds only.

### Highest-priority untried or under-tested experiments

1. **Posterior executable edge plus 5 ms shadow survival.** For direction `d`, price `p`, posterior arrival mean/variance `mu/v`, and required edge `c0`, compute `Pprof = s * Phi((d*(mu-p)-c0)/sqrt(v))` and `EVlot = s * (d*(mu-p)-c0)`, where `s` is measured by fixed-schedule virtual orders that mature after 5 ms. Keep current-touch orders, sizing and acknowledgement logic unchanged. Ablate **B1 posterior only (`s=1`)**, **B2 shadow survival only with the current point-estimate edge**, then **B3 joint**. Sample candidates independently of the final gate, in a tiny side × regime × signed-edge table, and audit 1/5/30 s markouts plus missed positive value. This is different from probability-scaled quote edges because it measures the actual execution race. Gao & Wang (2018, revised 2020) support the importance of latency, not this exchange-specific formula ([arXiv:1806.05849](https://arxiv.org/abs/1806.05849)).
2. **Sparse process-mixture KF.** On the same visible-limit reading, compare ordinary innovation variance `P+R` with displaced-process variance `P+J+R`, Bayes-reweight two scalar KF updates, and collapse mean plus total variance, including between-component disagreement. Run **A1 process mixture only**, **A2 clipped/ordinary-weighted offset and noise adaptation only**, then **A3 both**. Diagnose innovation calibration and nuisance drift before PnL. This is constant-memory and unlike the failed grid or ID-gap detector, but it is **unproven, adds no information, and detects/adapts rather than predicts jumps**. Aravkin, Burke & Pillonetto (2014) motivate process-heavy-tail tracking with an offline Student-t smoother; this online mixture is only an engineering analogue ([arXiv:1303.5588](https://arxiv.org/abs/1303.5588)).
3. **Separate continuous volatility from permanent jump exposure.** First only log decayed squared filtered-S increments and adjacent absolute-increment products for continuous variation, with a separate jump accumulator. Then replace **only** v2's volatile KF diffusion scalar with the continuous estimate, leaving HMM posterior, fair values, orders, cap and size unchanged; defer edge/exposure changes. Corsi, Pirino & Renò (2010) motivate separating continuous and discontinuous variation, but their asymptotics and forecasting results do not transfer directly to irregular tick-rounded observations ([paper record](https://openaccess.city.ac.uk/id/eprint/4435/)).
4. **Pre-sweep thin-depth plus signed-S predictor.** Diagnose held-out prediction of a trade through the current touch using visible touch depth/removals, spread, executed flow and `S_hat` lean—never invisible IOC size. If stable, cancel/shrink only the predicted adversely exposed opening quote while retaining unwinds, taking and fair values. Zheng, Moulines & Abergel (2012) motivate this executable-sweep target in equities; it is not evidence that exogenous S jumps here can be predicted ([arXiv:1204.1381](https://arxiv.org/abs/1204.1381)). This is before the sweep, unlike the failed post-sweep guard.
5. **Passive-unwind competing-risk survival.** Shadow queue positions must update from trades and snapshot quantities, not ghost cancel rows. Estimate fill and S-invalidation hazards in a few price-distance/queue/S-gap/spread cells. Change **only retention of inventory-reducing passive orders**: wait when fill-before-invalidation probability clears a frozen threshold, otherwise take the existing unwind action. Do not touch entries or directional taking. Lo, MacKinlay & Zhang (2002) support state-dependent execution survival and warn that hypothetical fills need careful validation ([author page](https://web.mit.edu/Alo/www/Papers/limit10.html)); Arroyo et al. (2023) supply the LOB survival target, not a deployable Transformer ([arXiv:2306.05479](https://arxiv.org/abs/2306.05479)).
6. **Isolated `RegimeFilter`.** First run it as a no-action shadow and score held-out calibration. Then replace only v2's policy regime-selector probability at `P_ON`, keeping KF diffusion, far triggers, momentum guards, quotes, caps, crosses and exits unchanged. Do not infer the filter's value from `storm_mm`.
7. **Marginal inventory opportunity cost, only after diagnostics.** Reuse shadow outcomes to measure whether opportunities actually blocked near ±100 have positive survival-adjusted forward value; stop if not. If present, add a small near-cap shadow price to **same-direction opening crosses only**, with no forced exits, current-edge size reduction, passive-unwind change or overshoot. Compare against a fixed near-cap extra edge of equal average strength. Existing evidence favors holding, so a negative result closes this line instead of triggering a large DP build.
8. **Carry/spread PnL decomposition.** On existing full logs, split prior-inventory carry (`I * delta_mid`) from signed execution capture (`mid_at_fill - fill_price`), reconciling final markout and penalties, by regime, jump window, opening/unwind and holding bucket. This is a diagnostic, not alpha: it reveals whether an apparent execution gain is just fortunate jump-facing inventory.

### Lower-priority ideas and research infrastructure

- **True-S Round-2 oracle and joint MLE regime fit:** high decision-value infrastructure, but the first twin was dropped because one 6,000 s tape did not identify regime durations, jumps and offsets consistently. The later path-based twin improved price-path calibration but still misrepresented flow-to-price and inventory responses: several synthetic tuning gains reversed on the real public tape. Restart only behind held-out acceptance tests for spread, offsets, sweeps, impact, regime durations, jumps, order-flow response and markouts. The joint MLE is offline infrastructure, not a reason to rerun the grid. Only after acceptance, replace v2's `S_hat` with true S while changing no policy and compare matched filtered/true-S seeds.
- **DP inventory thresholds and forward keep/cancel queue-value tables:** viable only after positive blocked-opportunity and survival diagnostics. Freeze one inventory-indexed opening threshold table, or one hazard-based keep/cancel rule, on training seeds; do not build a full HJB or repeat static queue-ahead thresholds. Guéant, Lehalle & Fernandez-Tapia (2012) motivate inventory control, not a locally optimal table ([arXiv:1105.3115](https://arxiv.org/abs/1105.3115)); Lehalle & Mounjid (2016, revised 2018) connect queue placement, adverse selection and latency ([arXiv:1610.00261](https://arxiv.org/abs/1610.00261)).
- **Post-jump acknowledgement-gated impulse, capped BOCPD, inventory-conditioned thin-queue step-back, queue hysteresis/shrink, original imbalance guard, inherited flow/markout gates, multilevel/participant OFI, and static deeper-book/microprice variants:** each needs a one-factor ablation and has less direct support than the top items. BOCPD follows only if the simpler process mixture is insufficient; Adams & MacKay (2007) provide the run-length framework, not jump prediction or PnL ([arXiv:0710.3742](https://arxiv.org/abs/0710.3742)). Round-1 queue gains are not established Round-2 gains.
- **Hawkes/ACD only after residual-memory diagnostics:** test time-integrated residuals separately for raw IOCs, executed IOCs, sweeps and large sweeps across held-out chunks. State-selected executions can look excited under Poisson raw arrivals. Only reproducible residual memory permits replacing the existing decayed arrival-count component with a small marked intensity. Xu & Zhou (2020) supply the time-rescaling diagnostic ([open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6953867/)); Bacry & Muzy (2014) distinguish excitation from impact/feedback ([arXiv:1301.1135](https://arxiv.org/abs/1301.1135)); Engle & Russell (1998) introduced ACD ([DOI](https://doi.org/10.2307/2999632)). Their empirical results are not evidence on this tape.

Engineering/evaluation work is complementary rather than direct alpha: use matched jump/no-jump seeds and mirrored jump signs, select on training seeds by **E[PnL] − 0.25 SD**, validate on unseen seeds, stress vol/IOC impact/jump rate/spread/cancels independently, and audit 5 ms acknowledgement races, duplicate orders, stuck states, rejected shrinks, actual sweep slippage and worst-case live exposure. Until a Round-2 generator passes acceptance, do not claim matched true-S or reliable synthetic tuning validation.

### Closed paths and execution order

Do **not** spend the next iteration on invisible/exact IOC reconstruction, ID-gap jump detection, another full grid/particle filter, forced exits/current-edge sizing/overshoot, passive signal-side entries, penny-jumping/deep calm sweeps/reactive sweep guards, blanket censoring/static queue thresholds/recent-flow filters/OLS fair values/mean IOC push, fixed execution schedules, full HJB/deep nets/RL, or contest-specific exploits. These either have direct negative local evidence, require unavailable information, or attack the wrong permanent-impact mechanism.

Run **diagnostic → single-change backtest → matched multi-seed validation** in this order: baseline/carry/race accounting; shadow 5 ms calibration then B1/B2/B3; mixture diagnostics then A1/A2/A3; continuous-vol D1; pre-sweep diagnostic; passive-unwind survival; isolated `RegimeFilter`; blocked-value diagnostic then marginal inventory; only then DP/queue and gated Hawkes/ACD. Use the public tape as confirmation, not a repeated tuning set. Its roughly 23k–26k SE and dependence on a few jumps make small gains and even the 13k v2 advantage inconclusive; pre-register one mechanism and stop on failed diagnostics rather than searching thresholds until one wins.

## Round 3: alpha -> target position -> execution: rejected (built as `strategies/round3/alpha_target.py`, since deleted, 2026-11-08)

**Why a new architecture.** Everything that scores on round 3 is a "fair ± edge" market maker with an inventory skew,
and all of them sit at |inventory| = 100 half the time. PnL scales with the cap (ladder_mm at 50 lots: 362k, hmm_mm at
100: 568k), and freeing capacity early loses (V_RECYCLE, K_INV sweeps). So the score is the carry of a ±100 position:
what matters is which way it points and how cheaply it flips. That is how the top IMC Prosperity / Optiver Ready Trader
teams treated their directional products: forecast → target position → cost-aware execution. Outside search turned up
no "leverage" trick beyond what the position limit allows (Prosperity teams hard-capped, and one found a bigger cap
scored worse); our flat 10k/s penalty overshoot was already ruled out on round 3 (see `round3_strategy.py`).

**What it does.** Online RLS (the `tools/online_rls.py` copy was deleted with it; forgetting about 17 min, clipped residuals) learns the 1 s mid
move from [S_hat − mid, size^0.7-weighted IOC flow, touch imbalance, mid − 60 s EMA, 1 s momentum], starting from the
prior "mid closes 90% of its gap to S_hat". Target = ±100 once |forecast| > FLIP, otherwise hold. Take every level
priced TAKE_EDGE through the expected mid; rest at the touch only towards the target.

**Data behind the features (round3_public analysis).** Passive fills from IOCs under 22 lots lose 0.2 ticks/lot by
1 s; from 45+ lots, 1.0. Big IOCs move S for good, so the impact-weighted flow is its own feature. bb_pctb / zscore
over 60 s have Spearman −0.15 with the 5 s move (mean reversion), and 1 s returns autocorrelate +0.11 (mid lag).

**Result on round3_public: 390,011** (hmm_mm 569,499), penalty 0, 7,949 fills, 607k lots, PnL steady in every
1,000 s slice (43k–87k). Rejected as it stands. Fill split (logs/r3-sample, markout against the 1 s mid):

| | lots | 1 s | 5 s | 30 s | PnL at 30 s |
|---|---|---|---|---|---|
| takes | 494k | +0.67 | +0.89 | +0.71 | +352k |
| passive (towards the target only) | 113k | +0.30 | −0.10 | −0.41 | −46k |

The takes earn what hmm_mm's do (538k lots at +0.93 by 5 s), so the online-learned forecast found no edge beyond
the S_hat gap: the take signal looks saturated. The 180k gap is the passive leg. hmm_mm trades 1.06M lots, about
550k of them passive (two-sided quiet-regime quotes: 95k opening at +0.94 by 30 s, 455k closing at about zero).
Resting only on the side towards the target gets filled exactly when the price runs against it.

**Next ideas if this shape works (untried):** policy distillation from the hindsight DP (`scripts/make_dp_oracle.py`):
fit a small hard-coded classifier from these features to the oracle's position sign; a second horizon (5 s) in the
RLS so the target uses the longer carry; per-inventory take thresholds from a DP over the opportunity flow.

## Round 3: fading the overshoot: rejected (built as `strategies/round3/fade_hmm.py`, since deleted, 2026-11-08)

Found in the 1 s mid series of `logs/analysis/round3_public.json` and `r3_eval_twin.json`. After a 1 s move of 3–20
ticks the mid keeps going for about 2 s, then gives back 1–7 ticks over 10–30 s (same sign in every size bucket on
both tapes, t up to −3.7 with overlapping windows). The strategy was `round3_strategy.py` plus a fair-price shift of
1.5 ticks against such a move, from 2 s to 22 s after it.

**Result on round3_public: 169,250** (the same strategy without the shift: 569,499), penalty 0, 4,771 fills. The
reversal is real on average, but it is a few ticks spread over 20–30 s, while the shift spends the whole ±100
position on it. That position no longer rides the lag takes worth +0.9 ticks/lot by 5 s, and each flip pays the
spread. A slow mean-reversion signal cannot compete with the fast lag edge for the same capacity.

## Round 4: structural alternatives (2026-11-08)

From the round 4 analysis in [DATA_ANALYSIS.md](DATA_ANALYSIS.md#round4_public-same-order-placement-trending-price).
Round 4 is the last round and counts 1.2, and we are 4th, so these are ranked by upside, not safety.

1. **Jump sniper with leverage** (`strategies/round4/r4m_snipe.py`). Far-placed new limit orders show a jump about 6 ms
   after it happens and before the touch moves. An IOC takes the stale side past the 100 cap, and the excess is
   closed within 0.5 s. Tape-only estimate: about 10 trades, +85k per 6,000 s. It is the only leverage design that
   fits Jakub's rule (enter before the move shows; [LEVERAGE.md](LEVERAGE.md)). Risk: about 10 events, and a false
   900-lot round trip costs 6–30k.
2. **Momentum rider (a third mode).** Moves continue for 1–5 s (past 1 s return → next 5 s, Spearman +0.20) and
   partly revert at about 30 s. Instead of quiet market making or storm sniping, hold ±100 in the direction of a
   strong 1 s move and flatten after about 10–20 s, before the reversal. First step: a momentum lean in fair
   (`r4m_alt.py` K_MOM / V_K_MOM).
3. **Taker-only quiet regime** (`r4m_alt.py` Q_PASSIVE 2) or reduce-only quoting (Q_PASSIVE 1). Passive fills at the
   touch earn about 0 by 30 s (quiet −0.17, storm −0.85 ticks/lot), but they fill the inventory that the takes
   (+2.96/lot in storm) need. 294k of stale-book bursts are missed with room left.
4. **Deep passive quotes in quiet.** Quiet passive fills behind the touch earn +1.89 ticks/lot by 30 s (57k lots),
   the only passive leg that pays. A ladder 1–3 ticks behind, instead of the touch.
5. **Position target with a 30 s exit clock** (the alpha_target design, revived). It failed on round 3 because the
   only edge was S_hat − mid. Round 4 has 3.6× the 1 s predictability (R² 0.13 vs 0.036) and a clear 1–5 s
   continuation / 30 s reversal shape that a target-position design can trade directly.
6. **Inventory exit timing from the 30 s reversal.** Past 10 s return → next 30 s is −0.14. Lean K_INV, or an exit
   rule, on how long and how far the position has run, not only on its size.

### Round 4 results (round4_public, 2026-11-08)

Base `r4m_safe` = hmm_deep_clear_r3 with K_S 0.95, SIG_Q 0.8, V_K_S 0.9 (quarter sweeps `sweep_r4a/b`: each one-step
winner was up in all 4 quarters): **1,073,369**, 0 penalty (hmm_deep_clear_r3 1,046,361).

| Variant | round4_public | penalty | note |
|---|---|---|---|
| far-order sniper (`r4m_snipe`, SNIPE_FAR 10, 2 confirming orders) | 1,065,984 | 27,691 | 11 snipes, every filled one +5 to +16 ticks/lot, but fills small |
| far-order sniper, SNIPE_FAR 15 | 1,082,973 | 3,786 | +10k |
| **id-gap sniper** (IDG_SNIPE 1) | **1,109,933** | 13,270 | +37k; catches jumps beyond the 20-tick public depth |
| id-gap + far 15 | 1,124,599 | 11,929 | `r4m_risk_b` |
| **id-gap + far 15 + momentum 0.1 / 0.5 s** | **1,156,545** | 7,963 | `r4m_risk_a`; robustness 6/6 chunks, mirrored 1,156,881, bootstrap 8/8 |

- **Why the far-order trigger misses the big jumps:** the public book shows 20 ticks from the best price, so an order
  placed 130 ticks away after a 130-tick jump is invisible. The two biggest jumps of the tape (−140 at 170 s, +130 at
  1740 s) never fired it. `IdGapJump` sees them through missing order ids plus orders clamped inside the stale touch.
  Trace of the id-gap sniper: 1394 s bought 505 at 9688, sold 302 at 9795 (+107 ticks/lot, about 32k); 2171 s +73/lot;
  2541 s +28/lot; 1740 s only 4 lots (the stale asks were gone in 5 ms). The settings sit on a plateau (IDG_DEPTH
  5/10/15, IDG_S_MIN 10/20/30, hold 0.2–1 s all within 1k); IDG_MISSING 2 fires falsely (penalty 34.6k).
- **Per quarter** (r4_q1..q4): sniper +30k / +23k / +2k / 0 (never loses, pays when a big jump comes); momentum
  +1k / −18k / +9k / +39k (net positive, inconsistent).
- **Closed on round 4:** quiet taker-only (Q_PASSIVE 2) −8%, reduce-only quoting −3%, deeper quotes (EDGE 0.6–1.5,
  KEEP_LEVELS 2–3, QTY 300) −4 to −10%. The quiet passive leg pays through turnover even though its 30 s markout is
  about 0. Momentum K_MOM / V_K_MOM 0.2 or a 2 s window lose; 0.1 over 0.5 s is the best of the grid.

### ML take rule: rejected (round 3, 2026-11-08)

`scripts/take_candidates.py` logs every take candidate of a strategy (21 features at decision time, the strategy's
own decision, and the executable 5 ms markout at 1/5/30 s); `scripts/meta_label_eval.py` compares policies on it
leave-one-quarter-out. Offline on hmm_deep_clear_ks / round3_public: filtering the strategy's takes (meta-labelling)
+1.5% ridge, −24% HGB; taking any candidate a ridge model rates positive doubled the summed markout (42k vs 22.5k
ticks·lots), also out of distribution on round 2. In the simulator (`r3_ks_mltake.py`, ridge fitted on q1+q2) it lost
**16–23% in every quarter, including its training quarters**, and every lower EDGE_TAKE / V_EDGE_TAKE lost too. The
per-lot markout label ignores that extra takes use up the ±100 room and displace passive fills. Any ML here needs the
marginal PnL with inventory as its label, which is what backtest sweeps already measure. The HGB storm gate
(`r3_ks_hgb.py`) lost 15–50% for a related reason: it predicts that a big move is coming, not the millisecond to take.

### Round 4: more tries on top of r4m_risk_a (all rejected, 2026-11-08)

Quarter tapes plus the full tape; `r4m_risk_a` scores 317,269 / 290,067 / 297,272 / 250,387 and 1,156,545 (`r4m_dev.py`
holds the switches).

- Momentum only past a threshold (MOM_TH 2/4/8 ticks, with K 0.1–0.3): 1,118–1,143k.
- Momentum split by regime (quiet 0–0.05, storm 0.15–0.2), although storm momentum is about twice as strong: 1,122–1,136k.
- Slow reversal lean against the 10–40 s move (K_REV / V_K_REV 0.02–0.1): −2% to −47%, worst in storm.
- Sniper re-firing within a jump (cooldown 0.1 / 0.25 s): identical (the stale side is gone after the first sweep);
  with IDG_DEPTH 20: −32k.
- Passive exit of the excess at S_new −2 / 0 / +2 for the post-jump IOC surge: within ±6k (too few excess lots).
- Trigger and constant nudges with the sniper on (SNIPE_FAR 12/20, CONFIRM_N 3, IDG_WINDOW 40 ms, SNIPE_POS 600,
  EDGE_TAKE 0.3, V_EDGE_TAKE 0.5/0.65, V_QTY 300, QTY 200, K_FLOW 1.2, V_K_FLOW 0.9, P_ON 0.25/0.35, K_INV 0.001,
  V_SWEEP, Q_SWEEP): all equal or lower.
- Reactive leverage bound on round 4 (`scripts/lever_oracle.py`, entry after the mid moved X ticks in 20 ms): the
  expected net per excursion is negative for every X, hold and Q (Q 900: −6k to −29k), as on round 3. Momentum does
  not rescue chasing; only the id-gap trigger enters early enough.

### Round 4: other families and a learned fair value (2026-11-08)

**Family screen** (19 existing strategies unchanged on round4_public): `r4m_risk_a` 1,156,545; HMM-skeleton variants
`conf_sniper` 996,610, `round3_strategy` 995,944, `r3_tp_mm` 952,786; **`alpha_as_mm` 943,575** (A-S market maker around an
online RLS fair value, the only competitive non-HMM design); `regime_mm_kf` 809,577; `ladder_mm` 770,170; `jump_mm`
526,403; `r3_lever_mm` 363,527 (penalty 277k); `sband_mm` 60,480; `trend_follower` −12; `lab_kimi_sniper` −29,778;
`lab_kimi_valuemm` −235,315; `lag_taker` −245,795; `adaptive_mm` −482,394; `imbalance_mm` −750,599; `queue_mm`
−846,641; `storm_mm` −1,713,582; `flow_momentum` −2,805,804; `penalty_mm` −6,273,220. Quoting without a hidden-price
estimate is run over by the momentum-driven IOCs, following momentum directly pays the spread on every trade, and
taking without making loses the passive turnover.

**Learned fair value on the skeleton** (`strategies/round4/r4m_rls.py`): `r4m_risk_a` plus an RLS per regime of the mid
move 0.5 s ahead on [1, S_hat − mid, imbalance, flow, 0.5 s momentum], fair blended toward mid + prediction by K_RLS.
Full tape: K_RLS 0.2 / 0.3 / 0.5 / 1.0 give 1,132,967 / 1,126,783 / 1,135,216 / 1,102,282; quiet-only 0.5 1,153,562;
storm-only 0.5 1,135,448; 1 s horizon 1,160,549 (+4k; quarters +16k / −2k / −5k / −3k, noise). The fixed-weight fair
already holds what an online linear model learns from these features, as on round 3.

### Round 4: how far can sniping go? (2026-11-08)

**Simulator ceiling** (`scripts/make_lever_oracle.py` on the round-4 hindsight schedule, Q 900, base `hmm_imm_b2`
1,039,171): all 349 excursions 2,185,632 (+1,146k, penalty 2.04M); the ones worth ≥ 5k on the tape 1,926,725 (+888k,
penalty 625k); ≥ 20k (about 9 jumps) 1,437,641 (+398k, penalty 59k). Perfect timing is worth about +0.9M per 6,000 s;
`r4m_risk_a`'s sniper takes about +51k of it (5%). The oracle enters before the jump; every trigger we have fires after it.

**New triggers and leverage rules on `r4m_risk_a`** (`r4m_dev2.py`, `r4m_dev3.py`; full tape, `r4m_risk_a` 1,156,545):

| Variant | round4_public | penalty | quarters vs risk_a |
|---|---|---|---|
| multi-level sweep trigger, 3 / 4 / 5 levels | 1,079k / 1,082k / 1,087k | ~20k | −2k to −53k, all 4 down |
| depth-collapse trigger (side < 0.3 / 0.2 of EMA with an id gap; 0.3 with 2 gaps) | 1,080k / 1,088k / 1,126k | 15–26k | mostly down |
| post-jump lean in fair (0.5–2 ticks for 3–10 s) | 1,154k–1,156k | 7,963 | ±1k (no effect) |
| clamp-confirmed id gaps (2 missing ids + 2 / 3 clamps) | 1,154k / 1,157k | 13k / 8k | −3k / 0 |
| edge leverage: fair 5 / 8 / 12 / 20 ticks through the book | 795k / 1,095k / 1,149k / 1,156k | 511k / 81k / 24k / 9k | worse the lower the threshold |

Every extra trigger adds false fires faster than real jumps, and leverage on our own S_hat edge is reactive (the
Kalman filter learns the jump from the same late orders). With the information the feed gives, the id-gap trigger is
the only one early enough; the remaining 95% of the ceiling needs knowing a jump before it happens, and calm
stretches are memoryless (DATA_ANALYSIS.md).

### Round 4: combined jump predictor (rejected, 2026-11-08)

`scripts/jump_feats.py` records 16 detector features every 5 ms (id gaps, clamped and visible new orders per side,
signed far orders, multi-level sweeps, spread and its jump, per-side depth against its EMA, p_vol, Kalman variance,
S_hat − mid and its last change, jump votes). `scripts/jump_eval.py` fits a regularised logistic (C 0.05, mirrored
up/down) for "mid moves ≥ L ticks in the next 300 ms" on one half of round4_public and scores the other half by tradable
value per fire: direction × (mid 300 ms later − mid when the order lands) − half spread, one fire per second at most.

| L = 20 | fires | value (ticks/lot, summed) | fires worth > 5 ticks |
|---|---|---|---|
| IdGapJump rule, test H1 | 7 | **+441** | 6 |
| ensemble, test H1, 4 thresholds | 8–29 | −212 to −368 | 0–4 |
| rule OR ensemble, test H1 | 13 | +318 | 7 |
| IdGapJump rule, test H2 | 0 | 0 | 0 |
| ensemble, test H2 | 2–18 | −15 to −66 | 1–2 |

L = 10 gives the same picture. The model's weight goes to p_vol and the spread: it learns that big moves happen in
storms, not when, so it fires during storms after the move has started and loses the spread. The weak signals carry
no timing information beyond the id-gap rule. Also note that the rule fired only in the first half of the tape, which
matches the sniper's per-quarter gain (+30k / +23k / +2k / 0).

### Round 4: passive-leg ideas from research_notes (rejected, 2026-11-08)

`strategies/round4/r4m_pass.py` (switches on `r4m_risk_a`, 1,156,545):

- **Queue-aware quoting** (`QMAX`: no new inventory-increasing passive order at the touch behind more than 300 / 600 /
  1000 / 2000 market lots; research_notes queue-position §1, §4): 1,153k–1,156k, every quarter within −1.4k. The
  touch is usually short enough that the rule rarely binds.
- **Convex inventory skew** (`INV_P` 1 / 2 / 4, GLFT boundary taper): 1,085k–1,097k, all 4 quarters down 7–26k; with
  K_INV 0.003 1,032k. Holding the position near ±100 is where the money is; pushing it back toward flat costs.
- **Adverse-selection-adapted edge** (`AS_K` 0.5 / 1 / 2, EWMA of the move 100 ms after our passive fills): identical
  to the base. Our passive fills are not adverse at 100 ms on average (the EWMA stays below zero), so the widening
  never switches on; the passive loss shows up at 30 s, not 100 ms.

### Round 4: r4m_risk_a vs Jaro's r4_best_snipe (2026-11-08)

Two independent builds of the same design (hmm_deep_clear_r3 + id-gap jump sniper past 100 + storm momentum lean) tie
on the arena validation tape: `r4m_risk_a` 1,965,048.04 (penalty 12,138), Jaro's `r4_best_snipe` 1,963,678.04 (penalty
11,261; public 1,142,427). Nothing transfers between them on round4_public: our constants (SIG_Q 0.8, V_K_S 0.9, K_S 0.95)
cost his file 4–23k, a quiet momentum lean or the far-order trigger add 1–2k (noise); his short storm fade (fair against
the 2–3 s move, 0.02–0.1) costs ours 5–74% (his works only next to his 1 s EMA momentum term, as a band-pass). Without his
quote hysteresis / 3 s fade his file loses 4k / 17k.

### Round 4: component alternatives on r4m_risk_a (2026-11-08)

`strategies/round4/r4m_comp.py` (r4m_risk_a plus Jaro's quote hysteresis switch), quarters and full tape (base 1,156,545):

- **Quote hysteresis** (new quiet quotes need EDGE_NEW, resting ones stay until below EDGE): 0.25 / 0.25 with EDGE 0.1 /
  0.3 with EDGE 0.15 / 0.35 → 1,157k / 1,160k / 1,160k / 1,160k. EDGE_NEW 0.3, EDGE 0.15 is up in all 4 quarters, but
  only by 0.5–1.4k each (+0.3%).
- **Regime detector:** IOC gaps as a second observation (H_IOC) 1,149k; H_LR 0.001 1,167k (quarters +21.5k / −1.3k /
  +1.0k / −9.6k, inconsistent); H_NMAX 3 1,115k.
- **Hidden-price filter:** deep readings off / DEEP_R 1.5 / 6 → 1,150k / 1,152k / 1,150k; jump boost JUMP_TH 4 / 10,
  JUMP_N 2, JUMP_P 50, off → 1,143k / 1,157k / 1,142k / 1,156k / 1,157k; KF_LEVELS 1 / 3 → 1,151k / 1,157k;
  KF_LEVELS_V 2 / 5 → 1,154k / 1,107k; C_LR 0.002 1,154k; R_MIN 0.3 1,147k; id-gap variance boost (IDG_P 25) no change.
- **Flow check off:** 1,157k (no change). **Order manager:** KEEP_LEVELS 0 / 2 → 1,149k / 1,156k; MIN_NEW 30 1,135k.

**A third base with the same sniper:** `strategies/round4/r4m_imm_snipe.py` = Jakub's `hmm_imm_b2` (IMM hidden-price
filter) plus the r4m_risk_a sniper and momentum lean. Plain port reproduces b2 (1,039,171); with the id-gap sniper
1,131,139, plus the far-order trigger 1,087,322 → with everything 1,143,398 (quarters 321k / 285k / 298k / 238k).
The sniper adds about +56k and momentum +34k on this base too.

Three independent bases (our deep_clear-based r4m_risk_a, Jaro's r4_best_snipe, Jakub's IMM base) all land at
1.14–1.16M on round4_public with the same sniper: the design family is at its ceiling with the information the feed
gives. GitHub has no other team's code for this arena (only QminersOrg/qminers-algo-arena itself, one commit, no forks).

### Round 4: ideas from public competition code (2026-11-08)

GitHub, worldwide: the closest analogues are the IMC Prosperity write-ups (TimoDiehm/imc-prosperity-3 "Frankfurt
Hedgehogs", ericcccsliu/imc-prosperity-2, chrispyroberts/imc-prosperity-3, CarterT27/imc-prosperity-3), Optiver Ready
Trader Go repos, hftbacktest, and Avellaneda-Stoikov implementations. No public code for this arena besides
QminersOrg/qminers-algo-arena. What carries over:

- **Their biggest edges came from reverse-engineering how bot flow moves the hidden price** (a "smart taker" filling
  orders near a hidden fair value, an "informed trader" buying at daily lows). Our analogue is the generator's permanent
  IOC impact (fit 0.093 · size^0.3). `strategies/round4/r4m_push.py` moves S_hat by PUSH_K · sign · size^PUSH_B on
  every market IOC at once: PUSH_K 0.03 / 0.06 / 0.093 / 0.15 → 1,124k / 1,088k / 1,028k / 918k (r4m_risk_a 1,156,545);
  as a replacement for the flow term (K_FLOW 0) 1,147k, half-and-half 1,152k. The flow term already prices this push;
  adding it again double-counts.
- **"Wall mid"** (fair = mid of the deepest levels, which a designated bot quotes around the true price): our analogue is
  the generator's new orders at S ± X, which the Kalman filter already reads.
- **Ready Trader Go** write-ups: mid / imbalance market makers with spread from recent volatility; nothing beyond
  imbalance_mm.

### Round 4: new-architecture pass (position age, weak jump tier, order-book layer; 2026-11-08)

Base `r4m_risk_a` 1,156,545 (quarters 317,269 / 290,067 / 297,272 / 250,387). Sweeps on .225 (quarters) and .171
(full tape); grids in `scripts/grid_{hold,ioc,tier,book,book2}.txt`.

**Where the storm money sits** (`scripts/markout_curve.py` on the r4m_risk_a full-logs run). Storm takes mark out
+1.8 / +3.2 / +3.8 ticks/lot at 1 / 5 / 30 s; storm passive fills that close a position −1.0 / −2.6 at 5 / 30 s
(34k lots). At the cap in storm (31% of seconds) the held ±100 carries +3.2 ticks/lot over the next 1 s and +7.0 over
5 s, and about 0 by 30 s **for a fixed position**. But by position age (time since the last fill that grew |inv|) the
next-5 s carry per lot *rises*: 4.1 (< 1 s), 7.1 (1–2 s), 8.9 (3–5 s), 11.8 (5–8 s), 14–15 (8–20 s). Positions that
survive sit on persistent trends; the 30 s reversal comes after our flips, not while we hold. So an exit clock
(architecture 3's "post-move" specialist) has nothing to take, and it was not built.

**Age-based specialists** (`strategies/round4/r4m_hold.py`): rejected.

| variant | q1 | q2 | q3 | q4 | full |
|---|---|---|---|---|---|
| storm flip hysteresis after 3 s, +0.25 / 0.5 / 1 / 2 ticks | 316k / 315k / 305k / 297k | 290k / 291k / 287k / 272k | 297k / 294k / 291k / 270k | 249k each, 243k | 1,155k / 1,151k / 1,134k / 1,084k |
| hysteresis 0.5 after 2 s / 5 s / ramped | 321k / 316k / 306k | 292k / 291k / 283k | 294k / 297k / 291k | 248k / 251k / 249k | 1,156k / 1,156k / 1,131k |
| quiet age skew (0.005–0.02 per lot after 5–10 s) | = | −0.3k | = | = | = (quiet positions never age) |
| storm takes as IOC (no resting leftovers), ± hysteresis / sweep | 311k / 310k / 311k | 291k | 298k / 295k / 298k | 251k / 250k | — |

Flips must stay fast: the fair value already prices trend persistence.

**Weak jump tier inside the cap** (`strategies/round4/r4m_tier.py`). A second `IdGapJump` with 2 missing ids (8 of
15 big jumps, against 4 for the 3-id rule, but 5 of 23 fires false) moves the position to ±100 with one IOC, never
past the cap, so a false fire costs the spread and no penalty. The strong 3-id rule still levers to 1000.

| variant | q1 | q2 | q3 | q4 | full |
|---|---|---|---|---|---|
| 2 missing ids (sweep depth 5; 2 / 10 / storm-only identical) | **+2.4k** | **+1.8k** | **+0.8k** | 0 | **1,161,578 (+5k)**, same penalty |
| same, to ±50 | +2.1k | +0.7k | +0.8k | 0 | 1,160,185 |
| single far order 10 / 15 / 20 ticks | −47k / −16k / −18k | −12k / −5k / −5k | −30k / −2k / +1k | −43k / −2k / +1k | — |
| 2 ids + far 15 | +4.6k | −3.3k | −0.9k | −1.6k | — |

The 2-id tier passes the rule (3 of 4 up, q4 equal, neighbours identical) but is +0.4%, noise level for a submission.
Far-order tiers lose: false fires in storm cost a 200-lot round trip and the quiet quotes trade the position back.

**Order-book layer** (`r4m_tier.py`, on top of the weak tier, base 1,161,578). The strategy reads new-order prices,
id gaps, taker flow and touch imbalance, but never cancels or depth beyond the touch. Tested: order-flow imbalance at
the touch (Cont et al., decayed over OFI_TAU, per mean touch depth) and depth imbalance within DIMB_LEVELS ticks, as
fair terms.

- **OFI: rejected at both signs.** +0.25 / 0.5 / 1 → 1,029k / 900k / 719k; quiet-only 0.5 935k; storm-only 0.5 1,124k;
  −0.1 / −0.25 quiet 1,084k / 929k. It adds noise to fair and triggers bad takes.
- **Depth imbalance in quiet: rejected** (+0.5 1,129k, −0.2 / −0.5 1,149k / 1,148k on quarters).
- **Depth imbalance in storm: promising shape, fails acceptance.** Quarters vs base: V_K_DIMB 0.5 +26k / +4k / −8k /
  −2.5k; 0.75 +26k / +22.5k / −3k / −5k; 0.5 over 3 levels +22.6k / +20.7k / −2k / −3.9k; 0.25 +4.5k / +1.4k / −5.6k /
  −1.4k. Full tape 0.5: 1,183,307 (`strategies/round4/r4m_book.py`). Paired robustness against r4m_risk_a (same 6 chunks,
  same 8 bootstrap sessions): chunks +6.5k / +22.5k / +4.5k / −5.1k / −2.1k / −2.7k, bootstrap up in 3 of 8 (mean
  +1.8k). The gain sits in the first half of the tape (heavier storms, 1 s storm vol 14 and 11.5 vs 8.6 and 10.8), so
  the expected gain on the hidden tape is about zero. Mechanism if it ever holds: in storm new orders land at S ± X
  and IOCs eat the stale side, so near-touch depth leans towards S, a second read on S − mid.

**Not built:** an ensemble of the Kalman and IMM hidden-price estimates. Both read the same new orders with the same
gap model and depth gate, so their errors are nearly the same. Also checked: the breach penalty is integrated
continuously (`simulator.py` `_accrue_breach_penalty`), so there are no free sub-second excursions.

**Verdict:** no new architecture beats r4m_risk_a beyond noise. r4m_risk_a stays the submission; `r4m_tier`
(+5k) and `r4m_book` (+27k, first-half only) are not worth a submission.

### Round 4: multi-level book features and the remaining untried ideas (2026-11-08)

`strategies/round4/r4m_ml.py` = `r4m_tier` (weak 2-id jump tier, base quarters 319,712 / 291,819 / 298,110 / 250,387,
full 1,161,578) plus switches; grids `scripts/grid_ml.txt`, `grid_ml2.txt`, `grid_all.txt`.

**Multi-level book features** (market orders only, per tick distance from each side's best, up to 20 ticks):

| feature | best setting: quarters vs base | full | verdict |
|---|---|---|---|
| distance-weighted depth imbalance exp(−d/λ), storm, 0.5, λ 3 | +21.5k / +22.6k / −4.4k / −4.6k | **1,197,048** | same first-half split as DIMB; λ 1 / 8, K 1 give 1,176–1,186k |
| same, scaled by the half spread (microprice form), 0.25–1 | +4k / +1k / −3k / −1k at best | — | worse than unscaled |
| same, quiet ±0.2 | −1k to −6k all quarters | 1,137k / 1,147k | rejected |
| multi-level microprice (VWAP of 3/5/10 levels), storm / quiet ± | down in all quarters (3 levels: +13.7k / +9k / −9k / −5k) | 1,119k | rejected |
| multi-level OFI (decayed Δ depth within 5 ticks), ± | −4% to −15% | — | rejected |
| far-band imbalance (5–20 or 10–20 ticks), ± | ±10k mixed | — | rejected |

The storm depth imbalance (DIMB or its distance-weighted form) is the one order-book signal with a consistent sign at
neighbouring settings, but it pays only in q1 / q2 (paired robustness chunks also first-half only). Kept off.

**Remaining untried ideas** (all on the same base):

- **Storm-end hazard** (storms last 84 s with CV 0.44, so lean against the position as the storm ages, V_HZ ticks per
  lot at HZ_US): 0.01 over 60 / 90 / 150 s → 1,036k / 1,093k / 1,134k; 0.02 / 90 s 854k. Down in every quarter.
- **Edge-sized storm takes** (half or quarter size below 1 / 1.5 ticks of edge): 1,155k–1,157k, noise.
- **Jump-direction prior** (storm lean toward the last jump's sign, 61% repeat): 0.3 / 0.6 / 0.3 for 30 s →
  1,129k / 1,090k / 1,134k, down in all quarters.
- **Pull quiet quotes on jump signals** (id-gap fires, single far orders 10 / 15 / 20 ticks, hold 50 / 200 ms):
  identical to the base. Far orders come in storms, when there are no quiet quotes to pull.
- **Bias to the tape's long-term trend:** not built. round4_public drifts 10,001 → 9,554 (−446 ticks over 6,000 s);
  the hidden validation tape ends at mark 10,349 (about +350 over 12,000 s). Each session drifts its own way, and at the
  ±100 cap a bias can only shift which way the cap is held: 20 extra lots over a 400-tick drift is about 8k.
- Not built: DP-policy distillation and per-inventory DP take thresholds (in-sample ML, the failure mode of the ML
  take rule); endgame handling (the strategy does not know the tape length; the end position is marked at the mid).

### Round 4: slow biases a short-horizon signal would miss (2026-11-08)

`scripts/slow_bias.py` on round4_public and round3_public (1 s mid grid):

- **Reversion to the 10,000 start:** corr(mid − 10000, next h move) round 4 −0.22 / −0.46 / −0.58 at 60 / 300 / 600 s,
  round 3 −0.19 / −0.42 / −0.68 (t about −2). **Not real:** paths rebuilt from 10,000 out of each tape's own shuffled
  60 s return blocks give r −0.16 / −0.35 / −0.47 (round 4) and −0.18 / −0.38 / −0.52 (round 3) by construction
  (the Dickey-Fuller bias of a random walk regressed on its level); P(null ≤ observed) 0.20–0.44. No reversion to a
  300 / 1200 s trailing mean either (|r| < 0.11).
- **Return autocorrelation** (non-overlapping, lag 1): round 4 +0.27 (1 s), +0.14 (5 s), −0.09 (10 s), −0.21 (30 s),
  0 from 60 s; round 3 +0.12, 0, −0.07, −0.10. The 1–5 s part is the momentum term; the 30 s reversal lean was
  rejected earlier.
- **Jumps ≥ 20 ticks:** round 4 19 up / 12 down, signed move after −6.9 (60 s) / −10.6 (300 s); round 3 11 / 9,
  +3.6 / +3.9. Coin-flip signs, after-jump drift flips sign between tapes.
- **Time of session** (600 s blocks): up-share 0.42–0.52, no pattern; storm vol falls over round 4 and rises over
  round 3. **Round-number magnets** (50 / 100): |t| < 1.2.
- **Drift:** round4_public −446, validation about +350, round3_public −128: each session its own way.

Nothing here is a bias to lean on, even accepting some overfit: each one is inside what a random walk with the tape's
own jumps and storms produces.

### Round 4: session-shape bet and simulator hacks (rejected, 2026-11-08)

**Bet that the hidden tape repeats round4_public's shape** (`strategies/round4/r4m_trend.py`, `scripts/fit_trend.py`):
cubic of the mid against u = t / 6000 s, S(u) = 9804 + 883u − 4881u² + 3859u³ (flat, decline, bottom near u 0.8,
recovery; residual sd 86 ticks); fair leans by TR_H seconds of the curve's slope, stretched over TR_T.

| TR_H (s) | round4_public, TR_T 6000 (in-sample, exact shape) | round3_public (other shape) | round4_public, TR_T 12000 |
|---|---|---|---|
| off | 1,156,545 | 603,651 | 1,156,545 |
| 5 | 900,554 | 355,392 | — |
| 10 | 617,167 | 205,336 | 885,820 |
| 20 | 334,059 | 101,618 | 597,671 |
| 40 | 217,087 | 29,844 | 336,058 |

It loses even in-sample: a fair lean of half a tick against 0.2–0.55 tick edges turns the strategy into a one-sided
taker. The ceiling is small anyway: −100 lots held through the whole public tape earn 100 × 446 ≈ 45k (about 90k on a
12,000 s tape if the shape repeated exactly), and the hidden validation tape ended about +350, opposite to the fit.

**Simulator hacks checked:** the public book shows only id, side, price, quantity, owner (no lifespan); cancel timers
are exponential, so order age says nothing about the remaining life; the final mark is a liquidation sweep against
market orders only (our orders cannot move it; ending flat would save only the sweep cost, hundreds of dollars); the breach
penalty is integrated continuously. The order-id gaps are the one leak, and the sniper already uses them.

### Round 4: long-shot candidate r4m_mldi (2026-11-08, chosen for public-tape PnL, no robustness gate)

The team asked for the highest public number, accepting light overfit. `strategies/round4/r4m_mldi.py` = r4m_risk_a
+ weak 2-id jump tier + storm distance-weighted depth imbalance (V_K_MLDI 0.6, λ 2) + Jaro's quote hysteresis
(EDGE_NEW 0.3, EDGE 0.15). **round4_public 1,204,809** (penalty 8,838; exact file re-run), quarters 344,258 / 314,775 /
297,613 / 247,518 (vs r4m_risk_a +27.0k / +24.7k / +0.3k / −2.9k).

Stacking grid (`scripts/grid_mldi.txt`, `grid_stack.txt`, full tape): V_K_MLDI 0.4 / 0.5 / 0.6 / 0.75 → 1,159k /
1,197k / 1,199k / 1,198k; λ 2 / 3 / 4 at 0.6 → 1,200k / 1,199k / 1,199k; + H_LR 0.001 1,171k–1,184k; + hysteresis
1,204,809 (EDGE_NEW 0.25: 1,203,757); + pre-storm lean K_PRE 0.5 / 1 / 2 (`r4m_aggr` logic) 1,191k / 1,174k / 1,191k.
Expected gain on the hidden tapes is smaller than on public (the depth term is first-half only on this tape; paired
robustness of the 0.5 version: chunks 3 / 6 up, bootstrap 3 / 8 up).

### Round 4: spectrum after removing the session swing, order lifetimes, combinations (2026-11-08)

**Spectrum** (`scripts/spectrum.py` + `scripts/plot_spectrum.py`; 1 s mid minus the session cubic, and |1 s move|,
against 300 paths of 60 s blocks shuffled in time). Detrended price: inside the null band at every period on
round4_public (a random walk, power ∝ 1/f²); on round3_public only the lowest bin (3000 s, the trend leftover).
Volatility: bumps at 2000 s (×9.8) and 240 s (×2.8) on round 4, 1500 s (×7.7) and 500 s (×4.7) on round 3. The
240 s bump is the storm/calm cycle (mean storm 84 s + calm 143 s), which the HMM already tracks live; the long
bumps are storm intensity drifting over the session, at a different period on each tape. No periodic price component
to bet on.

**Order lifetimes** (`scripts/lifetime_bias.py`): round 4 lifetimes are not exponential (mean 0.535 s, sd 1.04 s;
mean remaining life 0.54 s new, 1.15 s after 1 s, 2.65 s after 2 s; top size quartile 0.82 s vs 0.43 s), flat over the
session (0.49–0.58 s per 600 s block), equal by side. Bid-vs-ask cancelled lots per second vs the next move:
r −0.086 at 1 s when the counting second overlaps the move, −0.029 / −0.017 / −0.014 at 1 / 5 / 30 s without overlap
(and partly ghost cancels of filled orders, which the feed never shows). Nothing usable.

**Combinations on r4m_mldi** (`scripts/grid_combo.txt`, `grid_combo2.txt`; full 1,204,809): 18 single nudges (storm
take edge, S weight, flow, momentum, quiet momentum, DIMB on top, far band, edge sizing, quote pull, weak size, take
size, P_ON) all within −24k / +1.4k; the top four stacked 1,201k–1,202k. r4m_mldi sits on a flat top.

### Round 4: using the volatility spectrum (online vol on r4m_mldi, rejected, 2026-11-08)

Volatility clusters for about 100 s (spectrum knee), so `r4m_mldi.py` got an online vol estimate (EWMA of squared
1 s mid moves, VOL_TAU_S 60 s) and three switches, all off by default (base reproduces 1,204,809; `scripts/grid_vol.txt`):

| switch | full | quarters vs base |
|---|---|---|
| storm SIG_V = k · vol, k 0.8 / 1.0 / 1.2 / 1.5 | 1,204k / 1,188k / 1,182k / 1,154k | k 0.8: −8.2k / +4.1k / +3.6k / −0.9k |
| k 1.0 with tau 30 / 120 s | 1,201k / 1,188k | down in q1 |
| depth-imbalance term only when vol ≥ 6 / 8 / 10 / 12 | 1,163k–1,164k | q1 −23k, q2 −20k, q4 +4k |
| storm take edge × sqrt(vol / ref), ref 8 / 10 / 12 | 1,183k / 1,173k / 1,170k | down in q1–q3 |

The fixed SIG_V 10 and take edge are already right on average, and gating the depth term on vol throws its gain away
(it pays in moderate storms too; the vol estimate lags storm onsets). The volatility structure is already used
through the HMM regime.

### Round 4: storm-onset quote defence from the vol forecast (rejected, 2026-11-08)

`r4m_mldi.py` switches (off by default; `scripts/grid_pw.txt`): while P(storm) ≥ PW_LO (below P_ON 0.3) or realized
vol ≥ VW_VMIN, widen quiet quotes by PW_EDGE or quote only the inventory-reducing side. Full tape (base 1,204,809):
P(storm) 0.1 / 0.15 / 0.2 with +0.3 / +0.6 ticks 1,199k–1,206k (quarters within ±1k, except 0.1 / +0.3 q3 −6.7k and
0.2 / +1.0 q2 −9.7k); reduce-only 1,204.7k–1,204.8k; vol ≥ 3 / 5 with +0.5 1,179k / 1,179k; vol ≥ 5 / 8 reduce-only
1,185k / 1,198k. The HMM crosses 0.1 → 0.3 within moments of an onset, so the warning window holds few quotes; a vol
trigger lags and cuts quiet turnover.

Why a vol forecast pays so little here: the only instrument is the asset itself with a ±100 cap. A vol forecast earns
through vol-sensitive payoffs (options) or through size, and in storms the size is already pinned at the cap. The
forecast's one use, switching quiet making / storm taking at the right moment, is what the HMM already does.

### Round 4: quote size by the vol forecast (2026-11-08)

`r4m_mldi.py` (`scripts/grid_size.txt`; base 1,204,809): quiet quote size QTY · clamp((QV_REF / vol)^QV_POW,
QV_MIN, QV_MAX), and K_INV · vol / KI_REF.

| variant | full | quarters vs base |
|---|---|---|
| bigger when calm, ref 1.5 (max 1.5 / 2 / 3 identical: vol never below 1) | **1,214,033** | +5.9k / −0.6k / −1.5k / +0.3k |
| ref 2.0 / 1.0 | 1,203k / 1,204k | q4 −7k / −5k |
| sqrt form | 1,208k | +3.7k / −0.7k / −1.3k / +0.3k |
| min 0.25 (smaller when busy) | 1,195k | down |
| bigger when busy (reverse) | 1,182k | down in all 4 |
| K_INV scaled by vol, ref 1.5 / 3 | 917k / 1,131k | down in all 4 |

The direction agrees with the mechanism (bigger quotes in calm, the reverse loses), but the gain is thin and mostly
q1. Switched on in `strategies/round4/r4m_mldi_v2.py`: **1,214,033** (penalty 9,036); `r4m_mldi.py` stays the 1,204,809 version.

### Round 4: calm aggression and end-of-run exit on r4m_mldi_v2 (rejected, 2026-11-08)

Base `r4m_mldi_v2` 1,214,033 (switches off by default, base reproduced). `scripts/grid_calm.txt`: quiet quote and take
edge × clamp(vol / ref, min, 1): with the take edge 1,205k–1,207k (q4 −6 to −8k); quote edge only 1,213k–1,214k
(noise). `scripts/grid_end_{f,q}.txt`: reduce-only quiet quoting in the last 10 / 30 / 60 / 120 s (END_T = tape
length): full 1,214,033 / same / same / 1,213,959, quarters −0.1k to −4.4k. The end position is liquidated by a sweep
through the book (`_mark_out_value`), worth about half a spread plus depth on ≤ 100 lots, a few hundred dollars;
validation runs ended at −27 / −89 / −89 lots. Cash in result.json is the running cash balance, not idle capital:
there is no capital limit, and |inventory| is at 100 at the median, so the cap is the binding constraint.

### Round 4: vol-forecast quote size alone on r4m_risk_a (rejected, 2026-11-08)

`strategies/round4/r4m_risk_vs.py` (`scripts/grid_vs.txt`): r4m_risk_a + quiet size QTY · clamp(ref / vol, 0.5, max).
ref 1.5 / 1.2 / 2.0 → full 1,151,856 / 1,146,373 / 1,144,943 (r4m_risk_a 1,156,545); ref 1.5 quarters −5.2k / −1.4k /
+1.3k / +1.7k. Without the hysteresis and depth term of r4m_mldi the size term loses, so v2's +9k from it depends on
the rest of that file and is fragile.

### Round 4: validation results of the long shots (arena, 2026-11-08)

| submission | validation (12,000 s) | penalty | vs r4m_risk_a |
|---|---|---|---|
| **r4m_mldi.py** (09:32, marked final) | **2,034,207** | 12,743 | **+69k** |
| r4m_mldi_v2.py (10:11) | 2,029,878 | 12,793 | +65k |
| Jaro, 10:15 | 2,024,010 | 12,793 | +59k |
| r4m_risk_a.py (07:28) | 1,965,048 | 12,138 | — |

The storm depth-imbalance layer transferred (public +48k per 6,000 s, validation +69k per 12,000 s); the vol-forecast
quote size in v2 did not add (−4k, noise), as its dependence on the rest of the file suggested.
