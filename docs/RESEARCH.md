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
- **`strategies/adaptive_mm.py`** (first called merged.py) was tuned with `scripts/sweep.py` on the two halves of the sample (with a split script, since removed). Results:
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

`strategies/imbalance_mm.py` (JPod101010) makes 615,583 on round 1 against our 385,561. Its basis is in `docs/DATA_ANALYSIS.md`:
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
