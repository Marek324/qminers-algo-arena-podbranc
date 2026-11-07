# Data analysis: how the tapes are generated

Reverse-engineered from `data/sample.csv`, `data/sample_v2.csv`, `data/round1.csv` and `data/round2_public.csv` (2026-11-07). The full reports, scripts and online tools are in `findings/`, which is git-ignored.

## TL;DR

- **All three tapes are synthetic.** Even `round1`, the real arena round, comes from a simulator. It is not recorded market data, whatever the deck says.
- **`round1` uses the same engine as `sample_v2`, with different parameters.** `sample.csv` comes from a different toy generator and is not a subset of either.
- **Generator:**
  - Independent Poisson streams of limit orders and IOCs.
  - A hidden fair price S that follows a continuous random walk.
  - Limit prices drawn at a random distance X from S.
  - IOCs priced at S ± 1 tick, with side decided by a fair coin.
  - Each limit order gets a random cancel timer when it is inserted.
- **Main edge:**
  - The visible mid chases S with a half-life of about 200–260ms. You can estimate S from new limit orders and from trades that sweep past the best price.
  - The *side* of an IOC carries no information, but its *price* follows S. So when the book lags S, IOCs sweep one level deeper on the side S has moved toward.
- **`round2_public` adds jumps.** Same engine, but S also jumps 20–115 ticks instantly, about 8–9 times an hour, in random directions, with no warning and no reversion. Between jumps it moves like `round1`. Passive fills now lose money after 1s. See [round2_public](#round2_public-same-engine-plus-jumps).
- **Seed recovery:** no seed found among simple seeds yet. A larger sweep on HPC is pending review. See the seed-recovery section below.

## The tapes

| | `sample.csv` | `sample_v2.csv` | `round1.csv` | `round2_public.csv` |
|---|---|---|---|---|
| Rows / duration | 53.8k / 3,600s | 2.0M / 5,750s | 4.41M / 12,000s | 2.19M / 6,000s |
| Events/s | 15 | 348 | 368 | 366 |
| Generator | Toy re-quoting loop | Engine A, params v2 | Engine A, params r1 | Engine A, params r2, **plus jumps in S** |

**`sample.csv`** (very likely `scripts/make_null_tape.py`):
- One bid and one ask, re-quoted every ~0.27s on a 1-second clock. The spread is always 2 ticks.
- Limit sizes are U{10..40}.
- After an aggressive order the price is pushed further in the same direction (+3 ticks at 1s). So it trends, and makers lose about −9 ticks/lot at 5s.
- Nothing like the real rounds. Don't tune to it.

**`sample.csv` is not part of `sample_v2`.**
- 0 rows match exactly.
- No run of 2 or more consecutive events matches, even after shifting time and price and remapping IDs.
- The size ranges barely overlap.

## The engine (sample_v2 and round1)

### Parameters

| Component | Model | `sample_v2` | `round1` |
|---|---|---|---|
| Hidden price S | continuous random walk, dS = σ dW, S₀ = 10000 | σ = 0.50 ticks/√s | σ = 0.92 ticks/√s |
| Limit arrivals | Poisson; buy and sell are separate 50/50 streams | 179.66/s | 179.80/s |
| Limit price | buy = floor(S − X), sell = ceil(S + X), clamped so it never crosses | X ~ lognormal(−0.235, 0.341), median 0.79 | X ~ shifted gamma(k = 1.72, loc = 0.054, θ = 0.32), mean 0.61 |
| Limit size | iid, independent of everything | exact U{20..40} | heavy-tailed, roughly lognormal (log-mean 2.70, log-std 1.71): min 1, median 14, mean 72, max 124k, 10.5% size 1. KS D = 0.058, so the exact family is still open. |
| Cancels | timer drawn at insert (Exp), independent of price moves | mean life 0.70s | mean life 0.43–0.47s |
| Cancel rate by level (/s) | touch / L1 / L2 / L3 | 1.9 / 1.23 / 0.6 / 0.35 | 3.2 / 2.15 / 1.39 / 0.95 |
| IOC arrivals | Poisson; side a fair coin, no runs | 6.0/s | 8.0/s |
| IOC price | **round(S ± 1)**, not relative to the touch | mean offset from S +0.999 | mean offset from S +0.998 |
| IOC size | lognormal, iid | mu 3.712, sigma 1.397 (median 41) | mu 4.106, sigma 1.395 (median 61) |
| Order IDs | one counter, +1 per insert | yes | yes |

**Limit placement relative to the order's own best price**, as P(offset | spread). "Improve" is one tick inside the spread; "−1" and "−2" are ticks behind the best price. This is what S ± X looks like seen from the book.

| Spread | Tape | Improve | Join | −1 | −2 |
|---|---|---|---|---|---|
| 1 tick | `sample_v2` | – | 25% | 68% | 7% |
| 1 tick | `round1` | – | 55% | 39% | 5.5% |
| 2 ticks | `sample_v2` | 0.7% | 71% | 27% | 1% |
| 2 ticks | `round1` | 9.9% | 76% | 13% | 1.5% |

**IOC price relative to the opposite best price.** It depends on where S sits. On `round1`:
- When S is within 0.36 ticks of the opposite best price, 76% of IOCs go one tick through it.
- When S is 0.70 ticks or more away, 80% stop at the best price.

Overall, 52% stop at the best price and 46% go one tick through on `round1`. On `sample_v2` the split is 75% / 22%.

**Resulting market:**

| | `sample_v2` | `round1` |
|---|---|---|
| Mid vol | 0.50 ticks/√s | 0.92 ticks/√s |
| Spread = 1 tick | 41% | **93%** |
| Spread = 2 ticks | 58% | 7% |
| Mean spread | 1.59 | 1.07 |

### Architecture (code-level behaviour)

- **Event loop.** Independent Poisson streams merged in time order: limit buys at about 89.9/s, limit sells at about 89.9/s, IOCs, and the scheduled cancels.
  - Every stream passes an exponential test (CV within 0.1% of 1).
  - The sequence of event types has no memory.
  - The tape stops at `t < 12000` with no flush, leaving 66 orders resting at the end of `round1`.
- **Timestamps.** Exact whole microseconds, with a 1µs minimum gap: dt = max(1µs, round(dt)).
  - Not one of 6.4M rows has a gap of 0.
  - The 1µs gap appears 1.59× as often as expected, which is exactly the zero-gap mass folded into it.
- **The generator sees the book after matching.**
  - It never emits a limit order that crosses the spread.
  - After an IOC clears a level, the next limit order references the *new* best price (0% use the old one).
  - It starts from an empty book at t = 0 and seeds around 10000.
- **Cancels are scheduled at insert. In `round1` they ignore fills.**
  - `round1` has 2,157,591 cancels for 2,157,657 limit orders. 285,676 of those cancels (13.2%) target orders that were already fully filled, arriving between 3µs and 5.85s after the fill.
  - The exchange treats them as no-ops.
  - `sample_v2` never does this: it purged the timer when an order filled.
  - So in `round1`, cancels in the feed don't reliably mean the queue got shorter.
- **S ignores the mid, but not IOCs** (see [Synthetic twin](#synthetic-twin-of-round1)). On `round1` each IOC moves S for good, more for bigger orders.
  - The mid chases S with a half-life of about 260ms on `sample_v2` and about 200ms on `round1`.
  - S doesn't react to the mid: the gap between them explains under 0.6% of later changes in S.
  - S has no grid of tick fractions.
  - The variance of S between events grows with the time gap. That rules out "one step per event" and "fixed clock" versions of S.
- **Every column is independent of every other,** and nothing leaks through order IDs or timestamp bits (all correlations |r| < 10⁻³).

**Tests that rule out real data with re-drawn columns:**
- There is no volatility memory, even in event time.
- Order sides have no memory: the entropy rate is that of a fair coin.
- Prices and sizes show no round-number clustering.
- Cancels ignore price moves: more than 91% fire with the mid flat over the prior 50ms.

## The hidden fair price S

- **Estimating it.** S is estimated by a grid filter on limit-order prices. A cheap online version averages new limit-order prices with exponential decay, taking the mean of the buy and sell sides.
- **The signal:** δ = Ŝ − mid predicts the next move of the mid.
- **How strongly:** R² of the future mid move on δ:

| Horizon | `sample_v2` | `round1` (half-life 100ms) | Random-agent model, no S |
|---|---|---|---|
| 0.1s | 24% | 23% | 37% |
| 0.5s | 44% | 32% (t = +239) | 7.5% |
| 1s | 37% | 21% | 4.4% |
| 5s | 8% | 4% | 1.2% |

**Why we believe S exists:**
- A pure random-agent model placing orders relative to the best price can't produce the peak at 0.5–1s. A model with S reproduces the R² curve, the spread, the volatility and the stillness.
- Model vs real:

  | | Real | Model |
  |---|---|---|
  | Mean spread, `sample_v2` | 1.59 | 1.57 |
  | Mean spread, `round1` | 1.07 | 1.14 |
  | Chance the mid hasn't moved after 1s, `sample_v2` | 41.8% | 42.9% |

- Over and above queue imbalance and recent moves, δ still adds about 23% R² (on `sample_v2`).

**Online performance:**
- Using only what a strategy sees in its snapshots, a replay of `sample_v2` gave R² 29% at 0.5s with a 200ms half-life.
- On `round1`, 100–150ms is clearly better.
- The grid filter is likely better than the simple average, but has not yet been tested online.

## Synthetic twin of round1

`scripts/gen_tape.py` generates a `round1`-like tape and saves the true S next to it (`<out>_S.npz`: `t_us`, `S`, market-only `mid`). `scripts/fit_synth.py` compares generated tapes with `round1`.

**Missing piece: IOC impact on S.** With the parameters above and an exogenous S, the twin matches `round1` on rates, spread shares, depth, fill rates and IOC sweeps, but it is far too easy:

- Maker markout at 1s is +0.60 ticks/lot, against +0.07 on `round1`.
- `imbalance_mm` makes about 895k per 12,000s, against 616k.

`round1`'s mid move after a taker event grows with its size and does not decay: about +0.1 ticks below 18 lots and +0.6 above 192 lots, the same at 1s and at 60s. Adding a permanent jump `S += side · κ · size^α` at every IOC (size = order size), and lowering the diffusion to compensate, fixes this.

| Fit: κ 0.0075, α 0.7, σ 0.5 | twin | `round1` |
|---|---|---|
| Impact by taker size (1–5s mean), 5 bins | .11–.16 / .13–.19 / .20–.24 / .29 / .63–.68 | .09 / .17 / .21 / .33 / .59 |
| Maker markout at 1s | 0.04–0.08 | 0.07 |
| Realized vol (60s window) | 0.97–1.03 | 0.91 |
| Hindsight max PnL (`analysis.py` `max_pnl`, 12,000s) | 1.377–1.382M (3 seeds) | 1.389M |
| `imbalance_mm` PnL (12,000s) | 582k / 610k / 594k (3 seeds) | 616k |
| `adaptive_mm` PnL (12,000s, version 42b27bb) | 607k / 605k / 601k (3 seeds) | 651k |

PnL is sensitive to κ: κ 0.0065 gives about 660–700k. An independent fit in [round1.md](round1.md) (`analysis/generator.py`) found the same push, 0.009 · size^0.65 with σ 0.75. That model is slightly kind to market makers (`imbalance_mm` 194k/h vs 185k/h real); this one is slightly harsh (179k/h). Vol runs slightly high, so the exact form (order vs executed size, power vs other shapes) is still open.

**Using it to measure a strategy:**

- `imbalance_mm` gets 42–44% of the hindsight max on the twins and 44.3% on `round1`. `adaptive_mm` (version 42b27bb, with the hidden-price estimate) gets 43.6–43.9% on the twins and 46.8% on `round1`; the version before it got 24–26% and 27.8%. The hindsight max is perfect foresight (it only takes the fills that turn out good), so it is a loose upper bound, not an achievable target.
- `scripts/oracle_s_mm.py` is `imbalance_mm` reading the true S instead of Ŝ (local only, not submittable); `scripts/sweep_oracle.py` sweeps it. With the true S it scores **no better** than with Ŝ (best 68k vs 75k on a 1500s tape over K_S, K_INV and EDGE_TAKE; 451k–568k vs 582k on the full seed-1 twin). The Ŝ estimate is not what limits this strategy; inventory pinned at ±100, queue position and fills by big impactful IOCs are.

## Seed / RNG recovery

**Done so far (`agy3`):**
- Seeds 0..10⁶ and about 55 common seeds and dates were tried for Python `random`, NumPy `RandomState` and NumPy `default_rng`.
- It tested exponential draws at stream positions 0–2, against the first arrival gaps of all three tapes and with plausible rates.
- **No seed matched.**

**Why that is not conclusive:**
- The filter assumed the first two arrival gaps are *consecutive* random draws.
- The architecture analysis shows separate streams per event type. Each order also consumes draws for side, size, X, the cancel timer and S between arrivals.
- So the two gaps can be many draws apart. Only the simplest structures have been ruled out.

**Planned next:**
- A larger sweep: all 2³² seeds, with the first gap matched at any of the first K draws and a second observable confirming it.
- It is set up as a Slurm job array for IT4I Barbora: `findings/scripts/forensics_rng_*.py` and `.sbatch`.
- It gets a manual review before anything runs there.
- Rough cost: a few node-hours per combination of RNG and generator structure.

**Limits:**
- If the generator was seeded from OS entropy (`default_rng()` or `random.seed()` with no argument), brute force can't work.
- Recovering the state from observed outputs isn't realistic either. Each event reveals only part of the bits, the order of draws is unknown, and PCG64 isn't linear.

**Value check before spending HPC time:**
- A seed only helps if the scored tape's seed can be found or derived.
- If `round1` itself is the scored tape, the future is already in the file and a seed adds nothing.
- Check the hackathon rules on exploiting the generator.

## round2_public: same engine, plus jumps

`data/round2_public.csv` (2026-11-07): 6,000s, 2.19M rows, the same CSV format. Fitted with `analysis.py` and `scripts/estimate_model.py`; details in `findings/reports/round2_check.md`.

**Unchanged from `round1`:**
- Poisson limit orders (178.3/s vs 179.8/s) and IOCs (8.9/s vs 8.0/s), side a fair coin.
- Limit prices at floor(S − X) / ceil(S + X); IOCs priced from S (`ioc_mode` S).
- Exponential cancel timer, mean 0.44s (vs 0.47s), and cancel rows still written for filled orders (76% of fills get one).
- Sizes: median limit 14 lots, median IOC 60–61 lots.
- Each IOC pushes S: 0.0159 · size^0.63 (vs 0.0107 · size^0.66 in the re-run fit).

**New: S jumps.** The −311 tick fall over the tape (10000 → 9689.5) is not a trend; it comes from a few large jumps in S (reports: `findings/reports/round2_jumps.md`, `round2_check.md`).

- **Instant.** S moves from one level to another between two consecutive orders. Checked at the millisecond level on the 13 biggest, using the range between the best new buy and the best new sell price before and after: +58.5, −31.5, +41.5, −54.5, +30, +31, −54, −89.5, −37, +22.5, +47.5, +115, +50 ticks. There is no gradual move.
- **Permanent.** Large jumps don't revert: about 0–20% after 60s. They are shifts in fair value, not spikes.
- **Rate.** 15 jumps of 20 ticks or more in 100 minutes (about 9 an hour), 12 of 30 or more. A threshold detector at 8 ticks finds 94–116, but most of those small ones revert erratically and are noise in the S estimate. A likelihood fit with a jump term is under way to pin down the rate and size scale.
- **Big jumps are spaced out, not random.** The 20-tick-plus jumps (at 1559, 1794, 2264, 2467, 2763, 3056, 3209, 3423, 3621, 3873, 4267, 4414, 4825, 5399, 5570 s) are never closer than 147s apart. The mean gap is 286s and the gaps vary much less than Poisson arrivals would (CV 0.44 against 1). Under Poisson timing, gaps this regular have a probability of about 0.001. It looks like a schedule or a cool-down after each jump, so a strategy can expect about 2 minutes of calm after a big jump. But there is no big jump in the first 26 minutes, which fits neither a plain schedule nor Poisson timing; the jump process may switch on, or start after a long first wait. Smaller jumps (15–20 ticks) don't follow this spacing.
- **Sizes and signs.** Up and down are a coin flip (48 / 46 in the 8-tick catalogue, p = 0.92). Sizes have an exponential tail (it fits far better than a normal), reaching 115 ticks. Landing levels show no pattern (no pull to 10000 or round numbers).
- **No warning.** Nothing predicts the timing or the direction:
  - Around the 13 biggest jumps, the IOC rate from 5s to 1s before is 8.6/s, the normal 8.9/s. It is 10.2/s in the last second.
  - In the second before, order imbalance and IOC side don't predict the direction (|r| ≤ 0.16, not significant).
  - Jump times aren't tied to the clock (seconds mod 60/100/300/600), the order-id counter, the number of IOCs, big IOCs or round price levels.
  - Earlier "precursor" results came from windows that overlap earlier jumps.
- **IOCs surge after a jump.** In the first second after a big jump, IOC arrivals rise to about 47/s (normally 8.9/s), and they stay around 19/s over the next 4s. They are priced at the new level and sweep the stale quotes. So the generator seems to send more IOCs while the book is out of date. 74% of IOCs that go 2 or more ticks through the best price come in the windows around jumps. Their pricing rule is unchanged: they reach deep only because the book is stale.
- **Nothing else changes around jumps.** Limit-order rate and side balance, order sizes, and the push per IOC are the same inside and outside the jump windows.
- **`round1` and `sample_v2` have none.** The same detectors find 0 jumps on either: its largest 200ms move is 3 ticks, and the kurtosis of 1-second moves is 0.4. On `round2_public` it is 263.
- **Between jumps it is `round1` again.**
  - Quiet minutes move 0.9–1.4 ticks/√s.
  - Outside ±2.5s around each jump: X has mean 0.65 and sd 0.81 ticks (raw fit: 0.82 / 1.16), and the push per IOC is 0.0156 · size^0.59.
  - The single σ = 3.64 from `estimate_model.py` averages the jumps into the diffusion, which is why its fit error is 43.7 (`round1`: 8.6–15) and its spread share, impact and microprice predictability are far off.

**How fast a jump becomes visible** (median, from the jump):

| Event | Median | 90th percentile |
|---|---|---|
| First new limit order at the new level | 6 ms (the next order to arrive) | 54 ms |
| First trade against a stale resting order | 34 ms (10% within 1.5 ms) | |
| Best bid / ask moves | 19 ms | 168 ms |
| First IOC priced at the new level | 88 ms | 315 ms |

With our 5 ms latency, a strategy that cancels as soon as the first new order shows the jump beats the first stale fill in only about 48% of jumps. Large stale quotes get swept: around 1,000 lots of the market's own resting orders per jump.

**Resulting market:**

| | `round1` | `round2_public` |
|---|---|---|
| Mid vol at 1s / 60s / 300s (ticks/√s) | 0.92 / 0.95 / 0.78 | 3.80 / 4.29 / 5.13 |
| Spread = 1 / 2 / 3+ ticks | 93.0% / 6.9% / 0.1% | 83.9% / 11.6% / 4.5% |
| Median size at the best price (bid / ask) | 901 / 900 | 686 / 714 |
| IOCs one tick through / two or more through | 46% / 1% | 42% / 16% (up to 13 ticks) |
| Maker markout at the fill / 1s / 60s (ticks/lot) | +0.68 / +0.07 / +0.04 | +0.83 / **−0.40** / **−0.49** |
| Mid move 60s after an IOC, largest fifth by size | +0.60 | +1.72 |
| Hindsight max per hour (combined / making / taking) | 417k / 405k / 145k | 1,023k / 981k / 519k |

**Implications:**
- Passive fills lose money after 1s on average, so the `round1` market makers can't be assumed to work as they are. The loss comes from fills at the moment of a jump, when resting orders at the old price are swept.
- Holding 100 lots through a 50–100 tick jump costs 5–10k at once.
- **Detect jumps from new limit orders.** A new order far outside the current band shows the jump about 6ms after it happens, well before the best bid/ask (19ms) or IOCs (88ms). Then cancel resting orders on the side being run over.
- **Cancelling alone isn't enough.** It is too slow in about half the jumps. Defence has to be partly structural: less size resting at the touch, and smaller inventory when exposed.
- **Trade with the jump.** Jumps are permanent and stale quotes stay in the book for about 19ms. An IOC at the old best price in the jump direction, sent as soon as the jump shows, takes those quotes. The hindsight max more than doubled (1.02M/h).
- **Offline estimates** (`findings/reports/r2_exploit.md`, 5 ms latency, no queue model):
  - Pulling quotes when a new order lands 6–8 ticks from the mid avoids about a third of the market's stale-fill losses, with few false alarms.
  - An IOC of up to 100 lots in the jump direction, exited after 5–30s, makes +13k to +16k per hour.
- **After a big jump, expect a quiet stretch.** The next 20-tick jump hasn't come sooner than 147s.
- Between jumps the market is close to `round1`, so the `round1` quoting logic is still the base.

## What this means for strategies

1. **`round1` is almost always at a 1-tick spread (93%).** There is no 2-tick spread to step inside. Making means joining the touch queue, or resting one tick behind it.
2. **Quote toward S.** Ŝ leads the mid by hundreds of ms, far more than our 5ms latency. When δ is clearly positive, keep or join the bid and pull or back off the ask; do the reverse when it's negative.
3. **Watch IOC sweeps, which follow S.**
   - When S has moved past our side of the touch, IOCs go one tick through the best price, so resting one tick behind is *not* safe on that side.
   - A trade that sweeps past the best price is a second, faster reading of S.
4. **Executed trade direction follows the book state.**
   - The correlation between touch-depth imbalance and the sign of executed trades is +0.105 on `round1`.
   - IOC *sides* are still coin flips (|r| < 0.01 on `sample_v2`). The bias is in which IOCs actually execute, because only IOCs whose S ± 1 price reaches the opposite best price trade.
5. **Taker flow is pure noise in terms of side.** In `sample_v2`, fills from the back of the queue were *not* worse: +0.89 ticks/lot at 5s, against +0.74 at the front. Even fills from IOCs over 2,000 lots held +1.05.
   - The cost of the back of the queue is fill rate: about 16% of joins fill, and under 7% with more than 1,500 lots ahead.
   - Re-check this on `round1`. Its IOCs sweep deeper and its sizes and cancel rates differ.
6. **Queues turn over fast.** Touch orders live about 0.3s in `round1`. Ghost cancels mean you shouldn't count cancels as queue thinning; track the queue from snapshot quantities instead.
7. **Plan for parameters, and mechanisms, to change between rounds.** v2 and r1 share the engine but differ in every rate and size parameter, and `round2_public` adds jumps in S. Estimate them online (Ŝ half-life, vol, spread regime, jumps) instead of hard-coding values, and protect resting orders and inventory against jumps.
8. **The `GeneratorFingerprint` detector needs reworking before use.** It scores `round1` at 0.65, below its 0.70 threshold, only because it expects U{20..40} sizes. Its other 4 components score 0.97 or higher.

## Caveats

- The X distributions were fitted on 40k orders (about 223s of data), jointly with S, so the fit is somewhat circular. Refit on the full tape.
- The S model still underestimates touch depth: 606 vs 818 lots on v2, and 332 vs 650 on r1. On r1 it shows a 1-tick spread 86% of the time against 93% real. The cancel rule or the size handling is not yet exact.
- The size fit in the forensics report reports a maximum of 997, which contradicts the verified maximum of 124,248. The exact size family is open.
- The markout and queue analysis was done on `sample_v2` only.
- No numbers here come from a backtest. "456k PnL" (a hindsight best case) and "$60–80k" (a rough calculation) from the reports are not achievable estimates.
