# Data analysis: how the tapes are generated

Reverse-engineered from `data/sample.csv`, `data/sample_v2.csv` and `data/round1.csv` (2026-11-07). The full reports, scripts and online tools are in `findings/`, which is git-ignored.

## TL;DR

- **All three tapes are synthetic.** Even `round1`, the real arena round, comes from a simulator. It is not recorded market data, whatever the deck says.
- **`round1` uses the same engine as `sample_v2`, with different parameters.** `sample.csv` comes from a different toy generator and is not a subset of either.
- **Generator, in one sentence:** independent Poisson streams of limit orders and IOCs. Limit prices are placed around a **hidden fair price S** that follows a random walk. Each limit order gets a random cancel timer. IOC side is a fair coin flip.
- **Main edge:** the visible mid lags S by about 0.5s. You can estimate S from new limit orders and quote toward it. IOCs carry no information, so resting quotes don't suffer adverse selection from takers. The only risk is S moving.

## The tapes

| | `sample.csv` | `sample_v2.csv` | `round1.csv` |
|---|---|---|---|
| Rows / duration | 53.8k / 3,600s | 2.0M / 5,750s | 4.41M / 12,000s |
| Events/s | 15 | 348 | 368 |
| Generator | Toy re-quoting loop | Engine A, params v2 | Engine A, params r1 |

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

| Component | Model | `sample_v2` | `round1` |
|---|---|---|---|
| Limit arrivals | Poisson, side 50/50 iid | 179.66/s | 179.80/s |
| Limit size | iid, independent of everything | exact U{20..40} | heavy-tailed: min 1, median 14, mean 72, max 124k, 10.5% size 1 |
| Limit price | around hidden S (see below) | | |
| Cancels | per-order random timer, ignores price moves (>91% fire with the mid flat over the prior 50ms) | mean life 0.70s, CV ≈ 1.0 | mean life 0.43–0.47s, CV ≈ 0.95–0.98 |
| Cancel rate by level (/s) | touch / L1 / L2 | 1.9 / 1.23 / 0.6 | 3.2 / 2.15 / 1.39 |
| IOC arrivals | Poisson, side a fair coin, no runs (mean run length 2.00) | 6.0/s | 8.0/s |
| IOC size | lognormal, iid | mu 3.712, sigma 1.397 (median 41) | mu 4.106, sigma 1.395 (median 61) |
| IOC price vs opposite touch | at touch / one tick through | 75% / 22% | 52% / 46% |
| Order IDs | one counter, +1 per insert | yes | yes |

**Limit placement relative to the order's own best price**, as P(offset | spread). "Improve" is one tick inside the spread; "−1" and "−2" are ticks behind the best price.

| Spread | Tape | Improve | Join | −1 | −2 |
|---|---|---|---|---|---|
| 1 tick | `sample_v2` | – | 25% | 68% | 7% |
| 1 tick | `round1` | – | 55% | 39% | 5.5% |
| 2 ticks | `sample_v2` | 0.7% | 71% | 27% | 1% |
| 2 ticks | `round1` | 9.9% | 76% | 13% | 1.5% |

**Resulting market:**

| | `sample_v2` | `round1` |
|---|---|---|
| Mid vol | 0.50 ticks/√s | 0.92 ticks/√s |
| Spread = 1 tick | 41% | **93%** |
| Spread = 2 ticks | 58% | 7% |
| Mean spread | 1.59 | 1.07 |

Properties common to both:
- The price is a random walk: variance ratios ≈ 1 beyond a few seconds, with a short bid/ask bounce below that.
- No volatility clustering, in either clock time or event time.
- Thin tails at 5s and beyond.
- No intraday pattern and no regime changes. Rates are flat across all 10-minute blocks.

**Tests that rule out real data with re-drawn columns:**
- All columns are mutually independent.
- There is no volatility memory, even in event time.
- Order sides have no memory: the entropy rate is that of a fair coin.
- Prices show no round-number clustering.
- Cancels ignore price moves.

## The hidden fair price S

- **Model:** buy price = floor(S − X), sell price = ceil(S + X). S is a random walk, and X is an iid offset with mean about 1.3 ticks and std about 0.47 (`sample_v2`).
- **Signal:** δ = Ŝ − mid, where Ŝ is an exponentially weighted mean of new limit-order prices, with the buy and sell sides averaged. δ predicts the future mid move.
- **How strongly:** R² of the future mid move on δ:

| Horizon | `sample_v2` | `round1` (half-life 100ms) | Random-agent model, no S |
|---|---|---|---|
| 0.1s | 24% | 23% | 37% |
| 0.5s | 44% | 32% (t = +239) | 7.5% |
| 1s | 37% | 21% | 4.4% |
| 5s | 8% | 4% | 1.2% |

- **Why we believe S exists:** a pure random-agent model placing orders relative to the best price can't produce the peak at 0.5–1s, and a model with S does. Over and above queue imbalance and recent moves, δ still adds about 23% R² (on `sample_v2`).
- **Online estimate:** using only what a strategy sees in its snapshots, a replay of `sample_v2` gave R² 29% at 0.5s with a 200ms half-life.
  - In `round1`, a 100ms half-life clearly beats 200ms (32% vs 20% at 0.5s).
  - The best online half-life still needs tuning per round.

## What this means for strategies

1. **`round1` is almost always at a 1-tick spread (93%).** There is no 2-tick spread to step inside. Making means joining the touch queue, or resting one tick behind it, which IOCs reach 46% of the time.
2. **Quote toward S.** Ŝ leads the mid by hundreds of ms, far more than our 5ms latency. When δ is clearly positive, keep or join the bid and pull or back off the ask; do the reverse when it's negative.
3. **Taker flow is pure noise.** In `sample_v2`, fills from the back of the queue were *not* worse: +0.89 ticks/lot at 5s, against +0.74 at the front. Even fills from IOCs over 2,000 lots held +1.05.
   - The cost of the back of the queue is fill rate: about 16% of joins fill, and under 7% with more than 1,500 lots ahead.
   - Re-check this on `round1`. Its sizes and cancel rates differ.
4. **Queues turn over fast.** Touch orders live about 0.3s in `round1`, so queue position improves quickly.
5. **Plan for parameters to change between rounds.** v2 and r1 share the engine but differ in every rate and size parameter. Estimate them online (Ŝ half-life, vol, spread regime) instead of hard-coding v2 values.
6. **The `GeneratorFingerprint` detector needs reworking before use.** It scores `round1` at 0.65, below its 0.70 threshold, only because it expects U{20..40} sizes. Its other 4 components score 0.97 or higher.

## Caveats

- `sample_v2` is analysed most deeply. For `round1`, the same checks were rerun, but the markout and queue analysis was not.
- The hidden-S model still has a depth mismatch: about 603 lots at the touch in a back-of-envelope estimate, against 818 real. That suggests the X distribution is not yet exact.
- No numbers here come from a backtest. "456k PnL" (a hindsight best case) and "$60–80k" (a rough calculation) from the reports are not achievable estimates.
