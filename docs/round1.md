# Round 1: market model parameters

Parameters of our market model (`analysis/generator.py`) estimated from `data/round1.csv` (200 min, 4.41M rows) on
2026-11-07. The model can generate new sessions with the same statistics as the round, for testing strategies on more
than one tape. Background on the model structure: [DATA_ANALYSIS.md](DATA_ANALYSIS.md). The fitted parameter file is
`logs/generator/round1_params.json` (git-ignored); `scripts/estimate_model.py` produces it from a tape.

## The model in one paragraph

A hidden fair price S follows a random walk. Limit orders arrive as a Poisson stream and are placed a random distance X
from S, never crossing the book. Each gets a random cancel timer. Aggressive (IOC) orders arrive as a separate Poisson
stream, pick their side with a fair coin and are priced one tick beyond S. In round 1, each IOC also pushes S in its
direction, more for bigger orders.

## Parameters

### Hidden fair price S

| Parameter | Round 1 | How it was set |
|---|---|---|
| Process | Gaussian random walk, S₀ = 10000 | |
| Volatility σ | **0.75 ticks/√s** | fitted; together with the IOC pushes it gives the observed mid volatility of 0.92 |
| Push per IOC | **0.009 · size^0.65 ticks**, in the IOC's direction | measured by size bucket (+0.035 ticks at 8 lots, +0.79 at 1,000), then fitted |

### Limit orders

| Parameter | Round 1 | How it was set |
|---|---|---|
| Arrivals | Poisson, **179.80 per second**, side 50/50 | measured |
| Price | buy = floor(S − X), sell = ceil(S + X), moved back one tick if it would cross the book | from the tape analysis |
| X | **0.054 + Gamma(shape 1.72, scale 0.32)**, mean about 0.61 ticks | fitted |
| Size | drawn from the round's own limit sizes: min 1, median 14, mean 72, max 124,248, 10.5% are size 1 | measured |
| Lifetime | cancel timer ~ **Exponential, mean 0.467 s** (CV 0.95) | measured |
| Cancel rule | the cancel row is written even if the order already filled; 13% of cancels are no-ops | measured |

### Aggressive orders (IOC)

| Parameter | Round 1 | How it was set |
|---|---|---|
| Arrivals | Poisson, **8.01 per second** | measured |
| Side | fair coin, independent of S and of the book (P(buy) = 0.50 whatever S − mid is) | measured |
| Price | **round(S + 1)** for a buy, **round(S − 1)** for a sell; 52% stop at the best price, 46% go one tick through | from the tape analysis |
| Size | drawn from the round's own IOC sizes: median 61, roughly lognormal(μ 4.11, σ 1.40), max 23,224 | measured |

### Compared with sample_v2

| | `sample_v2` | `round1` |
|---|---|---|
| σ of S | 0.50 | 0.75 (+ pushes) |
| Push per IOC | none | 0.009 · size^0.65 |
| X | lognormal(−0.235, 0.341), median 0.79 | 0.054 + Gamma(1.72, 0.32), mean 0.61 |
| Limit arrivals | 179.66/s | 179.80/s |
| Limit size | uniform 20–40 | heavy-tailed, median 14 |
| Mean lifetime | 0.70 s | 0.47 s |
| Cancel after fill | no | yes |
| IOC arrivals | 6.0/s | 8.0/s |
| IOC median size | 41 | 61 |

Same structure, different settings. Prices sit closer to S in round 1, which gives the 1-tick spread 93% of the time,
and the push per IOC is what makes passive fills lose most of their edge within a second.

## How well the model matches the round

A 30-minute generated session analysed with `analysis.py`, next to the real round:

| Statistic | Real round 1 | Model |
|---|---|---|
| Share of time at a 1-tick spread | 0.930 | 0.928 |
| Mid volatility at 1 s (ticks/√s) | 0.921 | 0.984 |
| Price impact of an IOC after 60 s (ticks) | 0.302 | 0.373 |
| Passive fill value at the fill (ticks/lot) | 0.679 | 0.707 |
| Passive fill value after 1 s (ticks/lot) | 0.069 | 0.141 |
| Median size at the best price (lots) | 901 | 748 |
| Mid changes per second | 2.57 | 2.61 |
| Microprice signal, Spearman at 1 s | 0.25 | 0.33 |
| Share of new limit orders joining the best price | 0.56 | 0.48 |
| Trades per second | 31.3 | 28.2 |

Strategy check on fresh generated sessions (8 sessions of 20 min each):

| | Real round 1 | Model |
|---|---|---|
| `imbalance_mm.py` (615k version) | 185k per hour | 194k per hour (sd 8k) |
| Theoretical max (perfect hindsight) | 417k per hour | 435k per hour (sd 20k) |

**Known gaps:**
- The real book is less predictable than the model's: microprice signal 0.25 against 0.33.
- The real touch is about 20% deeper.
- Passive fills keep a little more value in the model after 1 s: 0.14 against 0.07 ticks per lot.
- Together these make the model slightly kind to market makers.
- The seed is unknown: the model makes new sessions with the round's statistics, not the round itself.
- Sizes are resampled from the round instead of following a known formula.

## What this means for a target

On generated round-1 sessions, `imbalance_mm.py` makes about **190–200k per hour**, which is **about 630–660k for a
200-minute round**. The real round gave 615k.

The same strategy fed the *true* S (re-tuned over 12 settings) made 188k per hour, no better. The strategy's own estimate
of S lags the true one in a way that matches what the book will do next, so a more precise estimate of S does not raise
the score. Getting closer to the 435k per hour hindsight maximum would take a different design, most likely in how
quotes and queue position are managed.

## Using generated data without overfitting

- Generate for search, and keep the real tapes for selection. Never tune on the real tapes.
- Track the gap between generated and real PnL for every candidate. A gain that shows up only on generated sessions is
  exploiting a gap in the model.
- Vary the parameters across sessions (σ, X, rates, sizes, push) over a range covering sample_v2 and round 1, because
  the next round will have different settings.
- Prefer settings on flat plateaus. Don't build an edge on the book signal or on queue position, the two places where the
  model is known to be optimistic.

## Reproducing

```bash
uv run analysis.py data/round1.csv                     # analysis report, needed for the fit check
uv run python scripts/estimate_model.py data/round1.csv  # estimate parameters, check the fit
```

`analysis/target.py` holds the generated-session runs (`compute_bracket` for the strategy check above). Run heavy jobs
on the compute server, not a laptop.
