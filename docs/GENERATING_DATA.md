# Generating synthetic market data

How to generate new sessions that behave like a round, for testing strategies on more than one tape. The model is
`analysis/generator.py`, and its structure is explained in [round1.md](round1.md). The steps to fit it to a new tape are
in [NEW_ROUND.md](NEW_ROUND.md).

## Quick start

```python
from pathlib import Path
from analysis.generator import PARAM_DIR, GenParams, generate, write_tape

p = GenParams.load(PARAM_DIR / 'round1_params.json')         # logs/generator/round1_params.json
df, s_times, s_values = generate(p, seed=1, duration_s=1200)   # one 20-minute session
write_tape(df, Path('data/synth_round1_s1.csv'))               # simulator CSV format
```

Then backtest on it like any tape:
`uv run exchange-simulator --strategy strategies/<mine>.py --backtest data/synth_round1_s1.csv --log-dir logs/<run> --no-logs`.

- `generate` returns the tape rows and the hidden price S (`s_times`, `s_values`). Use S to measure how good a
  strategy's fair-value estimate is. A strategy never sees S.
- The same seed always gives the same session. Use different seeds for training and for testing.

## What's needed

`logs/` is git-ignored, so a fresh checkout has no parameter files. Create them from the tape:

```bash
gunzip -k data/round1.csv.gz
.venv/bin/python scripts/estimate_model.py data/round1.csv --workers 12   # ~2 min
```

This writes `logs/generator/round1_params.json` and the size samples `round1_limit_sizes.npy` / `round1_ioc_sizes.npy`.
Order sizes are drawn from those samples. Without them, the generator falls back to fixed sizes of 30 and 40 lots,
which look nothing like the round.

To use a different parameter set below, load the params file, change the fields and save it under a new name:

```python
from dataclasses import replace
q = replace(p, name='round1_best', sigma_s=0.75, impact_k=0.009, impact_beta=0.65,
            x_dist='gamma', x_a=1.117, x_b=0.4007, x_loc=0.0168)
q.save(PARAM_DIR / 'round1_best_params.json')
```

## Round 1 parameter sets

All three share the measured parts:
- limit orders: 179.8/s;
- IOCs: 8.0/s;
- cancel timer: mean 0.467 s, with `cancel_filled=True`;
- IOCs priced at round(S ± 1) (`ioc_mode='S'`);
- `clamp=True`;
- sizes from the tape.

They differ only in σ of S, the push per IOC and X:

| Set | σ of S | Push per IOC | X (`x_loc` + Gamma(`x_a`, `x_b`)) | Fit error |
|---|---|---|---|---|
| Marek (doc) | 0.75 | 0.009 · size^0.65 | 0.054 + Gamma(1.72, 0.32), mean 0.61 | **8.6** |
| Marek (re-run) | 0.809 | 0.0107 · size^0.66 | 0.05 + Gamma(2.53, 0.188), mean 0.53 | 15.0 |
| Ours | 0.809 | 0.0107 · size^0.66 | **0.017 + Gamma(1.117, 0.401), mean 0.46** | 10.5 |

The fit error is the sum over 10 market statistics of |generated − real| / tolerance, so about 1 per statistic is a
close fit (details in [round1.md](round1.md)).

**Which to use:**
- **For testing strategies,** use Marek's documented set. It has the best overall fit.
- **For a market closer to the real touch,** use our X. Its depth at the touch is 891 lots against 901 real, the best of
  the three.
- **Best guess:** Marek's σ and push with our X (the `round1_best` example above). This combination is not tested yet:
  check its fit before relying on it.

Parameters change between rounds, so don't tune a strategy to one set. Generate sessions across a range (for example σ
0.7–0.9, X mean 0.45–0.6, push k 0.007–0.011) and keep the real tape for the final check.

## What the parameters mean

| Field | Meaning |
|---|---|
| `sigma_s` | volatility of the hidden fair price S, ticks per √second. With the IOC push added, the mid moves about 0.92 ticks/√s in round 1. |
| `impact_k`, `impact_beta` | every IOC permanently moves S by `impact_k · size^impact_beta` ticks in its direction |
| `x_dist`, `x_a`, `x_b`, `x_loc` | distance X of a new limit order from S. A buy is placed at floor(S − X) and a sell at ceil(S + X). Gamma: X = `x_loc` + Gamma(shape `x_a`, scale `x_b`). Lognormal: log-mean `x_a`, log-sd `x_b`. A smaller X gives a thicker touch and more 1-tick spreads. |
| `x_mean`, `x_std` | summary of X, only used by `x_dist='normal'` |
| `limit_rate`, `ioc_rate` | Poisson arrival rates per second; each order's side is a fair coin |
| `life_mean_s` | mean of the exponential cancel timer of each limit order |
| `cancel_filled` | True: a cancel row is written even for orders that already filled (round 1). False: only resting orders are cancelled (sample_v2). |
| `ioc_mode`, `ioc_offset` | `'S'`: IOCs priced at round(S ± 1). `'touch'`: priced relative to the opposite touch, using the `ioc_offset` probabilities. |
| `clamp` | a limit order that would cross the book is moved back to the touch |
| `s0`, `duration_s` | starting S and default session length |

## How each parameter was estimated

- **Measured directly from the tape** by `estimate_model.py`: the rates, the sizes, the cancel timer, the cancel rule and
  the IOC pricing.
- **σ, push and X (Marek's sets):** `estimate_model.py` generates tapes and adjusts these three until 10 statistics
  match the real tape: spread mix, mid volatility, IOC impact, fill markouts, touch depth and more. This is indirect, and
  it is noisy between runs. That's why the re-run differs from the documented values.
- **X (ours):** `scripts/fit_x.py` fits X straight from the limit-order prices, by maximum likelihood:
  1. It takes only the new limit orders, and fits 20 slices of about 2 minutes, spread across the tape.
  2. A buy at price p means X lies between S − p − 1 and S − p, so each order gives a probability for every possible S.
  3. A grid filter, a simple hidden Markov model on a 0.05-tick grid, tracks S between orders as a random walk with σ.
     It multiplies in each order's probability, which gives the exact likelihood of a guess of σ and X.
  4. A coarse search followed by Nelder-Mead picks the most likely σ and X for each slice.
  5. The slices are pooled with weights of 1/SE².

  This uses every order and the exact floor/ceil rule, so it pins down X better. It ignores the IOC push, though, so its
  σ (0.80–0.84) contains the push: take σ and the push from `estimate_model.py`. It also ignores the clamp. A lognormal
  X scores slightly better than the gamma (median 0.40), but the gamma is used so the sets compare directly.

## Expected PnL on generated sessions

Use `scripts/expected_pnl.py --name <set>`. It runs our strategies on 8 fresh 20-minute sessions, plus an S-aware
reference market maker. On round 1, per hour:

| | Marek (re-run) | Ours | Real round1 tape |
|---|---|---|---|
| imbalance_mm | 173.1k ± 8.4k | 159.9k ± 9.7k | 184.7k |
| adaptive_mm | 174.0k ± 7.8k | 165.5k ± 6.5k | 195.2k |
| Reference market maker | 137.4k | 134.5k | – |

The generated sessions come out 6–18% below the real tape, so they are slightly harder than the real round. The
reference market maker is a sanity check, not a ceiling: the hindsight maximum is about 435k/h.
