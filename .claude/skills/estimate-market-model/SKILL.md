---
name: estimate-market-model
description: Estimate the parameters of our market model for a new arena tape (a round file in data/), fit our synthetic-market model to it, check the fit, and compute the target PnL. Use when a new round or sample CSV arrives, when asked how a tape was generated, what changed between rounds, whether the generator model still fits, or what PnL to aim for ("target PnL", "expected value", "estimate parameters for round N").
---

# Estimating the market model for a tape

The arena tapes are synthetic. `sample_v2` and `round1` come from one engine with different parameters
(background: `docs/DATA_ANALYSIS.md`). Our model of that engine is `analysis/generator.py`:

- a hidden fair price **S**: a Gaussian random walk (`sigma_s` ticks/√s);
- **limit orders**: Poisson, side a coin, buy at `floor(S − X)` / sell at `ceil(S + X)`, clamped so they never cross the
  book; X is lognormal (sample_v2) or a shifted gamma (round1); sizes from the tape's own distribution; an exponential
  cancel timer (`life_mean_s`); `cancel_filled` says whether the cancel row is still written for orders that already filled;
- **IOCs**: Poisson, side a coin (independent of S), priced at `round(S ± 1)` (`ioc_mode='S'`);
- **impact**: each IOC pushes S by `impact_k · size^impact_beta` ticks (round1: 0.009 · size^0.65; sample_v2: none).
  The tape analysis in DATA_ANALYSIS.md call S exogenous; that holds for the mid, but in round1 IOCs do move S:
  without the push, generated permanent impact is half the real one.

## Workflow

Heavy steps (analysis of a multi-million-row tape, calibration, target) belong on the compute server, not the laptop:
see "Compute server" below.

1. **Get the tape.** `git pull`; tapes arrive as `data/<name>.csv.gz`. Unzip a plain copy: `gunzip -k data/<name>.csv.gz`
   (the simulator, the dashboard and the scripts read plain CSV).
2. **Analyse it:** `uv run analysis.py data/<name>.csv` (about 12 s per million rows when the machine is free). Compare
   it with the previous rounds. The useful comparison table is in the conversation history of this skill's first use:
   spread mix, volatility, impact, maker markout, touch depth, signal strength, theoretical max per hour.
3. **Fit the generator:**
   `uv run python scripts/estimate_model.py data/<name>.csv` (add `--target --strategies strategies/a.py,strategies/b.py`
   for the target). It measures rates, sizes, cancel timer, IOC pricing, X and impact, calibrates X / sigma_s /
   impact_k on generated tapes, and writes `logs/generator/<name>_params.json`, `<name>_fit.json` and `<name>_fit.md`.
4. **Read the fit** (`<name>_fit.md`): per metric, real vs generated. The fit error is the sum of |difference| /
   tolerance; about 1 per metric is a close fit. Then:
   - Everything close → the engine is unchanged, only the parameters moved. Report the parameter changes from the last round.
   - One family off → the model is missing a mechanism. Investigate before trusting targets (see "Checks that found
     mechanisms" below). Examples so far: touch too thin, so the X shape is wrong; markout too high, so impact is missing
     or too weak.
5. **Target PnL:** `analysis/target.py` tunes an S-aware reference market maker (it knows the current S, never the
   future) on generated sessions and reports its mean PnL per hour on fresh ones, plus our strategies on the same
   sessions. Present it as an estimate with its spread, not a promise. Also check how close the strategies' generated
   scores are to their real-tape scores: that is the best single test of the generator.
6. **Write it down:** update `docs/DATA_ANALYSIS.md` with the new round's column (parameters, resulting market, what
   changed, strategy implications) and commit the doc, not the logs. Follow the repo's commit rules.

## Checks that found mechanisms (reuse them on new rounds)

- **Are IOCs informed?** Correlate the IOC side with δ = (mean new-limit price over the last 100–500 ms) − mid. In
  sample_v2 and round1, P(buy) = 0.50 whatever δ is, so they are not informed.
- **Do IOCs move S?** Measure the signed mid move from just before an IOC to 10 s and 30 s later, bucketed by size.
  A flat, size-dependent permanent move means impact on S. Fit move = k · size^beta on the buckets (round1: about
  0.009 · size^0.65, from +0.035 ticks at 8 lots to +0.79 at 1,000); about 0 means no impact (sample_v2). Use the
  power law, not √size: the large IOCs dominate volume-weighted markouts.
- **X shape:** distance of limit prices from a centred ±0.5 s estimate of S. The measured X includes about 0.5 ticks of
  floor/ceil rounding, so the doc's "X ≈ 1.3" means an underlying mean of about 0.8 to 0.9.
- **Cancel rule:** cancel rows per limit order. 1.00 means every order is cancelled even after it fills (round1);
  about 0.90 means filled orders are not cancelled (sample_v2).

## Known residuals (as of round1, generator v3)

- sample_v2 (fit error 8.6 over 10 metrics): spread=1 share 0.51 against 0.41 real, touch about 20% thinner, the rest
  within a few percent.
- round1 (fit error 8.6): spread mix, mid change rate and impact are right, and the maker markout collapse is mostly
  there (+0.14 at 1 s against +0.07 real). Still the generated book is more predictable than the real one (microprice IC
  0.33 against 0.25) and the touch is about 15% thinner. Targets for round1-like parameters lean slightly optimistic.
- Fit history: X normal + touch-relative IOCs gave error 17 on round1. S-based IOC prices and gamma X brought it to 11.5,
  and power-law impact to 8.6.

## Compute server

The team server runs the dashboard (Streamlit) from a checkout of this repo. Use it for anything heavy so the laptop
stays usable:

- `ssh ubuntu@185.175.110.171`; the repo is at `~/.ssh/qminers-algo-arena-podbranc`; uv is at `~/.local/bin/uv`
  (not on the non-interactive PATH). 8 cores, 31 GB.
- **Keep it up to date:** after pushing, run `git pull --rebase --autostash` there. The server has a local,
  uncommitted `streamlit` dependency in `pyproject.toml` / `uv.lock`; the autostash keeps it. Streamlit picks up code
  changes on the next rerun.
- **Long jobs:** start them detached so the SSH call returns:
  `setsid nohup ~/.local/bin/uv run python <script> > <log> 2>&1 < /dev/null & disown`, then poll the log.
- Copy uncommitted work with `rsync` only for experiments. Commit, push and pull for anything the dashboard should show.
- The server's git remote URL contains a GitHub token. Never print it again or copy it anywhere.
