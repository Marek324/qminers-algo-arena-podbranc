# Estimating the market model when a new round drops

A checklist to get from a new tape to fitted parameters and an expected PnL in about 30–45 minutes on a 12-core
laptop. No cluster needed. The model and its parameter meanings are in [round1.md](round1.md). The engine background is
in [DATA_ANALYSIS.md](DATA_ANALYSIS.md). Generating sessions from the fitted parameters, and what each
parameter means: [GENERATING_DATA.md](GENERATING_DATA.md). The `estimate-market-model` skill (`.claude/skills/`) follows the same steps.

Use `.venv/bin/python` (or `uv run python`). Scripts that use multiprocessing need a normal shell, not a sandbox
that blocks Unix sockets.

## Steps

| # | Step | Command | Time (12 cores) |
|---|---|---|---|
| 1 | Get the tape | `git pull`, then `gunzip -k data/roundN.csv.gz` | seconds |
| 2 | Analyse it | `.venv/bin/python analysis.py data/roundN.csv` | ~1 min |
| 3 | Fit the model (rates, sizes, cancel timer, IOC pricing, X, σ, push) | `.venv/bin/python scripts/estimate_model.py data/roundN.csv --workers 12` | ~1–2 min |
| 4 | Read the fit | `logs/generator/roundN_fit.md` (about 1 per metric is close) | – |
| 5 | Optional: likelihood fit of X | `.venv/bin/python scripts/fit_x.py data/roundN.csv --slices 20 --workers 12` | ~8–10 min |
| 6 | Expected PnL with the fitted params | `.venv/bin/python scripts/expected_pnl.py --name roundN --real data/roundN.csv` | ~15–20 min |
| 7 | Same with the likelihood X | `.venv/bin/python scripts/expected_pnl.py --name roundN --x-from logs/generator/roundN_xfit.json` | ~15–20 min |
| 8 | Write it down | add a column for the new round to `docs/round1.md` / `DATA_ANALYSIS.md`, then commit the docs, not `logs/` | – |

Steps 5–7 can run together if the cores are split (`--workers 6` each).

## What to look at

- **The fit table (step 4).** If everything is close, the engine is unchanged and only the parameters moved: report
  what changed since the last round. If one family is off, a mechanism is missing. For example, a too-thin touch means
  the X shape is wrong, and a too-high maker markout means the push per IOC is missing or too weak. Use the checks in the
  skill before trusting any PnL.
- **Generated vs real PnL per strategy (step 6).** This is the best single test of the model. On round 1 it was within
  6–18%.
- **The reference market maker** (S-aware, from `analysis/target.py`) is a sanity line, not a ceiling. Our strategies
  beat it on round 1. For a real ceiling use the hindsight maximum (`compute_bracket`, or `theoretical_max` in the
  analysis report).

## Known pitfalls

- **The calibration in `estimate_model.py` is noisy between runs.** The round-1 re-run gave σ 0.81 and push 0.0107
  against the documented 0.75 and 0.009, with fit error 15 against 8.6. If the fit error is much above 9, run it again
  or keep the better earlier values.
- **The X from `fit_x.py` is the better estimate of X.** It is a likelihood fit on the whole tape. Its σ ignores the push,
  though, so take σ from `estimate_model.py`.
- **Cancels.** Check `cancel_filled` in the params. True means the round also writes cancels for orders that already
  filled (round 1); sample_v2 doesn't.
- **Parameters change every round.** Use the fitted model to generate many varied sessions for search, and keep the real
  tape for the final check. Never tune on the real tape.

## When a cluster helps

Only for big sweeps, such as all 96 slices of `fit_x.py` or many parameter variants. Everything above runs locally.
On IT4I Barbora, run 1-node jobs with short limits, because they start soonest. Use `qcpu_free` or `qcpu_exp`, never the
login node. The job needs the repo's pinned versions in a venv, because the cluster's SciPy module ships older pandas.
The experimental Barbora calibration (`findings/barbora/`, git-ignored) used a simplified simulator without the IOC
push, so don't reuse its results for anything but X.
