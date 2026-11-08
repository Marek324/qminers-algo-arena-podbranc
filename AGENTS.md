# AGENTS.md

Qminers Algo Trade Arena hackathon. Our job is to write a **trading strategy**. The exchange ("the market") is fixed and must not be touched. How the market works: [docs/MARKET.md](docs/MARKET.md).

## What you may change

- `strategies/`: new strategy files. Copy `strategies/template_strategy.py` into a new file and work there. Strategies are sorted by the round they were built for: `strategies/round1/`, `strategies/round2/`, and `strategies/roundN/` for each new round. Put new work in the current round's folder. The top level holds only the two reference files. Oracles (`oracle_overfit_round1`, `oracle_lever_b2_*`, `dumb`) and probes (`simple_probe_mm`, `probe_*`) live in their round's folder too; never submit them. File names stay unique across folders, because the dashboard lists them by name. When you move a strategy, update the paths that reference it.
- `tools/`: reusable building blocks (regressions, estimators, ...) that get **copied** into strategy files. See [Tools](#tools).
- `docs/`, `AGENTS.md`, and helper scripts for analysis or parameter sweeps (e.g. `scripts/`, notebooks). These never get submitted.
- `experiments/`: inactive experiments preserved for later, with their analysis, strategies, helpers and result summaries kept together. `experiments/inventory/` is the archived inventory-penalty/endgame study: its results are historical round-1 evidence, not active recommendations; re-run it on the new dataset before reconsidering it.
- `data/` and `logs/`: local data and backtest output. Both are git-ignored, so don't commit them.

## What you must not change

- `simulator/`: the exchange. It is the same code used for scoring, so changing it only makes local results wrong.
- `explorer/`: the backtest explorer app.
- `pyproject.toml`, `uv.lock`, `.python-version`: the package versions match the scoring sandbox. Don't add dependencies.
- `strategies/template_strategy.py` and `strategies/trend_follower.py`: keep them as reference.

If something in the simulator looks wrong, write it down. Work around it in the strategy, not in the simulator.

## Tools

Only **one `.py` file** is submitted, so a strategy can never `import tools` (or anything else local). `tools/` is a library to copy from, not to import:

- To use a tool, paste the class (and the imports listed at the top of its file) into the strategy file.
- Every tool must be **self-contained**: one class (or function) with no dependency on other tools or other files in the repo. If two tools share a helper, duplicate it.
- Tools must already pass the submission rules below: stdlib/`numpy`/`pandas`/`scipy`/`scikit-learn` only, no forbidden builtins, no dunder attributes besides `__init__` (so no `__slots__`, `__repr__`, ...), no `str.format`.
- Keep per-call cost low (O(1) or small O(k²)); they run on every market event.
- Analysis scripts may import from `tools/` directly.
- **Save every reusable building block to `tools/`.** When you write something generic for a strategy (a regression, filter, volatility estimator, signal, ...), put the canonical version in `tools/` as well, not only inline in the strategy. Strategies get rewritten and thrown away while we try things; `tools/` is where the pieces survive. If you improve a copied tool inside a strategy, port the fix back to `tools/` and add new tools to the list below.

Current tools:

- `tools/volatility.py`: `RealizedVol` (online volatility in ticks/√s from ~1 s samples, fast warm-up).
- `tools/markout.py`: `FillMarkout` (running per-lot markout of our own fills a fixed time later).
- `tools/kalman_price.py`: `HiddenPriceKalman` (1-D Kalman filter for the hidden price S from new limit-order prices, with jump inflation after big IOCs, online offset and noise, censored capped orders).
- `tools/regime_filter.py`: `RegimeFilter` (online 2-state calm/storm HMM forward filter on per-second counts of far-placed new orders, taker events, wide spread and optionally limit inserts; returns P(storm)).
- `tools/rls.py`: `RLS` (recursive least squares with forgetting; online fair-value prediction, used by `alpha_as_mm`).
- `tools/id_gap_jump.py`: `IdGapJump` (jump detector from missing market order ids plus orders clamped inside the stale touch; fires 5-25 ms into a jump; on round 4 it is the leverage trigger that pays, see docs/LEVERAGE.md).
- `tools/flow_reversion.py`: `FlowReversion` (online regression of the next ~5 s mid move on slow signed taker flow; uses the slope only when it shows mean reversion).
- `tools/take_gate.py`: `TakeGate` (taker edge threshold raised by filter inconsistency, i.e. the EWMA of innovation z², and by reversing a recent take; optional edge-sized take caps).
- `tools/breach_gate.py`: `BreachGate` (should we go over |I| = 100? Paper-trades each leveraged excursion on the live book and allows a signal bucket only when its mean net after the 10k $/s penalty is significantly positive).

## No tests

Don't write unit tests, integration tests, e2e tests or test fixtures, and don't add a test framework. To check a strategy, run it against the market with a backtest (see Workflow). The `pnl` in `result.json` and the explorer show whether it works.

## Submission rules (checked before it runs)

- **One `.py` file** with **one** `BaseStrategy` subclass that can be built with no arguments. If a file has more than one, the alphabetically first is used.
- Allowed imports: the Python 3.13 standard library, `numpy`, `pandas`, `scipy`, `scikit-learn`, `simulator.models`, and `from simulator.strategy import BaseStrategy`. Nothing else from `simulator`.
- Forbidden modules: `os`, `sys`, `io`, `socket`, `subprocess`, `threading`, `importlib`, `inspect`, `gc`, and anything else that gives system, network, file, threading or introspection access.
- Forbidden code: `open`, `eval`, `exec`, `getattr`, `type`, dunder attributes (`__init__` is fine), `str.format`. Use f-strings.
- No files, no network, no saved state between runs. All parameters are hard-coded in the file.

## Strategy hard limits

- Prices and quantities are positive integers (prices in ticks, sizes in lots).
- Keep `|inventory| <= 100`. Each second outside that range costs $10,000.
- An order is REJECTED if `|I + s·(R + q)| > 1000` (I = inventory, R = your resting size on that side, q = order size, s = +1 buy / −1 sell).
- Every send, cancel and modify takes effect 5 ms later. Track inventory from **fills** (`on_private_order_update`), not from what you sent.
- Callbacks run on every market event (tens of thousands per hour), so keep them light: no heavy work per tick.

## Workflow

```bash
uv sync
uv run python -m gzip -d data/<file>.csv.gz      # simulator reads plain CSV only
uv run exchange-simulator --strategy strategies/<mine>.py --backtest data/<file>.csv --log-dir logs/<run> --no-logs
uv run exchange-simulator --strategy strategies/<mine>.py --backtest data/<file>.csv --log-dir logs/<run> --progress   # full logs
uv run backtest-explorer --log-dir logs/<run>    # needs a full-logs run
```

- Analyze a tape before writing a strategy: `uv run analysis.py data/<file>.csv` writes `logs/analysis/<file>.json` and `.txt`. Report contents and schema: [docs/ANALYSIS.md](docs/ANALYSIS.md).
- Before trying leverage over |I| = 100, read [docs/LEVERAGE.md](docs/LEVERAGE.md): a hindsight bound, the real-simulator oracle and why the expected value is negative on round 3.
- Check a strategy for overfitting before trusting its score: `uv run python scripts/robustness.py strategies/<mine>.py` runs it on every tape in data/, over time chunks, on a mirrored market and reshuffled sessions (`--sensitivity all` adds parameter nudges); the dashboard's Robustness tab shows the report.
- Measure a strategy against a known generator: `.venv/bin/python scripts/gen_tape.py --seed 1` writes a `round1` twin to `data/synthetic/` (deterministic per seed) plus its true hidden price S (`_S.npz`). The tapes are git-ignored: regenerate the three used in the docs with `scripts/gen_tape.py --seed 1 2 3` (about 150 MB each); `analysis.py` on it gives the hindsight max, and `scripts/oracle_s_mm.py` / `scripts/sweep_oracle.py` run a true-S oracle. Fit and results: [docs/DATA_ANALYSIS.md](docs/DATA_ANALYSIS.md#synthetic-twin-of-round1).
- The score is `pnl` in `logs/<run>/result.json`. Also check `breach_penalty` (should be 0) and the `abs_inventory_q*` quantiles.
- `data/sample.csv` is synthetic and trends, and the real rounds look nothing like it, so don't tune to the sample. The deck calls the rounds "recorded market data", but analysis shows `round1` is synthetic too. It comes from the same engine as `sample_v2.csv`, with different parameters. The generator, its fitted parameters and the strategy implications are in [docs/DATA_ANALYSIS.md](docs/DATA_ANALYSIS.md). Expect parameters to change between rounds, so estimate them online instead of hard-coding them.
- Before submitting, check the file only uses allowed imports and builtins, then run a full backtest of that exact file.
- Submit at arena.qminers.com → Dashboard → Submit strategy (done by a human).
- Leaderboard, rounds, deadlines and our submission results: the read-only `arena` MCP server (`scripts/arena_mcp.py`, `.mcp.json`). Setup (an `ARENA_TOKEN` env var copied from the site), tools and troubleshooting: [.claude/skills/arena/SKILL.md](.claude/skills/arena/SKILL.md).
