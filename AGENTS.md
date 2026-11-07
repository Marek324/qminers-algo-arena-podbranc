# AGENTS.md

Qminers Algo Trade Arena hackathon. Our job is to write a **trading strategy**. The exchange ("the market") is fixed and must not be touched. How the market works: [docs/MARKET.md](docs/MARKET.md).

## What you may change

- `strategies/`: new strategy files. Copy `strategies/template_strategy.py` into a new file and work there.
- `tools/`: reusable building blocks (regressions, estimators, ...) that get **copied** into strategy files. See [Tools](#tools).
- `docs/`, `AGENTS.md`, and helper scripts for analysis or parameter sweeps (e.g. `scripts/`, notebooks). These never get submitted.
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

- `tools/regression.py`: `RollingLinReg` (exact rolling OLS), `EwLinReg` (time-decayed OLS), `RLS` (multivariate recursive least squares).

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
- The score is `pnl` in `logs/<run>/result.json`. Also check `breach_penalty` (should be 0) and the `abs_inventory_q*` quantiles.
- `data/sample.csv` is synthetic and trends. The real rounds are recorded market data and look nothing like it, so don't tune to the sample.
- Before submitting, check the file only uses allowed imports and builtins, then run a full backtest of that exact file.
- Submit at arena.qminers.com → Dashboard → Submit strategy (done by a human).
