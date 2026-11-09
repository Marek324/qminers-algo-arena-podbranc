# Qminers Algo Trade Arena

Team Podbranč's repo for the [Qminers Algo Trade Arena](https://arena.qminers.com) hackathon (November 2026), built on the organizers' starter template.

The task: write one Python file that trades a single anonymous instrument against a replayed limit order book.
Each round gave one public tape to train on.
Submissions were scored on a hidden validation tape, and the final pick on a private test tape.
PnL is the score.

## Results

**4th of 61 overall.**

| round | place | final strategy | test PnL |
|---|---|---|---:|
| 1 | 3rd / 61 | [`adaptive_mm_plus`](strategies/round1/adaptive_mm_plus.py) | 667,357 |
| 2 | 5th / 59 | [`hmm_mm_v2`](strategies/round2/hmm_mm_v2.py) | 1,195,019 |
| 3 | 4th / 53 | [`hmm_deep_clear_ks`](strategies/round3/hmm_deep_clear_ks.py) | 1,134,420 |
| 4 | 7th / 52 | [`jteam_r4`](strategies/round4/jteam_r4.py) | 1,885,793 |

Overall score = 0.8·z₁ + z₂ + z₃ + 1.2·z₄ (z = per-round z-score of PnL).
Every submission with train / validation / test PnL: [submissions.md](submissions.md).

## How the strategy evolved

1. **Round 1:** a market maker around an S estimate, keeping its queue place, plus a momentum taker that trades only while it would have paid.
2. **Round 2:** an online 2-state HMM picks the regime and a 1-D Kalman filter tracks S.
   It makes markets when calm and takes the stale book in a storm.
3. **Round 3:** deeper book orders count as Kalman readings, and runs of far orders snap S to the new level.
4. **Round 4:** a momentum lean, depth imbalance, quote hysteresis, and a jump sniper that briefly goes over the cap.

Full research log, including what didn't work: [docs/RESEARCH.md](docs/RESEARCH.md).
Written and read by our agents.

## Setup

```bash
# curl -LsSf https://astral.sh/uv/install.sh | sh  # if uv is missing
uv sync                                            # install dependencies
for f in data/*.csv.gz; do gzip -dkf "$f"; done    # unzip data
```

Run a strategy and open the result:

```bash
uv run exchange-simulator --strategy strategies/round4/jteam_r4.py --backtest data/round4_test.csv --log-dir logs/r4 --progress
uv run backtest-explorer --log-dir logs/r4
cat logs/r4/result.json                            # pnl, breach_penalty, inventory quantiles
```

Other entry points:

```bash
uv run analysis.py data/round4_train.csv                          # tape report -> logs/analysis/
uv run --with streamlit streamlit run dashboard/app.py            # tape and strategy dashboard
```

## Repo structure

```
strategies/       All submitted strategies, by round (roundN/)
data/             Each round's tapes: roundN_train / roundN_val / roundN_test; only *_train available during the hackathon

analysis/         Data analysis framework
dashboard/        Streamlit dashboard for tapes and strategy runs
scripts/          MCP server, analysis scripts and parameter sweeps
docs/             File-based inter-agent communication
tools/            Building blocks copied into strategies (a submission is one file)

simulator/        Provided in the template
explorer/         Provided in the template

submissions.md    PnL of every submitted strategy
```

`docs/` is a working log: it names many strategies that were never submitted, and those were deleted.
They are still in git history.

## How we worked

Most of the code was written by AI coding agents, steered and checked by the four of us.
[AGENTS.md](AGENTS.md) is the brief every agent read.
`.claude/skills/` holds repeatable workflows, and `scripts/arena_mcp.py` is a read-only MCP server for the arena site.

Strategies use only the Python standard library (`math`, `collections`), although the sandbox also allowed numpy, pandas, scipy and scikit-learn.
Analysis uses numpy, pandas, scipy, scikit-learn and matplotlib, and the dashboard uses Streamlit and Plotly.

## License

Our code is MIT ([LICENSE](LICENSE)).
`simulator/`, `explorer/`, the two reference strategies and the market data belong to Qminers.
