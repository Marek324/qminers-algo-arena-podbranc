# Playbook: picking a strategy for a new round

Repeat this process each time a new data file arrives. The idea: **find out who makes money in this market before writing a strategy.**

In a fixed tape, PnL comes from one of two places:

- **Aggressors are informed** (the price moves their way after they trade). Then whoever trades with them early wins, so **follow the flow** with IOC.
- **Aggressors are uninformed** (the price doesn't move). Then whoever rests orders and lets them trade wins, so **make markets** with LIMIT and earn the spread.

The diagnostic tells you which one, in about a minute.

## Step 0: unzip and run the old strategy

```bash
uv run python -m gzip -d data/<round>.csv.gz
uv run exchange-simulator --strategy strategies/round1/imbalance_mm.py --backtest data/<round>.csv --log-dir logs/<round>_old --no-logs
```

A large loss means the market type changed, not that a parameter is off. Don't tune; go to step 1.

## Step 1: run the diagnostic

```bash
uv run python scripts/diagnose.py data/<round>.csv
```

It replays the tape through the real engine without placing orders and prints five things. For each line: what to read from it, and what we saw on the two datasets.

| # | Measure | What to read | `sample.csv` | `sample_v2.csv` |
|---|---|---|---|---|
| 1 | Spread, depth at the best price, IOC rate, order lifetime | How many lots are available, how deep queues are, how fast the book churns | 2-tick spread, 25 lots, 0.17 IOC/s | 1–2 ticks, ~770 lots, 6 IOC/s |
| 2 | **Mid move after aggressive trades**, in the aggressor's direction | **The key number.** Compare it with the round-trip spread (2 × half-spread) | **+9.5 ticks at 3 s** | **+0.01 ticks** |
| 3 | **Maker edge per lot**: what a resting order filled by those trades earns against the later mid | Positive and steady over 1–30 s means market making pays and there's no adverse selection (getting run over after a fill) | −9 ticks | **+0.88 ticks** |
| 4 | Imbalance correlation: does the bid/ask size gap at the best price predict the next move | Above about 0.2: use it to skew quotes. It's usually too small to pay the spread on its own | 0.05 | 0.45 |
| 5 | Variance ratio | Above 1 means trending (be careful holding inventory). Below 1 means mean reverting (mean reversion may pay) | 2.3 (trends) | 1.1 (random walk) |

## Step 2: decide

```
mid move at 3 s  >  round-trip spread + 1 tick ?
    yes -> FLOW FOLLOWING   start from strategies/round1/flow_momentum.py
    no  -> maker edge at 5 s > 0.2 ticks/lot ?
             yes -> MARKET MAKING   start from strategies/round1/imbalance_mm.py
             no  -> neither is clean: split rows 2 and 3 by trade size and time of day
                    (big trades may be informed when small ones aren't)
```

The script prints this suggestion at the end.

## Step 3: build the simplest version, then measure your own fills

Get a positive baseline first. Then use a full-logs run (no `--no-logs`) and check the edge of **our own** fills against the later mid. This shows what actually limits PnL:

| What you see | Limit | Fix |
|---|---|---|
| Edge per lot good but few lots filled | Fill rate / queue position | Keep orders in their queue place instead of cancel/resend; quote larger; quote inside the spread when it's wide |
| Edge falls after the fill (worse at 5 s than at 0 s) | Adverse selection | Pull or step back the side the imbalance is against; widen |
| Inventory pinned at ±100 | Position limit | Inventory skew in the fair price; get back to flat at zero edge |
| PnL from a few big jumps | Inventory risk | Smaller position, stronger skew |
| Follow-flow: entry or exit size limited | Liquidity per price level | Enter early, exit with a ladder of resting orders |

This step took v2 from 113k to 276k. The edge per lot was already good; the fix was getting more lots filled.

## Step 4: tune, but pick the middle of the plateau

```bash
uv run python scripts/sweep_class.py strategies/<file>.py data/<round>.csv <scratch_dir> 'PARAM=a,b,c' 'PARAM2=x,y'
```

- Choose a value where **neighbouring settings score similarly**, not the single best run.
- If the best value sits at the edge of the grid, extend the grid in that direction.

## Step 5: check it isn't luck

- **Steady across time:** PnL per sixth of the run (from `pnl.csv`) should be similar in each slice, with no single cliff or jump.
- **Every earlier dataset:** run the new file on all previous rounds. A small loss on a different market type is acceptable; a large one means switching by market type should be added (see [STRATEGIES.md](STRATEGIES.md), section 4).
- **Result fields:** `breach_penalty` = 0, `abs_inventory_q997` ≤ 100, ending `inventory` small.
- **Submission rules:** only allowed imports and builtins (see [AGENTS.md](../AGENTS.md)).

## Step 6: write it down

Add the round's diagnostic numbers and results to [STRATEGIES.md](STRATEGIES.md), so later rounds can compare against them.
