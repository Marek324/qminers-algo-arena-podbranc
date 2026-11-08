# Leverage over 100: when it pays

Measured on round3_public (2026-11-08). Re-run the two scripts below on every new tape before deciding.

## The problem

The penalty is a flat **10,000 $/s while |I| > 100**, whatever the size of the excess (`simulator/simulator.py:1290`).
Going over is renting up to 900 extra lots (the 1000 reject line) at a fixed price per second. For Q extra lots
held for τ seconds:

    net = Q * move - roundtrip_cost(Q) - 10,000 * τ

The score is risk-neutral, so the optimal rule is **breach iff E[net | what we see now] > 0**. That is optimal
stopping with a running cost. With hindsight, the best set of excursions is a weighted interval scheduling problem
(backward DP). 10,000 $/s is 10 $/ms, or 0.011 ticks per lot per ms at Q = 900.

## Results (round3_public, 1.67 h)

Hindsight bound, from the tape (`scripts/lever_oracle.py`, 2 ms grid, IOC in and out walking the visible book):

| Extra lots Q | Hindsight | Reactive: mid moved >=1 tick / 20 ms, exit with hindsight | Reactive >=3 ticks |
|---:|---:|---:|---:|
| 100 | 79k (170 trades) | 57k | 3k |
| 300 | 245k (169) | 176k | 31k |
| 900 | 936k (222) | 633k | 148k |

Hindsight in the real simulator (`scripts/make_lever_oracle.py` writes `strategies/round3/oracle_lever_b2_*.py`,
hmm_imm_b2 plus the schedule; lever lots are tracked apart, so b2 quotes as before):

| Variant | PnL | Penalty | vs b2 728,501 |
|---|---:|---:|---:|
| all 222 hindsight excursions, Q 900 | 1,183,157 | 888,730 | **+455k** |
| top 35 (tape net >= 5k) | 1,103,251 | 233,363 | +375k |
| top 9 (tape net >= 20k) | 931,908 | 74,589 | +203k |

So perfect foresight is worth about +455k (+62%). Latency and our own impact eat about half of the tape bound. The
value is concentrated: the top 10 excursions carry 50% of it and the top 30 carry 68%. Those are the big jumps.

**Without hindsight, the expected net is negative everywhere.** We entered at the first point of each "mid moved >= X
ticks in 20 ms" episode, as seen 5 ms before entry, and held for a fixed time:

| Q = 900 | episodes | 2 ms | 20 ms | 100 ms | 200 ms | 1 s |
|---|---:|---:|---:|---:|---:|---:|
| X = 1 | 9,844 | -5.8k | -5.7k | -5.8k | -6.2k | -13.3k |
| X = 3 | 754 | -16.6k | -15.8k | -14.5k | -14.2k | -19.9k |
| X = 8 | 248 | -29.9k | -27.9k | -25.7k | -26.0k | -29.3k |

Q = 100 and Q = 300 are negative in every cell too. The reason: once a move is visible, the spread is wide and the
book is thin, so a 900-lot round trip costs 6-30k. The continuation after detection is only +2-4k. Holding longer
only adds penalty. Jumps arrive close to Poisson (memoryless), so they cannot be timed before they show.

## Rules for future strategies

1. A leverage idea must predict the move **before** it shows in the book, or enter **passively** (our quotes filled
   at stale prices). IOC chasing after detection is ruled out by the table above, for any trigger and hold.
2. Upper bound to beat: about +455k per 1.67 h with perfect foresight, about +200k from the 9 biggest jumps alone.
   A real signal recovers a fraction of that minus false positives. Each false 900-lot round trip costs 6-30k.
3. To gate leverage online, use `tools/breach_gate.py` (`BreachGate`). It paper-trades each candidate excursion on
   the live book and allows a signal bucket only once its mean net after the penalty is significantly positive. It
   would never fire on round 3, but it adapts if a later round's parameters make leverage pay.
4. Earlier failures agree: persistent caps 101-1000 and reactive overfill on round 1 (`experiments/inventory/`),
   and jump sniping, sweep catchers and jump-mode caps on round 3 (`docs/R3_IMM_KNOWLEDGE.md`).

Reproduce:

```bash
.venv/bin/python scripts/lever_oracle.py data/<tape>.csv --dump      # bounds + logs/lever_schedule_<tape>_q*.json
.venv/bin/python scripts/make_lever_oracle.py logs/lever_schedule_<tape>_q900.json 900 [min_net]
.venv/bin/exchange-simulator --strategy strategies/round3/oracle_lever_b2_q900.py --backtest data/<tape>.csv --log-dir logs/<run> --no-logs
```

The oracle strategies read the future: never submit them.
