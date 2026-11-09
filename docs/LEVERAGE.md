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

Hindsight in the real simulator (an oracle strategy, hmm_imm_b2 plus the schedule; its generator and base strategy are
in git history only; lever lots are tracked apart, so b2 quotes as before):

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
```

The oracle strategies read the future: never submit them.

## Round 4: an early trigger makes it pay (2026-11-08)

round4_public, `scripts/lever_oracle.py`: hindsight 128k / 422k / **1,878k** for Q = 100 / 300 / 900 (round 3: 79k /
245k / 936k), 349 excursions, the top 10 carry 38%. They are big discrete jumps (−140 ticks at 170 s, +130 at 1740 s).

**The 20-tick public depth hides the big jumps.** After a 130-tick jump the first new orders at the new level sit 130
ticks from the best price, outside the snapshot, so a "new order far behind the touch" trigger (`r3_fire`,
`r4m_snipe`) only sees jumps under about 20 ticks and fired on neither of the two biggest. `IdGapJump`
(`tools/id_gap_jump.py`) sees them: invisible orders leave gaps in the order ids, and the orders the generator
clamps at the stale touch give the direction. It fires 5–25 ms into a jump, before the stale side is gone.

Entering on it, as an IOC up to 900 extra lots of the stale side at most `IDG_DEPTH` ticks through the old touch, then
unwinding once the book reaches the new level (or through the touch after 0.5 s), in the real simulator
(`strategies/round4/r4m_risk_b.py`, `r4m_risk_a.py`):

| | round4_public | penalty |
|---|---|---|
| no leverage (`r4m_safe`) | 1,073,369 | 0 |
| far-order trigger, SNIPE_FAR 15 | 1,082,973 | 3,786 |
| id-gap trigger | 1,109,933 | 13,270 |
| both (`r4m_risk_b`) | 1,124,599 | 11,929 |
| both + momentum lean (`r4m_risk_a`) | 1,156,545 | 7,963 |

Per quarter the leverage adds +30k / +23k / +2k / 0: it never loses, it pays when a big jump comes. Filled trades
earn +28 to +107 ticks/lot (1394 s: 505 lots in at 9688, 302 out at 9795). Misses: when the stale side is gone within
our 5 ms (1740 s: 4 lots). So rule 1 above holds: the trigger has to beat the move into the book; the id-gap detector
does, the mid and visible far orders don't. On round 3 the same detector added only +0.7k as a filter boost; check
it as a sniper trigger on any new tape.
