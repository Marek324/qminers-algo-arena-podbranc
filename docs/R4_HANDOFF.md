# Round 4 handoff (2026-11-08, about 08:30 UTC)

Read this first, then `AGENTS.md`. Details and numbers: `docs/DATA_ANALYSIS.md` (section "round4_public") and
`docs/RESEARCH.md` (all "Round 4" sections), plus `docs/LEVERAGE.md` (round 4 section).

## Situation

- **Round 4 is the final round.** Submissions close **11:00 UTC 2026-11-08**; the round ends 12:00. Overall score
  = 0.8·z1 + z2 + z3 + 1.2·z4. We are 4th overall, 0.54 behind Codehorses (3rd). Goal: the **risk track** (aim for a big
  gain), with a safe fallback.
- **Submission budget:** 10 per round; the team agreed **2 per person + 2 reserve for the end**. Don't spend
  submissions on noise-level variants.
- **Arena so far (round 4):**
  - `r4m_risk_a.py` (Marek): validation **1,965,048** (penalty 12,138).
  - Jaro's `r4_best_snipe`: validation **1,963,678** (penalty 11,261), **currently marked final**. Code is in
    `data/validation_round4/0736_1e3df29e/strategy.py`.
  - Jaro has a second submission from 07:55; check it with the arena MCP (`arena_submissions`).
  - Validation SE ≈ 60k, so these two are tied. The round-4 leaderboard is hidden until the round ends.

## The current best design (all three bases land at 1.14–1.16M on round4_public)

`strategies/round4/r4m_risk_a.py` = `hmm_deep_clear_r3` (2-state HMM regime + Kalman hidden price, quiet market making
/ storm stale-book taking) with:
- round-4 constants K_S 0.95, SIG_Q 0.8, V_K_S 0.9;
- momentum lean K_MOM = V_K_MOM = 0.1 on the 0.5 s mid move;
- **jump sniper past the 100 cap** triggered by `IdGapJump` (missing order ids + clamped orders: big jumps land outside
  the 20-tick public depth) plus a far-order trigger (SNIPE_FAR 15, 2 confirmations); IOC up to 1000 position, unwind
  at S_new−2 or after 0.5 s; kill switches.
- round4_public 1,156,545, penalty 7,963; quarters 317k / 290k / 297k / 250k; robustness 6/6 chunks, mirrored equal,
  bootstrap 8/8.

Other files in `strategies/round4/` (all `r4m_*` are Marek's): `r4m_risk_b` (no momentum, 1,124,599), `r4m_safe` (no
leverage, 1,073,369), `r4m_imm_snipe` (Jakub's IMM base + same sniper, 1,143,398), and switch-files used for sweeps:
`r4m_alt`, `r4m_dev`, `r4m_dev2`, `r4m_dev3`, `r4m_rls`, `r4m_pass`, `r4m_comp` (with Jaro's quote hysteresis),
`r4m_push`, `r4m_aggr` (trade-through bounds TB_* and pre-storm lean PRE_*; its sweep was running at handoff,
results in `logs/sweep_aggr_q.log` on .225 and `logs/sweep_aggr_f.log` on .171).

## What the round-4 data says (mechanisms)

- **New in round 4: IOC side follows the last 1 s move** (P(buy) 0.56 after up, 0.45 after down; it was a coin flip in
  rounds 1–2). This makes the price trend at 1–10 s (variance ratio 2.0 at 10 s), roughly doubles impact, makes resting
  quotes lose −2.8 ticks/lot at 5 s, and breaks the generator fit (`estimate_model.py` error 89; the twin inverts
  strategy rankings, so do not use synthetic twins for round 4).
- Order placement (the HMM observation) is unchanged from round 3; the online HMM agrees 97.6% with an offline
  Baum-Welch fit (`scripts/regime_baum_welch.py`).
- Money: storms are 38% of the time and 81% of PnL. Storm takes earn +2.96 ticks/lot at 5 s; passive fills earn the
  spread at the fill and give it back by 30 s, but removing them loses 8% (turnover).
- **Theoretical max (±100, hindsight) = 2.51M** on the public tape (about 5M on the 12,000 s tapes); we take 46%.
  "Making only" is 97% of that max: it is perfect passive-fill selection, not reachable without knowing the next move.
  Our takes are at about 80% of the taking-only max.
- **Leverage:** perfect-foresight excursions are worth about +0.9M per 6,000 s in the simulator; our sniper gets about
  +51k (5%). Every trigger fires after the jump; calm stretches are memoryless; reactive leverage (after the mid moves)
  has negative expected value at every threshold and hold. `IdGapJump` catches 4 of the 15 jumps ≥ 40 ticks; misses show
  only 1–2 missing ids and unreliable taker-flow direction.
- Stale liquidity at big jumps: 662k of edge per tape, we take 8%; first market taker hits it ~8 ms after the jump.

## Rejected on round 4 (do not retry without a new reason)

ML take rule and HGB storm gate; lower take thresholds; taker-only / reduce-only / deeper quoting; slow reversal lean;
momentum threshold / regime split / bigger momentum; RLS learned fair value; queue-aware quoting; convex inventory skew;
adverse-selection-adapted edge; far-order sniper alone; sweep and depth-collapse triggers; post-jump lean;
clamp-confirmed id gaps; edge leverage on S_hat; re-firing snipes; passive exit of the excess; combined logistic jump
predictor (loses out of sample); IOC push into S_hat (duplicates the flow term); 19 other strategy families
(best non-HMM: `alpha_as_mm` 944k; pure market makers and momentum followers lose 0.5–2.8M); 27 component variants
(only Jaro's hysteresis is consistently positive, +0.3%); position-age exit / storm flip hysteresis; storm IOC takes;
weak jump tier inside the cap (+5k, noise; `r4m_tier`); OFI and depth-imbalance fair terms (storm depth imbalance
+27k but first half of the tape only; `r4m_book`). Numbers for each: `docs/RESEARCH.md`.

## How to work here

- **Never run backtests on the laptop.** Two servers:
  - `.225`: `ssh ubuntu@185.175.110.225`, repo `~/qminers-algo-arena-podbranc`, 16 cores, no GitHub. Sync: on the laptop
    `git push ssh://ubuntu@185.175.110.225/home/ubuntu/qminers-algo-arena-podbranc main:refs/heads/incoming`, then
    `git merge --ff-only incoming` there (delete untracked copies that would conflict first).
  - `.171`: `ssh ubuntu@185.175.110.171`, repo `~/.ssh/qminers-algo-arena-podbranc`, 8 cores; `git pull --rebase
    --autostash`. Streamlit dashboard on :8501.
  - Launch detached: `timeout 15 ssh -n HOST 'cd REPO && (nohup setsid CMD > log 2>&1 < /dev/null &)'`.
  - **Never `pgrep -f` a pattern that also appears in your own ssh command line** (the watcher matches itself and
    never exits).
- Tapes on both servers: `data/round4_train.csv` and quarters `data/q/r4_q1..q4.csv` (1,500 s each).
- Sweeps: `.venv/bin/python scripts/variant_sweep.py STRAT.py GRID.txt TAPES... --out logs/X -j N`; grid lines are
  `name: CONST=value, CONST=value` (empty = file as is).
- **Acceptance rule (against overfitting):** a change must be up in at least 3 of 4 quarters, not carried by one jump,
  with neighbouring settings agreeing. Expect ±1% noise from any nudge and ±25% PnL swings between quarters from the
  market alone. Judge by consistency, not totals.
- Useful scripts: `pnl_attribution.py`, `missed_opps.py`, `jump_value.py`, `lever_oracle.py`,
  `robustness.py`, `take_candidates.py`, `jump_feats.py` / `jump_eval.py`, `regime_baum_welch.py`. The backtest explorer
  for `r4m_risk_a` runs on .225 port 8051 (`ssh -L 8051:localhost:8051 ubuntu@185.175.110.225`).
- Arena: read-only MCP server `arena` (`arena_submissions`, `arena_leaderboard`, `arena_download`, ...). Submitting and
  marking final are done by a human.
- Commits: `type(scope): summary`, short, no AI attribution; strategies of this session are named `r4m_*`.
