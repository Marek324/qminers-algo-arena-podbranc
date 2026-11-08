---
name: arena
description: Read the Algo Trade Arena site (arena.qminers.com) without clicking it: leaderboards with final-score z-scores, rounds and deadlines, our submissions with public and private results. Use when asked about the leaderboard, our rank or score, how far behind the leader we are, round dates or deadlines, whether a submission finished, what our final submission scored (public or private), or anything else shown on the arena site. Read-only: never submits or marks a submission final.
---

# Arena site (read-only)

The `arena` MCP server (`scripts/arena_mcp.py`, registered in `.mcp.json`) reads the arena's API with our login token.
It can only send GET requests to a fixed allow-list, so it cannot submit a strategy or mark one final. Those stay
manual actions on the site, done by a human (AGENTS.md).

## Tools

| Tool | Use it for |
|---|---|
| `arena_me` | which account and team the token belongs to |
| `arena_rounds` | all rounds: names, start, submission deadline, end, whether the leaderboard is published |
| `arena_current_round` | the active round, or "no round is currently active" |
| `arena_leaderboard(round_name, top)` | standings with each team's z-score; `round_name` is "Round 2", "2", a round id, or empty (the active round, else the latest published one) |
| `arena_submissions(limit)` | our submissions, newest first: score, metrics, PUBLIC / PRIVATE type, `isFinal` |
| `arena_submission(id)` | one submission in full |
| `arena_download(round_name, evaluation_id)` | save submissions locally: `strategy.py` + the arena's `result.json` / `pnl.csv` / `metadata.json` + `evaluation.json`, into `data/validation_<round>/<HHMM>_<id8>/` (the anchor layout `scripts/twin_check.py` reads); one id, or every submission of a round |

## Setup (once per machine)

1. On the logged-in site, open the DevTools console (F12) and run `localStorage.getItem('qminers.accessToken')`.
2. `export ARENA_TOKEN='<value>'` in the shell that starts Claude Code, e.g. in the shell profile. Never write the
   token into a file in the repo or paste it into a commit, doc or chat.
3. Restart Claude Code in the repo and approve the `arena` server.

The token lasts about a week. If the MCP tools are not loaded, the same functions can be called directly:
`uv run --with mcp python -c "import sys; sys.path.insert(0, 'scripts'); import arena_mcp as A; print(A.arena_me())"`.

## Troubleshooting

The MCP client hides the server's error and shows only "Error executing tool". To see the real error, run the
direct call above in a shell that has `ARENA_TOKEN` set.

- **`ARENA_TOKEN is not set`**: the variable was not exported in the shell that started Claude Code. Export it,
  then restart Claude Code (or reconnect `arena` in `/mcp`).
- **401**: the token expired. Copy a fresh one and reconnect.
- **400 Bad Request (an nginx HTML page)**: the auth header is malformed. The server already removes whitespace
  from a wrapped paste, so check that the value is the bare token (no `Bearer ` prefix, no extra quotes).
- **403 "leaderboard ... hasn't been published"**: normal for the active round. Pass a finished round, e.g.
  `round_name="Round 1"`.
- **Not using Claude Code** (Codex, Copilot, ...): `.mcp.json` is Claude Code's format. Register the same command
  (`uv run --with mcp python scripts/arena_mcp.py`, with `ARENA_TOKEN` in its environment) in that tool's MCP config,
  or use the direct call above.

## How to read the results

- **Public vs private.** A submission is backtested on the round's public tape (type `PUBLIC`, the same file as
  `data/roundN.csv`). After the deadline, the organisers re-run the final submission on a private tape and store it as
  a `PRIVATE` evaluation, whose `sourceEvaluationId` points to the public one. The published leaderboard and the final
  score use the private result.
- **Metrics** match the simulator's `result.json`: `fills`, `tradedVolume`, `inventory` (end position), `cash`,
  `markPrice`, `breachPenalty` (should be 0), `pnlSe` (noise of the score; gaps smaller than this mean little) and the
  |position| quantiles `absInventoryQ50/68/95/997`. Score = cash + markout of the end position − penalty.
- **The final ranking** combines the private PnL of all four rounds:
  `Score = 0.8·z₁ + 1.0·z₂ + 1.0·z₃ + 1.2·z₄`. zᵣ is the PnL standardized against the other teams in round r. Teams
  with PnL ≤ 0 (or no final submission) are left out of that round's mean and standard deviation, and score as if their
  PnL were 0. The sd is the **sample** sd (n−1, `statistics.stdev`), not the population sd: `pstdev` overstates every
  z by √(n/(n−1)), about 1% (GammaStrike's round-1 z came out 2.60 against the official 2.58). Round to 2 decimals
  only at the end; the total is computed from unrounded z's. `arena_leaderboard` computes these z-scores the same way.
- **What follows for us:** a zero or negative round is catastrophic (round 1: z = −3.13; round 2: −1.24), so reliability beats a few
  percent of PnL. Later rounds weigh more. Tuning to the public tape buys nothing, because scoring uses the private one.

## Round 1 for reference

Our final was `strategies/round1/adaptive_mm_plus.py`: 678,057 public, 667,357 private, 3rd of 61 (z +1.68). Leader
GammaStrike at 792,696 (z +2.58). Mean of the 60 positive teams 434,481, sample sd 138,697.

Overall after round 2 (official board): we are 6th with z₁ 1.68, z₂ 1.40, total 2.74. Leaders GammaStrike 3.51,
delfini 3.24, Codehorses 2.91, Olexa namichej 2.90, Southeast Fortress 2.85.
