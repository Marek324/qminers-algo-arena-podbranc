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

## Setup (once per machine)

1. On the logged-in site, open the DevTools console (F12) and run `localStorage.getItem('qminers.accessToken')`.
2. `export ARENA_TOKEN='<value>'` in the shell that starts Claude Code, e.g. in the shell profile. Never write the
   token into a file in the repo or paste it into a commit, doc or chat.
3. Restart Claude Code in the repo and approve the `arena` server.

The token lasts about a week. A 401 error means it expired: copy a fresh one. If the MCP tools are not loaded, the
same functions can be called directly:
`ARENA_TOKEN=... uv run --with mcp python -c "import sys; sys.path.insert(0, 'scripts'); import arena_mcp as A; print(A.arena_leaderboard('Round 1'))"`.

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
  PnL were 0. `arena_leaderboard` computes these z-scores the same way.
- **What follows for us:** a zero or negative round is catastrophic (round 1: z = −3.16), so reliability beats a few
  percent of PnL. Later rounds weigh more. Tuning to the public tape buys nothing, because scoring uses the private one.

## Round 1 for reference

Our final was `strategies/adaptive_mm_plus.py`: 678,057 public, 667,357 private, 3rd of 61 (z +1.69). Leader
GammaStrike at 792,696 (z +2.60). Mean of the 60 positive teams 434,481, sd 137,534.
