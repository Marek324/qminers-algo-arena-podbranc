# Strategy notes

What winning teams did in IMC Prosperity, which of those ideas work with what we have here, and what we built and measured on our own data.

Our tools: **one instrument** and two order types:

- **LIMIT**: rests in the book. We are the maker and earn the spread.
- **IOC**: fills what it can right away and the rest is cancelled. We are the taker and pay the spread. It also stands in for a market order.

No fees, 5 ms latency, `|inventory| <= 100` (each second outside costs $10,000). See [MARKET.md](MARKET.md).

## 1. IMC Prosperity: what the top teams did

IMC Prosperity is IMC's yearly student trading competition. Bots trade simulated products for several rounds. Write-ups from top teams:

| Team / source | Placing |
|---|---|
| [Frankfurt Hedgehogs](https://github.com/TimoDiehm/imc-prosperity-3) | 2nd globally, Prosperity 3 (2025) |
| [Linear Utility](https://github.com/ericcccsliu/imc-prosperity-2) | 2nd globally, Prosperity 2 (2024) |
| [Alpha Animals](https://github.com/CarterT27/imc-prosperity-3) | 9th globally, Prosperity 3 |
| [Deanos Crew](https://github.com/JamesCole809/Deanos-Crew-IMC-Prosperity) | Prosperity 3 |
| [island of spacepirate](https://kinshu0.github.io/jekyll/update/2025/07/06/prosperity-3.html) | Prosperity 3 |
| [IMC UT team report](https://bhavyakansal1.github.io/IMC-UT-Prosperity-Team-Report/) | Prosperity 4 |

### What the IMC data looked like

- Several products per round, each with its own order book, plus the trades each step.
- Synthetic bots: one product had a fixed fair value (Amethysts / Rainforest Resin at 10,000), another a slow random walk (Starfruit / Kelp).
- Rounds also added baskets with their parts, options on an underlying, and products that could be converted on an outside market.
- Trader IDs were shown in the final round only.
- The book was rebuilt every step, so queue position never mattered. **That is not true here.**

### Strategies, and whether they apply to us

| # | Strategy | Who used it | Product and data | Orders used | Works here? |
|---|---|---|---|---|---|
| 1 | **Take mispriced orders.** Buy every ask below fair, sell every bid above it. | Both 2nd-place teams | Amethysts / Resin (fair fixed at 10,000), Starfruit / Kelp | IOC | **Yes**, if we have a good fair price |
| 2 | **Market making around fair.** Quote just inside the best bid and ask while keeping positive edge. Edge found by grid search. | Both 2nd-place teams | Same | LIMIT | **Yes.** This is our main strategy on v2 |
| 3 | **Fair price from the big orders.** Use the mid of the large, steady orders ("wall mid") or of the main market-maker bot, not the noisy best-price mid. | Frankfurt Hedgehogs (wall mid on Kelp). Linear Utility (the bot quoting 20–30 lots on Starfruit; they confirmed the scorer used that price). | Order book levels and sizes | none (it's a signal) | **Yes.** We see every order and its size. Our version is the size imbalance at the best bid and ask. |
| 4 | **Get back to flat at zero edge.** Trade at exactly fair to reduce inventory and free room under the position limit. | Linear Utility (about +3% PnL) | Amethysts | LIMIT or IOC | **Yes.** It matters more here: ±100 limit, and leftover inventory is valued by selling into the book at the end |
| 5 | **Follow the informed trader.** Copy a bot that bought at the daily low and sold at the daily high. | Frankfurt Hedgehogs (Squid Ink, "Olivia") | Trades, later trader IDs | IOC | **Partly.** There are no trader IDs, but each tape row shows whether an order is IOC or resting and how big it is. On the first sample, IOCs predicted a 10-tick move (section 2). On v2 they predict almost nothing. IOC *sides* are coin flips, but IOC *prices* follow the hidden fair price (see DATA_ANALYSIS.md). |
| 6 | **Mean reversion with a z-score or EMA.** Trade back toward a moving average at fixed thresholds. | Several teams (Squid Ink, Volcanic Rock) | Mid price history | IOC | **Weak.** Mixed results even at IMC. On v2 the price is a random walk with a slight trend, so not worth it |
| 7 | Basket vs parts arbitrage | Both 2nd-place teams (Picnic / Gift baskets) | Several related products | IOC | **No.** One instrument |
| 8 | Options: volatility trading and delta hedging | Both 2nd-place teams | Vouchers / coupons on an underlying | IOC + hedge | **No.** No options |
| 9 | Conversion arbitrage (Orchids, Macarons) | Both 2nd-place teams | Outside market prices and fees | LIMIT + conversion | **No.** No outside market |
| 10 | Predict prices from last year's data, then plan trades with dynamic programming | Linear Utility (Round 5, about 2.1M) | Previous year's prices | IOC | **No.** No related data |

The parts that carry over: **a good fair price (3), taking with IOC (1), making with LIMIT (2), and getting back to flat (4)**. A flow signal (5) is worth adding only if the data shows that aggressive orders predict the next move.

## 2. Our data

Analysis scripts: `scripts/record_book.py` (replays a tape through the engine and saves the best bid/ask and trades) and `scripts/sweep_class.py` (parameter grid search).

### `data/sample.csv` (synthetic, 1 h)

- One market order per side, constant 2-tick spread, requotes every ~0.37 s or 1 s.
- About 0.17 IOCs per second. After each one, the mid walks **about 10 ticks in the IOC's direction over 3–5 s**, then gives back about 3 ticks over the next 15 s.
- Best strategy: **follow the flow** (row 5 above). `strategies/flow_momentum.py` makes **290,944**.
- The market-making strategy loses 10,518 here.

### `data/sample_v2.csv` (about 96 min, 2M rows)

- Deep book: about 800 lots at the best price, about 2,000 within 5 ticks. Orders live about 0.5 s (median).
- Spread is 1 tick 41% of the time and 2 ticks 58%.
- About 6 IOCs per second, about 650 lots per second, with buys and sells balanced.
- The **mid barely moves after aggressive orders**: +0.02 ticks after 5 s. The flow is uninformed.
- Our resting fills earned **0.84 ticks per lot** against the mid, and that edge didn't decay over 60 s. So profit is limited only by how many lots we get filled.
- The size imbalance at the best bid and ask predicts the next 0.5 s move (correlation 0.45, about ±0.3 ticks at the extremes). That's too small to pay the spread, but useful for skewing quotes.
- The price is a random walk with a slight trend (variance ratio 1.1–1.2).
- Best strategy: **market making** (rows 2, 3 and 4 above). `strategies/imbalance_mm.py` makes **276,307**.
- `flow_momentum.py` loses 3,049,276 here.

### `data/round1.csv` (first scored round, 200 min, 4.4M rows)

- Like v2 but tighter: spread is 1 tick 93% of the time. About 900 lots at the best price; orders live about 0.34 s (median).
- About 8 IOCs per second, about 1,290 lots per second, balanced.
- **Big aggressors are informed, small ones are not.** Resting fills, 5 s later, by aggressor size: under 18 lots +0.44, 18–42 +0.37, 42–85 +0.35, 85–192 +0.26, over 192 lots −0.06 ticks per lot. The average is only +0.06, which made the first diagnostic say "UNCLEAR".
- A big aggressive order keeps moving the price: about +0.2 ticks within 5 ms, +0.3 by 100 ms and +0.6 by 1 s for orders over 192 lots.
- Fills when the spread is 2 ticks earn about +0.54, but quoting *inside* the spread lost money (as on v2).
- Signals: touch imbalance (Spearman 0.26 at 1 s) and aggressive flow over the last second (0.19) are the best; past returns and technical indicators are near 0.
- Ceiling from the analysis report (perfect hindsight): **1.39M**, almost all of it from resting orders (1.35M).
- The tape is synthetic (see [DATA_ANALYSIS.md](DATA_ANALYSIS.md)): limit orders are placed around a **hidden fair price S** that the mid lags by about 0.5 s. A running mean of new limit-order prices (each side averaged) estimates S. Offline, S estimate − mid explains 22% of the next 0.5 s mid move, against 11% for touch imbalance.
- Best strategy: **market making around the S estimate, crossing in when the signal is strong**. `imbalance_mm.py` makes **615,583**, about 44% of the ceiling.
- `adaptive_mm.py` makes 195,037; `flow_momentum.py` loses 5,142,910.
- More from the generator analysis: mid volatility is 0.92 ticks/√s, about 1.85× v2. IOCs are priced at S ± 1 tick, so 46% go one tick through the best price, against 22% on v2. Limit sizes are heavy-tailed (1 to 124k, median 14). 13% of cancels hit orders that were already filled; those are no-ops, so don't read cancels as queue thinning.

## 3. Our strategies

### `strategies/flow_momentum.py` (built for the first sample)

1. **Signal:** a trade between two market orders. Buy aggression means go long, sell aggression means go short.
2. **Entry:** IOC orders at the best price for 1.5 s, up to 95 lots, while the price is no more than 6 ticks past the signal.
3. **Exit:** a ladder of resting LIMIT orders 14, 20, 26, 32 and 38 ticks beyond the signal price.
4. **Time stop:** after 6 s, close what's left with IOCs.

Lessons:
- Each price level sees only one market order of about 25 lots as the price walks past, which limits both entry and exit size.
- Holding the full position to the top of the move beat selling early. Trailing stops, a stall exit and 1-tick ladders all did worse.

### `strategies/imbalance_mm.py` (the current choice)

1. **Fair price** = mid + `K_IMB` × imbalance + `K_FLOW` × flow / depth − `K_INV` × inventory. Flow is signed aggressive volume fading over `FLOW_TAU_US` (0.2 s); depth is the average size at the best price.
2. **Bid:** the highest price ≤ best bid with fair − price ≥ `EDGE`. Inside a 2-tick spread only if fair − price ≥ `EDGE_IN` (kept at 0.5, which in practice means never). The ask mirrors it.
3. **Keep queue position:** orders that end up `KEEP_LEVELS` behind the target price keep resting.
4. **Size:** each side rests as much as fits without inventory leaving ±`MAX_POS`.

How it improved:

| Version | PnL on v2 |
|---|---|
| First market maker | 113,197 |
| After tuning | 218,557 |
| Keep orders one tick behind the best price | 251,660 |
| Larger quote size | **276,307** |

- Nearby parameter values all land between 264k and 276k.
- PnL is 44–47k in every 16-minute slice.
- Inventory sits at ±100 most of the time. That's legal, since only above 100 is penalised. `MAX_POS = 95` costs about 8k.

Round 1 re-tune (current settings: `EDGE` 0.4, `K_FLOW` 0.5, `FLOW_TAU_US` 200,000, `K_IMB` 0.3, `K_INV` 0.002):

| Version | round 1 | round 1, 1st half | round 1, 2nd half | v2 | sample |
|---|---|---|---|---|---|
| v2 settings, no flow term | 347,799 | 167,407 | 180,392 | 276,307 | −10,518 |
| Add flow term (`K_FLOW` 0.25, 1 s) | 412,647 | | | | |
| Shorter flow memory (0.2 s), `EDGE` 0.4 | **468,499** | **228,527** | **240,057** | 251,092 | −2,488 |

- Sweeping each half separately ranks the settings in the same order (best on both: `EDGE` 0.35–0.4, `K_FLOW` 0.5), so the gain is not overfitting.
- `EDGE` 0.35 scored higher (up to 498k) but was uneven across `K_INV`; 0.4 sits in the stable part.
- Quoting inside the spread (`EDGE_IN` < 0.5) lost money at every setting.

Round 1, hidden-price version (current settings: `K_S` 0.75, `S_ALPHA` 0.1, `K_IMB` 0.15, `EDGE_TAKE` 0.25, `EDGE` 0.4, `K_FLOW` 0.5, `FLOW_TAU_US` 200,000). Tuned on the first half only, checked on the second:

| Version | 1st half (tuning) | 2nd half (held out) | full round 1 |
|---|---|---|---|
| Flow term only (previous) | 228,527 | 240,057 | 468,499 |
| + S estimate (`K_S` 0.25, `S_ALPHA` 0.25) | 247,198 | | |
| + cross in when fair − best ask ≥ 0.25 (`EDGE_TAKE`) | 273,168 | 271,564 | |
| Smoother S (`S_ALPHA` 0.1), `K_S` 0.75 | **309,298** | **306,213** | **615,583** |

- **S estimate:** new market orders are found by id (ids only grow), scanning the top 3 price levels of a side only when that side's snapshot list changed. Run time is about 90 s for round 1.
- **Pure taking doesn't pay:** even the strongest 5% of signals lose about 0.35 ticks per lot on an IOC round trip. Crossing in and letting the quotes unwind does pay.
- `EDGE_TAKE` 0.1 was unstable (one setting fell to 167k), so it stays at 0.25.
- `S_ALPHA` 0.08–0.12 with `K_S` 0.5–0.75 all score 293–309k per half; the two halves rank settings the same way.
- PnL is 74–83k in every 25-minute slice and never goes below 0.

## 4. Ideas not yet tried

- **Switch strategy by market type.** Measure how far the mid moves in the 2–3 s after IOCs. If it moves a lot, follow the flow; if not, market-make. That would handle both samples.
- **Better fair price.** Include deeper levels. (Recent aggressive flow is now in, as `K_FLOW`.)
- **Queue position.** Back-of-queue orders mostly fill when a big order empties the level, which is the bad case. Pulling orders that are deep in a long queue when the opposite side is building could avoid those fills.
- **Sharper S estimate.** Use a half-life of 100–200ms; the mid closes half its gap to S in about 200–260ms. Add IOCs that sweep past the best price as a second, faster reading of S, since IOCs are priced at S ± 1. See [DATA_ANALYSIS.md](DATA_ANALYSIS.md).
- **Get back to flat at zero edge** when inventory is pinned at ±100, so both sides can quote again.
- **End of round:** flatten near the end to avoid paying the spread on leftover inventory. We don't know the round length, so we would need a time guess or a rule that only uses the recent past.
