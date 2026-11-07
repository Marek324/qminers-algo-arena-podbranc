# Tape analysis (`analysis/`)

Replays a tape through the same matching rules as `simulator/simulator.py` (checked trade-for-trade against
the simulator's `trades.csv`) and writes one report per tape: JSON for the visualizer, indented text for reading.

```bash
uv run analysis.py data/<file>.csv [more.csv ...]          # -> logs/analysis/<file>.json + .txt
uv run analysis.py data/<file>.csv --format json --compact --series-step 0.5 --bin-step 300
```

In the dashboard (`uv run --with streamlit streamlit run dashboard/app.py`), the **Analysis** tab renders the report for the selected file, loading the saved JSON or running the analysis on request.

From Python: `from analysis import analyze; report = analyze(Path('data/x.csv'))` returns the same dict.

Speed: about 12 s per million rows, most of it the replay loop.

## Schema contract

- `meta.schema_version` (currently `1`) changes only when a key is renamed, removed or changes meaning. Adding keys
  does not bump it.
- Every key is always present, whatever the data. Missing or undefined values are `null`, never absent and
  never `NaN`.
- Lists of row objects (`insert_size_top`, `impact_by_size`, `join_fill_prob_by_queue_ahead`, `deciles`,
  `largest_taker_events`, `by_fill_size`) have fixed keys per row but can be shorter, or empty when there is
  too little data.
- Distribution objects (`dist`) always have `n, mean, std, min, p01, p05, p10, p25, p50, p75, p90, p95, p99, max`.
- Horizon keys are fixed strings: `0ms 5ms 100ms 1s 5s 10s 30s 60s 300s` (which ones depends on the section).

Units: prices in **ticks** (raw price / `tape.price.tick_size`), quantities in lots, times in seconds from the
first event unless the key ends in `_ms` / `_us`. Trade sign `+1` = buyer-initiated (ask lifted). "Time-weighted"
means each book state is weighted by how long it lasted.

Mid: when one side of the book is momentarily empty, time-sampled stats carry the last two-sided mid forward.
Spread and touch stats use only moments with both sides present.

## Sections

| Key | What it answers |
|---|---|
| `meta` | source, schema version, units, config (series/bin step, 100 ms grid, 5 ms latency), runtime per section |
| `highlights` | list of plain-language takeaways generated from the numbers below |
| `tape` | row counts, insert/cancel/IOC mix, buy/sell split, event timing (inter-arrival, events per second, events in the next 5 ms), insert sizes, price range and last-digit clustering, IOC fill rates, `anomalies` (cancels of filled/unknown orders, duplicate ids) |
| `orders` | outcome of every inserted order (filled / canceled / IOC remainder / still resting), marketable share, `passive_placement` = fill probability, cancel probability, lifetime and time to first fill by placement relative to the touch (`improve`, `join`, `behind_1`, `behind_2_5`, `behind_6_20`, `behind_gt20`, `side_empty`), lifetimes, cancel speed, where cancels happen, fill probability of touch-joiners by queue ahead |
| `book` | time with each side empty, spread distribution (time- and event-weighted, share by ticks), wide-spread episode length, touch quantity and order count, `depth_profile` (mean/median/cum quantity at k = 0..19 ticks from own best, 100 ms sampling), imbalance (touch, 5 and 20 levels), touch change rates and lifetimes, `latency_5ms` = how often the touch or mid moves while an order is in flight |
| `price` | mid first/last/range/TWAP, trade VWAP, drawdown/run-up, drift and its t-stat, `returns_ticks` per horizon (moments, quantiles, autocorrelation), `vol_ticks_per_sqrt_s` (signature plot), `variance_ratio_vs_1s` (<1 mean-reverting, >1 trending), Hurst estimate, 1 s jumps, event-time mid changes (size mix, autocorrelation, reversal probability, waiting time), Roll spread |
| `trades` | counts and volume per second, aggressor buy share, trade size and aggressive-order size, levels swept, IOC share of takers, aggressor sign autocorrelation and run lengths, maker age at fill, `impact_ticks` (signed mid move after an aggressive order per horizon) and `impact_by_size`, `kyle_lambda` (OLS of mid change on net aggressive volume per 10 s / 60 s), the 10 largest aggressive orders |
| `maker` | markout of every passive fill from the maker's side, per lot: `side * (mid(t+h) - fill_price)`. `0ms` uses the mid just before the aggressor, so it is the half-spread captured; later horizons subtract adverse selection. Split by at-touch vs through-touch, spread before the trade, taker type, maker side and fill size. Positive at 5 s and later means passive quoting at the touch has an edge before inventory costs |
| `max_pnl` | `theoretical_max` / `theoretical_min`: best and worst PnL any strategy could score knowing the future, under the simulator rules: 5 ms latency, price-time priority with queue position, \|inventory\| <= 100, flat at the end, own impact on later market behaviour ignored. One DP over inventory combines taking (best bid/ask at every state an order can land on, up to the size there) and making (for every market insert that crosses or improves the book: a new price level inside the spread or in a swept gap, where you are first; behind the queue at an existing level, joined right after that level last traded; or beyond the book at the insert's limit). `making_only` / `taking_only` restrict it to one way of trading. Each has `pnl`, `pnl_per_hour`, `inventory_before_markout`, `operations`. `opportunities` counts the make options; `curve` is the best (and worst) PnL achievable by each bin end if flat there. Model details in `analysis/sections/maxpnl.py` |
| `signals` | on the 100 ms grid (first 60 s skipped): for each directional feature (touch/depth imbalance, microprice offset, signed flow 1/10/60 s, past returns 1/10/60 s, last trade sign), Pearson/Spearman correlation and hit rate with the forward mid move at 100 ms/1/5/30 s, plus decile tables at 1 s and 5 s; activity features (spread, touch size, trade count, volume, absolute past return) against the absolute forward move; `linear_combination` = OLS of all directional features with in-sample fit and R² out of sample (fit on first half, test on second) |
| `indicators` | technical indicators on the mid, on the series grid (windows in seconds): EMA/SMA 10/30/60/300 s, MACD 12/26/9, RSI 14/60, Bollinger 60 s, z-score, ATR 14, Keltner 20, stochastic %K/%D 14, Williams %R, CCI 20, Donchian 60, ROC 10/60, ADX and ±DI 14, VWAP (session, 60 s), cumulative volume delta, realized vol 60 s. `evaluation`: each stationary form (distance from price, oscillator value) against the forward mid move at 1/5/30/60 s (Pearson, Spearman, hit rate) plus 5 s deciles; `ranking_by_abs_spearman_5s`; `series`: columnar table of every indicator for plotting |
| `regimes` | relationships across bins: volatility vs spread/volume/events, flow vs same and next bin return, return autocorrelation, volatility/volume/spread clustering, coefficients of variation |
| `bins` | columnar table, one row per `bin_step` (default 60 s): mid OHLC, return, realized vol (from 1 s changes), mean spread, touch sizes, imbalance, events/inserts/cancels/IOC, trades, volume, buy/sell volume, net flow, cancel/insert ratio, `bid_depth_mean`/`ask_depth_mean` (20 values per row, for a depth heatmap) |
| `series` | columnar table, one row per `series_step` (default 1 s): book at the end of the interval (best bid/ask, mid, spread, touch sizes, imbalance, last trade price) and flows during it (events, inserts, cancels, IOC, trades, volume, buy/sell volume, net flow) |

Columnar tables look like `{"step_s": 1.0, "rows": N, "columns": [...], "data": {"t_s": [...], "mid": [...], ...}}`.
Row k covers `(t_s - step_s, t_s]`.

## Caveats

- Signals use overlapping samples, so correlations are descriptive. Don't read t-stats into them.
- The 60 s Kyle lambda and the 300 s returns need tapes of at least a few minutes. Shorter tapes give `null`.
- `data/sample.csv` is synthetic: one quote pair re-posted every second plus IOCs. Most book stats on it are
  degenerate. Use the real round files.
