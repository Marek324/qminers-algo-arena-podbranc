# Market dashboard

A Streamlit dashboard for browsing a raw market-data CSV (`Time,Operation,OrderId,Side,Volume,Price,Lifespan`) over time.

```bash
uv run --with streamlit streamlit run dashboard/app.py
```

Streamlit comes in only through `--with`, so the project dependencies are unchanged. It opens `data/sample_v2.csv` (unzip `data/sample_v2.csv.gz` first) when it is there; pick another file from `data/*.csv` or upload one in the sidebar. The **How to read this page** button in the top right explains everything on the page in plain words.

## Moving through time
- **Play / Pause** runs the market forward. **Play speed** sets how fast (real time up to 300x).
- **Time slider:** drag it to any moment. It shows clock time (`mm:ss`).
- **Buttons:** Start / End, −1 min, −10 s, −1 s, +1 s, +10 s, +1 min, the previous or next trade (‹ Trade / Trade ›), and a single CSV row (‹ 1 event / 1 event ›).
- **Jump to time:** type `12:30` (or `1:05:00`) and press Enter.
- **Whole-session strip** under the chart: drag across it to make the main chart show that stretch, or click to jump there. **Back to presets** returns to the window buttons.

## What you see
- **Cards:** price now (mid), best bid, best ask, spread, last trade, and how much has traded so far.
- **Main chart:** candles (mid price by default, trade prices under *Chart options*) with traded volume per candle underneath. Green means the price rose or buyers pushed; red means it fell or sellers pushed. The dashed line is now, and the grey area hasn't happened yet.
  - *Chart shows* sets how much time is visible: 10 s up to the whole session.
  - *One candle =* sets how much time each candle and volume bar covers: Auto, or 100 ms up to 5 min.
  - **Mouse:** drag (left or middle button) to move, scroll to zoom, double-click to reset. Hover anything for details.
  - *Chart options* can add best bid/ask lines and individual trade arrows (off by default).
- **Order book now:** sell orders on top, buy orders below, the gap (spread) in the middle, and a bar for how many lots wait at each price.
- **Tabs:** recent trades, the event log (CSV rows around now in plain words, with the trades each row caused), order book depth, order types, and totals.
- **Advanced (sidebar):** order book rows, event log rows, go to CSV row #, show order ids.

## How trades are derived
Trades are not in the CSV. `dashboard/engine.py` replays the tape with the simulator's rules:
- price-time priority
- fills at the resting order's price
- a limit order's remainder rests, an IOC order's remainder is dropped
- cancels of unknown orders are ignored

The replay is cached and keeps a full-book checkpoint every few thousand events, so the book at any moment is rebuilt quickly. A 3M-row file loads in a few seconds.

## Analysis tab
The full statistical report from the `analysis/` package for the selected file: spread and depth, volatility across horizons, price impact, maker markouts, order lifecycles, signal correlations and deciles, per-bin regimes, a depth heatmap, a column explorer for the 1 s series, technical indicators (EMA, SMA, MACD, RSI, Bollinger, ATR, stochastic, ADX, VWAP, ...) with their predictive power, and the theoretical max and min PnL. It loads `logs/analysis/<file>.json` when that is newer than the CSV (written by `uv run analysis.py data/<file>.csv`); otherwise press **Run analysis**, which computes it (about 12 s per million rows) and saves it there. `docs/ANALYSIS.md` explains every number.
