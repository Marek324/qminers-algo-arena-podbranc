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

## Strategy replay
Pick a file from `strategies/` in the sidebar (**Strategy**). The dashboard runs it through the real simulator with the scoring settings
(5 ms latency, |inventory| 100 penalty, 1000-lot order limit; the score matches `exchange-simulator` exactly) and replays the market
**with your orders in it**: the order book is the true one, your orders are marked in their queues (● in the ladder, "YOU" when expanded),
and the market reacts mechanically to what you take and post. The run takes about as long as a backtest and is cached per strategy file,
CSV and parameter set.

- **Cards:** your inventory, resting lots, PnL now and fills so far.
- **Chart:** your fills (filled circles; hollow and faded = not happened yet).
- **‹ My action / My action ›, ‹ My fill / My fill ›:** jump between the rows where the strategy acted or got filled.
- **What it is thinking:** every number/bool/short value on the strategy object at that moment, plus anything it puts in a
  `self.debug` dict (e.g. `self.debug = {'forecast': f, 'reason': 'skew bid'}`); ▲▼ = changed since the previous capture. State is captured
  whenever the strategy acts and otherwise every 50 ms of market time.
- **What it did:** its sends, cancels and shrinks, and the exchange's replies (accepted, fills, cancels, rejects), newest first.
- **Plot strategy values:** inventory, PnL and any of its values over the chart window. The **Strategy run** tab has the score card
  (`result.json`, with the share of the theoretical max when the analysis has been run).

### Tuning parameters
**Parameters** in the sidebar lists the strategy's UPPERCASE number/bool constants, both module-level (`NAME = 1`) and inside the class
(`    NAME = 1`, marked "(class)"), with their trailing `# comment` as help. Change values and press **Run**: a fresh copy of the module is
loaded with the new values in memory, so the strategy file is never written. Changed values are listed as code to paste back into the
file by hand; **Reset** returns to the file's values. A constant computed from another constant when the file loads keeps its file value.

## Analysis tab
Opens with a short overview: six key numbers, four verdict cards (market making, direction, regime, risks; green / orange / red), the session chart, and three small charts (maker markout, top predictors of the 5 s move, fill probability by order placement). Below it a switcher opens one detailed section at a time: **Market & book**, **Flow & execution**, **Signals** (book signals and technical indicators), **PnL bounds** (theoretical max and min) and **Data** (series explorer, raw report and downloads). Only the chosen section is drawn, so the page stays fast.

The report comes from `logs/analysis/<file>.json` when that is newer than the CSV (written by `uv run analysis.py data/<file>.csv`); otherwise press **Run analysis**, which computes it (about 12 s per million rows) and saves it there. `docs/ANALYSIS.md` explains every number.
