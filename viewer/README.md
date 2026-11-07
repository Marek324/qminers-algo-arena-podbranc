# Raw market-data viewer

Streamlit app to browse a market-data CSV (`Time,Operation,OrderId,Side,Volume,Price,Lifespan`) over time.

```bash
uv run --with streamlit streamlit run viewer/app.py
```

Streamlit is added only through `--with`; project dependencies are unchanged. Pick a file from `data/*.csv` or upload one in the sidebar.

## Controls
- **Transport bar:** play/pause, previous/next trade, step back/forward (step size 1 / 10 / 100 / 1k events), play speed (simulated seconds per real second), and the time slider. The big readout is the cursor time and event number.
- **Window:** how much time the main chart shows (10 s ... All). The window trails the cursor (85 % past, 15 % future, the future is greyed). Candle size follows the window; override it under Advanced.
- **Main chart:** candlesticks of the mid price (or of trade prices), optional best bid/ask lines, trade markers (triangle up = buy aggressor, down = sell; filled = IOC, hollow = limit) and traded volume per candle (buys up, sells down). Scroll to zoom, drag to pan or box-zoom, double-click to reset. Zoom is kept while stepping or playing and resets when the window changes.
- **Navigator:** the whole session's mid price; click a point to move the cursor there.
- **Right panel:** best bid/ask, spread, last trade, and the order book ladder at the cursor.
- **Tabs:** recent trades, the raw event log around the cursor (with the trades each row caused), depth chart, order-flow breakdown (Insert/Cancel, limit/IOC, buy/sell) and session totals.
- **Advanced (sidebar):** candle size, ladder levels, log rows, go to event #, show order ids.

## How trades are derived
Trades are not in the CSV. `viewer/engine.py` replays the tape with the simulator's rules (price-time priority, fills at the resting order's price, limit residual rests, IOC residual dropped, unknown cancels ignored). The replay is cached and keeps a full-book checkpoint every few thousand events, so the book at any event is rebuilt quickly; a 3M-row file loads in a few seconds.
