# Qminers trading hackathon starter

```
strategies/   template_strategy.py, an empty strategy to copy and fill in; trend_follower.py, a simple working one
data/         sample.csv.gz, a synthetic hour of trending prices to start on; the rounds downloaded from the website go here too (git-ignored)
logs/         simulator output (git-ignored)
simulator/    the exchange simulator package
explorer/     the backtest explorer package
```

## Setup

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager; install it with `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `pip install uv`).

```bash
uv sync                                        # Python 3.13 and everything else, into .venv
uv run python -m gzip -d data/sample.csv.gz    # the simulator reads plain CSV only; one file at a time, the same for each downloaded round
```

`data/sample.csv.gz` is synthetic and its price trends: one resting order per side, and every few seconds an aggressive order takes part of the touch, after which the quotes keep moving in that direction tick by tick for a few seconds and then give part of the move back. `strategies/trend_follower.py` is a simple trend-following strategy: it sends a one-lot IOC order at the touch in the direction of the trades it has just seen. The rounds are recorded market data and look nothing like the sample.

## Writing a strategy

A strategy is one `.py` file defining one subclass of `BaseStrategy` (with several, the alphabetically first is used), constructible with no arguments:

```python
from simulator.models import OrderBookSnapshot, OrderRequest, OrderType, PrivateOrderEvent, Side, Trade
from simulator.strategy import BaseStrategy


class MyStrategy(BaseStrategy):
    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        """After every market event: the public book and all trades since the previous call."""

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        """Every state change of your own orders."""
```

| Call                                                   | Effect                                                         |
| ------------------------------------------------------ | -------------------------------------------------------------- |
| `self.send_order(OrderRequest(side, price, quantity, order_type, client_order_id=None))` | submit; returns the order id (`u000000001`, ...) immediately |
| `self.cancel_order(order_id)`                          | cancel your resting order                                      |
| `self.modify_order(order_id, new_quantity)`            | shrink your resting order                                      |
| `self.now_us()`                                        | simulation time in µs                                          |

`Side.BUY` / `Side.SELL`, `OrderType.LIMIT` / `OrderType.IOC`; prices and quantities are positive integers (prices in ticks).

Submissions run on Python 3.13 with numpy 2.5.4, pandas 3.0.6, scipy 1.18.1 and scikit-learn 1.9.1 available, and are statically checked first: besides the standard library and those packages, import only `simulator.models` and `simulator.strategy.BaseStrategy`. Modules giving system, network, file, threading or introspection access (`os`, `sys`, `io`, `socket`, `subprocess`, `threading`, `importlib`, `inspect`, `gc`, ...), builtins like `open`/`eval`/`exec`/`getattr`/`type`, dunder attributes and `str.format` are rejected.

## Running a backtest

```bash
uv run exchange-simulator --strategy strategies/trend_follower.py --backtest data/sample.csv --log-dir logs/sample --progress
```

`logs/sample/result.json` holds the outcome (`pnl`, `cash`, `inventory`, `mark_price`, `breach_penalty`, `traded_volume`, `fills`, `pnl_se`, time-weighted quantiles of `|inventory|`) and `pnl.csv` the PnL sampled every second. Without `--no-logs` the directory also gets `run.jsonl`, `events.csv`, `book_deltas.csv` and `trades.csv`, several hundred MB per full round. Use `--no-logs` when you only need the result and the PnL curve; it is much faster. `--public-depth N` (default 20) sets how many ticks of the book you see.

## Exploring a backtest

```bash
uv run backtest-explorer --log-dir logs/sample     # http://127.0.0.1:8050, --port to change
```

A Dash app over a full-logs run (not `--no-logs`): best bid/ask with your fills marked, your PnL over time, the reconstructed order book at any clicked time, and the event log.

## How the exchange works

**One instrument, replayed.** Each CSV row is a market participant's action:

```
Time,Operation,OrderId,Side,Volume,Price,Lifespan
0.003823,Insert,1000001,B,43,9999,G
```

`Time` in seconds; `Operation` is `Insert` or `Cancel` (of `OrderId`); `Side` `B`/`A`; `Lifespan` `G` = resting limit order, `F` = immediate-or-cancel. The tape is fixed and does not react to you; your orders only add to and take from the book it builds.

**Matching.** Price-time priority. An incoming order (market or yours) matches against the opposite side while prices cross, each fill at the resting (maker) order's price; a LIMIT residual rests at the back of its price level, an IOC residual is cancelled. Market orders can trade with yours; your orders never trade with each other. No fees.

**Latency.** Every `send_order`, `cancel_order` and `modify_order` takes effect at $t + 5\text{ms}$, where $t$ is the simulation time at which you call it: it is applied just before the first market event with timestamp $\ge t + 5\text{ms}$, so the book may have moved by then. An order you place is therefore never filled within 5 ms, and a cancel can lose the race against a fill.

**Feeds.**
- `on_order_book_change` fires after every market event, with a snapshot of every individual order (id, side, price, quantity, `owner` `"market"`/`"user"`) within `public_depth` ticks of the best price on each side, and the list of trades (market-vs-market included) since the previous call. Fills caused by your own order activating appear in the next call's `trades`.
- `on_private_order_update` fires synchronously for your orders: `ACCEPTED`, then one `PARTIALLY_FILLED`/`FILLED` per match (`fill_price`, `filled_quantity`, `remaining_quantity`), `CANCELED` (your cancel, or an IOC residual), `MODIFIED`, or `REJECTED`. Cancelling anything but your own resting order, or a modify that doesn't shrink it, is `REJECTED`; a modify of an unknown id is silently ignored.

**Limits and scoring.** With inventory $I$ (signed), resting same-side quantity $R$ and order quantity $q$ ($s = \pm 1$ for buy/sell):

$$|I + s (R + q)| > 1000 \Rightarrow \text{order REJECTED at acceptance}$$

$$\text{penalty} = 10{,}000 \cdot \int \mathbb{1}\big[|I(t)| > 100\big] dt \quad (\$ \text{ per second})$$

$$\text{PnL} = \text{cash} + \text{markout}(I) - \text{penalty}$$

$\text{markout}(I)$ is the cash from liquidating $I$ against the resting *market* orders on the opposite side, best price outwards (volume beyond the book at the last swept price, or at the last traded price if that side is empty). It is computed at the end and every second for `pnl.csv` and `pnl_se`.
