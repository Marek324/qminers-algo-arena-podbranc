# How the market works

A summary of the "Algo Trade Arena — Getting Started" deck and the README. The source of truth is `simulator/simulator.py`. Read it, never edit it.

## Setup

- **One instrument**, anonymised: no name, no news, only its order flow.
- A **tape** (CSV) of other traders' orders is replayed through a matching engine. Your bot trades alongside it. Your **PnL is the score**.
- The tape is **fixed**: it never reacts to your orders. Your orders only add liquidity to, or take it from, the book the tape builds.
- `sample.csv` is 1 synthetic hour (53,758 rows). The real rounds come from arena.qminers.com and are recorded market data.

## The tape

```
Time,Operation,OrderId,Side,Volume,Price,Lifespan
0.003823,Insert,1000001,B,43,9999,G
0.491183,Cancel,1000001,,,,
```

| Column | Meaning |
|---|---|
| Time | seconds |
| Operation | `Insert` or `Cancel` (of `OrderId`) |
| Side | `B` buy · `A` sell |
| Volume | lots |
| Price | ticks |
| Lifespan | `G` resting limit · `F` IOC |

The tape only holds inserts and cancels. Every trade comes from the exchange matching them.

## Order book and matching

- **Price-time priority.** Each price level is a FIFO queue. A new order joins the back of its queue.
- An incoming order (from the market or from you) matches the opposite side while prices cross. **Each fill happens at the resting (maker) order's price**, best price first and front of queue first.
- Whatever is left of a **LIMIT** order rests in the book. Whatever is left of an **IOC** order is cancelled.
- There is no separate market order: an IOC priced through the book does the same job.
- Market orders can fill your resting orders, because they wait in the same queues.
- Your orders **never trade with each other**: your buy skips past your own sell.
- **No fees.** Your only cost is the spread you cross.
- Whole numbers only: prices in ticks, sizes in lots.

| Term | Meaning |
|---|---|
| Bid / ask | buy / sell price waiting in the book |
| Touch | best bid and best ask |
| Spread | best ask − best bid |
| Mid | (best bid + best ask) / 2 |
| Maker | your order rests in the book, so you earn the spread |
| Taker | your order trades into the book, so you pay the spread |
| Depth | levels visible: 20 ticks per side by default (`--public-depth`) |

## Latency: 5 ms

- `send_order`, `cancel_order` and `modify_order` called at time `t` take effect **just before the first market event with timestamp ≥ t + 5 ms**. The book may have moved by then.
- So an order is never filled within 5 ms of sending it. In a busy market, several events can pass before it lands.
- **A cancel can lose the race.** Your resting order can be filled while the cancel is in flight. The cancel then comes back `REJECTED`. Sending a cancel does not mean the order is cancelled, so count inventory from fills.

## The event loop (once per tape row, in this order)

1. Your orders whose 5 ms are up are applied. They may trade at once, which fires `on_private_order_update`.
2. The market row is applied. It may fill your orders, which fires `on_private_order_update`.
3. `on_order_book_change(snapshot, trades)` fires with the new book and all trades since the last call.

Fills caused by your own order going live show up in the `trades` of the next call.

## What the strategy sees

**`on_order_book_change(snapshot, trades)`** runs after every market event.
- `snapshot.bids` and `snapshot.asks` list every individual order, best price first, within `public_depth` ticks of the best price. Each has `.order_id`, `.side`, `.price`, `.quantity` and `.owner` (`'market'` or `'user'`).
- `trades` holds every trade since the previous call, including market-vs-market trades. Each has `price`, `quantity`, `book_side` (`'ask'` means a buyer lifted the ask, `'bid'` means a seller hit the bid), `taker_owner` and `maker_owner`.

**`on_private_order_update(event)`** runs synchronously, only for your own orders.
- Fields: `status`, `order_id`, `filled_quantity` (per fill), `fill_price`, `remaining_quantity`, `message`.

## What the strategy can do

| Call | Effect |
|---|---|
| `self.send_order(OrderRequest(side, price, quantity, order_type, client_order_id=None))` | submit an order. Returns its id (`u000000001`, …) immediately. |
| `self.cancel_order(order_id)` | cancel your resting order |
| `self.modify_order(order_id, new_quantity)` | **shrink only**. The order keeps its place in the queue. |
| `self.now_us()` | simulation time in µs |

`Side.BUY` / `Side.SELL`, `OrderType.LIMIT` / `OrderType.IOC`.

## Order lifecycle

```
send_order ──(+5 ms)──► ACCEPTED ──► PARTIALLY_FILLED* ──► FILLED
          └─► REJECTED (position limit)  ├──► CANCELED  (your cancel, or IOC leftover)
                                         └──► MODIFIED  (shrunk)
```

- You get one `PARTIALLY_FILLED`/`FILLED` event per match.
- A cancel of anything other than your own resting order, or a modify that doesn't shrink the order, comes back `REJECTED`.
- A modify of an unknown id is **silently ignored**.

## Limits

With I = signed inventory, R = your resting quantity on the same side, q = order quantity, and s = +1 for buy, −1 for sell:

| Rule | Consequence |
|---|---|
| `|I + s·(R + q)| > 1000` when the order arrives | order **REJECTED** |
| `|I(t)| > 100` | **−$10,000 per second**, charged for the exact time spent outside ±100 |

In short: inside ±100 you pay nothing, between 100 and 1000 you pay the penalty, and past ±1000 new orders are rejected.

## Score

```
PnL = cash + markout(I) − penalty
```

- **cash** = money in from sells − money out on buys.
- **markout(I)** = the cash you would get by selling off the remaining inventory into the resting **market** orders on the opposite side, best price outward. Any volume beyond the visible book is valued at the last price swept, or at the last traded price if that side is empty. So holding inventory at the end costs you the spread and the book depth.
- Example: holding +30 against bids of 12@10000, 10@9999 and 25@9998 gives a markout of 299,974. Bought at 10001, the cash is −300,030, so PnL = −56.
- PnL is computed at the end, and also every second for `pnl.csv` / `pnl_se`.

`result.json` fields: `pnl`, `cash`, `inventory`, `mark_price`, `breach_penalty`, `traded_volume`, `fills`, `pnl_se`, and the time-weighted `abs_inventory_q50/q68/q95/q997`.

## Practical takeaways

- Aim to end the run (and spend most of it) close to flat. Leftover inventory pays the spread at markout.
- Keep a hard cap well inside ±100 that counts your resting orders, and leave room for fills on orders already in flight.
- Use one source of truth for inventory: fill events.
- Always handle `CANCELED` and `REJECTED`, or the strategy can wait forever for an answer that never comes.
- Keep callbacks O(1)-ish. They run on every row.
- Use the explorer when a result surprises you. Things to look for: fills away from the touch, price moving against you after a fill (you got picked off), cliffs in the PnL curve, and "Add Failed" / "Cancel Failed" rows.
