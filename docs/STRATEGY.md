# Adaptive market maker (`strategies/adaptive_mm.py`)

The strategy has two legs:

- A **maker leg** that keeps passive quotes at the best bid and ask and earns the spread. This leg makes the money on realistic data.
- A **taker leg** that holds a directional position when order flow predicts a move. It only trades while a paper-traded **shadow** copy of itself is proving that the move is real. On data without momentum the taker stays off.

Market rules (latency, limits, scoring) are in [MARKET.md](MARKET.md). Research and experiment history are in [RESEARCH.md](RESEARCH.md).

## What happens on each market event

`on_order_book_change` runs after every row of the tape:

1. **Read the touch.** Find the market's best bid and ask and the lots resting at each. Our own orders are ignored, so the signals never react to us.
2. **Fill the shadow's pending order**, if its 5 ms latency has passed (see *Momentum check*).
3. **Update the signals:** OFI, aggressive flow, and the average touch size. All of them fade exponentially over time.
4. **Forecast** = `OFI_WEIGHT × OFI / average touch size`. This is the expected price move in ticks.
5. **Taker leg:** set the target position and, if needed, send an IOC toward it.
6. **Shadow step:** move the shadow toward the same direction and add its PnL change to `edge`.
7. **Maker leg:** work out where the two quotes should be, and cancel or replace them if they have moved.

`on_private_order_update` keeps inventory (from fills only) and the list of our open orders up to date.

## Forecast: order flow imbalance (OFI)

This follows Cont, Kukanov & Stoikov. On each event the change at the touch is scored:

| Change since the previous event | Contribution |
|---|---|
| Bid queue grows / shrinks by q | +q / −q |
| Ask queue shrinks / grows by q | +q / −q |
| Best bid moves up / down | + new bid queue / − old bid queue |
| Best ask moves down / up | − new ask queue / + old ask queue |

The running sum fades with time constant `OFI_TAU_US` (1 s). Dividing it by the average touch size makes it scale-free, so the same thresholds work for a 25-lot book and an 800-lot book.

## Taker leg

- **Direction:** a forecast above `+TAKE_THRESHOLD` (1 tick) means long, and below `−TAKE_THRESHOLD` means short. Once the forecast is back within `±EXIT_THRESHOLD` (0.5 ticks), the direction is flat. In between, the direction stays as it was.
- **Target position:** direction × size, where size comes from the momentum check (0 to `TAKE_POS` = 50 lots).
- **Orders:** with one IOC in flight at most, it buys at the best ask or sells at the best bid, up to `TAKE_SIZE` lots, toward the target. It only takes liquidity in the direction the forecast points. Unwinding against the forecast is left to the maker leg's quotes.
- **When size is 0** the taker does nothing at all, not even flattening. This keeps it from paying the spread to unwind the maker's inventory.

## Momentum check (shadow taker)

The taker only pays when prices keep moving after order flow, as they do in the synthetic `sample.csv`. On tapes without that it loses about 860 per lot per hour. So the strategy measures this during the run:

- A **1-lot shadow** follows the same direction and is paper-traded with the real rules:
  - Orders arrive 5 ms later and fill against the book as it was then.
  - An IOC misses if the price moved away in the meantime.
  - It buys at the ask, sells at the bid, and unwinds by taking, which makes the estimate conservative.
- The shadow's PnL, marked at the touch, is summed into `edge`. The sum fades over `EDGE_TAU_US` (5 min).
- **Real taker size** = `TAKE_POS × clamp(edge / EDGE_FULL, 0, 1)`. `EDGE_FULL` is 60 ticks per lot. If the shadow is losing, the size is 0.

## Maker leg

- **One quote per side** of `QUOTE_SIZE` (60) lots, at the market's best bid and ask. It never quotes inside the spread, because in testing the first order at a new better price was filled by exactly the flow that then moved against it.
- **Skew:** `signal = forecast − (inventory − target) × SKEW_PER_LOT`. If the signal is above `BACKOFF_THRESHOLD` (0.3), the ask steps back `ceil(signal − 0.3)` ticks. If it is below −0.3, the bid steps back the same way. Holding inventory makes the side that would add more step back, while the side that unwinds stays at the touch.
- **Pull filter:** aggressive market volume (buys − sells), fading over 0.5 s, is tracked. If it exceeds `PULL_RATIO` × the average touch size, the quote on the side being swept is cancelled, unless that quote moves us toward the target.
- **Queue position:** a quote already at the right price is left alone, so it keeps its place. When the price changes, the old quote is cancelled and a new one sent straight away. At most 2 orders per side are open (one working, one being cancelled).

## Risk

- `_room(side)` makes sure that **even if every open order on one side fills**, |inventory| stays within `MAX_POS` (90). The exchange penalty starts above 100.
- Inventory changes only on fill events, so a cancel that loses the race against a fill can't put it out of sync.

## Parameters

| Name | Default | Meaning |
|---|---|---|
| `OFI_WEIGHT` | 0.25 | forecast ticks per average touch size of OFI |
| `OFI_TAU_US` | 1 s | OFI fade |
| `TAKE_THRESHOLD` / `EXIT_THRESHOLD` | 1.0 / 0.5 ticks | taker direction on / off |
| `TAKE_POS` / `TAKE_SIZE` | 50 / 50 lots | max taker position / max IOC size |
| `EDGE_TAU_US` / `EDGE_FULL` | 5 min / 60 ticks per lot | momentum check memory / evidence for full size |
| `QUOTE_SIZE` | 60 lots | maker quote size |
| `SKEW_PER_LOT` | 0.1 ticks | inventory skew |
| `BACKOFF_THRESHOLD` | 0.3 ticks | when a quote steps off the touch |
| `FLOW_TAU_US` / `PULL_RATIO` | 0.5 s / 1.0 | pull filter |
| `MAX_POS` | 90 lots | worst-case inventory cap |
| `TAKE_ENABLED` / `MAKE_ENABLED` / `EDGE_CHECK` | on | switches for experiments |

## Results (defaults)

| Tape | PnL | Notes |
|---|---|---|
| `sample_v2.csv` (realistic, 1.6 h) | **125,470** (±2,207) | almost all from the maker; the taker stays off |
| `sample.csv` (synthetic, trending) | 113,698 | almost all from the taker |
| no-momentum control tapes ×3 | +403 / −310 / −124 | |
| follow-through control tapes | −2,127 / −712 | worst case so far |
| trend → no momentum / no momentum → trend | 44,493 / 44,703 | the momentum check adapts in both directions |

Never any penalty or position-limit reject. On v2, inventory has a median of 30 and stays within ±64 lots 99.7% of the time.

## Known weaknesses

- **Only one realistic tape.** Its two halves agree, but every maker setting comes from 1.6 hours of one instrument.
- **Our place in the queue.** On v2 there are about 820 lots at the best price, so a new quote joins at the back. Being filled late in the queue is the riskiest kind of fill.
- **Quoting inside the spread on wide spreads** (2+ ticks) is untested on realistic data.
- **Leftover inventory at the end** (−30 on v2) is valued at the end price, which costs a little. There is no end-of-run flattening because the run length is unknown.
- **Momentum that comes and goes** is not well covered by the control tapes.

## Running it

```bash
uv run exchange-simulator --strategy strategies/adaptive_mm.py --backtest data/sample_v2.csv --log-dir logs/adaptive --no-logs
```

To tune a parameter on held-out halves and control tapes:

```bash
python3 -I scripts/split_tape.py data/sample_v2.csv 2
```

```bash
python3 -I scripts/sweep.py strategies/adaptive_mm.py data/sample_v2_p1of2.csv data/sample_v2_p2of2.csv data/null_1.csv -p QUOTE_SIZE=40,60,80
```
