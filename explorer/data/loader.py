"""Readers of the simulator's CSV logs, PnL curve, result and run metadata."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_data(log_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load book_deltas.csv, events.csv and trades.csv from `log_dir`.

    Columns per file:
      book_deltas.csv: timestamp_us, side, price, owner, qty_diff
      events.csv:      timestamp_us, source, operation, order_id, side, price, quantity
      trades.csv:      timestamp_us, price, taker_side, quantity, taker_order_id,
                       maker_order_id, taker_owner, maker_owner

    Returns:
        The (deltas, events, trades) frames.
    """
    deltas = pd.read_csv(log_dir / 'book_deltas.csv', dtype={'timestamp_us': 'int64'})
    events = pd.read_csv(log_dir / 'events.csv', dtype={'timestamp_us': 'int64'})
    trades = pd.read_csv(log_dir / 'trades.csv', dtype={'timestamp_us': 'int64'})
    return deltas, events, trades


def merge_trades_into_events(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Fold the trades into the event log as rows with operation 'trade'.

    A trade row takes the taker's order id and owner as `order_id` and
    `source`, the book side the taker hit as `side` (the flip of `taker_side`),
    and carries `maker_order_id` and `maker_owner` in two extra columns that are
    empty on every other row.

    Returns:
        The combined, unsorted frame.
    """
    maker_side = trades['taker_side'].map({'A': 'B', 'B': 'A'})
    trade_events = pd.DataFrame(
        {
            'timestamp_us': trades['timestamp_us'],
            'source': trades['taker_owner'],
            'operation': 'trade',
            'order_id': trades['taker_order_id'],
            'side': maker_side,
            'price': trades['price'],
            'quantity': trades['quantity'],
            'maker_order_id': trades['maker_order_id'],
            'maker_owner': trades['maker_owner'],
        }
    )

    events = events.copy()
    events['maker_order_id'] = ''
    events['maker_owner'] = ''

    columns = [
        'timestamp_us',
        'source',
        'operation',
        'order_id',
        'side',
        'price',
        'quantity',
        'maker_order_id',
        'maker_owner',
    ]
    merged = pd.concat([events, trade_events], ignore_index=True)[columns]
    merged = merged.astype({'source': 'category', 'side': 'category', 'operation': 'category'})
    return merged


def load_result(log_dir: Path) -> dict[str, Any]:
    """Load result.json from `log_dir`: the simulator's final figures, `pnl` among them."""
    with (log_dir / 'result.json').open() as file:
        return json.load(file)


def load_pnl(log_dir: Path) -> pd.DataFrame:
    """Load pnl.csv from `log_dir`: the simulator's mark-to-market PnL, sampled every second.

    Returns:
        A frame with `elapsed_s` (seconds since the backtest's first event, the
        charts' clock) and `pnl` columns.
    """
    return pd.read_csv(log_dir / 'pnl.csv')


def load_metadata(log_dir: Path) -> dict[str, Any]:
    """Load metadata.json from `log_dir`: who ran what, with which data and settings."""
    with (log_dir / 'metadata.json').open() as file:
        return json.load(file)
