"""Reconstruction of the order book from the book delta log."""

import heapq
from collections import defaultdict

import pandas as pd


def build_best_quotes(deltas: pd.DataFrame) -> pd.DataFrame:
    """Replay the book deltas in timestamp order into a best bid/ask history.

    Args:
        deltas: The book_deltas.csv frame: side ('B' for bid or 'A' for ask),
            price, and qty_diff (the signed quantity to add at that price).

    Returns:
        One row per distinct `timestamp_us` with `best_bid` and `best_ask` after
        every delta at that timestamp was applied (None on an empty side).
    """
    level_quantity: defaultdict[tuple[str, float], float] = defaultdict(float)
    bid_heap: list[float] = []
    ask_heap: list[float] = []
    previous_timestamp_us: int | None = None
    rows: list[dict[str, float | None]] = []

    def best_bid() -> float | None:
        while bid_heap and level_quantity[('B', -bid_heap[0])] <= 0:
            heapq.heappop(bid_heap)
        return -bid_heap[0] if bid_heap else None

    def best_ask() -> float | None:
        while ask_heap and level_quantity[('A', ask_heap[0])] <= 0:
            heapq.heappop(ask_heap)
        return ask_heap[0] if ask_heap else None

    def record(timestamp_us: int) -> None:
        rows.append({'timestamp_us': timestamp_us, 'best_bid': best_bid(), 'best_ask': best_ask()})

    for row in deltas.sort_values('timestamp_us').itertuples(index=False):
        timestamp_us = int(row.timestamp_us)

        if previous_timestamp_us is not None and timestamp_us != previous_timestamp_us:
            record(previous_timestamp_us)

        side = str(row.side)
        price = float(row.price)
        level_quantity[(side, price)] += float(row.qty_diff)
        if side == 'B':
            heapq.heappush(bid_heap, -price)
        else:
            heapq.heappush(ask_heap, price)
        previous_timestamp_us = timestamp_us

    if previous_timestamp_us is not None:
        record(previous_timestamp_us)

    return pd.DataFrame(rows)


_BOOK_CACHE: dict[int, pd.DataFrame] = {}


def get_book_at(deltas: pd.DataFrame, timestamp_us: int) -> pd.DataFrame:
    """Return the book just before `timestamp_us` as rows of side, price, owner, quantity.

    Deltas at `timestamp_us` itself are excluded. Results are cached per timestamp.
    """
    if timestamp_us in _BOOK_CACHE:
        return _BOOK_CACHE[timestamp_us]

    end = deltas['timestamp_us'].searchsorted(timestamp_us, side='left')
    deltas_before = deltas.iloc[:end]
    book = (
        deltas_before.groupby(['side', 'price', 'owner'])['qty_diff']
        .sum()
        .reset_index()
        .rename(columns={'qty_diff': 'quantity'})
    )
    book = book[book['quantity'] > 0]

    _BOOK_CACHE[timestamp_us] = book
    return book
