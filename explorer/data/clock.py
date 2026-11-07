"""Conversion between the logs' microsecond timestamps and the charts' clock.

The charts share pnl.csv's clock: seconds since the backtest's first event.
The CSV logs carry the backtest's own microsecond timestamps, so they are
converted with `start_us`, that first event's timestamp.
"""

import pandas as pd

MICROSECONDS_PER_SECOND = 1_000_000


def elapsed_seconds(timestamp_us: pd.Series | int, start_us: int) -> pd.Series | float:
    """Return the chart time of a log timestamp, or of each one in a Series."""
    return (timestamp_us - start_us) / MICROSECONDS_PER_SECOND


def to_timestamp_us(elapsed_s: float, start_us: int) -> int:
    """Return the log timestamp of a chart time."""
    return start_us + round(elapsed_s * MICROSECONDS_PER_SECOND)
