"""Matching engine, replay loop and run loggers.

`OrderBook` is a price-time priority book; `ExchangeSimulator` replays a
backtest CSV through it, delays the strategy's orders by the configured
latency and reports the result. The loggers write the run log and the CSV
logs the backtest explorer reads.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
import math
import statistics
import sys
import time
from bisect import insort
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, TextIO

from pydantic import BaseModel, ConfigDict, Field, model_validator

from simulator.models import (
    BookOrderView,
    LiveOrder,
    MarketEvent,
    MarketOperation,
    OrderBookSnapshot,
    OrderRequest,
    OrderStatus,
    OrderType,
    PrivateOrderEvent,
    Side,
    SimulationResult,
    Trade,
)
from simulator.strategy import BaseStrategy, StrategyContext

PNL_SAMPLE_INTERVAL_US = 1_000_000
_BACKTEST_BLOCK_BYTES = 1 << 20


_BUY = Side.BUY
_SELL = Side.SELL
_LIMIT = OrderType.LIMIT
_IOC = OrderType.IOC
_SIDE_BY_VALUE = {member.value: member for member in Side}
_FILL_STATUSES = frozenset({OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED})
_json_string = json.encoder.encode_basestring_ascii


def _backtest_lines(path: Path) -> Iterator[str]:
    """Yield the lines of the backtest CSV, keeping the file closed whenever a line is out.

    The strategy runs while this generator is suspended, in the same process. A
    descriptor held open across a `yield` is one it can read the rest of the
    backtest through, private rows included, with calls no audit event covers
    (`np.fromfile` on a wrapper around the fd, or `os.read`). So each block is
    read with its own open, seek and close, and only then handed out.
    """
    offset = 0
    partial_line = b''
    while True:
        with path.open('rb') as file:
            file.seek(offset)
            block = file.read(_BACKTEST_BLOCK_BYTES)
        if not block:
            break
        offset += len(block)
        data = partial_line + block
        end = data.rfind(b'\n') + 1
        partial_line = data[end:]
        yield from io.StringIO(data[:end].decode(), newline='')
    yield from io.StringIO(partial_line.decode(), newline='')


def _json_scalar(value: object) -> str:
    """Serialise one scalar exactly as `json.dumps(value, ensure_ascii=True)` would."""
    if type(value) is str:
        return _json_string(value)
    return json.dumps(value, separators=(',', ':'), ensure_ascii=True)


def _trade_json(trade: Trade) -> str:
    """Serialise a trade as `json.dumps(trade.model_dump(mode="json"))` would."""
    return (
        f'{{"timestamp_us":{trade.timestamp_us},"price":{trade.price},"quantity":{trade.quantity},'
        f'"taker_order_id":{_json_scalar(trade.taker_order_id)},'
        f'"maker_order_id":{_json_scalar(trade.maker_order_id)},'
        f'"taker_owner":"{trade.taker_owner}","maker_owner":"{trade.maker_owner}","book_side":"{trade.book_side}"}}'
    )


@dataclass(slots=True)
class PendingUserOrder:
    """An order submitted by the strategy, waiting out the accept delay."""

    order_id: str
    request: OrderRequest
    ready_at_us: int


@dataclass(slots=True)
class PendingUserCancel:
    """A cancel submitted by the strategy, waiting out the accept delay."""

    order_id: str
    ready_at_us: int


@dataclass(slots=True)
class PendingUserModify:
    """A size reduction submitted by the strategy, waiting out the accept delay."""

    order_id: str
    new_quantity: int
    ready_at_us: int


class _PriceLevel:
    """One price level: its FIFO queue and the cached public-feed views of it.

    `views` is None or exactly `[o.view for o in queue if o.remaining_quantity > 0]`,
    with every such `o.view` set. It is maintained incrementally by the book
    and rebuilt lazily whenever that is not possible.
    """

    __slots__ = ('queue', 'views')

    def __init__(self) -> None:
        self.queue: deque[LiveOrder] = deque()
        self.views: list[BookOrderView] | None = None

    def drop_view(self, view: BookOrderView | None) -> None:
        """Remove `view` (by identity) from the cache, or invalidate the cache."""
        views = self.views
        if views is None:
            return
        for index, candidate in enumerate(views):
            if candidate is view:
                del views[index]
                return
        self.views = None

    def replace_view(self, old_view: BookOrderView | None, new_view: BookOrderView) -> None:
        """Swap `old_view` for `new_view` (by identity) in the cache, or invalidate the cache."""
        views = self.views
        if views is None:
            return
        for index, candidate in enumerate(views):
            if candidate is old_view:
                views[index] = new_view
                return
        self.views = None


class OrderBook:
    """Price-time priority book with one FIFO queue per price level on each side."""

    def __init__(self) -> None:
        self._orders: dict[str, LiveOrder] = {}
        self._bids: dict[int, _PriceLevel] = {}
        self._asks: dict[int, _PriceLevel] = {}
        self._bid_prices: list[int] = []
        self._ask_prices: list[int] = []
        self._user_quantity: dict[Side, int] = {_BUY: 0, _SELL: 0}
        self.bid_version = 0
        self.ask_version = 0

    def get(self, order_id: str) -> LiveOrder | None:
        """Return the resting order with `order_id`, or None."""
        return self._orders.get(order_id)

    def insert(self, order: LiveOrder) -> None:
        """Append `order` to the queue of its price level."""
        if order.order_id in self._orders:
            raise ValueError(f'Order {order.order_id} already exists')
        self._orders[order.order_id] = order
        if order.side == _BUY:
            levels, prices = self._bids, self._bid_prices
        else:
            levels, prices = self._asks, self._ask_prices
        level = levels.get(order.price)
        if level is None:
            level = levels[order.price] = _PriceLevel()
            insort(prices, order.price)
        level.queue.append(order)
        if level.views is not None and order.remaining_quantity > 0:
            level.views.append(_view_of(order))
        if order.owner == 'user':
            self._user_quantity[order.side] += order.remaining_quantity
        self._bump_version(order.side)

    def remove(self, order_id: str) -> LiveOrder | None:
        """Take the order with `order_id` out of the book and return it, or None if unknown."""
        order = self._orders.pop(order_id, None)
        if order is None:
            return None

        if order.side == _BUY:
            levels, prices = self._bids, self._bid_prices
        else:
            levels, prices = self._asks, self._ask_prices
        level = levels.get(order.price)
        if level is not None:
            level.queue.remove(order)
            if order.remaining_quantity > 0:
                level.drop_view(order.view)
            if not level.queue:
                del levels[order.price]
                prices.remove(order.price)
        if order.owner == 'user':
            self._user_quantity[order.side] -= order.remaining_quantity
        self._bump_version(order.side)
        return order

    def reduce_quantity(self, order: LiveOrder, amount: int) -> None:
        """Lower a resting order's remaining quantity by `amount`."""
        was_live = order.remaining_quantity > 0
        order.remaining_quantity -= amount
        old_view = order.view
        order.view = None
        levels = self._bids if order.side == _BUY else self._asks
        level = levels.get(order.price)
        if level is not None and was_live:
            if order.remaining_quantity > 0:
                level.replace_view(old_view, _view_of(order))
            else:
                level.drop_view(old_view)
        if order.owner == 'user':
            self._user_quantity[order.side] -= amount
        self._bump_version(order.side)

    def _bump_version(self, side: Side) -> None:
        """Mark `side` as changed since any cached view of it."""
        if side == _BUY:
            self.bid_version += 1
        else:
            self.ask_version += 1

    def best_opposite_order(self, side: Side, *, allow_user_orders: bool = False) -> LiveOrder | None:
        """Return the first order a taker on `side` would match, or None if the opposite side is empty.

        Args:
            side: Side of the taker; the opposite side of the book is searched.
            allow_user_orders: Whether the strategy's own resting orders count;
                False prevents self-trading.
        """
        if side == _BUY:
            prices: Iterable[int] = self._ask_prices
            levels = self._asks
        else:
            prices = reversed(self._bid_prices)
            levels = self._bids
        for price in prices:
            for order in levels[price].queue:
                if allow_user_orders or order.owner == 'market':
                    assert order.side != side, 'Order side should match the queue it is in'
                    assert order.remaining_quantity > 0, 'Order with no remaining quantity should not be in the book'
                    return order
        return None

    def iter_levels(self, side: Side, depth: int | None = None) -> Iterable[LiveOrder]:
        """Yield live orders from the best price outwards.

        Args:
            side: Side of the book to iterate.
            depth: Number of price ticks to include, counted from the best live
                price and including empty prices. None means no limit.
        """
        if side == _BUY:
            prices: Iterable[int] = reversed(self._bid_prices)
            levels = self._bids
        else:
            prices = self._ask_prices
            levels = self._asks
        best_price: int | None = None
        for price in prices:
            for order in levels[price].queue:
                if order.remaining_quantity <= 0:
                    continue
                if best_price is None:
                    best_price = price
                if depth is not None and abs(price - best_price) >= depth:
                    return
                yield order

    def visible_views(self, side: Side, depth: int) -> list[BookOrderView]:
        """Return public-feed views of the orders within `depth` ticks on `side`.

        Same orders, same order, as `iter_levels(side, depth)`. Each level
        caches the views of its queue until the level changes, so an event
        typically rebuilds one level and the rest is list concatenation.
        """
        if side == _BUY:
            prices: Iterable[int] = reversed(self._bid_prices)
            levels = self._bids
        else:
            prices = self._ask_prices
            levels = self._asks
        views: list[BookOrderView] = []
        best_price: int | None = None
        for price in prices:
            level = levels[price]
            level_views = level.views
            if level_views is None:
                level_views = level.views = _build_level_views(level.queue)
            if not level_views:
                continue
            if best_price is None:
                best_price = price
            if abs(price - best_price) >= depth:
                break
            views.extend(level_views)
        return views

    def resting_user_quantity(self, side: Side) -> int:
        """Return the total resting user quantity on `side`."""
        return self._user_quantity[side]


def _view_of(order: LiveOrder) -> BookOrderView:
    """Return the order's cached public-feed view, building it if needed."""
    view = order.view
    if view is None:
        view = order.view = BookOrderView(
            order_id=order.order_id,
            side=order.side,
            price=order.price,
            quantity=order.remaining_quantity,
            owner=order.owner,
        )
    return view


def _build_level_views(queue: deque[LiveOrder]) -> list[BookOrderView]:
    """Build the public-feed views of every live order in `queue`, in order."""
    return [_view_of(order) for order in queue if order.remaining_quantity > 0]


class LogLevel(str, Enum):
    """Verbosity of the run log; DEBUG adds market events and full book snapshots."""

    DEBUG = 'DEBUG'
    INFO = 'INFO'


class SimulationConfig(BaseModel):
    """Parameters of a replay; a log path of None disables that log."""

    model_config = ConfigDict(frozen=True)

    order_accept_delay_ms: int = Field(default=5, ge=0)
    public_depth: int = Field(default=20, gt=0)
    run_log_path: Path | None = None
    book_delta_log_path: Path | None = None
    event_log_path: Path | None = None
    trade_log_path: Path | None = None
    max_position: int | None = Field(default=None, ge=0)
    max_potential_position: int | None = Field(default=None, ge=0)
    position_breach_penalty_per_second: float = Field(default=0.0, ge=0)
    min_log_level: LogLevel = LogLevel.INFO
    show_progress: bool = False
    progress_fd: int = Field(default=2, ge=0)

    @model_validator(mode='after')
    def _check_position_limits(self) -> SimulationConfig:
        if self.max_potential_position is not None:
            if self.max_position is None:
                raise ValueError('max_potential_position requires max_position to be set')
            if self.max_potential_position < self.max_position:
                raise ValueError('max_potential_position must be >= max_position')
        return self


_DEBUG = LogLevel.DEBUG
_INFO = LogLevel.INFO


class RunLogger:
    """JSON Lines run log: one record per simulator event, filtered by level."""

    def __init__(self, path: Path | None, min_level: LogLevel) -> None:
        self._path = path
        self._file: TextIO | None = None
        self._seq = 0
        self.min_level = min_level
        self._debug_enabled = min_level is LogLevel.DEBUG

    @property
    def enabled(self) -> bool:
        """Return True if a log path is configured."""
        return self._path is not None

    def should_log(self, level: LogLevel) -> bool:
        """Return whether a record at `level` would actually be written."""
        return self._file is not None and (level is LogLevel.INFO or self._debug_enabled)

    @property
    def debug_enabled(self) -> bool:
        """Return whether DEBUG records are currently being written."""
        return self._file is not None and self._debug_enabled

    def open(self) -> None:
        """Open the log file, creating its directory; a no-op when disabled or already open."""
        if self._path is None or self._file is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open('w', encoding='utf-8')

    def log(self, *, level: LogLevel, timestamp_us: int, kind: str, payload: dict[str, Any]) -> None:
        """Write one record, if `level` passes the filter."""
        if not self.should_log(level):
            return
        self._seq += 1
        row = {
            'seq': self._seq,
            'timestamp_us': timestamp_us,
            'kind': kind,
            'payload': payload,
            'level': level.value,
        }
        assert self._file is not None
        self._file.write(json.dumps(row, separators=(',', ':'), ensure_ascii=True) + '\n')

    def log_json(self, *, level: LogLevel, timestamp_us: int, kind: str, payload_json: str) -> None:
        """Write a record whose payload is already compact JSON.

        Produces exactly the bytes `log` would for the same record; `kind`
        must be a plain ASCII literal.
        """
        file = self._file
        if file is None or not self.should_log(level):
            return
        self._seq += 1
        file.write(
            f'{{"seq":{self._seq},"timestamp_us":{timestamp_us},"kind":"{kind}",'
            f'"payload":{payload_json},"level":"{level.value}"}}\n'
        )

    def close(self) -> None:
        """Close the log file, if open."""
        if self._file is not None:
            self._file.close()
            self._file = None


class CsvLogger:
    """CSV log file: the header on open, then the rows the subclasses' `log` append."""

    HEADER: ClassVar[str]

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._file: TextIO | None = None

    @property
    def enabled(self) -> bool:
        """Return True if a log path is configured."""
        return self._path is not None

    def open(self) -> None:
        """Open the CSV file and write the header row; a no-op when disabled or already open."""
        if self._path is None or self._file is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open('w', encoding='utf-8', newline='')
        self._file.write(self.HEADER)

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None


class BookDeltaLogger(CsvLogger):
    """CSV log of every order book quantity change."""

    HEADER = 'timestamp_us,side,price,owner,qty_diff\n'

    def log(self, timestamp_us: int, side: Side, price: int, owner: str, qty_diff: int) -> None:
        """Append one delta row."""
        assert self._file is not None
        self._file.write(f'{timestamp_us},{side.value},{price},{owner},{qty_diff}\n')


class OrderEventLogger(CsvLogger):
    """CSV log of order, cancel and modify lifecycle events."""

    HEADER = 'timestamp_us,source,operation,order_id,side,price,quantity\n'

    def log(
        self,
        timestamp_us: int,
        source: str,
        operation: str,
        order_id: str,
        side: str = '',
        price: str = '',
        quantity: str = '',
    ) -> None:
        """Append one event row."""
        assert self._file is not None
        self._file.write(f'{timestamp_us},{source},{operation},{order_id},{side},{price},{quantity}\n')


class TradeLogger(CsvLogger):
    """CSV log of every matched trade."""

    HEADER = 'timestamp_us,price,taker_side,quantity,taker_order_id,maker_order_id,taker_owner,maker_owner\n'

    def log(self, trade: Trade) -> None:
        """Append one trade row."""
        assert self._file is not None
        taker_side = 'B' if trade.book_side == 'ask' else 'A'
        self._file.write(
            f'{trade.timestamp_us},{trade.price},{taker_side},{trade.quantity},'
            f'{trade.taker_order_id},{trade.maker_order_id},{trade.taker_owner},{trade.maker_owner}\n'
        )


class ProgressBar:
    """Replay progress written to a file descriptor (stderr by default).

    On a terminal it is a single-line bar redrawn in place. Anywhere else (a
    pipe, a file, the evaluation sandbox) redraws would pile up as one endless
    line, so it writes one JSON line every `_LINE_INTERVAL_S` instead, which a
    supervising process can parse and log.
    """

    _WIDTH = 30
    _REDRAW_INTERVAL_S = 0.1
    _LINE_INTERVAL_S = 10.0

    def __init__(self, total: int, enabled: bool, fd: int = 2) -> None:
        self._total = total
        self._enabled = enabled
        self._start_s = time.monotonic()
        self._last_draw_s = 0.0
        self._drawn = False
        self._stream: TextIO | None = None
        self._owns_stream = False
        self._is_tty = False
        if enabled:
            if fd == 2:
                self._stream = sys.stderr
            else:
                self._stream = open(fd, 'w', buffering=1, closefd=False)
                self._owns_stream = True
            self._is_tty = self._stream.isatty()

    def update(self, done: int) -> None:
        """Redraw the bar for `done` processed events, at most every 100 ms.

        Args:
            done: Number of market events replayed so far.
        """
        if self._stream is None:
            return
        now_s = time.monotonic()
        interval_s = self._REDRAW_INTERVAL_S if self._is_tty else self._LINE_INTERVAL_S
        if done < self._total and self._drawn and now_s - self._last_draw_s < interval_s:
            return
        self._last_draw_s = now_s

        fraction = done / self._total if self._total else 1.0
        elapsed_s = now_s - self._start_s
        rate = done / elapsed_s if elapsed_s > 0 else 0.0
        eta_s = (self._total - done) / rate if rate > 0 else 0.0
        if self._is_tty:
            filled = int(self._WIDTH * fraction)
            bar = '#' * filled + '-' * (self._WIDTH - filled)
            self._stream.write(
                f'\r[{bar}] {fraction * 100:5.1f}%  {done:,}/{self._total:,} events  '
                f'{rate:,.0f} ev/s  ETA {eta_s:5.1f}s'
            )
        else:
            self._stream.write(
                json.dumps(
                    {
                        'kind': 'progress',
                        'done': done,
                        'total': self._total,
                        'percent': round(fraction * 100, 1),
                        'events_per_s': round(rate),
                        'elapsed_s': round(elapsed_s, 1),
                        'eta_s': round(eta_s, 1),
                    }
                )
                + '\n'
            )
        self._stream.flush()
        self._drawn = True

    def close(self) -> None:
        """Terminate the progress line so later output starts on a fresh line."""
        if self._stream is None:
            return
        if self._drawn and self._is_tty:
            self._stream.write('\n')
            self._stream.flush()
        self._drawn = False
        if self._owns_stream:
            self._stream.close()
        self._stream = None


class ExchangeSimulator:
    """Replay of a backtest through the matching engine with one strategy attached.

    `_trades` holds only the trades since the last public snapshot. Run totals are
    kept as counters, not as history: a backtest trades millions of times and the
    sandbox has 2 GiB.
    """

    def __init__(self, strategy: BaseStrategy, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.strategy = strategy
        self.book = OrderBook()
        self.current_ts_us = 0

        self._last_event_us: int | None = None
        self._pending: list[PendingUserOrder | PendingUserCancel | PendingUserModify] = []

        self._user_order_seq = 0
        self._trades: list[Trade] = []
        self._filled_order_ids: set[str] = set()
        self._user_traded_volume = 0
        self._visible_cache: dict[Side, tuple[int, list[BookOrderView]]] = {
            _BUY: (-1, []),
            _SELL: (-1, []),
        }

        self._inventory = 0
        self._cash = 0.0
        self._breach_penalty = 0.0
        self._max_position = self.config.max_position
        self._penalty_per_second = self.config.position_breach_penalty_per_second
        self._accept_delay_us = self.config.order_accept_delay_ms * 1_000
        self._last_trade_price: float | None = None
        self._next_pnl_sample_us: int | None = None
        self._pnl_sample_times_us: list[int] = []
        self._pnl_samples: list[float] = []
        self._abs_inventory_time_us: defaultdict[int, int] = defaultdict(int)
        self._logger = RunLogger(self.config.run_log_path, self.config.min_log_level)
        self._delta_logger = BookDeltaLogger(self.config.book_delta_log_path)
        self._event_logger = OrderEventLogger(self.config.event_log_path)
        self._trade_logger = TradeLogger(self.config.trade_log_path)
        self._log_run = self._logger.enabled
        self._log_deltas = self._delta_logger.enabled
        self._log_events = self._event_logger.enabled
        self._log_trades = self._trade_logger.enabled

        self.strategy._attach_context(
            StrategyContext(
                submit_order=self.submit_order,
                cancel_order=self.cancel_order,
                modify_order=self.modify_order,
                now_us=lambda: self.current_ts_us,
            )
        )

    def _log(self, *, level: LogLevel, kind: str, payload: dict[str, Any] | Callable[[], dict[str, Any]]) -> None:
        """Write a run-log record.

        `payload` may be a zero-argument callable, in which case it is only
        invoked if the record survives the level filter. The per-event payloads
        are `model_dump` calls, which are expensive and were previously built on
        every event and then discarded by `RunLogger.log` at the default INFO
        level.
        """
        if callable(payload):
            if not self._logger.should_log(level):
                return
            payload = payload()
        self._logger.log(level=level, timestamp_us=self.current_ts_us, kind=kind, payload=payload)

    @staticmethod
    def count_backtest_rows(path: Path) -> int:
        """Return the number of data rows in the backtest CSV, without parsing them."""
        lines = 0
        last_byte = b'\n'
        with path.open('rb') as file:
            while chunk := file.read(1 << 20):
                lines += chunk.count(b'\n')
                last_byte = chunk[-1:]
        if last_byte != b'\n':
            lines += 1
        return max(lines - 1, 0)

    @staticmethod
    def iter_backtest(path: Path) -> Iterator[MarketEvent]:
        """Parse the backtest CSV into market events, one row at a time (the CSV's `Time` is in seconds).

        A generator on purpose. Holding a whole backtest takes about 200 bytes per
        row, so a 300 MB CSV (8.8M rows) fills the sandbox's 2 GiB limit before the
        replay starts, and the kernel then evicts the interpreter's code pages until
        the run crawls and is killed.
        """
        insert_operation = MarketOperation.INSERT.value
        cancel_operation = MarketOperation.CANCEL.value
        reader = csv.reader(_backtest_lines(path))
        header = next(reader, None)
        if header is None:
            return
        column_index = {name: index for index, name in enumerate(header)}
        time_index = column_index['Time']
        operation_index = column_index['Operation']
        order_id_index = column_index['OrderId']
        side_index = column_index['Side']
        volume_index = column_index['Volume']
        price_index = column_index['Price']
        lifespan_index = column_index.get('Lifespan')
        for row in reader:
            operation = row[operation_index]
            timestamp_us = int(round(float(row[time_index]) * 1_000_000))
            order_id = row[order_id_index]

            if operation == insert_operation:
                yield MarketEvent(
                    timestamp_us=timestamp_us,
                    operation=MarketOperation.INSERT,
                    order_id=order_id,
                    side=_SIDE_BY_VALUE.get(row[side_index]) or Side(row[side_index]),
                    volume=int(row[volume_index]),
                    price=int(row[price_index]),
                    order_type=_IOC if lifespan_index is not None and row[lifespan_index] == 'F' else _LIMIT,
                )
            elif operation == cancel_operation:
                yield MarketEvent(
                    timestamp_us=timestamp_us,
                    operation=MarketOperation.CANCEL,
                    order_id=order_id,
                )
            else:
                raise ValueError(f'Unsupported operation: {operation}')

    def run(self, backtest_csv: Path) -> SimulationResult:
        """Replay `backtest_csv` through the strategy and return the result, writing the logs on the way."""
        self._logger.open()
        self._delta_logger.open()
        self._event_logger.open()
        self._trade_logger.open()
        self._log(
            level=_INFO,
            kind='run_started',
            payload={
                'backtest_csv': str(backtest_csv),
                'config': self.config.model_dump(mode='json'),
            },
        )
        event_count = self.count_backtest_rows(backtest_csv)
        self._log(
            level=_DEBUG,
            kind='backtest_loaded',
            payload={'events': event_count},
        )

        progress = ProgressBar(
            total=event_count,
            enabled=self.config.show_progress,
            fd=self.config.progress_fd,
        )
        try:
            logger = self._logger
            for index, event in enumerate(self.iter_backtest(backtest_csv), start=1):
                self._last_event_us = event.timestamp_us
                if self._pending:
                    self._activate_pending_orders_up_to(event.timestamp_us)

                self._advance_time(event.timestamp_us)
                if logger.debug_enabled:
                    self._log(
                        level=_DEBUG,
                        kind='market_event',
                        payload=event.model_dump(mode='json'),
                    )

                self._process_market_event(event)
                self._emit_public_snapshot()
                progress.update(index)

            self._activate_pending_orders_up_to(None)

            mark_price = self._mark_price()
            pnl = self._mark_to_market_pnl()

            result = SimulationResult(
                fills=len(self._filled_order_ids),
                traded_volume=self._user_traded_volume,
                inventory=self._inventory,
                cash=self._cash,
                mark_price=mark_price,
                breach_penalty=self._breach_penalty,
                pnl=pnl,
                pnl_se=self._pnl_standard_error(),
                abs_inventory_q50=self._abs_inventory_quantile(0.5),
                abs_inventory_q68=self._abs_inventory_quantile(0.68),
                abs_inventory_q95=self._abs_inventory_quantile(0.95),
                abs_inventory_q997=self._abs_inventory_quantile(0.997),
            )
            self._log(
                level=_INFO,
                kind='run_completed',
                payload=result.model_dump(mode='json'),
            )
            return result
        finally:
            progress.close()
            self._logger.close()
            self._delta_logger.close()
            self._event_logger.close()
            self._trade_logger.close()

    def submit_order(self, request: OrderRequest) -> str:
        """Queue a strategy order for activation after the accept delay and return its id."""
        self._user_order_seq += 1
        order_id = f'u{self._user_order_seq:09d}'
        ready_at_us = self.current_ts_us + self._accept_delay_us
        if self._log_run:
            self._logger.log_json(
                level=_INFO,
                timestamp_us=self.current_ts_us,
                kind='strategy_order_submitted',
                payload_json=(
                    f'{{"order_id":"{order_id}","request":{{"side":"{request.side.value}",'
                    f'"price":{_json_scalar(request.price)},"quantity":{_json_scalar(request.quantity)},'
                    f'"order_type":"{request.order_type.value}",'
                    f'"client_order_id":{_json_scalar(request.client_order_id)}}},"ready_at_us":{ready_at_us}}}'
                ),
            )
        if self._log_events:
            self._event_logger.log(
                self.current_ts_us,
                'user',
                'add_sent',
                order_id,
                request.side.value,
                str(request.price),
                str(request.quantity),
            )
        self._pending.append(
            PendingUserOrder(
                order_id=order_id,
                request=request,
                ready_at_us=ready_at_us,
            )
        )
        return order_id

    def cancel_order(self, order_id: str) -> None:
        """Queue a strategy cancel for activation after the accept delay."""
        ready_at_us = self.current_ts_us + self._accept_delay_us
        if self._log_run:
            self._logger.log_json(
                level=_INFO,
                timestamp_us=self.current_ts_us,
                kind='strategy_cancel_submitted',
                payload_json=f'{{"order_id":{_json_scalar(order_id)},"ready_at_us":{ready_at_us}}}',
            )
        if self._log_events:
            self._event_logger.log(self.current_ts_us, 'user', 'cancel_sent', order_id)
        self._pending.append(
            PendingUserCancel(
                order_id=order_id,
                ready_at_us=ready_at_us,
            )
        )

    def modify_order(self, order_id: str, new_quantity: int) -> None:
        """Queue a strategy size reduction for activation after the accept delay."""
        ready_at_us = self.current_ts_us + self._accept_delay_us
        if self._log_run:
            self._logger.log_json(
                level=_INFO,
                timestamp_us=self.current_ts_us,
                kind='strategy_modify_submitted',
                payload_json=(
                    f'{{"order_id":{_json_scalar(order_id)},"new_quantity":{_json_scalar(new_quantity)},'
                    f'"ready_at_us":{ready_at_us}}}'
                ),
            )
        if self._log_events:
            self._event_logger.log(
                self.current_ts_us,
                'user',
                'modify_sent',
                order_id,
                quantity=str(new_quantity),
            )
        self._pending.append(
            PendingUserModify(
                order_id=order_id,
                new_quantity=new_quantity,
                ready_at_us=ready_at_us,
            )
        )

    def _process_market_event(self, event: MarketEvent) -> None:
        """Apply one backtest row to the book: a market insert crosses then rests, a cancel removes."""
        if event.operation == MarketOperation.INSERT:
            assert event.side is not None and event.price is not None and event.volume is not None

            order_type = event.order_type
            assert order_type is not None
            order = LiveOrder(
                order_id=event.order_id,
                owner='market',
                side=event.side,
                price=event.price,
                initial_quantity=event.volume,
                remaining_quantity=event.volume,
                order_type=order_type,
                created_at_us=event.timestamp_us,
            )

            self._execute_taker(order)

            if order.order_type == _LIMIT and order.remaining_quantity > 0:
                self._rest(order)
            return

        if event.operation == MarketOperation.CANCEL:
            self._remove(event.order_id)
            return

        raise ValueError(f'Unsupported operation: {event.operation}')

    def _activate_pending_orders_up_to(self, timestamp_us: int | None) -> None:
        """Activate every pending item due by `timestamp_us` (all of them when None), in ready order."""
        pending = sorted(self._pending, key=lambda item: item.ready_at_us)
        self._pending = []

        for item in pending:
            if timestamp_us is not None and item.ready_at_us > timestamp_us:
                self._pending.append(item)
                continue

            self._advance_time(max(self.current_ts_us, item.ready_at_us))

            if isinstance(item, PendingUserOrder):
                self._activate_order(item)
            elif isinstance(item, PendingUserCancel):
                self._activate_cancel(item)
            else:
                self._activate_modify(item)

    def _activate_order(self, item: PendingUserOrder) -> None:
        """Accept or reject a strategy order at the exchange, then cross and rest it."""
        request = item.request

        if self._exceeds_position_limit(request.side, request.quantity):
            self._log(
                level=_DEBUG,
                kind='position_limit_exceeded',
                payload={
                    'order_id': item.order_id,
                    'side': request.side.value,
                    'quantity': request.quantity,
                    'current_position': self._inventory,
                    'max_position': self.config.max_position,
                },
            )
            if self._log_events:
                self._event_logger.log(
                    self.current_ts_us,
                    'user',
                    'add_failed',
                    item.order_id,
                    request.side.value,
                    str(request.price),
                    str(request.quantity),
                )
            self._private_notify(
                PrivateOrderEvent(
                    timestamp_us=self.current_ts_us,
                    order_id=item.order_id,
                    status=OrderStatus.REJECTED,
                )
            )
            return

        order = LiveOrder(
            order_id=item.order_id,
            owner='user',
            side=request.side,
            price=request.price,
            initial_quantity=request.quantity,
            remaining_quantity=request.quantity,
            order_type=request.order_type,
            created_at_us=item.ready_at_us,
            client_order_id=request.client_order_id,
        )

        self._private_notify(
            PrivateOrderEvent(
                timestamp_us=self.current_ts_us,
                order_id=order.order_id,
                status=OrderStatus.ACCEPTED,
                remaining_quantity=order.remaining_quantity,
            )
        )

        self._execute_taker(order)

        if order.order_type == _IOC or order.remaining_quantity == 0:
            return

        self._rest(order)

    def _activate_cancel(self, item: PendingUserCancel) -> None:
        """Remove the strategy's resting order, or reject the cancel if there is none with that id."""
        order = self.book.get(item.order_id)
        if order is None or order.owner != 'user':
            if self._log_events:
                self._event_logger.log(self.current_ts_us, 'user', 'cancel_failed', item.order_id)
            self._private_notify(
                PrivateOrderEvent(
                    timestamp_us=self.current_ts_us,
                    order_id=item.order_id,
                    status=OrderStatus.REJECTED,
                    message='cancel rejected: no resting order with this id',
                )
            )
            return

        removed = self._remove(item.order_id)
        assert removed is order
        self._private_notify(
            PrivateOrderEvent(
                timestamp_us=self.current_ts_us,
                order_id=item.order_id,
                status=OrderStatus.CANCELED,
                remaining_quantity=removed.remaining_quantity,
            )
        )

    def _rest(self, order: LiveOrder) -> None:
        """Put `order` in the book and log the book delta and the add event."""
        self.book.insert(order)
        if self._log_deltas:
            self._delta_logger.log(self.current_ts_us, order.side, order.price, order.owner, order.remaining_quantity)
        if self._log_events:
            self._event_logger.log(
                self.current_ts_us,
                order.owner,
                'add',
                order.order_id,
                order.side.value,
                str(order.price),
                str(order.initial_quantity),
            )

    def _remove(self, order_id: str) -> LiveOrder | None:
        """Take the order out of the book and log the delta and the cancel event; None if it is not resting."""
        removed = self.book.remove(order_id)
        if removed is None:
            return None
        if self._log_deltas:
            self._delta_logger.log(
                self.current_ts_us, removed.side, removed.price, removed.owner, -removed.remaining_quantity
            )
        if self._log_events:
            self._event_logger.log(
                self.current_ts_us,
                removed.owner,
                'cancel',
                removed.order_id,
                removed.side.value,
                str(removed.price),
                str(removed.remaining_quantity),
            )
        return removed

    def _activate_modify(self, item: PendingUserModify) -> None:
        """Shrink the strategy's resting order to `new_quantity`, or reject the modify."""
        order = self.book.get(item.order_id)
        if order is None or order.owner != 'user':
            if self._log_events:
                self._event_logger.log(self.current_ts_us, 'user', 'modify_failed', item.order_id)
            return

        if not (0 < item.new_quantity < order.remaining_quantity):
            if self._log_events:
                self._event_logger.log(self.current_ts_us, 'user', 'modify_failed', item.order_id)
            self._private_notify(
                PrivateOrderEvent(
                    timestamp_us=self.current_ts_us,
                    order_id=item.order_id,
                    status=OrderStatus.REJECTED,
                    remaining_quantity=order.remaining_quantity,
                )
            )
            return

        old_quantity = order.remaining_quantity
        self.book.reduce_quantity(order, old_quantity - item.new_quantity)
        order.initial_quantity = item.new_quantity

        if self._log_deltas:
            self._delta_logger.log(
                self.current_ts_us,
                order.side,
                order.price,
                order.owner,
                item.new_quantity - old_quantity,
            )
        if self._log_events:
            self._event_logger.log(
                self.current_ts_us,
                'user',
                'modify',
                order.order_id,
                order.side.value,
                str(order.price),
                str(item.new_quantity),
            )
        self._private_notify(
            PrivateOrderEvent(
                timestamp_us=self.current_ts_us,
                order_id=order.order_id,
                status=OrderStatus.MODIFIED,
                remaining_quantity=order.remaining_quantity,
            )
        )

    def _execute_taker(self, taker: LiveOrder) -> None:
        """Match `taker` against the opposite side of the book, best price first, until it stops crossing."""
        while taker.remaining_quantity > 0:
            maker = self.book.best_opposite_order(
                taker.side,
                allow_user_orders=taker.owner == 'market',
            )
            if maker is None:
                break

            crosses = taker.price >= maker.price if taker.side == _BUY else taker.price <= maker.price
            if not crosses:
                break

            traded_quantity = min(taker.remaining_quantity, maker.remaining_quantity)
            taker.remaining_quantity -= traded_quantity
            self.book.reduce_quantity(maker, traded_quantity)
            if self._log_deltas:
                self._delta_logger.log(self.current_ts_us, maker.side, maker.price, maker.owner, -traded_quantity)

            trade = Trade(
                timestamp_us=self.current_ts_us,
                price=maker.price,
                quantity=traded_quantity,
                taker_order_id=taker.order_id,
                maker_order_id=maker.order_id,
                taker_owner=taker.owner,
                maker_owner=maker.owner,
                book_side='bid' if maker.side == _BUY else 'ask',
            )
            self._trades.append(trade)
            if taker.owner == 'user' or maker.owner == 'user':
                self._user_traded_volume += traded_quantity
            if self._log_trades:
                self._trade_logger.log(trade)
            self._last_trade_price = float(trade.price)
            trade_level = _INFO if taker.owner == 'user' or maker.owner == 'user' else _DEBUG
            if self._logger.should_log(trade_level):
                self._logger.log_json(
                    level=trade_level,
                    timestamp_us=self.current_ts_us,
                    kind='trade',
                    payload_json=_trade_json(trade),
                )

            for participant in (taker, maker):
                if participant.owner != 'user':
                    continue
                sign = 1 if participant.side == _BUY else -1
                self._inventory += sign * traded_quantity
                self._cash -= sign * traded_quantity * trade.price

            if maker.owner == 'user':
                self._notify_fill(maker, traded_quantity, trade.price)
            if taker.owner == 'user':
                self._notify_fill(taker, traded_quantity, trade.price)

            if maker.remaining_quantity == 0:
                self.book.remove(maker.order_id)

        if taker.owner == 'user' and taker.order_type == _IOC and taker.remaining_quantity > 0:
            self._private_notify(
                PrivateOrderEvent(
                    timestamp_us=self.current_ts_us,
                    order_id=taker.order_id,
                    status=OrderStatus.CANCELED,
                    remaining_quantity=taker.remaining_quantity,
                )
            )

    def _notify_fill(self, order: LiveOrder, quantity: int, price: int) -> None:
        """Tell the strategy that `quantity` of its `order` just traded at `price`."""
        self._private_notify(
            PrivateOrderEvent(
                timestamp_us=self.current_ts_us,
                order_id=order.order_id,
                status=OrderStatus.FILLED if order.remaining_quantity == 0 else OrderStatus.PARTIALLY_FILLED,
                filled_quantity=quantity,
                remaining_quantity=order.remaining_quantity,
                fill_price=price,
            )
        )

    def _private_notify(self, event: PrivateOrderEvent) -> None:
        """Count `event` towards the run's fills and deliver it to the strategy's private feed."""
        if event.status in _FILL_STATUSES:
            self._filled_order_ids.add(event.order_id)
        self._log(
            level=_DEBUG,
            kind='private_order_event',
            payload=lambda: event.model_dump(mode='json'),
        )
        self.strategy.on_private_order_update(event)

    def _visible_orders(self, side: Side) -> list[BookOrderView]:
        """Return the public-feed views within `public_depth` ticks on `side`.

        The per-side list is cached until the book's version for that side
        changes, so an event that only touched one side rebuilds nothing for
        the other.
        """
        book = self.book
        version = book.bid_version if side == _BUY else book.ask_version
        cached_version, cached_views = self._visible_cache[side]
        if cached_version == version:
            return cached_views
        views = book.visible_views(side, self.config.public_depth)
        self._visible_cache[side] = (version, views)
        return views

    def _emit_public_snapshot(self) -> None:
        """Deliver the visible book and the trades since the previous call to the strategy."""
        snapshot = OrderBookSnapshot(
            timestamp_us=self.current_ts_us,
            bids=list(self._visible_orders(_BUY)),
            asks=list(self._visible_orders(_SELL)),
        )
        trades = self._trades
        self._trades = []
        if self._logger.debug_enabled:
            self._log(
                level=_DEBUG,
                kind='public_snapshot',
                payload=snapshot.model_dump(mode='json'),
            )
        self.strategy.on_order_book_change(snapshot, trades)

    def _exceeds_position_limit(self, side: Side, quantity: int) -> bool:
        """Return whether an order's worst-case fill would breach the (potential) position limit."""
        limit = self.config.max_potential_position
        if limit is None:
            limit = self.config.max_position
        if limit is None:
            return False
        sign = 1 if side == _BUY else -1
        pending_same_side = self.book.resting_user_quantity(side)
        worst_case = self._inventory + sign * (pending_same_side + quantity)
        return abs(worst_case) > limit

    def _advance_time(self, timestamp_us: int) -> None:
        """Move current time forward to `timestamp_us`, accruing the penalty and PnL samples for the gap."""
        if self._next_pnl_sample_us is None:
            self.current_ts_us = timestamp_us
            self._pnl_sample_times_us.append(timestamp_us)
            self._pnl_samples.append(self._mark_to_market_pnl())
            self._next_pnl_sample_us = timestamp_us + PNL_SAMPLE_INTERVAL_US
            return
        if timestamp_us > self.current_ts_us:
            self._abs_inventory_time_us[abs(self._inventory)] += timestamp_us - self.current_ts_us
            while self._next_pnl_sample_us <= timestamp_us:
                self._accrue_breach_penalty(self._next_pnl_sample_us)
                self._pnl_sample_times_us.append(self._next_pnl_sample_us)
                self._pnl_samples.append(self._mark_to_market_pnl())
                self._next_pnl_sample_us += PNL_SAMPLE_INTERVAL_US
            self._accrue_breach_penalty(timestamp_us)

    def _accrue_breach_penalty(self, timestamp_us: int) -> None:
        """Move current time forward to `timestamp_us`, accruing any position-breach penalty for the gap."""
        limit = self._max_position
        if limit is not None and abs(self._inventory) > limit:
            elapsed_s = (timestamp_us - self.current_ts_us) / 1_000_000
            self._breach_penalty += elapsed_s * self._penalty_per_second
        self.current_ts_us = timestamp_us

    def pnl_curve(self, final_pnl: float) -> list[tuple[int, float]]:
        """Return the run's (timestamp_us, PnL) curve for comparing runs on one backtest.

        The 1s samples before the last market event, then `final_pnl` at that event's
        time. Orders still pending after the last event activate up to the accept
        delay past it, which moves `current_ts_us` and can add a sample there; cutting
        the curve at the last event instead gives every run on the same backtest
        exactly the same timestamps, so their curves line up point for point.
        """
        end_us = self._last_event_us
        if end_us is None:
            return [(self.current_ts_us, final_pnl)]
        samples = zip(self._pnl_sample_times_us, self._pnl_samples, strict=True)
        return [*((ts_us, pnl) for ts_us, pnl in samples if ts_us < end_us), (end_us, final_pnl)]

    def _mark_to_market_pnl(self) -> float:
        """Return cash plus the marked-out inventory, net of the breach penalty."""
        return self._cash + self._mark_out_value() - self._breach_penalty

    def _pnl_standard_error(self) -> float | None:
        """Return stdev * sqrt(count) of the whole-interval PnL increments."""
        increments = [later - earlier for earlier, later in itertools.pairwise(self._pnl_samples)]
        if len(increments) < 2:
            return None
        return statistics.stdev(increments) * math.sqrt(len(increments))

    def _abs_inventory_quantile(self, quantile: float) -> int:
        """Return the time-weighted `quantile` of abs(inventory)."""
        total_us = sum(self._abs_inventory_time_us.values())
        held_us = 0
        for level in sorted(self._abs_inventory_time_us):
            held_us += self._abs_inventory_time_us[level]
            if held_us >= quantile * total_us:
                return level
        return 0

    def _mark_out_value(self) -> float:
        """Return the cash flow of IOC-liquidating the inventory against the book.

        The sweep walks resting market orders on the opposite side from the
        best price outwards. Inventory beyond the swept volume is marked at the
        last sweep fill, or at the last trade if there is nothing to sweep.
        """
        if self._inventory == 0:
            return 0.0
        assert self._last_trade_price is not None
        remaining = abs(self._inventory)
        last_price = self._last_trade_price
        value = 0.0
        for order in self.book.iter_levels(_BUY if self._inventory > 0 else _SELL):
            if order.owner != 'market':
                continue
            filled = min(remaining, order.remaining_quantity)
            value += filled * order.price
            last_price = float(order.price)
            remaining -= filled
            if remaining == 0:
                break
        value += remaining * last_price
        return value if self._inventory > 0 else -value

    def _mark_price(self) -> float:
        """Return the average mark-out price, or the last trade price when flat."""
        if self._inventory == 0:
            return self._last_trade_price or 0.0
        return self._mark_out_value() / self._inventory
