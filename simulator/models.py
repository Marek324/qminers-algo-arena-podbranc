"""Domain types shared by the engine, the strategies and the logs.

`OrderRequest`, `PrivateOrderEvent`, `Trade` and `SimulationResult` are
pydantic models. The hot-path types `LiveOrder`, `MarketEvent`,
`BookOrderView` and `OrderBookSnapshot` are slots dataclasses for speed; the
last three keep only a `model_dump(mode)` shim of the pydantic API.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Side(str, Enum):
    """Order side, with the values used by the backtest CSV."""

    BUY = 'B'
    SELL = 'A'

    @property
    def opposite(self) -> Side:
        """Return the other side."""
        return Side.SELL if self == Side.BUY else Side.BUY


class OrderType(str, Enum):
    """Whether an unfilled order rests in the book (LIMIT) or is cancelled (IOC)."""

    LIMIT = 'limit'
    IOC = 'ioc'


class MarketOperation(str, Enum):
    """Operation of a backtest CSV row."""

    INSERT = 'Insert'
    CANCEL = 'Cancel'


class OrderStatus(str, Enum):
    """State transition reported by a `PrivateOrderEvent`."""

    ACCEPTED = 'accepted'
    PARTIALLY_FILLED = 'partially_filled'
    FILLED = 'filled'
    MODIFIED = 'modified'
    CANCELED = 'canceled'
    REJECTED = 'rejected'


class OrderRequest(BaseModel):
    """An order a strategy submits with `BaseStrategy.send_order`."""

    model_config = ConfigDict(frozen=True)

    side: Side
    price: int = Field(gt=0)
    quantity: int = Field(gt=0)
    order_type: OrderType
    client_order_id: str | None = None


@dataclass(slots=True, eq=False)
class LiveOrder:
    """A resting or in-flight order as tracked by the engine.

    A slots dataclass rather than a pydantic model: one is built per market
    event and per user order, and its fields come from already-validated
    `MarketEvent` / `OrderRequest` values. `eq=False` keeps identity
    comparison, which the book relies on to remove an order from its queue.
    """

    order_id: str
    owner: Literal['market', 'user']
    side: Side
    price: int
    initial_quantity: int
    remaining_quantity: int
    order_type: OrderType
    created_at_us: int
    client_order_id: str | None = None
    view: BookOrderView | None = None

    def __post_init__(self) -> None:
        if self.price <= 0 or self.initial_quantity <= 0 or self.remaining_quantity < 0 or self.created_at_us < 0:
            raise ValueError(f'Invalid LiveOrder fields: {self!r}')


@dataclass(slots=True)
class MarketEvent:
    """One row of the backtest CSV.

    A slots dataclass with explicit range checks instead of a pydantic model:
    one is built per CSV row and the per-row validation overhead dominated
    `iter_backtest`.
    """

    timestamp_us: int
    operation: MarketOperation
    order_id: str
    side: Side | None = None
    volume: int | None = None
    price: int | None = None
    order_type: OrderType | None = None

    def __post_init__(self) -> None:
        if (
            self.timestamp_us < 0
            or (self.volume is not None and self.volume <= 0)
            or (self.price is not None and self.price <= 0)
        ):
            raise ValueError(f'Invalid MarketEvent fields: {self!r}')

    def model_dump(self, mode: str = 'python') -> dict[str, Any]:
        """Return the fields as a dict, with enum values when `mode` is 'json'.

        The order type is reported as the CSV's `lifespan`, so the run log
        keeps the backtest's own vocabulary.
        """
        as_json = mode == 'json'
        return {
            'timestamp_us': self.timestamp_us,
            'operation': self.operation.value if as_json else self.operation,
            'order_id': self.order_id,
            'side': (self.side.value if as_json else self.side) if self.side is not None else None,
            'volume': self.volume,
            'price': self.price,
            'lifespan': None if self.order_type is None else ('F' if self.order_type is OrderType.IOC else 'G'),
        }


@dataclass(slots=True, frozen=True)
class BookOrderView:
    """One resting order as shown on the public feed.

    A plain dataclass rather than a pydantic model: the simulator builds up to
    `public_depth` of these per side on *every* market event, and re-validating
    fields it has just copied out of its own `LiveOrder` records dominated the
    replay cost. `slots=True` additionally drops the per-instance `__dict__`.
    Frozen because the engine reuses one view across consecutive snapshots
    while the underlying order is unchanged.
    """

    order_id: str
    side: Side
    price: int
    quantity: int
    owner: Literal['market', 'user']

    def model_dump(self, mode: str = 'python') -> dict[str, Any]:
        """Return the fields as a dict, with enum values when `mode` is 'json'."""
        return {
            'order_id': self.order_id,
            'side': self.side.value if mode == 'json' else self.side,
            'price': self.price,
            'quantity': self.quantity,
            'owner': self.owner,
        }


@dataclass(slots=True)
class OrderBookSnapshot:
    """The public book at a point in time, best price first on each side.

    See `BookOrderView` for why this is a dataclass rather than a pydantic model.
    """

    timestamp_us: int
    bids: list[BookOrderView]
    asks: list[BookOrderView]

    def model_dump(self, mode: str = 'python') -> dict[str, Any]:
        """Return the fields as a dict, with enum values when `mode` is 'json'."""
        return {
            'timestamp_us': self.timestamp_us,
            'bids': [bid.model_dump(mode) for bid in self.bids],
            'asks': [ask.model_dump(mode) for ask in self.asks],
        }


class PrivateOrderEvent(BaseModel):
    """A state change of one of the strategy's own orders.

    Fills carry `fill_price` and the per-match `filled_quantity`; rejects may
    carry a `message`.
    """

    timestamp_us: int
    order_id: str
    status: OrderStatus
    filled_quantity: int = 0
    remaining_quantity: int = 0
    fill_price: int | None = None
    message: str | None = None


class Trade(BaseModel):
    """One match between a taker and a resting maker order.

    `book_side` is the side of the book the maker rested on.
    """

    timestamp_us: int
    price: int
    quantity: int
    taker_order_id: str
    maker_order_id: str
    taker_owner: Literal['market', 'user']
    maker_owner: Literal['market', 'user']
    book_side: Literal['bid', 'ask']


class SimulationResult(BaseModel):
    """Outcome of a replay, written to `result.json`."""

    fills: int
    traded_volume: int
    inventory: int
    cash: float
    mark_price: float
    breach_penalty: float
    pnl: float
    pnl_se: float | None
    abs_inventory_q50: int
    abs_inventory_q68: int
    abs_inventory_q95: int
    abs_inventory_q997: int
