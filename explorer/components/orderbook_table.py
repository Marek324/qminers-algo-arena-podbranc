"""Price-ladder table of the order book at the selected time."""

from typing import Any

import pandas as pd
from dash import dash_table

LADDER_DEPTH = 14
LADDER_DIVIDER = {'bid_user': '─', 'bid_mkt': '─', 'price': '─', 'ask_mkt': '─', 'ask_user': '─'}


def build_orderbook_table(initial_rows: list[dict[str, Any]]) -> dash_table.DataTable:
    """Return a price-ladder DataTable split into bid/ask and user/market columns.

    `initial_rows` are shaped like `format_orderbook`'s output: dicts with the
    keys bid_user, bid_mkt, price, ask_mkt, ask_user.
    """
    return dash_table.DataTable(
        id='ob-table',
        columns=[
            {'name': 'Bid User', 'id': 'bid_user'},
            {'name': 'Bid Mkt', 'id': 'bid_mkt'},
            {'name': 'Price', 'id': 'price'},
            {'name': 'Ask Mkt', 'id': 'ask_mkt'},
            {'name': 'Ask User', 'id': 'ask_user'},
        ],
        data=initial_rows,
        page_action='none',
        virtualization=True,
        fixed_rows={'headers': True},
        style_table={
            'height': '100%',
            'overflowY': 'auto',
            'backgroundColor': 'transparent',
        },
        style_header={
            'backgroundColor': '#1e1e2f',
            'color': '#ffffff',
            'fontWeight': 'bold',
            'border': 'none',
            'textAlign': 'center',
            'fontFamily': 'monospace',
            'fontSize': '12px',
        },
        style_cell={
            'backgroundColor': '#2a2a3c',
            'color': '#ffffff',
            'border': 'none',
            'textAlign': 'center',
            'padding': '4px 8px',
            'fontFamily': 'monospace',
            'fontSize': '12px',
        },
        style_data_conditional=[
            {
                'if': {'filter_query': '{ask_mkt} != ""', 'column_id': 'ask_mkt'},
                'color': '#ff8899',
            },
            {
                'if': {'filter_query': '{ask_user} != ""', 'column_id': 'ask_user'},
                'color': '#ffaaaa',
            },
            {
                'if': {'filter_query': '{bid_mkt} != ""', 'column_id': 'bid_mkt'},
                'color': '#66dd99',
            },
            {
                'if': {'filter_query': '{bid_user} != ""', 'column_id': 'bid_user'},
                'color': '#aaffcc',
            },
            {
                'if': {'filter_query': '{price} = "─"'},
                'backgroundColor': '#1a2030',
                'color': '#445566',
            },
            {
                'if': {'state': 'selected'},
                'backgroundColor': '#46465c',
                'border': '1px solid #6a6a85',
            },
        ],
    )


def format_orderbook(book: pd.DataFrame, depth: int = LADDER_DEPTH) -> list[dict[str, Any]]:
    """Convert a book snapshot into price-ladder rows: asks top-down, a divider, then bids.

    Args:
        book: Rows of side, price, owner, quantity, as `data.orderbook.get_book_at` returns.
        depth: Price levels shown per side.

    Returns:
        Row dicts with the keys bid_user, bid_mkt, price, ask_mkt, ask_user.
    """
    if book.empty:
        return [
            {
                'bid_user': '',
                'bid_mkt': '',
                'price': '(empty)',
                'ask_mkt': '',
                'ask_user': '',
            }
        ]

    pivot = book.pivot_table(
        index=['side', 'price'],
        columns='owner',
        values='quantity',
        aggfunc='sum',
        fill_value=0,
    ).reset_index()
    for owner in ('market', 'user'):
        if owner not in pivot.columns:
            pivot[owner] = 0

    rows: list[dict[str, Any]] = []

    asks = (
        pivot[pivot['side'] == 'A']
        .sort_values('price', ascending=True)
        .head(depth)
        .sort_values('price', ascending=False)
    )
    for row in asks.itertuples(index=False):
        rows.append(
            {
                'bid_user': '',
                'bid_mkt': '',
                'price': int(row.price),
                'ask_mkt': int(row.market) if row.market > 0 else '',
                'ask_user': int(row.user) if row.user > 0 else '',
            }
        )

    rows.append(dict(LADDER_DIVIDER))

    bids = pivot[pivot['side'] == 'B'].sort_values('price', ascending=False).head(depth)
    for row in bids.itertuples(index=False):
        rows.append(
            {
                'bid_user': int(row.user) if row.user > 0 else '',
                'bid_mkt': int(row.market) if row.market > 0 else '',
                'price': int(row.price),
                'ask_mkt': '',
                'ask_user': '',
            }
        )

    return rows
