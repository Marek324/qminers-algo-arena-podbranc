"""Paginated table of the event log."""

from typing import Any

import pandas as pd
from dash import dash_table
from dash.dash_table.Format import Format, Group

COLUMN_LABELS = {
    'timestamp_us': 'Time (us)',
    'source': 'Source',
    'operation': 'Operation',
    'order_id': 'Order ID',
    'side': 'Side',
    'price': 'Price',
    'quantity': 'Quantity',
    'maker_order_id': 'Maker ID',
    'maker_owner': 'Maker',
}


def page_records(events: pd.DataFrame, page: int, page_size: int) -> list[dict[str, Any]]:
    """Return one page's rows as DataTable records, with absolute-index ids."""
    start = page * page_size
    page_frame = events.iloc[start : start + page_size]
    records = page_frame.to_dict('records')
    for offset, row in enumerate(records):
        row['id'] = start + offset
    return records


def build_event_log(events: pd.DataFrame, page_size: int) -> dash_table.DataTable:
    """Return a paginated DataTable listing the events.

    Expects the columns of `data.loader.merge_trades_into_events`; the last two
    (`maker_order_id`, `maker_owner`) are empty except on 'trade' rows.
    """
    return dash_table.DataTable(
        id='event-log-table',
        columns=[
            {
                'name': COLUMN_LABELS[str(column)],
                'id': str(column),
                **({'type': 'numeric', 'format': Format(group=Group.yes)} if column == 'timestamp_us' else {}),
            }
            for column in events.columns
        ],
        data=page_records(events, 0, page_size),
        page_action='custom',
        page_current=0,
        page_size=page_size,
        page_count=max(1, -(-len(events) // page_size)),
        style_table={'width': '100%', 'tableLayout': 'fixed'},
        style_cell={
            'backgroundColor': '#161b22',
            'color': '#c9d1d9',
            'border': '1px solid #30363d',
            'textAlign': 'left',
            'padding': '5px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'minWidth': '60px',
            'width': '60px',
            'maxWidth': '120px',
        },
        style_header={
            'backgroundColor': '#21262d',
            'fontWeight': 'bold',
            'borderBottom': '2px solid #30363d',
        },
    )
