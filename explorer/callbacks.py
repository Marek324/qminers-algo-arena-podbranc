"""Dash callbacks: chart zoom synchronisation and the time selection shared by every panel."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import pandas as pd
from dash import Dash, Input, Output, Patch, callback_context, no_update
from plotly_resampler import FigureResampler

from explorer.components.event_tables import page_records
from explorer.components.orderbook_table import format_orderbook
from explorer.data.clock import elapsed_seconds, to_timestamp_us
from explorer.data.orderbook import get_book_at

_Y_RANGE_PAD_FRAC = 0.05

_XYSeries = list[tuple[Any, Any]]

_FILTER_IDS = ('filter-source', 'filter-side', 'filter-op')

_ACTIVE_CELL_STYLE = {
    'if': {'state': 'active'},
    'backgroundColor': '#161b22',
    'border': '1px solid #30363d',
    'color': '#c9d1d9',
}


@dataclass
class _SelectionUpdate:
    """The selection callback's outputs, in the order they are declared; a field left at no_update keeps its value."""

    spread_figure: Any = no_update
    pnl_figure: Any = no_update
    page_current: Any = no_update
    style_data_conditional: Any = no_update
    active_cell: Any = no_update
    selected_cells: Any = no_update
    book_rows: Any = no_update
    page_rows: Any = no_update
    page_count: Any = no_update

    def as_tuple(self) -> tuple[Any, ...]:
        """Return the outputs in declaration order, as the callback returns them."""
        return tuple(getattr(self, field.name) for field in fields(self))


def _cleared_selection() -> _SelectionUpdate:
    """Return an update that drops the table's highlighted row and selection."""
    return _SelectionUpdate(style_data_conditional=[_ACTIVE_CELL_STYLE], active_cell=None, selected_cells=[])


def _select(deltas: pd.DataFrame, timestamp_us: int, crosshair_s: float, relative_row: int) -> _SelectionUpdate:
    """Return the update that selects one moment: a crosshair on both charts, the highlighted table row, the book.

    Args:
        deltas: The book_deltas.csv frame, for `data.orderbook.get_book_at`.
        timestamp_us: The selected log timestamp, which the book is rebuilt for.
        crosshair_s: Where to draw the crosshair, in chart seconds.
        relative_row: The selected row's index within the table's current page.
    """
    update = _cleared_selection()
    update.style_data_conditional = [
        _ACTIVE_CELL_STYLE,
        {
            'if': {'row_index': relative_row},
            'backgroundColor': '#2d333b',
            'borderTop': '2px solid #58a6ff',
            'borderBottom': '2px solid #58a6ff',
            'color': '#ffffff',
        },
    ]
    crosshair = dict(
        type='line',
        x0=crosshair_s,
        x1=crosshair_s,
        y0=0,
        y1=1,
        yref='paper',
        line=dict(color='white', width=1, dash='dash'),
    )
    update.spread_figure, update.pnl_figure = Patch(), Patch()
    update.spread_figure['layout']['shapes'] = [crosshair]
    update.pnl_figure['layout']['shapes'] = [crosshair]
    update.book_rows = format_orderbook(get_book_at(deltas, timestamp_us))
    return update


def _y_range_for_window(series: _XYSeries, x_min: float, x_max: float) -> list[float] | None:
    """Return a padded [min, max] covering every y of `series` visible in [x_min, x_max].

    That is every sample inside the window plus, per pair, the last one
    before it: the step lines hold that value into the window, and the PnL
    curve has only one sample a second, so a short window may hold nothing
    else. The bounds are combined across all the pairs, so overlaid traces
    (best bid/ask, trade prices) all stay in view. Each `x` must be sorted.
    """
    windows = []
    for x, y in series:
        mask = ((x >= x_min) & (x <= x_max)).to_numpy(copy=True)
        last_before = int(x.searchsorted(x_min)) - 1
        if last_before >= 0:
            mask[last_before] = True
        if mask.any():
            windows.append(y[mask])
    if not windows:
        return None
    combined = pd.concat(windows).dropna()
    if combined.empty:
        return None
    low, high = combined.min(), combined.max()
    pad = (high - low) * _Y_RANGE_PAD_FRAC if high > low else (abs(low) * _Y_RANGE_PAD_FRAC or 1.0)
    return [low - pad, high + pad]


def _extract_x_range(relayout: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return the new x bounds of a relayoutData dict, or (None, None).

    A box zoom sends split 'xaxis.range[0]' / 'xaxis.range[1]' keys, a pan
    sends one 'xaxis.range': [x_min, x_max] key; plotly_resampler's own
    construct_update_data_patch applies the same fallback.
    """
    x_min = relayout.get('xaxis.range[0]')
    x_max = relayout.get('xaxis.range[1]')
    if x_min is not None and x_max is not None:
        return x_min, x_max
    x_range = relayout.get('xaxis.range')
    if isinstance(x_range, (list, tuple)) and len(x_range) == 2:
        return x_range[0], x_range[1]
    return None, None


def _is_reset(relayout: dict[str, Any]) -> bool:
    """Return whether the relayout is an autorange ("Reset axes" or double click) rather than a zoom or pan.

    plotly_resampler also requires a 'showspikes' key to tell a "Reset axes"
    click from a double click, but that key's presence depends on hover state
    and is often absent after a pan. Both cases mean "autorange", and asking
    the resampler to reload data when it is not needed is a safe no-op.
    """
    return any(key.endswith('.autorange') and relayout[key] for key in relayout)


def register_chart_sync_callback(
    app: Dash,
    best_quotes: pd.DataFrame,
    pnl: pd.DataFrame,
    trades: pd.DataFrame,
    spread_fig: FigureResampler,
    pnl_fig: FigureResampler,
) -> None:
    """Keep the spread and PnL charts' axes and resampled data in sync.

    Whichever chart the user zooms, pans or resets, both get the same x range,
    a y range fitted to the data now visible, and reloaded high-frequency
    data. One callback drives both charts because the chart the user did not
    touch never fires its own relayoutData; left to itself it would keep the
    slice it last loaded, so a reset on the other chart would look like a no-op.
    """
    figures: dict[str, FigureResampler] = {
        'spread-chart': spread_fig,
        'pnl-chart': pnl_fig,
    }
    series_by_chart: dict[str, _XYSeries] = {
        'spread-chart': [
            (best_quotes['elapsed_s'], best_quotes['best_bid']),
            (best_quotes['elapsed_s'], best_quotes['best_ask']),
            (trades['elapsed_s'], trades['price']),
        ],
        'pnl-chart': [(pnl['elapsed_s'], pnl['pnl'])],
    }

    @app.callback(
        [
            Output('spread-chart', 'figure', allow_duplicate=True),
            Output('pnl-chart', 'figure', allow_duplicate=True),
        ],
        [
            Input('spread-chart', 'relayoutData'),
            Input('pnl-chart', 'relayoutData'),
        ],
        prevent_initial_call=True,
    )
    def sync_charts(
        spread_relayout: dict[str, Any] | None,
        pnl_relayout: dict[str, Any] | None,
    ) -> tuple[Any, Any]:
        context = callback_context
        if not context.triggered:
            return no_update, no_update

        trigger_id = context.triggered[0]['prop_id']
        triggered_chart = 'spread-chart' if 'spread-chart' in trigger_id else 'pnl-chart'
        relayout = spread_relayout if triggered_chart == 'spread-chart' else pnl_relayout
        if not relayout:
            return no_update, no_update

        reset = _is_reset(relayout)
        x_min, x_max = (None, None) if reset else _extract_x_range(relayout)
        if not reset and (x_min is None or x_max is None):
            return no_update, no_update

        data_relayout = (
            {
                'xaxis.autorange': True,
                'xaxis.showspikes': False,
                'yaxis.autorange': True,
                'yaxis.showspikes': False,
            }
            if reset
            else relayout
        )

        patches = {}
        for chart_id, fig in figures.items():
            patch = fig.construct_update_data_patch(data_relayout)
            patches[chart_id] = patch if patch is not no_update else Patch()

        for chart_id, patch in patches.items():
            if reset:
                patch['layout']['xaxis']['autorange'] = True
                patch['layout']['yaxis']['autorange'] = True
                continue
            assert x_min is not None and x_max is not None
            patch['layout']['xaxis']['range'] = [x_min, x_max]
            patch['layout']['xaxis']['autorange'] = False
            y_range = _y_range_for_window(series_by_chart[chart_id], x_min, x_max)
            if y_range is not None:
                patch['layout']['yaxis']['range'] = y_range
                patch['layout']['yaxis']['autorange'] = False

        return patches['spread-chart'], patches['pnl-chart']


def register_selection_callback(
    app: Dash,
    events: pd.DataFrame,
    deltas: pd.DataFrame,
    page_size: int,
    start_us: int,
) -> None:
    """Register the one callback that reacts to chart clicks, table clicks, paging and the filters.

    Args:
        app: The Dash app.
        events: The event log, sorted by `timestamp_us`.
        deltas: The book_deltas.csv frame, passed on to `data.orderbook.get_book_at`.
        page_size: Rows per page of the event table.
        start_us: The log timestamp at the charts' time zero (see `data.clock`).
    """

    @app.callback(
        [
            Output('spread-chart', 'figure', allow_duplicate=True),
            Output('pnl-chart', 'figure', allow_duplicate=True),
            Output('event-log-table', 'page_current'),
            Output('event-log-table', 'style_data_conditional'),
            Output('event-log-table', 'active_cell'),
            Output('event-log-table', 'selected_cells'),
            Output('ob-table', 'data'),
            Output('event-log-table', 'data'),
            Output('event-log-table', 'page_count'),
        ],
        [
            Input('spread-chart', 'clickData'),
            Input('pnl-chart', 'clickData'),
            Input('event-log-table', 'active_cell'),
            Input('event-log-table', 'page_current'),
            Input('filter-source', 'value'),
            Input('filter-side', 'value'),
            Input('filter-op', 'value'),
        ],
        prevent_initial_call=True,
    )
    def update_selection(
        spread_click: dict[str, Any] | None,
        pnl_click: dict[str, Any] | None,
        active_cell: dict[str, Any] | None,
        page_current: int | None,
        source_filter: str | None,
        side_filter: str | None,
        operation_filter: str | None,
    ) -> tuple[Any, ...]:
        context = callback_context
        if not context.triggered:
            return _SelectionUpdate().as_tuple()

        trigger_id = context.triggered[0]['prop_id']
        page = page_current if page_current else 0

        filtered = events.copy()
        if source_filter and source_filter != 'All':
            filtered = filtered[filtered['source'] == source_filter]
        if side_filter and side_filter != 'All':
            filtered = filtered[filtered['side'] == side_filter]
        if operation_filter and operation_filter != 'All':
            filtered = filtered[filtered['operation'] == operation_filter]

        if any(filter_id in trigger_id for filter_id in _FILTER_IDS):
            update = _cleared_selection()
            update.page_current = 0
            update.page_rows = page_records(filtered, 0, page_size)
            update.page_count = max(1, -(-len(filtered) // page_size))
            return update.as_tuple()

        if len(filtered) == 0:
            return _SelectionUpdate().as_tuple()

        if 'page_current' in trigger_id:
            update = _cleared_selection()
            update.page_rows = page_records(filtered, page, page_size)
            return update.as_tuple()

        if 'active_cell' in trigger_id and active_cell:
            relative_row = active_cell['row']
            absolute_index = (page * page_size) + relative_row
            if absolute_index < 0 or absolute_index >= len(filtered):
                return _SelectionUpdate().as_tuple()
            timestamp_us = filtered.iloc[absolute_index]['timestamp_us']
            selected_s = float(elapsed_seconds(timestamp_us, start_us))
            return _select(deltas, timestamp_us, selected_s, relative_row).as_tuple()

        if 'spread-chart' in trigger_id or 'pnl-chart' in trigger_id:
            click_data = spread_click if 'spread-chart' in trigger_id else pnl_click
            if not click_data:
                return _SelectionUpdate().as_tuple()

            clicked_s = click_data['points'][0]['x']
            timestamp_us = to_timestamp_us(clicked_s, start_us)

            index = filtered['timestamp_us'].searchsorted(timestamp_us)
            index = min(index, len(filtered) - 1)
            if index > 0:
                distance_right = abs(filtered.iloc[index]['timestamp_us'] - timestamp_us)
                distance_left = abs(filtered.iloc[index - 1]['timestamp_us'] - timestamp_us)
                if distance_left < distance_right:
                    index = index - 1

            target_page = int(index // page_size)
            update = _select(deltas, timestamp_us, clicked_s, int(index % page_size))
            update.page_current = target_page
            update.page_rows = page_records(filtered, target_page, page_size)
            return update.as_tuple()

        return _SelectionUpdate().as_tuple()
