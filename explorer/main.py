"""CLI entry point: load one simulator run from --log-dir and serve the Dash app."""

import argparse
from pathlib import Path

from dash import Dash, html

from explorer.callbacks import register_chart_sync_callback, register_selection_callback
from explorer.components.event_tables import build_event_log
from explorer.components.filters import build_filters
from explorer.components.orderbook_table import build_orderbook_table, format_orderbook
from explorer.components.pnl_chart import build_pnl_chart
from explorer.components.spread_chart import build_spread_chart
from explorer.components.topbar import build_topbar
from explorer.data.clock import elapsed_seconds
from explorer.data.loader import load_data, load_metadata, load_pnl, load_result, merge_trades_into_events
from explorer.data.orderbook import build_best_quotes, get_book_at

EVENTS_PAGE_SIZE = 20


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description='Backtest explorer: inspect one exchange-simulator run')
    parser.add_argument(
        '--log-dir',
        type=Path,
        required=True,
        help='Directory the simulator wrote without --no-logs (the CSV logs, pnl.csv, result.json, metadata.json)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Port to run the dashboard on (default: 8050)',
    )
    parser.add_argument('--debug', action='store_true', help='Enable Dash debug mode and hot-reloading')
    return parser.parse_args()


def main() -> None:
    """Load the run, build the page and serve it."""
    args = parse_args()

    metadata = load_metadata(args.log_dir)
    result = load_result(args.log_dir)
    deltas, events, trades = load_data(args.log_dir)

    deltas = deltas.sort_values('timestamp_us').reset_index(drop=True)
    trades = trades.sort_values('timestamp_us').reset_index(drop=True)
    events = merge_trades_into_events(events, trades)
    events = events.sort_values('timestamp_us').reset_index(drop=True)

    best_quotes = build_best_quotes(deltas)
    first_timestamp_us = int(best_quotes['timestamp_us'].iloc[0])
    best_quotes['elapsed_s'] = elapsed_seconds(best_quotes['timestamp_us'], first_timestamp_us)
    trades['elapsed_s'] = elapsed_seconds(trades['timestamp_us'], first_timestamp_us)
    pnl = load_pnl(args.log_dir)
    initial_book_rows = format_orderbook(get_book_at(deltas, first_timestamp_us))

    app = Dash(__name__)

    spread_graph, spread_fig = build_spread_chart(best_quotes, trades)
    pnl_graph, pnl_fig = build_pnl_chart(pnl)

    app.layout = html.Div(
        [
            html.Div(build_topbar(metadata, result['pnl'])),
            html.Div(
                [
                    html.Div(spread_graph, className='chart-container spread-panel'),
                    html.Div(build_orderbook_table(initial_book_rows), className='chart-container ladder-panel'),
                ],
                className='top-row',
            ),
            html.Div(
                [
                    html.Div(html.Div(pnl_graph, className='pnl-panel')),
                    html.Div(
                        [
                            build_filters(),
                            html.Div(build_event_log(events, EVENTS_PAGE_SIZE), className='event-log'),
                        ],
                        className='events-panel',
                    ),
                ],
                className='bottom-row',
            ),
        ],
        className='app-container',
    )

    register_selection_callback(app, events, deltas, EVENTS_PAGE_SIZE, first_timestamp_us)
    register_chart_sync_callback(app, best_quotes, pnl, trades, spread_fig, pnl_fig)

    app.run(debug=args.debug, port=args.port, dev_tools_ui=args.debug)


if __name__ == '__main__':
    main()
