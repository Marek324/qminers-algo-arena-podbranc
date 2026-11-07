"""Header bar with the run's metadata and final PnL."""

import os
from typing import Any

from dash import html


def build_topbar(metadata: dict[str, Any], final_pnl: float) -> html.Div:
    """Return the header bar populated with the run's metadata and the simulator's final PnL.

    Args:
        metadata: The simulator's metadata.json, as `loader.load_metadata` returns it.
        final_pnl: The `pnl` of the simulator's result.json.
    """
    fields = [
        ('Trader', metadata['trader']),
        ('Data', os.path.basename(metadata['data_file'])),
        ('Strategy', metadata['trader_class']),
        ('Latency', f'{float(metadata["latency"]) * 1000}ms'),
        ('Position Limit', f'±{metadata["position_limit"]}'),
    ]

    items = [
        html.Div(
            [html.Span(f'{label}: ', className='topbar-label'), html.Span(value)],
            className='topbar-item',
        )
        for label, value in fields
    ]

    pnl_class = 'topbar-pnl-positive' if final_pnl >= 0 else 'topbar-pnl-negative'

    return html.Div(
        [
            html.Span(
                [
                    html.Span('Q', style={'color': '#FE0000'}),
                    'miners Trading Hackathon',
                ],
                className='topbar-title',
            ),
            *items,
            html.Div(
                f'Final PnL: {final_pnl:.2f}',
                className=f'topbar-item topbar-pnl {pnl_class}',
            ),
        ],
        className='topbar',
    )
