"""Step chart of the strategy's PnL over time."""

import pandas as pd
import plotly.graph_objects as go
from dash import dcc
from plotly_resampler import FigureResampler


def build_pnl_chart(pnl: pd.DataFrame) -> tuple[dcc.Graph, FigureResampler]:
    """Return a step plot of the simulator's PnL curve, plus its figure.

    Args:
        pnl: The `loader.load_pnl` frame: `elapsed_s` and `pnl` columns.

    Returns:
        The graph and its FigureResampler, which `callbacks.register_chart_sync_callback`
        drives together with the spread chart's.
    """
    fig = FigureResampler(go.Figure())
    fig.add_trace(
        go.Scattergl(
            mode='lines',
            name='PnL',
            line=dict(color='green'),
            line_shape='hv',
        ),
        hf_x=pnl['elapsed_s'],
        hf_y=pnl['pnl'],
        max_n_samples=len(pnl),
    )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        autosize=True,
        uirevision=True,
        margin=dict(l=60, r=20, t=10, b=50),
    )
    fig.update_xaxes(title_text='Time (s)', showgrid=True)
    fig.update_yaxes(title_text='PnL', showgrid=True)
    graph = dcc.Graph(
        figure=fig,
        id='pnl-chart',
        style={'flex': 1, 'minHeight': 0, 'height': '100%', 'width': '100%'},
    )
    return graph, fig
