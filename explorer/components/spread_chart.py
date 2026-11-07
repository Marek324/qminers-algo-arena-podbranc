"""Chart of the best bid and ask with the trades marked on it."""

import pandas as pd
import plotly.graph_objects as go
from dash import dcc
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import NoGapHandler

DEFAULT_N_SHOWN_SAMPLES = 2000


def build_spread_chart(best_quotes: pd.DataFrame, trades: pd.DataFrame) -> tuple[dcc.Graph, FigureResampler]:
    """Return a downsampled best bid/ask step plot with trade markers, plus its resampler.

    Args:
        best_quotes: Frame with `elapsed_s`, `best_bid` and `best_ask` columns.
        trades: The trades.csv frame; `taker_side`, `taker_owner` and `maker_owner`
            tell the strategy's buys and sells apart from market trades.

    Returns:
        The graph and its FigureResampler, which `callbacks.register_chart_sync_callback`
        needs to reload full-resolution data for the visible range.
    """
    fig = FigureResampler(
        go.Figure(),
        default_n_shown_samples=DEFAULT_N_SHOWN_SAMPLES,
        default_gap_handler=NoGapHandler(),
    )

    fig.add_trace(
        go.Scattergl(
            mode='lines',
            name='Best Bid',
            line=dict(color='#2979ff', width=1.5, shape='hv'),
        ),
        hf_x=best_quotes['elapsed_s'],
        hf_y=best_quotes['best_bid'],
    )
    fig.add_trace(
        go.Scattergl(
            mode='lines',
            name='Best Ask',
            line=dict(color='#ff5252', width=1.5, shape='hv'),
        ),
        hf_x=best_quotes['elapsed_s'],
        hf_y=best_quotes['best_ask'],
    )

    market_trades = trades[(trades['taker_owner'] != 'user') & (trades['maker_owner'] != 'user')]
    fig.add_trace(
        go.Scattergl(
            mode='markers',
            name='Market Trade',
            marker=dict(color='rgba(255, 255, 255, 0.3)', size=4, symbol='circle'),
        ),
        hf_x=market_trades['elapsed_s'],
        hf_y=market_trades['price'],
    )

    user_taker = trades['taker_owner'] == 'user'
    user_maker = trades['maker_owner'] == 'user'
    user_buys = trades[(user_taker & (trades['taker_side'] == 'B')) | (user_maker & (trades['taker_side'] == 'A'))]
    user_sells = trades[(user_taker & (trades['taker_side'] == 'A')) | (user_maker & (trades['taker_side'] == 'B'))]

    fig.add_trace(
        go.Scattergl(
            mode='markers',
            name='User Buy',
            marker=dict(color='#00E5FF', size=10, symbol='triangle-up'),
        ),
        hf_x=user_buys['elapsed_s'],
        hf_y=user_buys['price'],
    )
    fig.add_trace(
        go.Scattergl(
            mode='markers',
            name='User Sell',
            marker=dict(color='#FF00FF', size=10, symbol='triangle-down'),
        ),
        hf_x=user_sells['elapsed_s'],
        hf_y=user_sells['price'],
    )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        autosize=True,
        uirevision=True,
        margin=dict(l=60, r=20, t=30, b=50),
    )
    fig.update_xaxes(title_text='Time (s)', showgrid=True)
    fig.update_yaxes(title_text='Price', showgrid=True)
    graph = dcc.Graph(figure=fig, id='spread-chart', style={'flex': 1, 'minHeight': 0})
    return graph, fig
