"""Filter bar of the event log: one dropdown each for source, side and operation."""

from dash import dcc, html


def build_filters() -> html.Div:
    """Return the filter bar; `callbacks.register_selection_callback` reads the three dropdowns by id."""
    return html.Div(
        [
            dcc.Dropdown(
                id='filter-source',
                options=[
                    {'label': 'All Sources', 'value': 'All'},
                    {'label': 'User', 'value': 'user'},
                    {'label': 'Market', 'value': 'market'},
                ],
                value='All',
                clearable=False,
                className='filter-dropdown filter-source',
            ),
            dcc.Dropdown(
                id='filter-side',
                options=[
                    {'label': 'All Sides', 'value': 'All'},
                    {'label': 'Bid', 'value': 'B'},
                    {'label': 'Ask', 'value': 'A'},
                ],
                value='All',
                clearable=False,
                className='filter-dropdown filter-side',
            ),
            dcc.Dropdown(
                id='filter-op',
                options=[
                    {'label': 'All Types', 'value': 'All'},
                    {'label': 'Trade', 'value': 'trade'},
                    {'label': 'Add', 'value': 'add'},
                    {'label': 'Cancel', 'value': 'cancel'},
                    {'label': 'Modify', 'value': 'modify'},
                    {'label': 'Add Sent', 'value': 'add_sent'},
                    {'label': 'Cancel Sent', 'value': 'cancel_sent'},
                    {'label': 'Modify Sent', 'value': 'modify_sent'},
                    {'label': 'Add Failed', 'value': 'add_failed'},
                    {'label': 'Cancel Failed', 'value': 'cancel_failed'},
                    {'label': 'Modify Failed', 'value': 'modify_failed'},
                ],
                value='All',
                clearable=False,
                className='filter-dropdown filter-op',
            ),
        ],
        className='filter-bar',
    )
