"""Analysis tab: renders the report from the `analysis` package (see docs/ANALYSIS.md) for the selected tape.

The report is read from logs/analysis/<name>.json when that file is newer than the CSV and has the current
schema version; otherwise it is computed on request and saved there, so the CLI and the dashboard share it.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis import SCHEMA_VERSION, analyze  # noqa: E402
from analysis.report import SECTION_ORDER, to_json, to_text  # noqa: E402

OUT_DIR = ROOT / 'logs' / 'analysis'
UP, DOWN, ACCENT, NEUTRAL = '#26a69a', '#ef5350', '#f5c542', '#7e8ce0'
GRID = 'rgba(128,128,128,0.15)'
CLEAR = 'rgba(0,0,0,0)'
PLOT_CONFIG = {'scrollZoom': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']}
DIST_COLS = ['n', 'mean', 'std', 'min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max']
PALETTE = [NEUTRAL, ACCENT, UP, DOWN, '#b07cc6', '#4fb3d9', '#e08a3c', '#9aa0a6']


# ---------------------------------------------------------------- loading
def report_path(name: str) -> Path:
    return OUT_DIR / f'{name.removesuffix(".gz").removesuffix(".csv")}.json'


def saved_report(csv: Path | None, name: str) -> dict | None:
    """The saved report for this tape if it is at least as new as the CSV and has the current schema."""
    p = report_path(name)
    if not p.exists() or (csv is not None and p.stat().st_mtime < csv.stat().st_mtime):
        return None
    try:
        rep = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    current = rep.get('meta', {}).get('schema_version') == SCHEMA_VERSION and all(k in rep for k in SECTION_ORDER)
    return rep if current else None


@st.cache_data(show_spinner=False, max_entries=4)
def _load_saved(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


@st.cache_data(show_spinner='Running the analysis (about 12 s per million rows)...', max_entries=4)
def _run(source: str, csv: str | None, _data: bytes | None, name: str, series_step: float, bin_step: float) -> dict:
    if csv is not None:
        rep = analyze(Path(csv), series_step, bin_step)
    else:
        rep = analyze(io.BytesIO(_data), series_step, bin_step, name=name)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path(name).write_text(to_json(rep))
    return rep


# ---------------------------------------------------------------- helpers
def arr(x) -> np.ndarray:
    """List with nulls -> float array with NaN."""
    return np.array([np.nan if v is None else v for v in (x or [])], dtype=float)


def num(v, fmt: str = ',.3g', suffix: str = '') -> str:
    if v is None:
        return '-'
    return f'{v:{fmt}}{suffix}'


def big(v) -> str:
    """1,234 / 56.7k / 3.0M"""
    if v is None:
        return '-'
    a = abs(v)
    return f'{v / 1e6:.2f}M' if a >= 1e6 else f'{v / 1e3:.1f}k' if a >= 1e4 else f'{v:,.0f}'


def cell(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '-'
    if isinstance(v, (bool, np.bool_)):
        return 'yes' if v else 'no'
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer() and abs(v) < 1e15):
        return f'{int(v):,}'
    if isinstance(v, float):
        return f'{v:.4g}'
    return str(v)


def pct(v) -> str:
    return '-' if v is None else f'{100 * v:.1f}%'


def ts(sec) -> np.ndarray:
    return (np.asarray(sec, dtype=np.float64) * 1e6).round().astype(np.int64).astype('datetime64[us]')


def style(fig: go.Figure, height: int, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=30), paper_bgcolor=CLEAR, plot_bgcolor=CLEAR,
                      legend=dict(orientation='h', y=-0.12, x=0, yanchor='top', font_size=11), **layout)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width='stretch', config=PLOT_CONFIG, theme='streamlit', key=key)


def time_axes(fig: go.Figure, hrs: bool) -> None:
    f = '%H:%M:%S' if hrs else '%M:%S'
    fig.update_xaxes(type='date', hoverformat=f, tickformat=f)


def dist_table(rows: dict[str, dict | None]) -> pd.DataFrame:
    """One row per named distribution object."""
    out = []
    for name, d in rows.items():
        d = d or {}
        out.append({'': name, **{c: d.get(c) for c in DIST_COLS}})
    return pd.DataFrame(out)


def table(df: pd.DataFrame, height: int | None = None) -> None:
    """Read-only table: integers with separators, floats to 4 significant digits, nulls as '-', numbers right-aligned."""
    shown = df.copy()
    numeric = [c for c in shown.columns if all(isinstance(v, (int, float, np.integer, np.floating, type(None))) and not isinstance(v, bool)
                                              for v in shown[c].tolist())]
    for c in shown.columns:
        shown[c] = [cell(v) for v in shown[c].tolist()]
    sty = shown.style.set_properties(subset=numeric, **{'text-align': 'right'}) if numeric else shown.style
    kw = {} if height is None else {'height': height}
    st.dataframe(sty, hide_index=True, width='stretch', **kw)


def bar(labels, values, title: str, color=NEUTRAL, height: int = 260, pct_axis: bool = False, key: str = '') -> None:
    fig = go.Figure(go.Bar(x=[str(v) for v in labels], y=arr(values), marker_color=color,
                           hovertemplate='%{x}: %{y:.4g}<extra></extra>'))
    style(fig, height, title=dict(text=title, font_size=13, x=0))
    fig.update_xaxes(type='category')
    if pct_axis:
        fig.update_yaxes(tickformat='.0%')
    show(fig, key)


def horizon_lines(series: dict[str, dict], horizons: list[str], title: str, ytitle: str, key: str, height: int = 340) -> None:
    """Lines over a fixed horizon order, one per named {horizon: value} dict."""
    fig = go.Figure()
    for i, (name, row) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=horizons, y=arr([row.get(h) for h in horizons]), name=name, mode='lines+markers',
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=2 if i == 0 else 1.4)))
    fig.add_hline(y=0, line=dict(color='rgba(128,128,128,0.6)', width=1, dash='dot'))
    style(fig, height, title=dict(text=title, font_size=13, x=0), yaxis_title=ytitle)
    fig.update_xaxes(type='category')
    show(fig, key)


# ---------------------------------------------------------------- sections
STATUS = {'good': ('green', 'looks good'), 'warn': ('orange', 'mixed'), 'bad': ('red', 'unfavourable'), 'info': ('blue', 'context')}


def kpis(rep: dict) -> None:
    b, p, m, mp = rep['book'], rep['price'], rep['maker'], rep['max_pnl']
    c = st.columns(6)
    off = dict(delta_color='off', delta_arrow='off')
    c[0].metric('Theoretical max PnL', num(mp['theoretical_max']['pnl'], ',.0f'), f'{num(mp["theoretical_max"]["pnl_per_hour"], ",.0f")} / hour', **off,
                help='Best score any strategy could reach knowing the future, under the simulator rules. See PnL bounds.')
    c[1].metric('Maker markout 5 s (ticks/lot)', num(m['markout_ticks'].get('5s'), '+.2f'), f'half-spread {num(m["markout_ticks"].get("0ms"), "+.2f")}',
                **off, help='What a passive fill is worth 5 s later, from the maker side. Positive = making money.')
    c[2].metric('Typical spread (ticks)', num(b['modal_spread_ticks'], 'g'), f'1 tick {pct(b["spread_share"]["1"])} of time', **off)
    c[3].metric('Volatility at 1 s (ticks/√s)', num(p['vol_ticks_per_sqrt_s']['1s'], '.3g'), f'per 60 s: {num(p["vol_ticks_per_sqrt_s"]["60s"], ".3g")}/√s',
                **off)
    vr = p['variance_ratio_vs_1s']['60s']
    c[4].metric('Trend vs revert (VR 60 s)', num(vr, '.2f'), '<1 reverts, >1 trends', **off)
    c[5].metric('Best price moves in 5 ms', pct(b['latency_5ms']['event_weighted']['touch_price_moved_share']), 'after an event', **off,
                help='How often the touch changes while your order is in flight.')


def verdict_cards(rep: dict) -> None:
    vs = rep.get('verdicts') or []
    cols = st.columns(len(vs) or 1)
    for col, v in zip(cols, vs, strict=False):
        color, word = STATUS.get(v['status'], ('gray', ''))
        with col.container(border=True, height='stretch'):
            st.markdown(f'**{v["topic"]}** &nbsp; :{color}-badge[{word}]')
            st.markdown(v['headline'])
            st.caption(v['detail'])


def overview_page(rep: dict, hrs: bool) -> None:
    kpis(rep)
    verdict_cards(rep)
    session_chart(rep, hrs, 'ov_session')
    c = st.columns(3)
    with c[0]:
        m = rep['maker']
        hs = [h for h in m['markout_ticks'] if h not in ('fills', 'volume')]
        horizon_lines({'all fills': m['markout_ticks'], 'at touch': m['at_touch']}, hs, 'maker markout (ticks/lot)', '', 'ov_markout', 300)
    with c[1]:
        cand = []
        for n, row in rep['signals']['directional'].items():
            cand.append((n, row['5s']['spearman']))
        for n, row in rep['indicators']['evaluation'].items():
            cand.append((n, row['5s']['spearman']))
        cand = sorted((x for x in cand if x[1] is not None), key=lambda x: -abs(x[1]))[:6][::-1]
        fig = go.Figure(go.Bar(y=[n for n, _ in cand], x=[v for _, v in cand], orientation='h',
                               marker_color=[UP if v >= 0 else DOWN for _, v in cand], hovertemplate='%{y}: %{x:+.3f}<extra></extra>'))
        style(fig, 300, title=dict(text='top predictors of the 5 s move (Spearman)', font_size=13, x=0))
        show(fig, 'ov_predictors')
    with c[2]:
        pp = rep['orders']['passive_placement']
        order = ['improve', 'join', 'behind_1', 'behind_2_5', 'behind_6_20']
        bar(order, [pp[k]['any_fill_prob'] for k in order], 'fill probability by order placement', UP, height=300, pct_axis=True, key='ov_fill')
    with st.expander('All findings'):
        st.markdown('\n'.join(f'- {h}' for h in rep['highlights']) or 'No findings.')


def market_section(rep: dict, hrs: bool) -> None:
    st.subheader('Order book')
    book_tab(rep)
    st.subheader('Price')
    price_tab(rep)
    st.subheader('Over the session')
    overview(rep, hrs)
    st.subheader('Regimes across bins')
    regimes_tab(rep)
    with st.expander('Tape composition and anomalies'):
        tape_tab(rep)


def flow_section(rep: dict) -> None:
    st.subheader('Maker markouts')
    maker_tab(rep)
    st.subheader('Trades and price impact')
    trades_tab(rep)
    st.subheader('Order lifecycle')
    orders_tab(rep)


def signals_section(rep: dict, hrs: bool) -> None:
    st.subheader('Order book and flow signals')
    signals_tab(rep)
    st.subheader('Technical indicators')
    indicators_tab(rep, hrs)


def data_section(rep: dict, name: str, hrs: bool) -> None:
    st.subheader('Series explorer')
    series_explorer(rep, hrs)
    st.subheader('Raw report')
    raw_tab(rep, name)


def session_chart(rep: dict, hrs: bool, key: str) -> None:
    """Price, aggressive volume and spread per bin over the whole session."""
    bins = rep['bins']
    d = bins['data']
    start = rep['tape']['start_s'] or 0.0
    x = ts(start + (arr(d['t_start_s']) + arr(d['t_end_s'])) / 2)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.58, 0.24, 0.18], vertical_spacing=0.04,
                        subplot_titles=(f'mid price per {bins["step_s"]:g} s', 'aggressive volume (buy up, sell down)', 'mean spread (ticks)'))
    fig.add_trace(go.Candlestick(x=x, open=arr(d['mid_open']), high=arr(d['mid_high']), low=arr(d['mid_low']), close=arr(d['mid_close']),
                                 name='mid', showlegend=False, whiskerwidth=0, increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
                                 decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN)), row=1, col=1)
    w = bins['step_s'] * 1e3 * 0.8
    fig.add_trace(go.Bar(x=x, y=arr(d['buy_volume']), width=w, marker_color=UP, name='buy volume', showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=x, y=-arr(d['sell_volume']), width=w, marker_color=DOWN, name='sell volume', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=arr(d['spread_mean']), mode='lines', line=dict(color=ACCENT, width=1.4, shape='hv'), name='spread',
                             showlegend=False), row=3, col=1)
    style(fig, 520, barmode='relative', bargap=0.1, dragmode='pan', hovermode='x unified')
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_annotations(font_size=12, x=0, xanchor='left')
    time_axes(fig, hrs)
    show(fig, key)


def overview(rep: dict, hrs: bool) -> None:
    """Session detail per bin: volatility, activity, cancel ratio and imbalance, then the depth heatmap."""
    bins = rep['bins']
    d = bins['data']
    start = rep['tape']['start_s'] or 0.0
    x = ts(start + (arr(d['t_start_s']) + arr(d['t_end_s'])) / 2)
    rows = (('realized_vol_ticks', 'realized volatility (ticks)', NEUTRAL), ('events', 'tape events', ACCENT),
            ('cancel_insert_ratio', 'cancels per insert', DOWN), ('imbalance_mean', 'touch imbalance (bid − ask) / total', UP))
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=[r[1] for r in rows])
    for i, (col, _, color) in enumerate(rows, 1):
        fig.add_trace(go.Scatter(x=x, y=arr(d[col]), mode='lines', line=dict(color=color, width=1.4, shape='hv'), name=col, showlegend=False),
                      row=i, col=1)
    style(fig, 560, dragmode='pan', hovermode='x unified')
    fig.update_annotations(font_size=12, x=0, xanchor='left')
    time_axes(fig, hrs)
    st.caption(f'{bins["rows"]} bins of {bins["step_s"]:g} s.')
    show(fig, 'an_overview')

    # depth heatmap
    bd, ad = np.array(d['bid_depth_mean'], dtype=float), np.array(d['ask_depth_mean'], dtype=float)
    if bd.ndim == 2 and bd.size:
        levels = bd.shape[1]
        z = np.column_stack([bd[:, ::-1], ad]).T
        ylab = [f'bid −{k}' for k in range(levels - 1, -1, -1)] + [f'ask +{k}' for k in range(levels)]
        hm = go.Figure(go.Heatmap(x=x, y=ylab, z=z, colorscale='Viridis', colorbar=dict(title='lots', thickness=10),
                                  hovertemplate='%{x}<br>%{y} ticks from best<br>mean %{z:,.1f} lots<extra></extra>'))
        style(hm, 420, title=dict(text='mean depth per bin by distance from own-side best', font_size=13, x=0), dragmode='pan')
        time_axes(hm, hrs)
        show(hm, 'an_depth_heat')


def series_explorer(rep: dict, hrs: bool) -> None:
    s = rep['series']
    d = s['data']
    cols = [c for c in s['columns'] if c != 't_s']
    pick = st.multiselect('Columns', cols, default=[c for c in ('mid', 'spread', 'imbalance', 'volume', 'net_flow') if c in cols], key='an_series_cols')
    if not pick:
        return
    x = ts((rep['tape']['start_s'] or 0.0) + arr(d['t_s']))
    fig = make_subplots(rows=len(pick), cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=pick)
    for i, c in enumerate(pick, 1):
        y = arr(d[c])
        if c in ('volume', 'buy_volume', 'sell_volume', 'trades', 'events', 'inserts', 'cancels', 'ioc_inserts', 'net_flow'):
            fig.add_trace(go.Bar(x=x, y=y, name=c, showlegend=False, marker_color=np.where(y >= 0, UP, DOWN) if c == 'net_flow' else NEUTRAL,
                                 marker_line_width=0), row=i, col=1)
        else:
            fig.add_trace(go.Scatter(x=x, y=y, name=c, showlegend=False, mode='lines', line=dict(width=1.2, shape='hv', color=PALETTE[(i - 1) % len(PALETTE)])),
                          row=i, col=1)
    style(fig, 140 + 150 * len(pick), dragmode='pan', hovermode='x unified', bargap=0)
    fig.update_annotations(font_size=12, x=0, xanchor='left')
    time_axes(fig, hrs)
    st.caption(f'{s["rows"]:,} rows of {s["step_s"]:g} s. Book columns are the state at the end of each row, flows are sums over it.')
    show(fig, 'an_series')


def book_tab(rep: dict) -> None:
    b = rep['book']
    c = st.columns(4)
    c[0].metric('Both sides present', pct(b['time_share']['both_sides']))
    c[1].metric('Mid changes / s', num(b['dynamics']['mid_changes_per_s'], '.3g'))
    c[2].metric('Touch moves within 5 ms (time-weighted)', pct(b['latency_5ms']['time_weighted']['touch_price_moved_share']))
    c[3].metric('Mean |mid move| in 5 ms (ticks)', num(b['latency_5ms']['event_weighted']['mean_abs_mid_move_ticks'], '.3g'))
    left, right = st.columns(2)
    with left:
        ss = b['spread_share']
        bar(list(ss), list(ss.values()), 'spread, share of time (ticks)', ACCENT, pct_axis=True, key='an_spread')
    with right:
        dp = b['depth_profile']
        lv = list(range(dp['levels']))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=lv, y=arr(dp['bid_mean']), name='bid mean', marker_color=UP, opacity=0.8))
        fig.add_trace(go.Bar(x=lv, y=arr(dp['ask_mean']), name='ask mean', marker_color=DOWN, opacity=0.8))
        fig.add_trace(go.Scatter(x=lv, y=arr(dp['bid_nonempty_share']), name='bid level occupied', yaxis='y2', line=dict(color=UP, dash='dot')))
        fig.add_trace(go.Scatter(x=lv, y=arr(dp['ask_nonempty_share']), name='ask level occupied', yaxis='y2', line=dict(color=DOWN, dash='dot')))
        style(fig, 260, title=dict(text='depth by ticks from own-side best', font_size=13, x=0), barmode='group',
              yaxis2=dict(overlaying='y', side='right', range=[0, 1.05], tickformat='.0%', showgrid=False))
        show(fig, 'an_depth')
    with st.expander('Book distributions (time-weighted unless noted)'):
        table(dist_table({
            'spread (ticks)': b['spread_ticks'],
            'spread, event-weighted': b['spread_ticks_event_weighted'],
            'wide-spread episode (ms)': b['wide_spread_episode_ms'],
            'touch qty bid': b['touch_qty']['bid'],
            'touch qty ask': b['touch_qty']['ask'],
            'touch orders bid': b['touch_orders']['bid'],
            'touch orders ask': b['touch_orders']['ask'],
            'imbalance touch': b['imbalance']['touch'],
            'imbalance 5 levels': b['imbalance']['depth_5'],
            'imbalance 20 levels': b['imbalance']['depth_20'],
            'touch price lifetime (ms)': b['dynamics']['touch_price_lifetime_ms'],
        }))


def price_tab(rep: dict) -> None:
    p = rep['price']
    c = st.columns(6)
    c[0].metric('Mid change (ticks)', num(p['mid']['change_ticks'], '+,.1f'), f'range {num(p["mid"]["range_ticks"], ",.1f")}', delta_color='off', delta_arrow='off')
    c[1].metric('Drift (ticks/hour)', num(p['trend']['drift_ticks_per_hour'], '+.1f'), f't = {num(p["trend"]["t_stat_60s_returns"], "+.2f")}', delta_color='off', delta_arrow='off')
    c[2].metric('Hurst', num(p['hurst_estimate'], '.2f'), '0.5 = random walk', delta_color='off', delta_arrow='off')
    c[3].metric('Reversal prob', pct(p['event_time']['reversal_prob']), 'next mid change flips', delta_color='off', delta_arrow='off')
    c[4].metric('Max drawdown (ticks)', num(p['max_drawdown_ticks'], ',.1f'), f'run-up {num(p["max_runup_ticks"], ",.1f")}', delta_color='off', delta_arrow='off')
    c[5].metric('Roll spread (ticks)', num(p['trade_price']['roll_spread_ticks'], '.2f'))
    left, right = st.columns(2)
    hs = list(p['vol_ticks_per_sqrt_s'])
    with left:
        horizon_lines({'vol per √s': p['vol_ticks_per_sqrt_s']}, hs, 'signature plot: volatility per √s by sampling horizon', 'ticks / √s', 'an_sig', 300)
    with right:
        fig = go.Figure(go.Scatter(x=hs, y=arr([p['variance_ratio_vs_1s'][h] for h in hs]), mode='lines+markers', line=dict(color=ACCENT)))
        fig.add_hline(y=1, line=dict(color='rgba(128,128,128,0.6)', dash='dot'))
        style(fig, 300, title=dict(text='variance ratio vs 1 s (<1 mean-reverting, >1 trending)', font_size=13, x=0))
        fig.update_xaxes(type='category')
        show(fig, 'an_vr')
    rows = []
    for h, r in p['returns_ticks'].items():
        rows.append({'horizon': h, 'n': r['n'], 'mean': r['mean'], 'std': r['std'], 'skew': r['skew'], 'excess kurtosis': r['kurtosis'],
                     'mean |r|': r['abs_mean'], 'zero share': r['zero_share'], 'up share': r['up_share'], 'autocorr lag1': r['autocorr']['lag1'],
                     'p01': r['quantiles']['p01'], 'p99': r['quantiles']['p99']})
    with st.expander('Mid returns by horizon (ticks, non-overlapping)'):
        table(pd.DataFrame(rows))
    left, right = st.columns(2)
    et = p['event_time']
    with left:
        ac = et['autocorr_nonzero_changes']
        bar(list(ac), list(ac.values()), 'autocorrelation of successive mid changes (event time)', NEUTRAL, key='an_ac')
    with right:
        sz = et['mid_change_size_share']
        bar([f'{k}' if k != '2.5' else '≥2.5' for k in sz], list(sz.values()), 'size of mid changes (ticks)', NEUTRAL, pct_axis=True, key='an_mcs')


def trades_tab(rep: dict) -> None:
    t = rep['trades']
    c = st.columns(6)
    c[0].metric('Trades', big(t['count']), f'{num(t["trades_per_s"], ".3g")} / s', delta_color='off', delta_arrow='off')
    c[1].metric('Aggressive orders', big(t['taker_events']), f'{num(t["taker_events_per_s"], ".3g")} / s', delta_color='off', delta_arrow='off')
    c[2].metric('Buy-initiated', pct(t['buy_initiated_share']['by_volume']), 'by volume', delta_color='off', delta_arrow='off')
    c[3].metric('Single-level sweeps', pct(t['single_level_share']))
    c[4].metric('IOC takers', pct(t['taker_ioc_share']['by_volume']), 'by volume', delta_color='off', delta_arrow='off')
    c[5].metric('Sign autocorr lag 1', num(t['sign_autocorr']['lag1'], '+.3f'))
    hs = list(t['impact_ticks'])
    series = {'all': {h: t['impact_ticks'][h]['mean_ticks'] for h in hs}}
    for row in t['impact_by_size']:
        series[f'size {row["size_lo"]:g}–{row["size_hi"]:g}'] = row
    left, right = st.columns([3, 2])
    with left:
        horizon_lines(series, hs, 'impact: signed mid move after an aggressive order', 'ticks', 'an_impact')
    with right:
        sa = t['sign_autocorr']
        bar(list(sa), list(sa.values()), 'aggressor sign autocorrelation', NEUTRAL, height=200, key='an_signac')
        ls = t['levels_swept_share']
        bar([k if k != '4' else '4+' for k in ls], list(ls.values()), 'price levels per aggressive order', NEUTRAL, height=200, pct_axis=True,
            key='an_levels')
    with st.expander('Trade distributions'):
        table(dist_table({
        'trade size': t['trade_size'],
        'aggressive order size': t['taker_event_size'],
        'same-sign run length': t['same_sign_run_length'],
        'time between aggressive orders (ms)': t['taker_interarrival_ms'],
        'maker age at fill (ms)': t['maker_age_at_fill_ms'],
        'spread before aggressor (ticks)': t['spread_before_taker_ticks'],
    }))
    left, right = st.columns([2, 3])
    with left:
        st.markdown('**Kyle lambda**: mid change per lot of net aggressive volume')
        table(pd.DataFrame([{'bucket': k, 'lambda': v['coef'][0], 't': v['t'][0], 'R²': v['r2'], 'n': v['n']} for k, v in t['kyle_lambda'].items()]))
    with right:
        st.markdown('**Largest aggressive orders**')
        table(pd.DataFrame(t['largest_taker_events']))


def maker_tab(rep: dict) -> None:
    m = rep['maker']
    st.caption('Markout of every passive fill, per lot, from the maker side: side × (mid(t+h) − fill price). 0 ms = half-spread captured; '
               'the drop after it is adverse selection. Positive late markouts mean quoting at the touch has an edge before inventory costs.')
    hs = [h for h in m['markout_ticks'] if h not in ('fills', 'volume')]
    left, right = st.columns(2)
    with left:
        horizon_lines({'all fills': m['markout_ticks'], 'at touch': m['at_touch'], 'through touch': m['through_touch'],
                       'taker IOC': m['by_taker_type']['ioc'], 'taker limit': m['by_taker_type']['limit']}, hs, 'markout by fill type', 'ticks / lot',
                      'an_mk1')
    with right:
        horizon_lines({'spread 1': m['by_spread_before']['1'], 'spread 2': m['by_spread_before']['2'], 'spread >2': m['by_spread_before']['gt2'],
                       'maker bid': m['by_maker_side']['bid'], 'maker ask': m['by_maker_side']['ask']}, hs, 'markout by spread and side', 'ticks / lot', 'an_mk2')
    left, right = st.columns([2, 3])
    with left:
        ps = m['profitable_share']
        bar(list(ps), list(ps.values()), 'share of fills in profit', UP, pct_axis=True, key='an_mkprof')
        st.metric('Touch fill volume (lots/s)', num(m['touch_fill_volume_per_s'], ',.3g'))
    with right:
        st.markdown('**By fill size**')
        table(pd.DataFrame(m['by_fill_size']))
        groups = {'all': m['markout_ticks'], 'at touch': m['at_touch'], 'through touch': m['through_touch'], **{f'spread {k}': v for k, v in m['by_spread_before'].items()}}
        st.markdown('**Volume per group**')
        table(pd.DataFrame([{'group': k, 'fills': v['fills'], 'volume': v['volume']} for k, v in groups.items()]))


def max_pnl_tab(rep: dict, hrs: bool) -> None:
    mp = rep['max_pnl']
    hi, lo = mp['theoretical_max'], mp['theoretical_min']
    st.caption('Best and worst PnL any strategy could score on this tape knowing the future, under the simulator rules (5 ms latency, '
               'price-time priority including queue position, |inventory| <= 100, flat at the end). Same units as the simulator score. '
               'Your own fills are assumed not to change what the market does next.')
    c = st.columns(4)
    c[0].metric('Theoretical max', num(hi['pnl'], ',.0f'), f'{num(hi["pnl_per_hour"], ",.0f")} / hour', delta_color='off', delta_arrow='off')
    c[1].metric('Making only', num(mp['making_only']['pnl'], ',.0f'), f'{pct(safe_ratio(mp["making_only"]["pnl"], hi["pnl"]))} of max',
                delta_color='off', delta_arrow='off', help='Only passive fills: resting orders filled by incoming market orders.')
    c[2].metric('Taking only', num(mp['taking_only']['pnl'], ',.0f'), f'{pct(safe_ratio(mp["taking_only"]["pnl"], hi["pnl"]))} of max',
                delta_color='off', delta_arrow='off', help='Only crossing the spread at the best bid/ask.')
    c[3].metric('Theoretical min', num(lo['pnl'], ',.0f'), f'{num(lo["pnl_per_hour"], ",.0f")} / hour', delta_color='off', delta_arrow='off',
                help='Worst possible score: mostly from crossing the spread back and forth at full size.')
    cv = mp['curve']
    d = cv['data']
    x = ts((rep['tape']['start_s'] or 0.0) + arr(d['t_s']))
    left, right = st.columns(2)
    for col, keys, title, key in ((left, ('theoretical_max', 'making_only', 'taking_only'), 'best PnL achievable by time t (flat at t)', 'an_mp_hi'),
                                  (right, ('theoretical_min',), 'worst PnL by time t (flat at t)', 'an_mp_lo')):
        with col:
            fig = go.Figure()
            for i, k in enumerate(keys):
                fig.add_trace(go.Scatter(x=x, y=arr(d[k]), mode='lines', name=k.replace('_', ' '),
                                         line=dict(width=2 if i == 0 else 1.3, color=(DOWN if k == 'theoretical_min' else PALETTE[i % len(PALETTE)]))))
            style(fig, 320, title=dict(text=title, font_size=13, x=0), hovermode='x unified', dragmode='pan')
            time_axes(fig, hrs)
            show(fig, key)
    o = mp['opportunities']
    table(pd.DataFrame([{'': k.replace('_', ' '), 'value': v} for k, v in o.items()]))


def safe_ratio(a, b):
    return a / b if a is not None and b else None


def orders_tab(rep: dict) -> None:
    o = rep['orders']
    c = st.columns(4)
    c[0].metric('Orders', big(o['count']))
    c[1].metric('Marketable on arrival', pct(o['marketable_share']))
    c[2].metric('Canceled', pct(o['outcome_share']['canceled']))
    c[3].metric('Cancels within 1 s', pct(o['cancel_within_share']['1s']))
    pp = o['passive_placement']
    order = ['improve', 'join', 'behind_1', 'behind_2_5', 'behind_6_20', 'behind_gt20', 'side_empty']
    fig = go.Figure()
    for key, name, col in (('any_fill_prob', 'any fill', UP), ('full_fill_prob', 'full fill', NEUTRAL), ('cancel_prob', 'canceled', DOWN)):
        fig.add_trace(go.Bar(x=order, y=arr([pp[k][key] for k in order]), name=name, marker_color=col))
    fig.add_trace(go.Scatter(x=order, y=arr([pp[k]['share'] for k in order]), name='share of passive orders', mode='lines+markers',
                             line=dict(color=ACCENT, dash='dot')))
    style(fig, 320, title=dict(text='passive orders by placement vs own-side best: what happens to them', font_size=13, x=0), barmode='group')
    fig.update_yaxes(tickformat='.0%')
    show(fig, 'an_place')
    table(pd.DataFrame([{'placement': k, **pp[k]} for k in order]))
    left, mid_, right = st.columns(3)
    with left:
        os_ = o['outcome_share']
        bar(list(os_), list(os_.values()), 'order outcome', NEUTRAL, pct_axis=True, key='an_outcome')
    with mid_:
        cw = o['cancel_within_share']
        bar(list(cw), list(cw.values()), 'cancels within (cumulative)', DOWN, pct_axis=True, key='an_cw')
    with right:
        cp = o['cancel_placement_share']
        bar(list(cp), list(cp.values()), 'where cancels happen', DOWN, pct_axis=True, key='an_cp')
    with st.expander('Order lifetimes'):
        table(dist_table({'lifetime, all rested (ms)': o['lifetime_ms']['all_rested'], 'lifetime, canceled (ms)': o['lifetime_ms']['canceled'],
                      'lifetime, filled (ms)': o['lifetime_ms']['filled'], 'queue ahead on join (lots)': o['queue_ahead_on_join']}))
    if o['join_fill_prob_by_queue_ahead']:
        st.markdown('**Touch joiners: fill probability by queue ahead**')
        table(pd.DataFrame(o['join_fill_prob_by_queue_ahead']))


def tape_tab(rep: dict) -> None:
    t = rep['tape']
    c = st.columns(6)
    c[0].metric('Rows', big(t['rows']), f'{num(t["rows_per_s"], ",.0f")} / s', delta_color='off', delta_arrow='off')
    c[1].metric('Inserts', big(t['inserts']), f'IOC {big(t["ioc_inserts"])}', delta_color='off', delta_arrow='off')
    c[2].metric('Cancels / insert', num(t['cancel_to_insert_ratio'], '.2f'))
    c[3].metric('Duration (s)', num(t['duration_s'], ',.0f'))
    c[4].metric('IOC any fill', pct(t['ioc']['share_any_fill']))
    c[5].metric('Gaps < 5 ms', pct(t['timing']['share_interarrival_lt_5ms']))
    left, right = st.columns(2)
    with left:
        ld = t['price']['last_digit_share_by_volume']
        bar(list(ld), list(ld.values()), 'price last digit, by insert volume (10% each = no clustering)', NEUTRAL, pct_axis=True, key='an_digit')
    with right:
        top = t['insert_size_top']
        bar([r['size'] for r in top], [r['share'] for r in top], 'most common insert sizes', NEUTRAL, pct_axis=True, key='an_sizes')
    with st.expander('Tape distributions'):
        table(dist_table({'insert size': t['insert_size'], 'inter-arrival (µs)': t['timing']['interarrival_us'],
                      'events per second': t['timing']['events_per_second'], 'events in next 5 ms': t['timing']['events_in_next_5ms']}))
    st.markdown('**Tape anomalies**')
    table(pd.DataFrame([{'anomaly': k, 'count': v} for k, v in t['anomalies'].items()]))


def signals_tab(rep: dict) -> None:
    s = rep['signals']
    st.caption(s['note'] + '. Correlations are descriptive: samples overlap.')
    hs = list(s['forward_returns'])
    feats = list(s['directional'])
    metric = st.segmented_control('Metric', ['spearman', 'pearson', 'hit_rate'], default='spearman', key='an_sig_metric') or 'spearman'
    z = np.array([[s['directional'][f][h].get(metric) for h in hs] for f in feats], dtype=float)
    center = 0.5 if metric == 'hit_rate' else 0.0
    span = np.nanmax(np.abs(z - center)) if np.isfinite(z).any() else 1.0
    fig = go.Figure(go.Heatmap(x=hs, y=feats, z=z, zmid=center, zmin=center - span, zmax=center + span, colorscale='RdBu',
                               text=[[f'{v:+.3f}' if np.isfinite(v) else '' for v in row] for row in z], texttemplate='%{text}',
                               hovertemplate='%{y} → fwd %{x}: %{z:.4f}<extra></extra>', colorbar=dict(thickness=10)))
    style(fig, 120 + 30 * len(feats), title=dict(text=f'{metric} of feature vs forward mid move', font_size=13, x=0))
    fig.update_yaxes(autorange='reversed')
    show(fig, 'an_ic')

    c = st.columns([2, 1])
    feat = c[0].selectbox('Decile chart for', feats, key='an_dec_feat')
    dh = list(s['directional'][feat]['deciles'])
    h = c[1].segmented_control('Horizon', dh, default=dh[0] if dh else None, key='an_dec_h') or (dh[0] if dh else None)
    rows = s['directional'][feat]['deciles'].get(h, []) if h else []
    if rows:
        lab = [f'{r["x_lo"]:.3g} … {r["x_hi"]:.3g}' for r in rows]
        y = arr([r['mean_fwd_ticks'] for r in rows])
        fig = go.Figure(go.Bar(x=lab, y=y, marker_color=np.where(y >= 0, UP, DOWN),
                               customdata=np.column_stack([[r['n'] for r in rows], [r['up_share'] for r in rows], [r['down_share'] for r in rows]]),
                               hovertemplate='%{x}<br>mean fwd %{y:+.4f} ticks<br>n %{customdata[0]:,}<br>up %{customdata[1]:.1%} / down %{customdata[2]:.1%}<extra></extra>'))
        style(fig, 300, title=dict(text=f'mean forward {h} mid move by {feat} decile', font_size=13, x=0), yaxis_title='ticks')
        fig.update_xaxes(type='category')
        show(fig, 'an_dec')
    else:
        st.caption('Not enough samples for deciles.')

    left, right = st.columns(2)
    with left:
        st.markdown('**Activity features vs |forward move|** (Spearman)')
        table(pd.DataFrame([{'feature': f, **{hh: s['activity_vs_abs_move'][f][hh]['spearman'] for hh in hs}} for f in s['activity_vs_abs_move']]))
    with right:
        st.markdown('**Linear combination of directional features**')
        lc = s['linear_combination']
        table(pd.DataFrame([{'horizon': k, 'R² in sample': v['in_sample']['r2'], 'R² out of sample': v['r2_out_of_sample'], 'n': v['in_sample']['n']}
                            for k, v in lc.items()]))
        k0 = next(iter(lc), None)
        if k0:
            v = lc[k0]
            table(pd.DataFrame({'feature': v['features'], f'coef ({k0})': v['in_sample']['coef'], f't ({k0})': v['in_sample']['t']}))


OVERLAYS = {
    'EMA': ['ema_10s', 'ema_30s', 'ema_60s', 'ema_300s'],
    'SMA': ['sma_10s', 'sma_30s', 'sma_60s', 'sma_300s'],
    'Bollinger 60 s': ['bb_upper_60s', 'bb_lower_60s'],
    'Keltner 20 s': ['keltner_upper_20s', 'keltner_lower_20s'],
    'Donchian 60 s': ['donchian_high_60s', 'donchian_low_60s'],
    'VWAP': ['vwap_session', 'vwap_60s'],
}
OSCILLATORS = {
    'RSI': ['rsi_14s', 'rsi_60s'],
    'MACD': ['macd', 'macd_signal', 'macd_hist'],
    'Stochastic': ['stoch_k_14s', 'stoch_d_14s'],
    'Williams %R': ['williams_r_14s'],
    'CCI': ['cci_20s'],
    'ADX / DI': ['adx_14s', 'plus_di_14s', 'minus_di_14s'],
    'ATR': ['atr_14s'],
    'ROC': ['roc_10s', 'roc_60s'],
    'Z-score': ['zscore_60s'],
    'CVD': ['cvd'],
    'Realized vol': ['realized_vol_60s'],
}


def indicators_tab(rep: dict, hrs: bool) -> None:
    ind = rep['indicators']
    st.caption(ind['note'] + '.')
    sr = ind['series']
    d = sr['data']
    x = ts((rep['tape']['start_s'] or 0.0) + arr(d['t_s']))
    c = st.columns(2)
    ov = c[0].multiselect('Overlays on price', list(OVERLAYS), default=['EMA', 'Bollinger 60 s'], key='an_ind_ov')
    osc = c[1].multiselect('Panels below', list(OSCILLATORS), default=['RSI', 'MACD'], key='an_ind_osc')
    rows = 1 + len(osc)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5] + [0.5 / max(len(osc), 1)] * len(osc),
                        subplot_titles=['mid + overlays'] + osc)
    Line = go.Scattergl if len(x) > 20000 else go.Scatter
    fig.add_trace(Line(x=x, y=arr(d['mid']), name='mid', mode='lines', line=dict(color='rgba(200,200,200,0.9)', width=1.2)), row=1, col=1)
    i = 0
    for g in ov:
        for col in OVERLAYS[g]:
            if col in d:
                fig.add_trace(Line(x=x, y=arr(d[col]), name=col, mode='lines', line=dict(width=1.1, color=PALETTE[i % len(PALETTE)])), row=1, col=1)
                i += 1
    for r_, g in enumerate(osc, 2):
        for j, col in enumerate(OSCILLATORS[g]):
            if col not in d:
                continue
            if col == 'macd_hist':
                y = arr(d[col])
                fig.add_trace(go.Bar(x=x, y=y, name=col, marker_color=np.where(y >= 0, UP, DOWN), marker_line_width=0), row=r_, col=1)
            else:
                fig.add_trace(Line(x=x, y=arr(d[col]), name=col, mode='lines', line=dict(width=1.1, color=PALETTE[j % len(PALETTE)])), row=r_, col=1)
    style(fig, 380 + 170 * len(osc), hovermode='x unified', dragmode='pan', bargap=0)
    fig.update_annotations(font_size=12, x=0, xanchor='left')
    time_axes(fig, hrs)
    show(fig, 'an_ind_chart')

    ev = ind['evaluation']
    names = [r['indicator'] for r in ind['ranking_by_abs_spearman_5s']]
    hs = [h for h in next(iter(ev.values())) if h not in ('feature', 'deciles_5s')] if ev else []
    metric = st.segmented_control('Metric', ['spearman', 'pearson', 'hit_rate'], default='spearman', key='an_ind_metric') or 'spearman'
    z = np.array([[ev[n][h].get(metric) if ev[n][h].get(metric) is not None else np.nan for h in hs] for n in names], dtype=float)
    center = 0.5 if metric == 'hit_rate' else 0.0
    span = np.nanmax(np.abs(z - center)) if np.isfinite(z).any() else 1.0
    hm = go.Figure(go.Heatmap(x=hs, y=names, z=z, zmid=center, zmin=center - span, zmax=center + span, colorscale='RdBu',
                              text=[[f'{v:+.3f}' if np.isfinite(v) else '' for v in row] for row in z], texttemplate='%{text}',
                              hovertemplate='%{y} → fwd %{x}: %{z:.4f}<extra></extra>', colorbar=dict(thickness=10)))
    style(hm, 120 + 22 * len(names), title=dict(text=f'{metric} of indicator vs forward mid move (sorted by |Spearman 5 s|)', font_size=13, x=0))
    hm.update_yaxes(autorange='reversed')
    show(hm, 'an_ind_ic')
    feat = st.selectbox('Decile chart (5 s forward move) for', names, key='an_ind_dec')
    rows_ = ev[feat]['deciles_5s'] if feat else []
    if rows_:
        y = arr([r['mean_fwd_ticks'] for r in rows_])
        f2 = go.Figure(go.Bar(x=[f'{r["x_lo"]:.3g} … {r["x_hi"]:.3g}' for r in rows_], y=y, marker_color=np.where(y >= 0, UP, DOWN),
                              hovertemplate='%{x}<br>mean fwd %{y:+.4f} ticks<extra></extra>'))
        style(f2, 300, title=dict(text=f'mean forward 5 s mid move by {feat} decile', font_size=13, x=0), yaxis_title='ticks')
        f2.update_xaxes(type='category')
        show(f2, 'an_ind_dec_chart')


def regimes_tab(rep: dict) -> None:
    g = rep['regimes']
    st.caption(f'Relationships across the {g["bins"]} bins of {rep["bins"]["step_s"]:g} s.')
    left, right = st.columns(2)
    with left:
        rows = [{'pair': k.removeprefix('corr_').replace('_', ' '), 'pearson': v['pearson'], 'spearman': v['spearman'], 'n': v['n']}
                for k, v in g.items() if k.startswith('corr_')]
        table(pd.DataFrame(rows))
        c = st.columns(2)
        c[0].metric('CV of volume', num(g['cv_volume'], '.2f'))
        c[1].metric('CV of realized vol', num(g['cv_realized_vol'], '.2f'))
    with right:
        rows = [{'series': k.removeprefix('autocorr_'), **v} for k, v in g.items() if k.startswith('autocorr_')]
        table(pd.DataFrame(rows))
    d = rep['bins']['data']
    cols = [k for k in rep['bins']['columns'] if k not in ('t_start_s', 't_end_s', 'bid_depth_mean', 'ask_depth_mean')]
    c = st.columns(2)
    xc = c[0].selectbox('x', cols, index=cols.index('realized_vol_ticks') if 'realized_vol_ticks' in cols else 0, key='an_sc_x')
    yc = c[1].selectbox('y', cols, index=cols.index('spread_mean') if 'spread_mean' in cols else 0, key='an_sc_y')
    fig = go.Figure(go.Scatter(x=arr(d[xc]), y=arr(d[yc]), mode='markers', marker=dict(color=arr(d['t_start_s']), colorscale='Viridis', size=7,
                                                                                        colorbar=dict(title='t (s)', thickness=10)),
                               hovertemplate=f'{xc} %{{x:.4g}}<br>{yc} %{{y:.4g}}<extra></extra>'))
    style(fig, 380, title=dict(text=f'{yc} vs {xc}, one point per bin', font_size=13, x=0), xaxis_title=xc, yaxis_title=yc)
    show(fig, 'an_scatter')


def raw_tab(rep: dict, name: str) -> None:
    stem = name.removesuffix('.gz').removesuffix('.csv')
    c = st.columns(3)
    c[0].download_button('Download JSON', to_json(rep), f'{stem}.json', 'application/json', width='stretch')
    c[1].download_button('Download text', to_text(rep), f'{stem}.txt', 'text/plain', width='stretch')
    c[2].caption(f'schema v{rep["meta"]["schema_version"]} · generated {rep["meta"]["generated_at"]} · {num(rep["meta"]["runtime_s"]["total"], ".1f")} s')
    st.json({k: ({kk: vv for kk, vv in v.items() if kk != 'series'} if k == 'indicators' else v) for k, v in rep.items() if k not in ('series', 'bins')}, expanded=1)


# ---------------------------------------------------------------- entry point
def render(source: str, name: str, csv: Path | None, data: bytes | None) -> None:
    """Draw the Analysis tab for one tape: a path in data/ (csv) or an upload (data)."""
    rep = None
    saved = saved_report(csv, name) if csv is not None else None
    if saved is not None:
        p = report_path(name)
        rep = _load_saved(str(p), p.stat().st_mtime)
    with st.expander('Analysis settings', expanded=rep is None):
        c = st.columns([1, 1, 2], vertical_alignment='bottom')
        series_step = c[0].number_input('Series step (s)', 0.1, 600.0, 1.0, key='an_series_step')
        bin_step = c[1].number_input('Bin step (s)', 1.0, 3600.0, 60.0, key='an_bin_step')
        if c[2].button('Run analysis' if rep is None else 'Re-run analysis', type='primary' if rep is None else 'secondary', key='an_run'):
            st.session_state[f'an_requested:{source}'] = (series_step, bin_step)
        if rep is not None:
            st.caption(f'Loaded {report_path(name).relative_to(ROOT)} (generated {rep["meta"]["generated_at"]}).')
    req = st.session_state.get(f'an_requested:{source}')
    if req is not None:
        rep = _run(source, None if csv is None else str(csv), data, name, float(req[0]), float(req[1]))
    if rep is None:
        st.info(f'No saved analysis for {name}. Press **Run analysis** (about 12 s per million rows), '
                f'or run `uv run analysis.py data/{name}` and reload.')
        return

    hrs = (rep['tape']['end_s'] or 0) >= 3600
    overview_page(rep, hrs)
    st.divider()
    section = st.segmented_control('Details', ['Market & book', 'Flow & execution', 'Signals', 'PnL bounds', 'Data'], default='Market & book',
                                   key='an_section') or 'Market & book'
    if section == 'Market & book':
        market_section(rep, hrs)
    elif section == 'Flow & execution':
        flow_section(rep)
    elif section == 'Signals':
        signals_section(rep, hrs)
    elif section == 'PnL bounds':
        max_pnl_tab(rep, hrs)
    else:
        data_section(rep, name, hrs)
