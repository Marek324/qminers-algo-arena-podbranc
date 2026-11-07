"""Streamlit viewer for raw market-data CSVs (tape): candles, trades, order book and breakdowns over time.

Run:  uv run --with streamlit streamlit run viewer/app.py
"""

from __future__ import annotations

import hashlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
UP, DOWN, ACCENT = '#26a69a', '#ef5350', '#f5c542'
GRID = 'rgba(128,128,128,0.15)'
CLEAR = 'rgba(0,0,0,0)'
WINDOWS = {'10 s': 10, '1 min': 60, '5 min': 300, '15 min': 900, '1 h': 3600, 'All': None}
AUTO_BIN = {'10 s': 0.1, '1 min': 1, '5 min': 5, '15 min': 10, '1 h': 30, 'All': 60}
BIN_CHOICES = {'Auto': None, '100 ms': 0.1, '1 s': 1, '5 s': 5, '10 s': 10, '30 s': 30, '1 min': 60, '5 min': 300}
PLOT_CONFIG = {'scrollZoom': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']}
CSS = f"""<style>
.block-container {{ padding-top: 1.5rem; }}
.stApp {{ font-variant-numeric: tabular-nums; }}
[data-testid="stMetricValue"] {{ font-size: 1.3rem; }}
.ttl {{ font-size: 1.15rem; font-weight: 600; margin-right: .8rem; }}
.muted {{ opacity: .6; font-size: .85rem; }}
.tread {{ font: 600 2rem/1.1 ui-monospace, SFMono-Regular, monospace; text-align: right; }}
.tsub {{ opacity: .6; font-size: .8rem; text-align: right; }}
.ladder {{ font-family: ui-monospace, SFMono-Regular, monospace; font-size: .85rem; }}
.lrow {{ display: grid; grid-template-columns: 1.2fr 1fr .6fr; text-align: right; padding: 1px 8px; --c: transparent;
  background: linear-gradient(to left, var(--c) var(--w, 0%), transparent var(--w, 0%)); }}
.lrow.ask {{ --c: rgba(239,83,80,.30); }} .lrow.ask span:first-child {{ color: {DOWN}; }}
.lrow.bid {{ --c: rgba(38,166,154,.30); }} .lrow.bid span:first-child {{ color: {UP}; }}
.lrow.best {{ font-weight: 700; }}
.lhead {{ opacity: .6; font-size: .75rem; }}
.lspread {{ text-align: center; opacity: .75; padding: 3px 0; margin: 2px 0; border-top: 1px solid rgba(128,128,128,.3);
  border-bottom: 1px solid rgba(128,128,128,.3); }}
</style>"""

st.set_page_config(page_title='Market viewer', layout='wide')


# ---------------------------------------------------------------- loading (cached)
@st.cache_resource(show_spinner='Parsing the CSV and replaying it through the matching engine...')
def load_path(path: str, mtime: float, size: int) -> engine.Replay:
    return engine.replay(engine.parse_csv(path))


@st.cache_resource(show_spinner='Parsing the CSV and replaying it through the matching engine...')
def load_bytes(digest: str, _data: bytes) -> engine.Replay:
    return engine.replay(engine.parse_csv(io.BytesIO(_data)))


@st.cache_resource(show_spinner=False)
def mid_series(source: str, _r: engine.Replay) -> np.ndarray:
    """Mid price after every event; NaN while a side is empty."""
    mid = (_r.best_bid.astype(np.float64) + _r.best_ask.astype(np.float64)) / 2
    mid[(_r.best_bid < 0) | (_r.best_ask < 0)] = np.nan
    return mid


@st.cache_data(show_spinner=False)
def nav_series(source: str, _r: engine.Replay) -> tuple[np.ndarray, np.ndarray]:
    """Whole-session mid price as a min/max envelope of about 3k points: (x seconds, y)."""
    mid = mid_series(source, _r)
    ok = ~np.isnan(mid)
    x, y = _r.tape.time[ok], mid[ok]
    if len(y) == 0:
        return np.array([0.0, 1.0]), np.array([np.nan, np.nan])
    idx = np.arange(0, len(y), max(1, -(-len(y) // 1500)))
    return np.repeat(x[idx], 2), np.column_stack([np.minimum.reduceat(y, idx), np.maximum.reduceat(y, idx)]).ravel()


def fmt_t(t: float, hrs: bool = False) -> str:
    ms = int(round(t * 1000))
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    if hrs:
        h, m = divmod(m, 60)
        return f'{h:d}:{m:02d}:{s:02d}.{ms:03d}'
    return f'{m:02d}:{s:02d}.{ms:03d}'


# ---------------------------------------------------------------- state callbacks
def set_pos(pos: int) -> None:
    st.session_state['pos'] = int(min(max(pos, 0), st.session_state['_r'].n - 1))


def on_slider() -> None:
    set_pos(int(np.searchsorted(st.session_state['_r'].tape.time, st.session_state['tslider'], side='right')) - 1)


def on_step(sign: int) -> None:
    set_pos(st.session_state['pos'] + sign * (st.session_state.get('stepsize') or 1))


def on_goto() -> None:
    set_pos(st.session_state['goto'])


def on_trade_jump(direction: int) -> None:
    r = st.session_state['_r']
    pos = st.session_state['pos']
    if direction > 0:
        k = int(np.searchsorted(r.t_ev, pos, side='right'))
        if k < len(r.t_ev):
            set_pos(int(r.t_ev[k]))
    else:
        k = int(np.searchsorted(r.t_ev, pos, side='left')) - 1
        if k >= 0:
            set_pos(int(r.t_ev[k]))


# ---------------------------------------------------------------- binning (numpy only)
def ohlc(x: np.ndarray, y: np.ndarray, base: float, bw: float, nb: int, carry_open: bool):
    """Per-bin open/high/low/close of y over the events x (sorted). Returns (bin index, o, h, l, c) for non-empty bins."""
    if len(x) == 0:
        return None
    idx = np.searchsorted(x, base + np.arange(nb + 1) * bw, side='left')
    s, e = idx[:-1], idx[1:]
    ne = e > s
    if not ne.any():
        return None
    bins, s, e = np.nonzero(ne)[0], s[ne], e[ne]
    yy = y[: idx[-1]]
    hi, lo = np.maximum.reduceat(yy, s), np.minimum.reduceat(yy, s)
    o, c = y[s], y[e - 1]
    if carry_open:  # state-like series (mid): a candle opens where the previous one closed
        o = np.where(s > 0, y[np.maximum(s - 1, 0)], o)
        hi, lo = np.maximum(hi, o), np.minimum(lo, o)
    return bins, o, hi, lo, c


def window_range(t: float, wsec, t0: float, t1: float) -> tuple[float, float]:
    """Trailing window [t - 0.85 W, t + 0.15 W], shifted to stay inside the session."""
    if wsec is None:
        return t0, t1
    wsec = min(wsec, t1 - t0)
    x0, x1 = t - 0.85 * wsec, t + 0.15 * wsec
    if x0 < t0:
        x0, x1 = t0, t0 + wsec
    if x1 > t1:
        x0, x1 = t1 - wsec, t1
    return x0, x1


# ---------------------------------------------------------------- figures
def style_time_axes(fig: go.Figure, hrs: bool, **kw) -> None:
    f = '%H:%M:%S' if hrs else '%M:%S'
    fig.update_xaxes(type='date', gridcolor=GRID, hoverformat=f + '.%L', zeroline=False,
                     tickformatstops=[dict(dtickrange=[None, 1000], value=f + '.%L'), dict(dtickrange=[1000, None], value=f)], **kw)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)


def cursor_shapes(t: float, x1: float | None) -> list[dict]:
    """Dashed cursor line, plus a faint grey rectangle over the not-yet-happened part when x1 is given."""
    shapes = [dict(type='line', xref='x', yref='paper', x0=t * 1000, x1=t * 1000, y0=0, y1=1, line=dict(color=ACCENT, width=1, dash='dash'))]
    if x1 is not None:
        shapes.insert(0, dict(type='rect', xref='x', yref='paper', x0=t * 1000, x1=x1 * 1000, y0=0, y1=1, fillcolor='rgba(128,128,128,0.10)',
                              line_width=0, layer='above'))
    return shapes


def main_fig(r: engine.Replay, mid: np.ndarray, x0: float, x1: float, t: float, bw: float, cand_src: str, show_ba: bool, hrs: bool,
             revision: str) -> go.Figure:
    tm = r.tape.time
    base = np.floor(x0 / bw) * bw
    nb = int(np.ceil((x1 - base) / bw)) + 1
    ta = r.t_time
    a, b = int(np.searchsorted(ta, x0, 'left')), int(np.searchsorted(ta, x1, 'right'))
    i0 = max(int(np.searchsorted(tm, x0, 'left')) - 1, 0)
    i1 = int(np.searchsorted(tm, x1, 'right'))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.02)

    # candles
    if cand_src == 'Mid':
        y = mid[i0:i1]
        ok = ~np.isnan(y)
        res = ohlc(tm[i0:i1][ok], y[ok], base, bw, nb, True)
    else:
        res = ohlc(ta[a:b], r.t_price[a:b].astype(float), base, bw, nb, False)
    ymin = ymax = None
    if res is not None:
        bins, o, h, l, c = res
        ymin, ymax = float(l.min()), float(h.max())
        text = [f'{fmt_t(base + k * bw, hrs)}<br>O {p:,.1f}  H {q:,.1f}<br>L {u:,.1f}  C {v:,.1f}' for k, p, q, u, v in zip(bins, o, h, l, c)]
        fig.add_trace(go.Candlestick(x=(base + (bins + 0.5) * bw) * 1000, open=o, high=h, low=l, close=c, name='Candles', showlegend=False, text=text,
                                     hoverinfo='text', increasing=dict(line=dict(color=UP), fillcolor=UP),
                                     decreasing=dict(line=dict(color=DOWN), fillcolor=DOWN)), row=1, col=1)

    # best bid / ask overlay
    if show_ba:
        for name, arr, color in (('best bid', r.best_bid, UP), ('best ask', r.best_ask, DOWN)):
            yy = arr[i0:i1].astype(float)
            yy[yy < 0] = np.nan
            xx = tm[i0:i1]
            shape = 'hv'
            if len(yy) > 6000:
                idx = np.arange(0, len(yy), -(-len(yy) // 3000))
                xx, yy = np.repeat(xx[idx], 2), np.column_stack([np.fmin.reduceat(yy, idx), np.fmax.reduceat(yy, idx)]).ravel()
                shape = 'linear'
            fig.add_trace(go.Scatter(x=xx * 1000, y=yy, mode='lines', name=name, opacity=0.4, line=dict(color=color, width=1, shape=shape),
                                     hoverinfo='skip'), row=1, col=1)

    # trade markers
    sel = np.arange(a, b)
    capped = len(sel) > 4000
    if capped:
        sel = sel[:: -(-len(sel) // 4000)]
    if len(sel):
        pr = r.t_price[sel].astype(float)
        off = max(0.03 * ((ymax - ymin) if ymax is not None and ymax > ymin else 4.0), 0.3)
        for side, ioc, sym, col, nm in ((0, 1, 'triangle-up', UP, 'Buy aggressor, IOC'), (0, 0, 'triangle-up-open', UP, 'Buy aggressor, limit'),
                                        (1, 1, 'triangle-down', DOWN, 'Sell aggressor, IOC'), (1, 0, 'triangle-down-open', DOWN, 'Sell aggressor, limit')):
            m = (r.t_side[sel] == side) & (r.t_ioc[sel] == ioc)
            if not m.any():
                continue
            s = sel[m]
            q = r.t_qty[s]
            fig.add_trace(go.Scatter(
                x=ta[s] * 1000, y=pr[m] + (-off if side == 0 else off), mode='markers', name=nm,
                marker=dict(symbol=sym, color=col, size=np.clip(6 + 1.6 * np.sqrt(q), 7, 20), line=dict(width=1.5, color=col)),
                customdata=np.column_stack([[fmt_t(v, hrs) for v in ta[s]], r.t_price[s], q]),
                hovertemplate='%{customdata[0]}<br>price %{customdata[1]:,}  qty %{customdata[2]:,}<extra>' + nm + '</extra>'), row=1, col=1)

    # volume bars (same bins as the candles)
    k_ = np.clip(((ta[a:b] - base) // bw).astype(int), 0, nb - 1)
    side_, qty_ = r.t_side[a:b], r.t_qty[a:b]
    buy = np.bincount(k_[side_ == 0], weights=qty_[side_ == 0], minlength=nb)
    sell = np.bincount(k_[side_ == 1], weights=qty_[side_ == 1], minlength=nb)
    nz = np.nonzero((buy > 0) | (sell > 0))[0]
    cd = np.column_stack([[fmt_t(base + k * bw, hrs) for k in nz], buy[nz], sell[nz], buy[nz] - sell[nz]]) if len(nz) else None
    for name, v, col in (('Buy aggressor', buy[nz], UP), ('Sell aggressor', -sell[nz], DOWN)):
        fig.add_trace(go.Bar(x=(base + (nz + 0.5) * bw) * 1000, y=v, width=bw * 1000 * 0.9, name=name, marker_color=col, showlegend=False, customdata=cd,
                             hovertemplate='%{customdata[0]}<br>buy %{customdata[1]:,}  sell %{customdata[2]:,}  net %{customdata[3]:,}<extra></extra>'),
                      row=2, col=1)

    style_time_axes(fig, hrs)
    fig.update_xaxes(range=[x0 * 1000, x1 * 1000], rangeslider_visible=False)
    fig.update_yaxes(tickformat=',.2~f', row=1, col=1)
    fig.update_yaxes(title_text='lots', row=2, col=1)
    fig.update_layout(height=560, margin=dict(l=50, r=10, t=10, b=30), paper_bgcolor=CLEAR, plot_bgcolor=CLEAR, dragmode='zoom', barmode='relative',
                      hovermode='closest', uirevision=revision, shapes=cursor_shapes(t, x1),
                      legend=dict(orientation='h', y=1.0, x=0, yanchor='bottom', font_size=11))
    if capped:
        fig.add_annotation(text='trade markers subsampled', xref='paper', yref='paper', x=1, y=1.0, yanchor='bottom', xanchor='right', showarrow=False,
                           font=dict(size=10), opacity=0.6)
    return fig


def nav_fig(nx: np.ndarray, ny: np.ndarray, x0: float, x1: float, t: float, hrs: bool) -> go.Figure:
    fig = go.Figure(go.Scatter(x=nx * 1000, y=ny, mode='lines+markers', line=dict(color='rgba(128,128,128,0.8)', width=1),
                               marker=dict(size=7, opacity=0), selected=dict(marker=dict(opacity=0)), unselected=dict(marker=dict(opacity=0)),
                               hoverinfo='skip'))
    fig.update_layout(height=110, margin=dict(l=50, r=10, t=4, b=24), paper_bgcolor=CLEAR, plot_bgcolor=CLEAR, showlegend=False, uirevision='nav',
                      shapes=[dict(type='rect', xref='x', yref='paper', x0=x0 * 1000, x1=x1 * 1000, y0=0, y1=1, fillcolor='rgba(245,197,66,0.15)', line_width=0),
                              dict(type='line', xref='x', yref='paper', x0=t * 1000, x1=t * 1000, y0=0, y1=1, line=dict(color=ACCENT, width=1.5))])
    style_time_axes(fig, hrs, fixedrange=True)
    fig.update_yaxes(fixedrange=True, tickformat=',.2~f', nticks=3)
    return fig


def depth_fig(book: engine.Book, ticks: int = 40) -> go.Figure:
    bids, asks = book.levels()
    fig = go.Figure()
    if bids:
        b = [x for x in bids if x[0] >= bids[0][0] - ticks]
        p, v = np.array([x[0] for x in b]), np.array([x[1] for x in b])
        fig.add_trace(go.Bar(x=p, y=v, name='Bid level', marker_color=UP, opacity=0.5))
        fig.add_trace(go.Scatter(x=p[::-1], y=np.cumsum(v)[::-1], name='Bid cumulative', line=dict(color=UP, shape='vh'), fill='tozeroy',
                                 fillcolor='rgba(38,166,154,0.15)'))
    if asks:
        a = [x for x in asks if x[0] <= asks[0][0] + ticks]
        p, v = np.array([x[0] for x in a]), np.array([x[1] for x in a])
        fig.add_trace(go.Bar(x=p, y=v, name='Ask level', marker_color=DOWN, opacity=0.5))
        fig.add_trace(go.Scatter(x=p, y=np.cumsum(v), name='Ask cumulative', line=dict(color=DOWN, shape='hv'), fill='tozeroy',
                                 fillcolor='rgba(239,83,80,0.15)'))
    fig.update_layout(height=320, margin=dict(l=50, r=10, t=10, b=30), paper_bgcolor=CLEAR, plot_bgcolor=CLEAR, barmode='overlay',
                      legend=dict(orientation='h', y=1.0, x=0, yanchor='bottom'), xaxis_title='price (ticks)', yaxis_title='lots')
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def cumulative_fig(r: engine.Replay, t: float, t0: float, hrs: bool, revision: str) -> go.Figure:
    T = len(r.t_ev)
    idx = np.arange(0, T + 1, max(1, T // 4000))
    tx = np.concatenate([[t0], r.t_time])[idx] * 1000
    fig = go.Figure()
    for name, rows_, col in (('Buy aggressor', (0, 2), UP), ('Sell aggressor', (1, 3), DOWN)):
        fig.add_trace(go.Scatter(x=tx, y=r.cum_qty[list(rows_)].sum(0)[idx], name=name, line=dict(color=col, shape='hv', width=1.5)))
    style_time_axes(fig, hrs)
    fig.update_layout(height=300, margin=dict(l=50, r=10, t=10, b=30), paper_bgcolor=CLEAR, plot_bgcolor=CLEAR, uirevision=revision,
                      shapes=cursor_shapes(t, None), legend=dict(orientation='h', y=1.0, x=0, yanchor='bottom'), yaxis_title='cumulative lots')
    return fig


# ---------------------------------------------------------------- ladder (HTML)
def ladder_html(bids: list, asks: list, n: int) -> str:
    a, b = asks[:n], bids[:n]
    mx = max([v for _, v, _ in a + b], default=1)

    def row(kind: str, lv: tuple, best: bool) -> str:
        p, v, c = lv
        return f'<div class="lrow {kind}{" best" if best else ""}" style="--w:{100 * v / mx:.1f}%"><span>{p:,}</span><span>{v:,}</span><span>{c}</span></div>'

    out = ['<div class="ladder"><div class="lrow lhead"><span>Price</span><span>Lots</span><span>Orders</span></div>']
    out += [row('ask', lv, j == len(a) - 1) for j, lv in enumerate(reversed(a))] if a else ['<div class="lspread">no asks</div>']
    if a and b:
        out.append(f'<div class="lspread">spread {asks[0][0] - bids[0][0]:,} &middot; mid {(asks[0][0] + bids[0][0]) / 2:,.1f}</div>')
    else:
        out.append('<div class="lspread">&nbsp;</div>')
    out += [row('bid', lv, j == 0) for j, lv in enumerate(b)] if b else ['<div class="lspread">no bids</div>']
    return ''.join(out) + '</div>'


# ---------------------------------------------------------------- tables
def trades_table(r: engine.Replay, k: int, hrs: bool, ids: bool) -> None:
    if not k:
        st.caption('No trades yet.')
        return
    j = np.arange(k - 1, max(0, k - 50) - 1, -1)
    df = pd.DataFrame({'Time': [fmt_t(v, hrs) for v in r.t_time[j]], 'Side': np.where(r.t_side[j] == 0, 'BUY', 'SELL'), 'Price': r.t_price[j],
                       'Qty': r.t_qty[j], 'Type': np.where(r.t_ioc[j] == 1, 'IOC', 'Limit')})
    if ids:
        df['Taker id'], df['Maker id'] = r.t_taker[j], r.t_maker[j]
    colour = lambda v: f'color: {UP if v == "BUY" else DOWN}; font-weight: 600'  # noqa: E731
    sty = (df.style.map(colour, subset=['Side']) if hasattr(df.style, 'map') else df.style.applymap(colour, subset=['Side'])).format(
        {'Price': '{:,}', 'Qty': '{:,}'})
    st.dataframe(sty, hide_index=True, width='stretch', height=min(40 + 35 * len(df), 520))


def event_log(r: engine.Replay, pos: int, half: int, hrs: bool, ids: bool) -> None:
    tp = r.tape
    lo, hi = max(0, pos - half), min(r.n, pos + half + 1)
    k = int(np.searchsorted(r.t_ev, lo, 'left'))
    rows, styles = [], []
    for i in range(lo, hi):
        fills = []
        while k < len(r.t_ev) and r.t_ev[k] == i:
            fills.append(k)
            k += 1
        style = 'background-color: rgba(245,197,66,0.25); font-weight: 600' if i == pos else ('opacity: 0.5' if i > pos else '')
        if tp.is_cancel[i]:
            q = int(r.cancel_qty[i])
            res = f'Cancelled, removed {q:,} lots' if q else 'Ignored, order already gone'
            rows.append([fmt_t(tp.time[i], hrs), 'Cancel', '', None, None, '', res, int(tp.oid[i])])
        else:
            traded = int(sum(r.t_qty[f] for f in fills))
            vol = int(tp.vol[i])
            rem = vol - traded
            if traded and rem == 0:
                res = f'Filled {traded:,} lots'
            elif traded:
                res = f'Filled {traded:,} of {vol:,}, ' + ('rest cancelled' if tp.ioc[i] else f'{rem:,} rest in book')
            else:
                res = 'No fill, cancelled' if tp.ioc[i] else 'Rests in book'
            rows.append([fmt_t(tp.time[i], hrs), 'Insert', 'Buy' if tp.side[i] == 0 else 'Sell', int(tp.price[i]), vol,
                         'IOC' if tp.ioc[i] else 'Limit', res, int(tp.oid[i])])
        styles.append(style)
        for f in fills:
            tint = 'background-color: rgba(38,166,154,0.12)' if r.t_side[f] == 0 else 'background-color: rgba(239,83,80,0.12)'
            rows.append(['', '   ↳ trade', 'Buy' if r.t_side[f] == 0 else 'Sell', int(r.t_price[f]), int(r.t_qty[f]), 'IOC' if r.t_ioc[f] else 'Limit',
                         f'{"Buyer" if r.t_side[f] == 0 else "Seller"} took lots from a resting order', int(r.t_maker[f])])
            styles.append(style if i == pos else tint + ('; opacity: 0.5' if i > pos else ''))
    df = pd.DataFrame(rows, columns=['Time', 'Event', 'Side', 'Price', 'Qty', 'Type', 'Result', 'OrderId'])
    for col in ('Price', 'Qty'):
        df[col] = df[col].astype('Int64')
    if not ids:
        df = df.drop(columns='OrderId')
    sty = df.style.apply(lambda row: [styles[row.name]] * len(row), axis=1).format({'Price': '{:,}', 'Qty': '{:,}'}, na_rep='')
    st.dataframe(sty, hide_index=True, width='stretch', height=min(40 + 35 * len(df), 900))
    st.caption('Highlighted row = current event (already applied); faded rows have not happened yet.')


def order_flow(cur: np.ndarray, tot: np.ndarray, filled_cur: float, filled_tot: float) -> None:
    def s(c, idx):
        return int(c[list(idx)].sum())

    spec = [('Inserts, limit', (0, 1), (4, 5)), ('Inserts, IOC', (2, 3), (6, 7)), ('Buys', (0, 2), (4, 6)), ('Sells', (1, 3), (5, 7))]
    rows = [{'': n, 'Orders to t': s(cur, ci), 'Orders in file': s(tot, ci), 'Lots to t': s(cur, vi), 'Lots in file': s(tot, vi)} for n, ci, vi in spec]
    rows.append({'': 'Cancels that removed an order', 'Orders to t': int(cur[8]), 'Orders in file': int(tot[8]), 'Lots to t': int(cur[10]),
                 'Lots in file': int(tot[10])})
    rows.append({'': 'Cancels ignored (order already gone)', 'Orders to t': int(cur[9]), 'Orders in file': int(tot[9]), 'Lots to t': None,
                 'Lots in file': None})
    df = pd.DataFrame(rows)
    num = list(df.columns[1:])
    for c in num:
        df[c] = df[c].astype('Int64')
    st.dataframe(df.style.format('{:,}', subset=num, na_rep=''), hide_index=True, width='stretch')
    fig = go.Figure()
    for name, filled, ins in (('Whole file', filled_tot, tot[6] + tot[7]), ('To t', filled_cur, cur[6] + cur[7])):
        fig.add_trace(go.Bar(y=[name], x=[filled], orientation='h', name='IOC lots filled', marker_color=UP, showlegend=name == 'To t'))
        fig.add_trace(go.Bar(y=[name], x=[max(ins - filled, 0)], orientation='h', name='IOC lots cancelled', marker_color=DOWN, showlegend=name == 'To t'))
    fig.update_layout(barmode='stack', height=170, margin=dict(l=70, r=10, t=30, b=20), paper_bgcolor=CLEAR, plot_bgcolor=CLEAR,
                      title=dict(text='IOC orders: filled vs cancelled lots', font_size=13, x=0),
                      legend=dict(orientation='h', y=1.0, x=1, xanchor='right', yanchor='bottom'))
    fig.update_xaxes(gridcolor=GRID)
    st.plotly_chart(fig, width='stretch', key='ioc', theme='streamlit')


# ---------------------------------------------------------------- main view (a fragment: reruns alone while playing)
def view(r: engine.Replay, source: str, opts: dict) -> None:
    ss = st.session_state
    tp = r.tape
    n = r.n
    t0, t1 = float(tp.time[0]), float(tp.time[-1])
    if t1 <= t0:
        t1 = t0 + 1
    hrs = t1 >= 3600

    # navigator click: handled once per new selection, before anything uses `pos`
    nx, ny = nav_series(source, r)
    try:
        pts = ss['nav']['selection']['points']
    except (KeyError, TypeError):
        pts = []
    if pts:
        pi = pts[0].get('point_index')
        if pi is not None and ss.get('_nav_last') != (source, pi):
            ss['_nav_last'] = (source, pi)
            ss['pos'] = max(int(np.searchsorted(tp.time, nx[min(int(pi), len(nx) - 1)], side='right')) - 1, 0)
            ss['playing'] = False
    else:
        ss['_nav_last'] = None

    if ss.get('playing'):
        now = time.time()
        dt = min(now - ss.get('last_tick', now), 5.0)
        ss['last_tick'] = now
        pos = int(np.searchsorted(tp.time, tp.time[ss['pos']] + dt * (ss.get('speed') or 1), side='right')) - 1
        pos = max(pos, ss['pos'] + 1)
        if pos >= n - 1:
            pos, ss['playing'] = n - 1, False
        ss['pos'] = pos
    pos = ss['pos']
    t = float(tp.time[pos])
    ss['tslider'] = t

    # ---- transport bar
    with st.container(border=True):
        c = st.columns([0.7, 0.55, 0.55, 0.55, 0.55, 1.5, 1.5, 1.6], vertical_alignment='center')
        if c[0].button(':material/pause:' if ss.get('playing') else ':material/play_arrow:', type='primary', help='Play / pause', width='stretch'):
            ss['playing'] = not ss.get('playing', False)
            if ss['playing'] and pos >= n - 1:
                ss['pos'] = 0
            ss['last_tick'] = time.time()
            st.rerun()
        c[1].button(':material/skip_previous:', help='Previous trade', on_click=on_trade_jump, args=(-1,), width='stretch', key='prevtrade')
        c[2].button(':material/chevron_left:', help='Step back', on_click=on_step, args=(-1,), width='stretch', key='back')
        c[3].button(':material/chevron_right:', help='Step forward', on_click=on_step, args=(1,), width='stretch', key='fwd')
        c[4].button(':material/skip_next:', help='Next trade', on_click=on_trade_jump, args=(1,), width='stretch', key='nexttrade')
        c[5].segmented_control('Step (events)', [1, 10, 100, 1000], default=1, key='stepsize', format_func=lambda v: f'{v:,}' if v < 1000 else '1k',
                               help='Events per step', label_visibility='collapsed')
        c[6].segmented_control('Speed', [1, 10, 60, 300], default=10, key='speed', format_func=lambda v: f'{v}x',
                               help='Play speed: simulated seconds per real second', label_visibility='collapsed')
        c[7].markdown(f'<div class="tread">{fmt_t(t, hrs)}</div><div class="tsub">event {pos:,} / {n - 1:,}</div>', unsafe_allow_html=True)
        st.slider('Time', t0, t1, key='tslider', step=max((t1 - t0) / 20000, 1e-6), format='%.3f', on_change=on_slider, label_visibility='collapsed')

    # ---- window + chart options
    o = st.columns([3, 1.4, 1.2], vertical_alignment='center')
    wlabel = o[0].segmented_control('Window', list(WINDOWS), default='5 min', key='window') or '5 min'
    cand_src = o[1].segmented_control('Candles from', ['Mid', 'Trades'], default='Mid', key='candsrc') or 'Mid'
    show_ba = o[2].toggle('Bid/ask lines', value=False, key='showba')
    x0, x1 = window_range(t, WINDOWS[wlabel], t0, t1)
    bw = opts['bin'] or AUTO_BIN[wlabel]
    bw *= max(1, int(np.ceil((x1 - x0) / bw / 2000)))

    book = engine.book_at(r, pos)
    bids, asks = book.levels()
    k = engine.trade_count_through(r, pos)
    cur = engine.counters_at(r, pos)
    tot = r.chunk_counters[-1]
    mid = mid_series(source, r)

    left, right = st.columns([3, 1])
    with left:
        st.plotly_chart(main_fig(r, mid, x0, x1, t, bw, cand_src, show_ba, hrs, f'{source}|{wlabel}'), width='stretch', config=PLOT_CONFIG,
                        theme='streamlit', key='main')
        st.plotly_chart(nav_fig(nx, ny, x0, x1, t, hrs), width='stretch', config={'displayModeBar': False}, theme='streamlit', key='nav',
                        on_select='rerun', selection_mode='points')
    with right:
        m = st.columns(3)
        m[0].metric('Bid', f'{bids[0][0]:,}' if bids else '-')
        m[1].metric('Ask', f'{asks[0][0]:,}' if asks else '-')
        m[2].metric('Spread', f'{asks[0][0] - bids[0][0]:,}' if bids and asks else '-')
        m = st.columns(2)
        m[0].metric('Last trade', f'{int(r.t_price[k - 1]):,}' if k else '-')
        ka = min(int(np.searchsorted(r.t_time, x0, 'left')), k)
        m[1].metric('Vol in window', f'{int(r.cum_qty[:, k].sum() - r.cum_qty[:, ka].sum()):,}')
        if k:
            st.caption(f'{int(r.t_qty[k - 1]):,} lots, {"buy" if r.t_side[k - 1] == 0 else "sell"} aggressor, {"IOC" if r.t_ioc[k - 1] else "limit"}, '
                       f'at {fmt_t(r.t_time[k - 1], hrs)}')
        st.markdown(ladder_html(bids, asks, opts['levels']), unsafe_allow_html=True)

    # ---- tabs
    tabs = st.tabs(['Trades', 'Event log', 'Depth', 'Order flow', 'Session totals'])
    with tabs[0]:
        trades_table(r, k, hrs, opts['ids'])
    with tabs[1]:
        event_log(r, pos, opts['log_half'], hrs, opts['ids'])
    with tabs[2]:
        st.plotly_chart(depth_fig(book), width='stretch', theme='streamlit', key='depth')
    with tabs[3]:
        order_flow(cur, tot, float(r.cum_qty[[2, 3], k].sum()), float(r.cum_qty[[2, 3], -1].sum()))
    with tabs[4]:
        total_q = r.cum_qty[:, k].sum()
        mm = st.columns(3)
        mm[0].metric('Trades to t', f'{int(r.cum_cnt[:, k].sum()):,}')
        mm[1].metric('Volume to t', f'{int(total_q):,} lots')
        mm[2].metric('VWAP to t', f'{r.cum_notional[k] / total_q:,.2f}' if total_q else '-')
        st.plotly_chart(cumulative_fig(r, t, t0, hrs, source), width='stretch', config=PLOT_CONFIG, theme='streamlit', key='cum')


# ---------------------------------------------------------------- app
st.markdown(CSS, unsafe_allow_html=True)
with st.sidebar:
    st.header('Data')
    files = sorted((ROOT / 'data').glob('*.csv'))
    choice = st.selectbox('CSV in data/', files, format_func=lambda p: p.name) if files else None
    upload = st.file_uploader('...or upload a CSV', type='csv')
    with st.expander('Advanced', expanded=False):
        bin_label = st.selectbox('Candle size', list(BIN_CHOICES), help='Auto picks a size from the window')
        levels = st.number_input('Ladder levels per side', 1, 40, 10)
        log_half = st.number_input('Event log rows each side', 3, 40, 12)
        st.number_input('Go to event #', 0, None, 0, key='goto', on_change=on_goto)
        ids = st.checkbox('Show ids', value=False)

if upload is not None:
    data = upload.getvalue()
    source = f'upload:{upload.name}:{hashlib.blake2b(data, digest_size=8).hexdigest()}'
    replay_ = load_bytes(source, data)
    fname = upload.name
elif choice is not None:
    stt = choice.stat()
    source = f'{choice}:{stt.st_mtime}:{stt.st_size}'
    replay_ = load_path(str(choice), stt.st_mtime, stt.st_size)
    fname = choice.name
else:
    st.info('Put a CSV in data/ or upload one.')
    st.stop()

if st.session_state.get('_src') != source:
    st.session_state.update(_src=source, pos=0, playing=False, _nav_last=None)
st.session_state['_r'] = replay_
tpe = replay_.tape
st.markdown(f'<span class="ttl">Market viewer</span><span class="muted">{fname} &middot; {replay_.n:,} rows &middot; '
            f'{tpe.time[-1] - tpe.time[0]:,.0f} s &middot; {len(replay_.t_ev):,} trades &middot; {int(replay_.t_qty.sum()):,} lots traded</span>',
            unsafe_allow_html=True)
if tpe.time_fixed:
    st.warning(f'{tpe.time_fixed} rows had decreasing timestamps (clamped).')
if replay_.dups:
    st.warning(f'{replay_.dups} inserts reused a live order id (the simulator would raise).')

options = dict(bin=BIN_CHOICES[bin_label], levels=int(levels), log_half=int(log_half), ids=ids)
st.fragment(view, run_every=0.8 if st.session_state.get('playing') else None)(replay_, source, options)
