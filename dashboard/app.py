"""Streamlit dashboard for raw market-data CSVs (tape): candles, trades, order book and breakdowns over time.

Run:  uv run --with streamlit streamlit run dashboard/app.py
The Analysis tab renders reports from the analysis/ package (docs/ANALYSIS.md).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_tab  # noqa: E402
import engine  # noqa: E402
import robustness_tab  # noqa: E402
import strategy_view  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = 'round1_train.csv'  # opened first when it is in data/
UP, DOWN, ACCENT = '#26a69a', '#ef5350', '#f5c542'
GRID = 'rgba(128,128,128,0.15)'
CLEAR = 'rgba(0,0,0,0)'
WINDOWS = {'10 s': 10, '1 min': 60, '5 min': 300, '15 min': 900, '1 h': 3600, 'All': None}
AUTO_BIN = {'10 s': 0.1, '1 min': 1, '5 min': 5, '15 min': 10, '1 h': 30, 'All': 60}
BIN_CHOICES = {'Auto': None, '100 ms': 0.1, '1 s': 1, '5 s': 5, '10 s': 10, '30 s': 30, '1 min': 60, '5 min': 300}
MAX_MARKERS = 300
PLOT_CONFIG = {'scrollZoom': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']}
CSS = f"""<style>
.block-container {{ padding-top: 1.5rem; }}
.stApp {{ font-variant-numeric: tabular-nums; }}
[data-testid="stMetricValue"] {{ font-size: 1.3rem; }}
.ttl {{ font-size: 1.15rem; font-weight: 600; margin-right: .8rem; }}
.muted {{ opacity: .6; font-size: .85rem; }}
.tread {{ font: 700 2.2rem/1.1 ui-monospace, SFMono-Regular, monospace; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .6rem; margin: .4rem 0 .8rem; }}
.card {{ border: 1px solid rgba(128,128,128,.25); border-radius: .5rem; padding: .5rem .75rem; }}
.clabel {{ font-size: .78rem; opacity: .65; text-transform: uppercase; letter-spacing: .03em; }}
.cvalue {{ font-size: 1.45rem; font-weight: 700; line-height: 1.3; }}
.csub {{ font-size: .78rem; opacity: .65; }}
.legend {{ display: flex; flex-wrap: wrap; gap: .4rem 1.2rem; font-size: .82rem; margin: -.4rem 0 .6rem; }}
.legend i {{ display: inline-block; width: .8rem; height: .8rem; border-radius: 2px; margin-right: .35rem; vertical-align: -1px; }}
.legend i.line {{ width: 1.2rem; height: 0; border-top: 2px dashed {ACCENT}; border-radius: 0; vertical-align: 3px; }}
.lsec {{ font-size: .75rem; font-weight: 600; padding: 4px 8px 2px; }}
.lsec.ask {{ color: {DOWN}; }} .lsec.bid {{ color: {UP}; }}
.ladder {{ font-family: ui-monospace, SFMono-Regular, monospace; font-size: .85rem; }}
.lrow {{ display: grid; grid-template-columns: 1.2fr 1fr .6fr; text-align: right; padding: 1px 8px; --c: transparent;
  background: linear-gradient(to left, var(--c) var(--w, 0%), transparent var(--w, 0%)); }}
.lrow.ask {{ --c: rgba(239,83,80,.30); }} .lrow.ask span:first-child {{ color: {DOWN}; }}
.lrow.bid {{ --c: rgba(38,166,154,.30); }} .lrow.bid span:first-child {{ color: {UP}; }}
.lrow.best {{ font-weight: 700; }}
.lhead {{ opacity: .6; font-size: .75rem; }}
.lvl summary {{ list-style: none; cursor: pointer; }} .lvl summary::-webkit-details-marker {{ display: none; }}
.lvl summary:hover {{ outline: 1px solid rgba(128,128,128,.35); }}
.lorders {{ margin: 0 0 3px 14px; border-left: 2px solid rgba(128,128,128,.3); font-size: .78rem; }}
.lord {{ display: grid; grid-template-columns: 1.4fr 1fr .9fr; text-align: right; padding: 0 8px; opacity: .85; --c: transparent;
  background: linear-gradient(to left, var(--c) var(--w, 0%), transparent var(--w, 0%)); }}
.lorders.ask .lord {{ --c: rgba(239,83,80,.15); }} .lorders.bid .lord {{ --c: rgba(38,166,154,.15); }}
.lord span:first-child {{ text-align: left; }}
.lord.mine {{ color: #4fb3d9; font-weight: 700; opacity: 1; }}
.lrow .mine {{ color: #4fb3d9; }}
.lspread {{ text-align: center; opacity: .75; padding: 3px 0; margin: 2px 0; border-top: 1px solid rgba(128,128,128,.3);
  border-bottom: 1px solid rgba(128,128,128,.3); }}
</style>"""

st.set_page_config(page_title='Market dashboard', layout='wide')


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


def fmt_dur(sec: float) -> str:
    """A candle length in words: 100 ms, 5 s, 2 min."""
    if sec < 1:
        return f'{sec * 1000:g} ms'
    return f'{sec:g} s' if sec < 60 else f'{sec / 60:g} min'


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


def pos_at(sec: float) -> int:
    """Last event at or before `sec` (the state of the market at that moment)."""
    return int(np.searchsorted(st.session_state['_r'].tape.time, sec + 1e-9, side='right')) - 1


def on_slider() -> None:
    set_pos(pos_at(from_dt(st.session_state['tslider'])))


def on_step(events: int) -> None:
    set_pos(st.session_state['pos'] + events)


def on_time_step(sec: float) -> None:
    """Move by `sec` seconds of market time; always moves at least one event so a button press is never a no-op."""
    tp = st.session_state['_r'].tape
    pos = st.session_state['pos']
    new = pos_at(float(tp.time[pos]) + sec)
    set_pos(max(new, pos + 1) if sec > 0 else min(new, pos - 1))


def on_edge(end: bool) -> None:
    set_pos(st.session_state['_r'].n - 1 if end else 0)
    st.session_state['playing'] = False


def parse_time(text: str) -> float | None:
    """'12:30', '1:02:03.5', '754.2' -> seconds; None if it can't be read."""
    try:
        sec = 0.0
        for part in text.strip().split(':'):
            sec = sec * 60 + float(part)
        return sec
    except ValueError:
        return None


def on_jump() -> None:
    text = st.session_state.get('jumpto', '')
    sec = parse_time(text) if text.strip() else None
    if sec is None:
        st.session_state['_jump_err'] = text
    else:
        set_pos(max(pos_at(sec), 0))
        st.session_state['playing'] = False


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


def on_mine_jump(kind: str, direction: int) -> None:
    """Jump to the previous/next row where the strategy acted ('action') or got filled ('fill')."""
    rows = st.session_state.get('_mine', {}).get(kind)
    if rows is None or not len(rows):
        return
    pos = st.session_state['pos']
    k = int(np.searchsorted(rows, pos, side='right' if direction > 0 else 'left'))
    k = k if direction > 0 else k - 1
    if 0 <= k < len(rows):
        set_pos(int(rows[k]))
        st.session_state['playing'] = False


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
# Session seconds become real timestamps from the epoch: Plotly sizes candles and bars correctly only on true dates
# (plain millisecond numbers on a date axis give wildly wrong candle widths).
def ts(sec) -> np.ndarray:
    return (np.asarray(sec, dtype=np.float64) * 1e6).round().astype(np.int64).astype('datetime64[us]')


def ts1(sec: float) -> str:
    return str(ts(sec))


EPOCH = dt.datetime(1970, 1, 1)


def to_dt(sec: float) -> dt.datetime:
    """Session seconds as a datetime, so the time slider can show clock time instead of raw seconds."""
    return EPOCH + dt.timedelta(milliseconds=round(sec * 1000))


def from_dt(v: dt.datetime) -> float:
    return (v - EPOCH).total_seconds()


def nav_seconds(v) -> float:
    """A navigator selection x value (a date string such as '1970-01-01 00:01:17.72', or epoch ms) as session seconds."""
    if isinstance(v, (int, float)):
        return float(v) / 1000
    return (np.datetime64(str(v).replace(' ', 'T'), 'us') - np.datetime64('1970-01-01T00:00:00', 'us')) / np.timedelta64(1, 's')


SPIKE = dict(showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=1, spikedash='solid', spikecolor='rgba(128,128,128,0.6)')

# Middle mouse button drag pans any Plotly chart on the page. Plotly's drag box pans when a drag starts with Shift held,
# so a middle-button press is swallowed and replayed as a Shift+left press; the real moves and release finish the drag.
MIDDLE_PAN_JS = """<script>
const w = window.parent, doc = w.document;
if (!w.__middlePan) {
  w.__middlePan = true;
  doc.addEventListener('mousedown', e => {
    if (e.button !== 1 || e.__synthetic) return;
    const drag = e.target.closest && e.target.closest('.nsewdrag');
    if (!drag) return;
    e.preventDefault(); e.stopImmediatePropagation();
    const gd = drag.closest('.js-plotly-plot');
    const inPan = gd && gd._fullLayout && gd._fullLayout.dragmode === 'pan';  // Shift flips pan to zoom, so only add it when not panning already
    const ev = new w.MouseEvent('mousedown', {bubbles: true, cancelable: true, view: w, clientX: e.clientX, clientY: e.clientY,
      screenX: e.screenX, screenY: e.screenY, button: 0, buttons: 1, shiftKey: !inPan});
    ev.__synthetic = true;
    drag.dispatchEvent(ev);
  }, true);
  // The whole-session strip is in drag-to-select mode, where Plotly reports no plain click. Replay a click there as a tiny
  // drag; the app treats a very narrow selection as "jump here".
  let lastClick = 0, downX = null;
  doc.addEventListener('mousedown', e => { if (!e.__synthetic && e.button === 0) downX = e.clientX; }, true);
  doc.addEventListener('click', e => {
    // a real drag also ends in a click event: only a press and release in the same spot is a click
    if (e.__synthetic || Date.now() - lastClick < 400 || downX === null || Math.abs(e.clientX - downX) > 3) return;
    const drag = e.target.closest && e.target.closest('.nsewdrag');
    const gd = drag && drag.closest('.js-plotly-plot');
    if (!gd || !gd._fullLayout || gd._fullLayout.dragmode !== 'select') return;
    lastClick = Date.now();
    const mk = (type, x, target) => {
      const ev = new w.MouseEvent(type, {bubbles: true, cancelable: true, view: w, clientX: x, clientY: e.clientY, button: 0, buttons: type === 'mouseup' ? 0 : 1});
      ev.__synthetic = true; target.dispatchEvent(ev);
    };
    setTimeout(() => { mk('mousedown', e.clientX - 6, drag); mk('mousemove', e.clientX + 6, doc); mk('mouseup', e.clientX + 6, doc); }, 0);
  }, true);
  doc.addEventListener('auxclick', e => { if (e.button === 1 && e.target.closest && e.target.closest('.nsewdrag')) e.preventDefault(); }, true);
}
</script>"""


def style_time_axes(fig: go.Figure, hrs: bool, **kw) -> None:
    f = '%H:%M:%S' if hrs else '%M:%S'
    fig.update_xaxes(type='date', gridcolor=GRID, hoverformat=f + '.%L', zeroline=False,
                     tickformatstops=[dict(dtickrange=[None, 1000], value=f + '.%L'), dict(dtickrange=[1000, None], value=f)], **kw)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)


def cursor_shapes(t: float, x1: float | None) -> list[dict]:
    """Dashed cursor line, plus a faint grey rectangle over the not-yet-happened part when x1 is given."""
    shapes = [dict(type='line', xref='x', yref='paper', x0=ts1(t), x1=ts1(t), y0=0, y1=1, line=dict(color=ACCENT, width=1, dash='dash'))]
    if x1 is not None and x1 > t:
        shapes.insert(0, dict(type='rect', xref='x', yref='paper', x0=ts1(t), x1=ts1(x1), y0=0, y1=1, fillcolor='rgba(128,128,128,0.10)',
                              line_width=0, layer='below'))
    return shapes


def main_fig(r: engine.Replay, mid: np.ndarray, x0: float, x1: float, t: float, bw: float, cand_src: str, show_ba: bool, hrs: bool,
             revision: str, show_trades: bool = False) -> go.Figure:
    tm = r.tape.time
    base = np.floor(x0 / bw) * bw
    nb = int(np.ceil((x1 - base) / bw)) + 1
    ta = r.t_time
    a, b = int(np.searchsorted(ta, x0, 'left')), int(np.searchsorted(ta, x1, 'right'))
    i0 = max(int(np.searchsorted(tm, x0, 'left')) - 1, 0)
    i1 = int(np.searchsorted(tm, x1, 'right'))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.02)

    centres = base + (np.arange(nb) + 0.5) * bw
    # per-slot trade and event stats, shared by the candle and volume hovers
    k_ = np.clip(((ta[a:b] - base) // bw).astype(int), 0, nb - 1)
    side_, qty_ = r.t_side[a:b], r.t_qty[a:b]
    buy = np.bincount(k_[side_ == 0], weights=qty_[side_ == 0], minlength=nb)
    sell = np.bincount(k_[side_ == 1], weights=qty_[side_ == 1], minlength=nb)
    ntr = np.bincount(k_, minlength=nb)
    ev_edges = np.searchsorted(tm, base + np.arange(nb + 1) * bw, side='left')
    nev = np.diff(ev_edges)

    def slot(k: int) -> str:
        return f'{fmt_t(base + k * bw, hrs)} → {fmt_t(base + (k + 1) * bw, hrs)}'

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
        if cand_src == 'Mid':
            # The mid price is a state, so a quiet slot is a flat candle at the last close: one candle per slot, like any standard chart.
            fill = np.full(nb, -1)
            fill[bins] = np.arange(len(bins))
            fill = np.maximum.accumulate(fill)
            keep = fill >= 0
            src_i, has = fill[keep], np.isin(np.arange(nb)[keep], bins)
            bins = np.nonzero(keep)[0]
            o, h, l, c = (np.where(has, v[src_i], c[src_i]) for v in (o, h, l, c))
        ymin, ymax = float(l.min()), float(h.max())
        what = 'mid price' if cand_src == 'Mid' else 'trade price'
        text = [f'<b>{slot(k)}</b>  ({what})<br>open {p:,.1f}   high {q:,.1f}<br>low {u:,.1f}   close {v:,.1f}<br>'
                f'change {v - p:+,.1f} ({100 * (v - p) / p:+.3f}%)   range {q - u:,.1f}<br>'
                f'trades {ntr[k]:,}   volume {int(buy[k] + sell[k]):,} (buy {int(buy[k]):,} / sell {int(sell[k]):,})<br>tape events {nev[k]:,}'
                for k, p, q, u, v in zip(bins, o, h, l, c)]
        # An invisible candle in every slot pins the candle width to the slot width, even when trade candles leave gaps.
        fig.add_trace(go.Candlestick(x=ts(centres), open=np.full(nb, c[0]), high=np.full(nb, c[0]), low=np.full(nb, c[0]), close=np.full(nb, c[0]),
                                     opacity=0, hoverinfo='skip', showlegend=False, name='grid'), row=1, col=1)
        fig.add_trace(go.Candlestick(x=ts(centres[bins]), open=o, high=h, low=l, close=c, name='Candles', showlegend=False, text=text,
                                     hoverinfo='text', whiskerwidth=0, increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
                                     decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN)), row=1, col=1)

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
            fig.add_trace(go.Scatter(x=ts(xx), y=yy, mode='lines', name=name, opacity=0.4, line=dict(color=color, width=1, shape=shape),
                                     hovertemplate='%{x}<br>' + name + ' %{y:,}<extra></extra>'), row=1, col=1)

    # trade markers
    # Off by default: the candles and volume bars already aggregate the trades (and their hovers list them).
    sel = np.arange(a, b) if show_trades else np.arange(0)
    capped = len(sel) > MAX_MARKERS  # a busy window would bury the candles under markers, so they appear only once zoomed in
    if capped:
        sel = sel[:0]
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
                x=ts(ta[s]), y=pr[m] + (-off if side == 0 else off), mode='markers', name=nm,
                marker=dict(symbol=sym, color=col, size=np.clip(5 + 1.2 * np.sqrt(q), 6, 14), line=dict(width=1.2, color=col)),
                customdata=np.column_stack([[fmt_t(v, hrs) for v in ta[s]], r.t_price[s], q, r.t_ev[s], r.t_taker[s], r.t_maker[s]]),
                hovertemplate='<b>Trade at %{customdata[0]}</b><br>' + nm + '<br>price %{customdata[1]:,}   qty %{customdata[2]:,} lots<br>'
                              'taker order %{customdata[4]} → maker order %{customdata[5]}<br>tape event #%{customdata[3]:,}<extra></extra>'),
                row=1, col=1)

    # volume bars (same bins as the candles)
    nz = np.nonzero((buy > 0) | (sell > 0))[0]
    if len(nz):
        # Standard volume pane: one bar per candle slot, total traded lots, coloured by which side was more aggressive.
        cd = np.column_stack([[slot(k) for k in nz], buy[nz], sell[nz], buy[nz] - sell[nz], ntr[nz]])
        fig.add_trace(go.Bar(x=ts(centres[nz]), y=buy[nz] + sell[nz], width=bw * 1e3 * 0.8, name='Volume', showlegend=False, customdata=cd,
                             marker=dict(color=np.where(buy[nz] >= sell[nz], UP, DOWN), opacity=0.7, line_width=0),
                             hovertemplate='<b>%{customdata[0]}</b><br>volume %{y:,} lots in %{customdata[4]} trades<br>'
                                           'buy aggressor %{customdata[1]:,}   sell aggressor %{customdata[2]:,}<br>net %{customdata[3]:+,}'
                                           '<extra></extra>'), row=2, col=1)

    style_time_axes(fig, hrs)
    fig.update_xaxes(range=[ts1(x0), ts1(x1)], rangeslider_visible=False, **SPIKE)
    fig.update_yaxes(side='right', tickformat=',.2~f', **SPIKE, row=1, col=1)
    fig.update_yaxes(side='right', rangemode='tozero', tickformat=',~s', title_text='lots', row=2, col=1)
    fig.update_layout(height=560, margin=dict(l=10, r=60, t=10, b=30), paper_bgcolor=CLEAR, plot_bgcolor=CLEAR, dragmode='pan', bargap=0.2,
                      hovermode='closest', hoverdistance=30, spikedistance=-1, uirevision=revision, shapes=cursor_shapes(t, x1),
                      legend=dict(orientation='h', y=1.0, x=0, yanchor='bottom', font_size=11))
    if capped:
        fig.add_annotation(text=f'{b - a:,} trades in view: trade arrows hidden, pick a shorter window', xref='paper', yref='paper', x=1, y=1.0, yanchor='bottom', xanchor='right', showarrow=False,
                           font=dict(size=10), opacity=0.6)
    return fig


def nav_fig(nx: np.ndarray, ny: np.ndarray, x0: float, x1: float, t: float, hrs: bool) -> go.Figure:
    fig = go.Figure(go.Scatter(x=ts(nx), y=ny, mode='lines+markers', line=dict(color='rgba(128,128,128,0.8)', width=1),
                               marker=dict(size=7, opacity=0), selected=dict(marker=dict(opacity=0)), unselected=dict(marker=dict(opacity=0)),
                               hovertemplate='%{x}<br>mid %{y:,.1f}<br>click to jump here<extra></extra>'))
    fig.update_layout(height=110, margin=dict(l=10, r=60, t=4, b=24), paper_bgcolor=CLEAR, plot_bgcolor=CLEAR, showlegend=False, uirevision='nav',
                      dragmode='select', selectdirection='h', hovermode='x', clickmode='event+select', newselection=dict(line=dict(color=ACCENT, width=1)),
                      shapes=[dict(type='rect', xref='x', yref='paper', x0=ts1(x0), x1=ts1(x1), y0=0, y1=1, fillcolor='rgba(245,197,66,0.15)', line_width=0),
                              dict(type='line', xref='x', yref='paper', x0=ts1(t), x1=ts1(t), y0=0, y1=1, line=dict(color=ACCENT, width=1.5))])
    style_time_axes(fig, hrs, fixedrange=True)
    fig.update_yaxes(fixedrange=True, tickformat=',.2~f', nticks=3, side='right')
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
    tx = ts(np.concatenate([[t0], r.t_time])[idx])
    fig = go.Figure()
    for name, rows_, col in (('Buy aggressor', (0, 2), UP), ('Sell aggressor', (1, 3), DOWN)):
        fig.add_trace(go.Scatter(x=tx, y=r.cum_qty[list(rows_)].sum(0)[idx], name=name, line=dict(color=col, shape='hv', width=1.5)))
    style_time_axes(fig, hrs)
    fig.update_layout(height=300, margin=dict(l=50, r=10, t=10, b=30), paper_bgcolor=CLEAR, plot_bgcolor=CLEAR, uirevision=revision,
                      shapes=cursor_shapes(t, None), legend=dict(orientation='h', y=1.0, x=0, yanchor='bottom'), yaxis_title='cumulative lots')
    return fig


# ---------------------------------------------------------------- ladder (HTML)
def ladder_html(book: engine.Book, bids: list, asks: list, n: int, expand: bool, ids: bool) -> str:
    """Bid/ask ladder. Each price level is a <details>: click it (or tick "Show every order") to see its orders in queue order."""
    a, b = asks[:n], bids[:n]
    mx = max([v for _, v, _ in a + b], default=1)

    def row(kind: str, lv: tuple, best: bool) -> str:
        p, v, c = lv
        orders = book.level_orders(0 if kind == 'bid' else 1, p)
        omx = max((q for _, q in orders), default=1)
        lines = ''.join(f'<div class="lord{" mine" if isinstance(oid, str) else ""}" style="--w:{100 * q / omx:.1f}%"><span>{j + 1}.'
                        f'{" YOU" if isinstance(oid, str) else ""}{" next to trade" if j == 0 else ""}</span>'
                        f'<span>{q:,} lots</span><span>{oid if isinstance(oid, str) else (f"#{oid}" if ids else "")}</span></div>'
                        for j, (oid, q) in enumerate(orders))
        yours = sum(q for oid, q in orders if isinstance(oid, str))
        mark = f' <span class="mine" title="your lots at this price">&#9679;{yours:,}</span>' if yours else ''
        return (f'<details class="lvl"{" open" if expand else ""}><summary class="lrow {kind}{" best" if best else ""}" style="--w:{100 * v / mx:.1f}%">'
                f'<span>{p:,}</span><span>{v:,}{mark}</span><span>{c} &#9662;</span></summary><div class="lorders {kind}">{lines}</div></details>')

    out = ['<div class="ladder"><div class="lrow lhead"><span>Price</span><span>Lots</span><span>Orders</span></div>',
           '<div class="lsec ask">Waiting to SELL (asks) &uarr; cheapest at the bottom</div>']
    out += [row('ask', lv, j == len(a) - 1) for j, lv in enumerate(reversed(a))] if a else ['<div class="lspread">nobody is selling</div>']
    if a and b:
        out.append(f'<div class="lspread">gap (spread) {asks[0][0] - bids[0][0]:,} &middot; mid {(asks[0][0] + bids[0][0]) / 2:,.1f}</div>')
    else:
        out.append('<div class="lspread">&nbsp;</div>')
    out += [row('bid', lv, j == 0) for j, lv in enumerate(b)] if b else ['<div class="lspread">nobody is buying</div>']
    out.append('<div class="lsec bid">Waiting to BUY (bids) &darr; highest at the top</div>')
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
SPEEDS = [1, 10, 60, 300]
HELP_TEXT = f"""
**What is this?** A replay of one trading session. Every row of the CSV is somebody placing or cancelling an order; the page rebuilds
what the market looked like at any moment and which trades happened.

**Moving through time.** Press **Play** to watch it run, or drag the time slider. The buttons jump by a fixed amount of market time
(−1 min … +1 min), to the previous/next trade, or by a single CSV row (‹ 1 event ›). Type a time such as `12:30` in *Jump to time*.
You can also click the thin whole-session chart under the main chart.

**The main chart.** Each candle covers one time slot (pick its length under *One candle =*).
<span style="color:{UP}">Green</span> = the price ended the slot higher than it started, <span style="color:{DOWN}">red</span> = lower.
The bars underneath are how many lots traded in that slot, green when buyers did most of the pushing, red when sellers did.
The dashed yellow line is *now*; the grey area to its right has not happened yet. Hover anything for details.
Drag to move the chart, scroll to zoom, double-click to reset.

**Words.** *Bid* = the best price someone is waiting to buy at. *Ask* = the best price someone is waiting to sell at.
*Spread* = ask − bid. *Mid* = halfway between them. *Lot* = one unit of size. *IOC* = an order that trades immediately
or is cancelled; *limit* = an order that waits in the book.
"""


def card(label: str, value: str, sub: str = '', color: str | None = None) -> str:
    style = f' style="color:{color}"' if color else ''
    return f'<div class="card"><div class="clabel">{label}</div><div class="cvalue"{style}>{value}</div><div class="csub">{sub or "&nbsp;"}</div></div>'


def view(r: engine.Replay, source: str, opts: dict) -> None:
    ss = st.session_state
    tp = r.tape
    n = r.n
    t0, t1 = float(tp.time[0]), float(tp.time[-1])
    if t1 <= t0:
        t1 = t0 + 1
    hrs = t1 >= 3600

    # navigator: a drag picks the stretch the main chart shows (and moves "now" to its end), a click jumps there.
    # Handled once per new selection, before anything uses `pos`.
    nx, ny = nav_series(source, r)
    try:
        sel = ss[f'nav{ss.get("_navgen", 0)}']['selection']
        boxes, pts = sel.get('box') or [], sel.get('points') or []
    except (KeyError, TypeError, AttributeError):
        boxes, pts = [], []
    key = (source, repr(boxes[-1]['x']) if boxes else None, pts[0].get('point_index') if pts and not boxes else None)
    if (boxes or pts) and ss.get('_nav_last') != key:
        ss['_nav_last'] = key
        ss['playing'] = False
        ss['_navgen'] = ss.get('_navgen', 0) + 1  # a fresh strip widget, so the old selection can't swallow the next click or drag
        if boxes:
            a_, b_ = sorted(nav_seconds(v) for v in boxes[-1]['x'][:2])
            a_, b_ = max(a_, t0), min(b_, t1)
            if b_ - a_ < 0.025 * (t1 - t0):  # a tiny drag (or a click, which the page script turns into one) means "jump here"
                a_, b_ = None, (a_ + b_) / 2
            if a_ is not None:
                ss['custom_win'] = (a_, b_)
            ss['pos'] = max(pos_at(b_), 0)
        else:
            ss['pos'] = max(pos_at(float(nx[min(int(pts[0]['point_index']), len(nx) - 1)])), 0)
    elif not (boxes or pts):
        ss['_nav_last'] = None

    if ss.get('playing'):
        now = time.time()
        step = min(now - ss.get('last_tick', now), 5.0)
        ss['last_tick'] = now
        pos = int(np.searchsorted(tp.time, tp.time[ss['pos']] + step * (ss.get('speed') or 1), side='right')) - 1
        pos = max(pos, ss['pos'] + 1)
        if pos >= n - 1:
            pos, ss['playing'] = n - 1, False
        ss['pos'] = pos
    pos = ss['pos']
    t = float(tp.time[pos])
    ss['tslider'] = to_dt(t)

    # ---- 1. where am I in time, and how do I move
    with st.container(border=True):
        top = st.columns([2.4, 1.2, 1.3, 2.6], vertical_alignment='center')
        top[0].markdown(f'<div class="clabel">Market time</div><div class="tread">{fmt_t(t, hrs)}</div>'
                        f'<div class="csub">of {fmt_t(t1, hrs)} &middot; row {pos + 1:,} of {n:,} in the CSV</div>', unsafe_allow_html=True)
        playing = ss.get('playing', False)
        if top[1].button(':material/pause: Pause' if playing else ':material/play_arrow: Play', type='primary', width='stretch',
                         help='Run the market forward on its own'):
            ss['playing'] = not playing
            if ss['playing'] and pos >= n - 1:
                ss['pos'] = 0
            ss['last_tick'] = time.time()
            st.rerun()
        top[2].selectbox('Play speed', SPEEDS, index=1, key='speed', format_func=lambda v: 'real time' if v == 1 else f'{v}x faster',
                         help='How fast Play runs compared to real time')
        top[3].text_input('Jump to time', key='jumpto', placeholder='e.g. 12:30 or 1:05:00', on_change=on_jump,
                          help='Type a market time (minutes:seconds) and press Enter')
        if 'jumpto' in ss and ss.get('_jump_err') is not None:
            st.warning(f'Could not read "{ss.pop("_jump_err")}" as a time. Use minutes:seconds, e.g. 12:30.')
        fmt = 'H:mm:ss' if hrs else 'mm:ss'
        st.slider('Time', to_dt(t0), to_dt(t1), key='tslider', step=dt.timedelta(milliseconds=100 if t1 - t0 <= 7200 else 1000),
                  format=fmt, on_change=on_slider, label_visibility='collapsed')
        b = st.columns(12)
        moves = [(':material/first_page: Start', on_edge, (False,), 'Go to the first row'),
                 ('−1 min', on_time_step, (-60,), 'Back one minute'), ('−10 s', on_time_step, (-10,), 'Back ten seconds'),
                 ('−1 s', on_time_step, (-1,), 'Back one second'), ('‹ 1 event', on_step, (-1,), 'Back one CSV row'),
                 ('‹ Trade', on_trade_jump, (-1,), 'Back to the previous trade'), ('Trade ›', on_trade_jump, (1,), 'On to the next trade'),
                 ('1 event ›', on_step, (1,), 'Forward one CSV row'), ('+1 s', on_time_step, (1,), 'Forward one second'),
                 ('+10 s', on_time_step, (10,), 'Forward ten seconds'), ('+1 min', on_time_step, (60,), 'Forward one minute'),
                 ('End :material/last_page:', on_edge, (True,), 'Go to the last row')]
        for col, (label, fn, args, tip) in zip(b, moves):
            col.button(label, on_click=fn, args=args, help=tip, width='stretch', key=f'mv{label}')

    run = opts.get('run')
    if run is not None:
        ss['_mine'] = strategy_view.jump_targets(run)
        jb = st.columns([1.3, 1, 1, 1, 1, 3], vertical_alignment='center')
        jb[0].markdown(f'**{run.strategy_name}** <span class="muted">replayed with your orders</span>', unsafe_allow_html=True)
        for col, (label, kind, d, tip) in zip(jb[1:5], [('‹ My action', 'action', -1, 'Previous row where the strategy sent/cancelled/shrank'),
                                                       ('My action ›', 'action', 1, 'Next row where the strategy sent/cancelled/shrank'),
                                                       ('‹ My fill', 'fill', -1, 'Previous fill of one of your orders'),
                                                       ('My fill ›', 'fill', 1, 'Next fill of one of your orders')]):
            col.button(label, on_click=on_mine_jump, args=(kind, d), help=tip, width='stretch', key=f'mine{label}')

    # ---- 2. the market right now, in plain words
    book = run.book_at(pos) if run is not None else engine.book_at(r, pos)
    bids, asks = book.levels()
    k = engine.trade_count_through(r, pos)
    cur = engine.counters_at(r, pos)
    tot = r.chunk_counters[-1]
    mid = mid_series(source, r)
    bb, ba = (bids[0] if bids else None), (asks[0] if asks else None)
    cards = [card('Price now (mid)', f'{(bb[0] + ba[0]) / 2:,.1f}' if bb and ba else '-', 'halfway between bid and ask'),
             card('Best bid', f'{bb[0]:,}' if bb else 'none', f'{bb[1]:,} lots waiting to buy' if bb else 'nobody waiting to buy', UP),
             card('Best ask', f'{ba[0]:,}' if ba else 'none', f'{ba[1]:,} lots waiting to sell' if ba else 'nobody waiting to sell', DOWN),
             card('Spread', f'{ba[0] - bb[0]:,}' if bb and ba else '-', 'ask minus bid')]
    if k:
        j = k - 1
        who = 'a buyer' if r.t_side[j] == 0 else 'a seller'
        cards.append(card('Last trade', f'{int(r.t_price[j]):,}', f'{int(r.t_qty[j]):,} lots, {who} hit the book, {fmt_t(t - r.t_time[j], False)} ago',
                          UP if r.t_side[j] == 0 else DOWN))
    else:
        cards.append(card('Last trade', '-', 'no trades yet'))
    total_q = r.cum_qty[:, k].sum()
    cards.append(card('Traded so far', f'{int(total_q):,} lots', f'in {int(r.cum_cnt[:, k].sum()):,} trades'))
    if run is not None:
        cards += [card(*c_) for c_ in strategy_view.cards(run, pos, t)]
    st.markdown(f'<div class="cards">{"".join(cards)}</div>', unsafe_allow_html=True)

    # ---- 3. chart controls
    o = st.columns([2.6, 3.6, 1.1], vertical_alignment='bottom')
    wlabel = o[0].segmented_control('Chart shows', list(WINDOWS), default='5 min', key='window', on_change=lambda: ss.pop('custom_win', None),
                                    help='How much market time the chart shows around now. You can also drag across the whole-session strip.') or '5 min'
    custom = ss.get('custom_win')
    if custom:
        # the stretch dragged on the strip: kept while "now" is inside it, then it follows "now" at the same width
        cw = custom[1] - custom[0]
        x0, x1 = custom if custom[0] <= t <= custom[1] else window_range(t, cw, t0, t1)
        auto_bw = next((b_ for lim, b_ in ((10, 0.1), (60, 1), (300, 5), (900, 10), (3600, 30)) if cw <= lim), 60)
        win_name = f'custom {fmt_t(cw, hrs)}'
    else:
        x0, x1 = window_range(t, WINDOWS[wlabel], t0, t1)
        auto_bw, win_name = AUTO_BIN[wlabel], wlabel
    bin_label = o[1].segmented_control('One candle =', list(BIN_CHOICES) + ['Custom'], default='Auto', key='candlebin',
                                       format_func=lambda v: f'Auto ({fmt_dur(auto_bw)})' if v == 'Auto' else v,
                                       help='How much time each candle and volume bar covers. Auto picks one from the window '
                                            '(shown in brackets); Custom lets you type any length.') or 'Auto'
    if bin_label == 'Custom':
        custom_bw = o[1].number_input('Seconds per candle', min_value=0.001, value=float(ss.get('custombin', 2.0)), step=0.5, format='%g',
                                      key='custombin', help='Any length in seconds, e.g. 0.25 or 15')
    with o[2].popover('Chart options', width='stretch'):
        cand_src = st.radio('Candles from', ['Mid', 'Trades'], key='candsrc', format_func=lambda v: 'Mid price' if v == 'Mid' else 'Trade prices',
                            help='Mid price changes all the time; trade prices only when something trades, so trade candles have gaps.')
        show_ba = st.toggle('Best bid/ask lines', value=False, key='showba')
        show_trades = st.toggle('Show individual trades', value=False, key='showtrades',
                                help='Small arrows for every trade (up = buyer, down = seller). Hidden when more than 300 are in view.')
    if custom:
        cc = st.columns([6, 1], vertical_alignment='center')
        cc[0].caption(f'Showing the stretch you dragged on the whole-session strip: {fmt_t(x0, hrs)} to {fmt_t(x1, hrs)}.')
        cc[1].button('Back to presets', on_click=lambda: ss.pop('custom_win', None), width='stretch')
    bw = float(custom_bw) if bin_label == 'Custom' else (BIN_CHOICES[bin_label] or auto_bw)
    grow = max(1, int(np.ceil((x1 - x0) / bw / 2000)))
    if grow > 1:
        o[1].caption(f'{fmt_dur(bw)} would give too many candles here; using {fmt_dur(bw * grow)}')
        bw *= grow

    # ---- 4. chart + order book
    left, right = st.columns([3, 1])
    with left:
        fig_main = main_fig(r, mid, x0, x1, t, bw, cand_src, show_ba, hrs, f'{source}|{win_name}|{custom}', show_trades)
        if run is not None:
            strategy_view.add_fills_to_chart(fig_main, run, x0, x1, ts, fmt_t, hrs, t)
        st.plotly_chart(fig_main, width='stretch', config=PLOT_CONFIG, theme='streamlit', key='main')
        st.markdown(f'<div class="legend"><span><i style="background:{UP}"></i>price went up / buyers pushed</span>'
                    f'<span><i style="background:{DOWN}"></i>price went down / sellers pushed</span>'
                    f'<span><i class="line"></i>now</span><span><i style="background:rgba(128,128,128,.35)"></i>not happened yet</span>'
                    f'<span class="muted">drag = move &middot; scroll = zoom &middot; double-click = reset &middot; hover = details</span></div>',
                    unsafe_allow_html=True)
        st.caption('Whole session (yellow = what the chart shows). **Drag** across it to show that stretch, **click** to jump there.')
        st.plotly_chart(nav_fig(nx, ny, x0, x1, t, hrs), width='stretch', config={'displayModeBar': False}, theme='streamlit', key=f'nav{ss.get("_navgen", 0)}',
                        on_select='rerun', selection_mode=('points', 'box'))
    with right:
        hc = st.columns([1.4, 1], vertical_alignment='center')
        hc[0].markdown('**Order book now**', help='Orders waiting to be traded. Sellers on top (red), buyers below (green). '
                       'The bar shows how many lots wait at each price. Click a price to see each order there and its lots.')
        expand = hc[1].toggle('Show every order', key='expandbook', help='Open every price level to list its orders (first = next to trade)')
        st.markdown(ladder_html(book, bids, asks, opts['levels'], expand, opts['ids']), unsafe_allow_html=True)
        st.caption('Click a price to list its orders.')

    # ---- 4b. the strategy at this moment
    if run is not None:
        with st.container(border=True):
            strategy_view.thinking_panel(run, pos, tp, fmt_t, hrs)
            strategy_view.state_chart(run, x0, x1, t, ts, style_time_axes, cursor_shapes, hrs, 'stv')

    # ---- 5. details
    names = ['Recent trades', 'What happened (event log)', 'Order book depth', 'Order types', 'Totals']
    if run is not None:
        names = ['Strategy run'] + names
    tabs = st.tabs(names)
    if run is not None:
        with tabs[0]:
            strategy_view.summary(run, opts.get('theo_max'))
        tabs = tabs[1:]
    with tabs[0]:
        st.caption('The last 50 trades up to now, newest first. BUY = a buyer took a resting sell order; SELL = the other way round.')
        trades_table(r, k, hrs, opts['ids'])
    with tabs[1]:
        st.caption('CSV rows around now, in plain words. Trades caused by a row are listed under it.')
        event_log(r, pos, opts['log_half'], hrs, opts['ids'])
    with tabs[2]:
        st.caption('All waiting orders by price: bars per price, lines = total lots available from the best price outwards.')
        st.plotly_chart(depth_fig(book), width='stretch', theme='streamlit', key='depth')
    with tabs[3]:
        st.caption('What kind of orders the market sent, up to now and in the whole file.')
        order_flow(cur, tot, float(r.cum_qty[[2, 3], k].sum()), float(r.cum_qty[[2, 3], -1].sum()))
    with tabs[4]:
        mm = st.columns(3)
        mm[0].metric('Trades so far', f'{int(r.cum_cnt[:, k].sum()):,}')
        mm[1].metric('Lots traded so far', f'{int(total_q):,}')
        mm[2].metric('Average trade price so far', f'{r.cum_notional[k] / total_q:,.2f}' if total_q else '-',
                     help='Volume-weighted: big trades count more (VWAP)')
        st.plotly_chart(cumulative_fig(r, t, t0, hrs, source), width='stretch', config=PLOT_CONFIG, theme='streamlit', key='cum')


# ---------------------------------------------------------------- app
st.markdown(CSS, unsafe_allow_html=True)
components.html(MIDDLE_PAN_JS, height=0)
with st.sidebar:
    st.header('Data')
    # real tapes in data/, then synthetic tapes in data/synthetic/ (scripts/gen_tape.py, scripts/make_synthetic.py)
    files = sorted((ROOT / 'data').glob('*.csv')) + sorted((ROOT / 'data' / 'synthetic').glob('*.csv'))
    names = [f.name for f in files]
    choice = st.selectbox('CSV in data/ or data/synthetic/', files,
                          format_func=lambda p: f'synthetic/{p.name}' if p.parent.name == 'synthetic' else p.name,
                          help='synthetic/* are generated tapes (scripts/gen_tape.py, scripts/make_synthetic.py), with the true hidden '
                               'price saved next to them (docs/DATA_ANALYSIS.md, "Synthetic twin of round1").',
                          index=names.index(DEFAULT_FILE) if DEFAULT_FILE in names else 0) if files else None
    upload = st.file_uploader('...or upload a CSV', type='csv')
    with st.expander('Advanced', expanded=False):
        levels = st.number_input('Order book rows per side', 1, 40, 10)
        log_half = st.number_input('Event log rows before/after now', 3, 40, 12)
        st.number_input('Go to CSV row #', 0, None, 0, key='goto', on_change=on_goto)
        ids = st.checkbox('Show order ids', value=False)

if upload is not None:
    data = upload.getvalue()
    source = f'upload:{upload.name}:{hashlib.blake2b(data, digest_size=8).hexdigest()}'
    replay_ = load_bytes(source, data)
    fname, csv_path = upload.name, None
elif choice is not None:
    stt = choice.stat()
    source = f'{choice}:{stt.st_mtime}:{stt.st_size}'
    replay_ = load_path(str(choice), stt.st_mtime, stt.st_size)
    fname, csv_path, data = choice.name, choice, None
else:
    st.info('Put a CSV in data/ (or generate one into data/synthetic/ with scripts/gen_tape.py --seed 1), or upload one in the sidebar.')
    st.stop()

with st.sidebar:
    strategy_run_ = strategy_view.sidebar(csv_path, source, replay_.tape)
theo_max = None
if strategy_run_ is not None:
    replay_ = strategy_run_.replay
    saved = analysis_tab.saved_report(csv_path, fname) if csv_path is not None else None
    theo_max = ((saved or {}).get('max_pnl') or {}).get('theoretical_max', {}).get('pnl')

if st.session_state.get('_src') != source:
    st.session_state.update(_src=source, pos=0, playing=False, _nav_last=None)
# Streamlit keeps a tab's state when the app code is reloaded, so a tab that was playing would carry on by itself: pause instead.
code_id = Path(__file__).stat().st_mtime
if st.session_state.get('_code') != code_id:
    st.session_state.update(_code=code_id, playing=False)
st.session_state['_r'] = replay_
tpe = replay_.tape
head = st.columns([5, 1.3], vertical_alignment='center')
head[0].markdown(f'<span class="ttl">Market dashboard</span><span class="muted">{fname} &middot; {(tpe.time[-1] - tpe.time[0]) / 60:,.0f} min of trading &middot; '
                 f'{len(replay_.t_ev):,} trades &middot; {int(replay_.t_qty.sum()):,} lots traded</span>', unsafe_allow_html=True)
with head[1].popover(':material/help: How to read this page', width='stretch'):
    st.markdown(HELP_TEXT, unsafe_allow_html=True)
if tpe.time_fixed:
    st.warning(f'{tpe.time_fixed} rows had decreasing timestamps (clamped).')
if replay_.dups:
    st.warning(f'{replay_.dups} inserts reused a live order id (the simulator would raise).')

options = dict(levels=int(levels), log_half=int(log_half), ids=ids, run=strategy_run_, theo_max=theo_max)
replay_tab, analysis_view, robustness_view = st.tabs(['Replay', 'Analysis', 'Robustness'])
with replay_tab:
    st.fragment(view, run_every=0.8 if st.session_state.get('playing') else None)(replay_, source, options)
with analysis_view:
    analysis_tab.render(source, fname, csv_path, data)
with robustness_view:
    robustness_tab.render()
