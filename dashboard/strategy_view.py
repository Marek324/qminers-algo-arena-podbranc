"""Strategy replay UI: pick a strategy, tune its constants in memory, run it in the simulator, then see what it thinks and does.

The run itself is in `strategy_run.py`; this module only draws. Nothing here writes to the strategy file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import engine
import strategy_run as R

NONE_LABEL = 'None: market only'
UP, DOWN, ACCENT, MINE = '#26a69a', '#ef5350', '#f5c542', '#4fb3d9'
GRID = 'rgba(128,128,128,0.15)'
CLEAR = 'rgba(0,0,0,0)'
ACT_NAMES = {R.ACT_SEND: 'send', R.ACT_CANCEL: 'cancel', R.ACT_MODIFY: 'shrink'}


# ---------------------------------------------------------------- running (cached)
@st.cache_resource(show_spinner=False, max_entries=6)
def _run(strategy_path: str, strategy_mtime: float, tape_path: str, tape_key: str, overrides_key: tuple, _tape: engine.Tape) -> R.StrategyRun:
    return R.run_strategy(Path(strategy_path), Path(tape_path), _tape, dict(overrides_key))


@st.cache_data(show_spinner=False)
def _params(strategy_path: str, mtime: float) -> tuple[str, list[R.Param]]:
    return R.discover_params(Path(strategy_path))


def _fmt_value(v) -> str:
    if isinstance(v, float):
        return f'{v:.6g}'
    return str(v)


def sidebar(csv_path: Path | None, tape_key: str, tape: engine.Tape) -> R.StrategyRun | None:
    """Strategy picker and parameter editor in the sidebar. Returns the run to replay, or None for market only."""
    st.header('Strategy')
    files = R.strategy_files()
    choice = st.selectbox('Strategy in strategies/', [NONE_LABEL] + [f.name for f in files], key='strat_file',
                          help='Runs the strategy through the real simulator (5 ms latency, same scoring as exchange-simulator) '
                               'and replays the market with your orders in it.')
    if choice == NONE_LABEL:
        return None
    if csv_path is None:
        st.warning('Strategies run on files in data/ or data/synthetic/, not uploads.')
        return None
    path = next(f for f in files if f.name == choice)
    mtime = path.stat().st_mtime
    try:
        cls_name, params = _params(str(path), mtime)
    except Exception as exc:  # syntax error etc. in the strategy file
        st.error(f'Cannot load {choice}: {exc}')
        return None

    applied_key = f'strat_applied:{choice}'
    applied: dict = st.session_state.get(applied_key, {})
    with st.expander(f'Parameters ({len(params)})', expanded=bool(params)):
        if not params:
            st.caption(f'{cls_name} has no UPPERCASE number/bool constants to tune.')
        edited: dict = {}
        for p in params:
            k = f'param:{choice}:{p.name}'
            cur = applied.get(p.name, (p.scope, p.default))[1]
            label = f'{p.name}{" (class)" if p.scope == "class" else ""}'
            if isinstance(p.default, bool):
                v = st.checkbox(label, value=bool(cur), key=k, help=p.comment or None)
            elif isinstance(p.default, int):
                v = st.number_input(label, value=int(cur), step=max(1, abs(p.default) // 10 or 1), key=k, help=p.comment or None)
            else:
                v = st.number_input(label, value=float(cur), format='%g', key=k, help=p.comment or None)
            if v != p.default:
                edited[p.name] = (p.scope, type(p.default)(v))
        c = st.columns(2)
        run_clicked = c[0].button('Run', type='primary', width='stretch', key='strat_run',
                                  help='Run with these values. The strategy file is not changed.')
        if c[1].button('Reset', width='stretch', key='strat_reset', help='Back to the values in the file'):
            for p in params:
                st.session_state.pop(f'param:{choice}:{p.name}', None)
            st.session_state[applied_key] = {}
            st.rerun()
        if run_clicked:
            st.session_state[applied_key] = edited
            applied = edited
        if edited != applied:
            st.caption(':orange[Changed values not run yet: press Run.]')
        if applied:
            st.caption('Running with (copy into the file to keep):')
            st.code('\n'.join(f'{"    " if s == "class" else ""}{n} = {_fmt_value(v)}' for n, (s, v) in applied.items()), language='python')

    key = tuple(sorted(applied.items()))
    with st.spinner(f'Running {choice} through the simulator (about 25-30 s per million rows, cached afterwards)...'):
        run = _run(str(path), mtime, str(csv_path), tape_key, key, tape)
    if run.error:
        st.error(f'The strategy crashed: {run.error}. Showing what it did up to the crash.')
    return run


# ---------------------------------------------------------------- pieces used by the replay view
def cards(run: R.StrategyRun, pos: int, t: float) -> list[tuple[str, str, str, str | None]]:
    """(label, value, sub, colour) cards for the strategy at row `pos`."""
    i, cur, _ = run.state_at(pos)
    inv = cur.get('sim.inventory')
    pnl = cur.get('sim.pnl')
    rb, rs = cur.get('sim.resting_buy') or 0, cur.get('sim.resting_sell') or 0
    k = int(np.searchsorted(run.replay.t_ev, pos, side='right'))
    mine = run.t_taker_user[:k] | run.t_maker_user[:k]
    pen = cur.get('sim.breach_penalty') or 0
    return [
        ('Your inventory', f'{inv:+,}' if inv is not None else '-', f'resting buy {rb:,} / sell {rs:,} lots',
         DOWN if inv is not None and abs(inv) > 100 else None),
        ('Your PnL now', f'{pnl:,.0f}' if pnl is not None else '-', f'penalty {pen:,.0f}' if pen else 'marked out at the book', None),
        ('Your fills so far', f'{int(mine.sum()):,}', f'{int(run.replay.t_qty[:k][mine].sum()):,} lots', MINE),
    ]


def add_fills_to_chart(fig: go.Figure, run: R.StrategyRun, x0: float, x1: float, ts, fmt_t, hrs: bool, t: float | None = None) -> None:
    """Your fills as large markers on the main price chart (row 1); fills after `t` (not happened yet) are faded."""
    r = run.replay
    ta = r.t_time
    a, b = int(np.searchsorted(ta, x0, 'left')), int(np.searchsorted(ta, x1, 'right'))
    sel = np.arange(a, b)
    mine = sel[(run.t_taker_user[sel] | run.t_maker_user[sel])]
    if not len(mine):
        return
    # your side: if you were the taker, the aggressor side; if the maker, the opposite
    your_buy = np.where(run.t_taker_user[mine], r.t_side[mine] == 0, r.t_side[mine] == 1)
    done = ta[mine] <= (t if t is not None else np.inf)
    for buy, past, col, name in ((True, True, UP, 'you bought'), (False, True, DOWN, 'you sold'),
                                 (True, False, UP, 'you will buy'), (False, False, DOWN, 'you will sell')):
        m = mine[(your_buy == buy) & (done == past)]
        if not len(m):
            continue
        sym = 'circle' if past else 'circle-open'
        how = np.where(run.t_taker_user[m], 'took liquidity', 'your resting order was hit')
        fig.add_trace(go.Scatter(
            x=ts(ta[m]), y=r.t_price[m], mode='markers', name=name,
            marker=dict(symbol=sym, size=np.clip(6 + np.sqrt(r.t_qty[m]), 8, 18), color=col, line=dict(width=2, color=MINE),
                        opacity=1.0 if past else 0.3),
            customdata=np.column_stack([[fmt_t(v, hrs) for v in ta[m]], r.t_price[m], r.t_qty[m], how]),
            hovertemplate='<b>' + name + ' at %{customdata[0]}</b><br>%{customdata[2]} lots at %{customdata[1]:,}<br>%{customdata[3]}<extra></extra>'),
            row=1, col=1)


def jump_targets(run: R.StrategyRun) -> dict[str, np.ndarray]:
    """Rows to jump between: your actions and your fills."""
    fills = run.replay.t_ev[run.t_taker_user | run.t_maker_user]
    return {'action': np.unique(run.act_ev), 'fill': np.unique(fills)}


def _actions_table(run: R.StrategyRun, pos: int, tape, fmt_t, hrs: bool, n: int = 14) -> pd.DataFrame:
    """Your actions and the exchange's replies around row `pos`, newest first."""
    rows = []
    a_hi = int(np.searchsorted(run.act_ev, pos, side='right'))
    for j in range(max(0, a_hi - n), min(len(run.act_ev), a_hi + 3)):
        e = int(run.act_ev[j])
        kind = int(run.act_kind[j])
        if kind == R.ACT_SEND:
            what = f'{"BUY" if run.act_side[j] == 0 else "SELL"} {int(run.act_qty[j]):,} @ {int(run.act_price[j]):,} {"IOC" if run.act_ioc[j] else "limit"}'
        elif kind == R.ACT_CANCEL:
            what = 'cancel'
        else:
            what = f'shrink to {int(run.act_qty[j]):,}'
        rows.append({'row': e, 'time': fmt_t(float(tape.time[e]), hrs), 'who': 'you', 'what': what, 'order': f'#{-int(run.act_oid[j])}',
                     'future': e > pos})
    p_hi = int(np.searchsorted(run.pv_ev, pos, side='right'))
    for j in range(max(0, p_hi - n), min(len(run.pv_ev), p_hi + 3)):
        e = int(run.pv_ev[j])
        stt = run.pv_status[j]
        detail = stt
        if run.pv_fill_qty[j]:
            detail += f' {int(run.pv_fill_qty[j]):,} @ {int(run.pv_fill_price[j]):,}'
        if run.pv_message[j]:
            detail += f' ({run.pv_message[j]})'
        rows.append({'row': e, 'time': fmt_t(float(tape.time[min(e, len(tape.time) - 1)]), hrs), 'who': 'exchange', 'what': detail,
                     'order': f'#{-int(run.pv_oid[j])}', 'future': e > pos})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(['row', 'who'], kind='stable').tail(2 * n)
    return df.iloc[::-1].reset_index(drop=True)  # newest first


def thinking_panel(run: R.StrategyRun, pos: int, tape, fmt_t, hrs: bool) -> None:
    """What the strategy is thinking (state) and doing (actions, replies) at row `pos`."""
    left, right = st.columns([1, 1.25])
    with left:
        i, cur, prev = run.state_at(pos)
        if i < 0:
            st.caption('No state captured yet.')
        else:
            e = int(run.st_ev[i])
            st.markdown(f'**What it is thinking** <span class="muted">at row {e + 1:,} ({fmt_t(float(tape.time[e]), hrs)})</span>',
                        unsafe_allow_html=True,
                        help='Every number/bool/short value on the strategy object, plus its `self.debug` dict if it has one. '
                             'Captured whenever it acts and otherwise every 50 ms of market time. ▲▼ = changed since the previous capture.')
            rows = []
            for k in run.st_keys:
                v = cur.get(k)
                if v is None and k not in prev:
                    continue
                pv = prev.get(k)
                mark = ''
                if pv is not None and v != pv:
                    mark = '▲' if isinstance(v, (int, float)) and isinstance(pv, (int, float)) and v > pv else ('▼' if isinstance(v, (int, float)) and isinstance(pv, (int, float)) else '•')
                group = 'simulator' if k.startswith('sim.') else ('debug' if k.startswith('debug.') else 'strategy')
                rows.append({'': mark, 'name': k.removeprefix('sim.').removeprefix('debug.'), 'value': _fmt_value(v), 'from': group})
            df = pd.DataFrame(rows)
            order = {'debug': 0, 'strategy': 1, 'simulator': 2}
            df = df.sort_values('from', key=lambda s: s.map(order), kind='stable')
            st.dataframe(df, hide_index=True, width='stretch', height=min(40 + 35 * len(df), 560),
                         column_config={'': st.column_config.TextColumn(width='small'), 'value': st.column_config.TextColumn(width='large')})
    with right:
        st.markdown('**What it did** <span class="muted">your orders and the exchange\'s replies, newest first; faded = not happened yet</span>',
                    unsafe_allow_html=True)
        df = _actions_table(run, pos, tape, fmt_t, hrs)
        if df.empty:
            st.caption('No actions yet.')
        else:
            fut = df.pop('future').tolist()
            sty = df.style.apply(lambda r: ['opacity: 0.45' if fut[r.name] else ('color: #4fb3d9' if r['who'] == 'you' else '')] * len(r), axis=1)
            st.dataframe(sty.format({'row': '{:,}'}), hide_index=True, width='stretch', height=min(40 + 35 * len(df), 560))


def state_chart(run: R.StrategyRun, x0: float, x1: float, t: float, ts, style_time_axes, cursor_shapes, hrs: bool, key_prefix: str) -> None:
    """Inventory, PnL and any chosen state values over the chart window."""
    tape = run.replay.tape
    numeric = [k for k in run.st_keys if not k.startswith('sim.') and np.isfinite(run.series(k)).any()]
    default = [k for k in numeric if k.startswith('debug.')][:2] or numeric[:1]
    pick = st.multiselect('Plot strategy values', numeric, default=default, key=f'{key_prefix}_pick',
                          help='Numbers the strategy keeps (and its self.debug values), over the window shown above.')
    xt = tape.time[np.minimum(run.st_ev, len(tape.time) - 1)]
    m = (xt >= x0) & (xt <= x1)
    rows = 2 + len(pick)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=['your inventory (lots)', 'your PnL'] + pick)
    fig.add_trace(go.Scatter(x=ts(xt[m]), y=run.series('sim.inventory')[m], mode='lines', line=dict(color=MINE, shape='hv', width=1.4),
                             name='inventory', showlegend=False), row=1, col=1)
    fig.add_hline(y=100, line=dict(color=DOWN, width=1, dash='dot'), row=1, col=1)
    fig.add_hline(y=-100, line=dict(color=DOWN, width=1, dash='dot'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts(xt[m]), y=run.series('sim.pnl')[m], mode='lines', line=dict(color=ACCENT, width=1.4), name='pnl',
                             showlegend=False), row=2, col=1)
    for j, k in enumerate(pick, 3):
        fig.add_trace(go.Scatter(x=ts(xt[m]), y=run.series(k)[m], mode='lines', line=dict(width=1.2, shape='hv'), name=k, showlegend=False),
                      row=j, col=1)
    style_time_axes(fig, hrs)
    fig.update_xaxes(range=[ts(np.array([x0]))[0], ts(np.array([x1]))[0]])
    fig.update_layout(height=150 + 140 * rows, margin=dict(l=10, r=60, t=30, b=30), paper_bgcolor=CLEAR, plot_bgcolor=CLEAR, dragmode='pan',
                      hovermode='x unified', shapes=cursor_shapes(t, None))
    fig.update_annotations(font_size=12, x=0, xanchor='left')
    st.plotly_chart(fig, width='stretch', theme='streamlit', key=f'{key_prefix}_chart')


def summary(run: R.StrategyRun, analysis_max: float | None) -> None:
    """The run's score card (result.json) and how it compares to the theoretical max."""
    res = run.result
    c = st.columns(6)
    c[0].metric('PnL (score)', f'{res.get("pnl", 0):,.0f}' if res else '-',
                f'{100 * res["pnl"] / analysis_max:.1f}% of theoretical max' if res and analysis_max else None, delta_color='off', delta_arrow='off')
    c[1].metric('Breach penalty', f'{res.get("breach_penalty", 0):,.0f}' if res else '-')
    c[2].metric('Fills', f'{res.get("fills", 0):,}' if res else '-', f'{res.get("traded_volume", 0):,} lots' if res else None,
                delta_color='off', delta_arrow='off')
    c[3].metric('End inventory', f'{res.get("inventory", 0):+,}' if res else '-')
    c[4].metric('|inventory| p95 / p99.7', f'{res.get("abs_inventory_q95", 0)} / {res.get("abs_inventory_q997", 0)}' if res else '-')
    c[5].metric('Run took', f'{run.runtime_s:.0f} s', f'{run.strategy_class}', delta_color='off', delta_arrow='off')
    if run.overrides:
        st.caption('Parameters changed for this run: ' + ', '.join(f'`{n} = {_fmt_value(v)}`' for n, (_, v) in run.overrides.items()))
