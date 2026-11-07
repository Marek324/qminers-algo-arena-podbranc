"""Robustness tab: is a strategy stable across tapes, over time and around its parameter values?

Reads the reports `scripts/robustness.py` writes to logs/robustness/ and can start that script in the background.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis_tab as A
import strategy_run as R
import strategy_view as V

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / 'logs' / 'robustness'
SCRIPT = ROOT / 'scripts' / 'robustness.py'


def _reports() -> list[Path]:
    return sorted(OUT_DIR.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True) if OUT_DIR.exists() else []


@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


def _launch(strategy: Path, overrides: dict, sens: list[str], chunks: int, boot: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = OUT_DIR / f'{strategy.stem}.log'
    cmd = [sys.executable, str(SCRIPT), str(strategy), '--chunks', str(chunks), '--boot', str(boot)]
    for name, (_, v) in overrides.items():
        cmd += ['--set', f'{name}={v}']
    if sens:
        cmd += ['--sensitivity', ','.join(sens)]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log.open('w'), stderr=subprocess.STDOUT)
    st.session_state['rb_proc'] = (proc, str(log), strategy.name)


def _status() -> None:
    item = st.session_state.get('rb_proc')
    if not item:
        return
    proc, log, name = item
    tail = Path(log).read_text().strip().splitlines()[-1:] if Path(log).exists() else []
    if proc.poll() is None:
        st.info(f'Robustness run for {name} in progress: {tail[0] if tail else "starting"}. Press **Refresh** to update.')
        st.button('Refresh', key='rb_refresh')
    else:
        if proc.returncode == 0:
            st.success(f'Robustness run for {name} finished.')
        else:
            st.error(f'Robustness run for {name} failed (exit {proc.returncode}); see {log}.')
        del st.session_state['rb_proc']


def controls() -> None:
    files = R.strategy_files()
    with st.expander('Run a robustness check', expanded=not _reports()):
        c = st.columns([2, 1, 1, 3], vertical_alignment='bottom')
        names = [f.name for f in files]
        side = st.session_state.get('strat_file')
        strat = c[0].selectbox('Strategy', names, index=names.index(side) if side in names else 0, key='rb_strat')
        chunks = c[1].number_input('Time chunks per tape', 2, 24, 6, key='rb_chunks')
        boot = c[2].number_input('Bootstrap sessions', 0, 50, 8, key='rb_boot')
        path = next(f for f in files if f.name == strat)
        try:
            params = [p.name for p in V._params(str(path), path.stat().st_mtime)[1]]
        except Exception:  # unloadable strategy file
            params = []
        sens = c[3].multiselect('Sensitivity on (optional, slower)', params, key='rb_sens',
                                help='Each chosen constant at x0.5 ... x1.5 (bools flipped), on both halves of every tape.')
        overrides = st.session_state.get(f'strat_applied:{strat}', {})
        if overrides:
            st.caption('Uses the parameter values applied in the sidebar: ' + ', '.join(f'`{k} = {v}`' for k, (_, v) in overrides.items()))
        tapes = ', '.join(p.name for p in sorted((ROOT / 'data').glob('*.csv*')))
        st.caption(f'Runs on every tape in data/ ({tapes}). About 1 min per million rows of tape on all cores, '
                   'plus about 10 s per sensitivity variant per tape.')
        busy = bool(st.session_state.get('rb_proc'))
        if st.button('Run robustness check', type='primary', disabled=busy, key='rb_go'):
            _launch(path, overrides, sens, int(chunks), int(boot))
            st.rerun()
    _status()


def render() -> None:
    st.caption('Is the strategy stable or fitted to one session? Each check runs the real simulator (scoring settings) on variants of '
               'every tape: whole tapes, time chunks from a cold start, a mirror-image market, reshuffled sessions and, optionally, '
               'parameters nudged around their values.')
    controls()
    reps = _reports()
    if not reps:
        st.info('No robustness reports yet. Run one above, or `uv run python scripts/robustness.py strategies/<file>.py`.')
        return
    labels = {p.name: p for p in reps}
    pick = st.selectbox('Report', list(labels), key='rb_report',
                        format_func=lambda n: f'{n.removesuffix(".json")}  ·  {_load(str(labels[n]), labels[n].stat().st_mtime)["meta"]["generated_at"]}')
    p = labels[pick]
    rep = _load(str(p), p.stat().st_mtime)
    meta = rep['meta']
    st.caption(f'{meta["strategy"]} ({meta["strategy_class"]}) on {", ".join(meta["tapes"])}; {meta["chunks"]} chunks per tape, '
               f'{meta["bootstrap"]} bootstrap sessions; {meta["jobs"]} backtests in {meta["runtime_s"]:.0f} s'
               + (f'; overrides {meta["overrides"]}' if meta.get('overrides') else ''))
    A.verdict_cards(rep)

    files = rep['files']
    st.subheader('Across tapes')
    rows = [{'tape': f['name'], 'minutes': f['duration_s'] / 60, 'PnL': f['full'].get('pnl'), 'PnL / hour': f['pnl_per_hour'],
             'share of max': f['share_of_max'], 'theoretical max': f['theoretical_max'], 'chunks profitable': f'{f["chunks_profitable"]}/{len(f["chunks"])}',
             'mirrored PnL': f['mirrored'].get('pnl'), 'bootstrap p05': f['bootstrap_summary']['p05'],
             'bootstrap median': f['bootstrap_summary']['median'], 'breach penalty': f['full'].get('breach_penalty'),
             '|inv| p99.7': f['full'].get('abs_inventory_q997')} for f in files]
    A.table(pd.DataFrame(rows))

    st.subheader('Over time')
    fig = go.Figure()
    for i, f in enumerate(files):
        ch = f['chunks']
        fig.add_trace(go.Bar(x=[c['label'] for c in ch], y=A.arr([c['pnl_per_hour'] for c in ch]), name=f['name'],
                             marker_color=A.PALETTE[i % len(A.PALETTE)],
                             customdata=np.column_stack([[c['pnl'] or 0 for c in ch], [100 * (c['share_of_max'] or 0) for c in ch]]),
                             hovertemplate='%{x}: %{y:,.0f}/hour<br>PnL %{customdata[0]:,.0f}<br>%{customdata[1]:.1f}% of max<extra>' + f['name'] + '</extra>'))
    fig.add_hline(y=0, line=dict(color='rgba(128,128,128,0.6)', dash='dot'))
    A.style(fig, 340, title=dict(text='PnL per hour in each time chunk (cold start each)', font_size=13, x=0), barmode='group')
    A.show(fig, 'rbc_chunks_chart')

    left, right = st.columns(2)
    ch = [c for f in files for c in f['chunks'] if c.get('pnl') is not None and c.get('mid_change_ticks') is not None]
    for col, key, title, xlab in ((left, 'mid_change_ticks', 'chunk PnL vs price drift', 'price drift over the chunk (ticks)'),
                                  (right, 'vol_ticks_per_sqrt_s', 'chunk PnL vs volatility', 'volatility (ticks per √s)')):
        with col:
            fig = go.Figure()
            for i, f in enumerate(files):
                cc = [c for c in ch if c['file'] == f['name']]
                fig.add_trace(go.Scatter(x=[c[key] for c in cc], y=[c['pnl_per_hour'] for c in cc], mode='markers', name=f['name'],
                                         marker=dict(size=10, color=A.PALETTE[i % len(A.PALETTE)]), text=[c['label'] for c in cc],
                                         hovertemplate='%{text}<br>x %{x:.3g}<br>%{y:,.0f}/hour<extra>' + f['name'] + '</extra>'))
            fig.add_hline(y=0, line=dict(color='rgba(128,128,128,0.6)', dash='dot'))
            A.style(fig, 320, title=dict(text=title, font_size=13, x=0), xaxis_title=xlab, yaxis_title='PnL per hour')
            A.show(fig, f'rbc_sc_{key}')

    boots = [(f['name'], [b['pnl'] for b in f['bootstrap'] if b.get('pnl') is not None]) for f in files]
    if any(b for _, b in boots):
        st.subheader('Reshuffled sessions (bootstrap)')
        fig = go.Figure()
        for i, (name, b) in enumerate(boots):
            if b:
                fig.add_trace(go.Box(y=b, name=name, boxpoints='all', jitter=0.4, marker_color=A.PALETTE[i % len(A.PALETTE)]))
        fig.add_hline(y=0, line=dict(color='rgba(128,128,128,0.6)', dash='dot'))
        A.style(fig, 320, title=dict(text=f'PnL of {meta["bootstrap"]} sessions of {meta["boot_minutes"]:g} min built from random 60 s blocks',
                                     font_size=13, x=0), showlegend=False)
        A.show(fig, 'rbc_boot')

    sens = rep.get('sensitivity') or {}
    if sens.get('params'):
        st.subheader('Parameter sensitivity')
        st.caption('Total PnL over both halves of every tape with one constant changed at a time (x = multiple of the current value). '
                   'A plateau around 1.0 is robust; a spike at 1.0 is a lucky peak.')
        items = list(sens['params'].items())
        for k in range(0, len(items), 3):
            cols = st.columns(3)
            for col, (name, d) in zip(cols, items[k:k + 3], strict=False):
                with col:
                    rows_ = d['rows']
                    x = [r['scale'] if r['scale'] is not None else 'flipped' for r in rows_]
                    fig = go.Figure(go.Scatter(x=[str(v) for v in x], y=A.arr([r['total_pnl'] for r in rows_]), mode='lines+markers',
                                               text=[str(r['value']) for r in rows_], hovertemplate='x%{x} (= %{text}): %{y:,.0f}<extra></extra>',
                                               line=dict(color=A.ACCENT)))
                    drop = d.get('drop_at_25pct')
                    A.style(fig, 240, title=dict(text=f'{name}' + (f' · drop at ±25%: {100 * drop:.0f}%' if drop is not None else ''), font_size=12, x=0))
                    fig.update_xaxes(type='category')
                    A.show(fig, f'rbc_sens_{name}')
        wf = sens.get('walk_forward') or {}
        if wf:
            st.markdown('**Walk-forward**: does the variant that wins on the first half also win on the second?')
            A.table(pd.DataFrame([{'tape': n, 'rank correlation first vs second half': w['rank_corr'], 'best on first half': w['best_on_first'],
                                   'its PnL on second half': w['its_pnl_second'], 'current values on second half': w['default_pnl_second']}
                                  for n, w in wf.items()]))
