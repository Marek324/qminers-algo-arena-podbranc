"""Building the report dict and rendering it as JSON or indented text."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

from .context import GRID_STEP_US, Context
from .replay import replay
from .sections.book import book_section
from .sections.flow import orders_section, tape_section
from .sections.indicators import indicators_section
from .sections.maxpnl import max_pnl_section
from .sections.price import price_section
from .sections.series import bins_section, series_section
from .sections.signals import signals_section
from .sections.trades import maker_section, trades_section
from .tape import load_tape
from .util import US, clean

SCHEMA_VERSION = 1
SECTION_ORDER = (
    'meta',
    'highlights',
    'tape',
    'orders',
    'book',
    'price',
    'trades',
    'maker',
    'max_pnl',
    'signals',
    'indicators',
    'regimes',
    'bins',
    'series',
)


def analyze(
    path: Path | IO[bytes], series_step_s: float = 1.0, bin_step_s: float = 60.0, progress: bool = False, name: str | None = None
) -> dict:
    """Run every section on one tape (a path, or an open binary file plus `name`) and return the JSON-ready report."""
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    def timed(name, fn, *args):
        t = time.perf_counter()
        out = fn(*args)
        timings[name] = time.perf_counter() - t
        if progress:
            print(f'  {name:<8} {timings[name]:6.2f}s', flush=True)
        return out

    tape = timed('load', load_tape, path)
    if tape.empty:
        raise ValueError(f'{path} has no rows')
    rp = timed('replay', replay, tape, GRID_STEP_US)
    ctx = Context(tape=tape, rp=rp, series_step_us=int(series_step_s * US), bin_step_us=int(bin_step_s * US))

    report: dict = {}
    report['tape'] = timed('tape', tape_section, ctx)
    report['orders'] = timed('orders', orders_section, ctx)
    report['book'] = timed('book', book_section, ctx)
    report['price'] = timed('price', price_section, ctx)
    report['trades'] = timed('trades', trades_section, ctx)
    report['maker'] = timed('maker', maker_section, ctx)
    report['max_pnl'] = timed('max_pnl', max_pnl_section, ctx, ctx.bin_step_us)
    report['signals'] = timed('signals', signals_section, ctx)
    report['indicators'] = timed('indicators', indicators_section, ctx)
    bins, regimes = timed('bins', bins_section, ctx)
    report['regimes'] = regimes
    report['bins'] = bins
    report['series'] = timed('series', series_section, ctx)
    report = clean(report)
    report['highlights'] = highlights(report)
    report['meta'] = {
        'schema_version': SCHEMA_VERSION,
        'source': str(path) if isinstance(path, Path) else name,
        'source_name': path.name if isinstance(path, Path) else name,
        'generated_at': datetime.now(UTC).isoformat(timespec='seconds'),
        'units': {
            'price': 'ticks (raw tape price / tick_size)',
            'time': 'seconds from first event unless the key ends in _ms or _us',
            'quantity': 'lots',
            'signs': 'trade sign +1 = buyer-initiated; book side +1 = bid',
        },
        'config': {
            'series_step_s': series_step_s,
            'bin_step_s': bin_step_s,
            'grid_step_s': GRID_STEP_US / US,
            'latency_s': 0.005,
        },
        'runtime_s': clean({**timings, 'total': time.perf_counter() - t_start}),
    }
    return {k: report[k] for k in SECTION_ORDER}


def _get(d: dict, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return default if d is None else d


def highlights(r: dict) -> list[str]:
    """Plain-language takeaways derived from the numbers. Rules, not models."""
    out: list[str] = []
    ss = _get(r, 'book', 'spread_share', default={})
    modal = _get(r, 'book', 'modal_spread_ticks')
    if modal is not None:
        out.append(f'Spread is mostly {modal:g} tick(s); 1-tick share of time {_pct(ss.get("1"))}, >=3 ticks {_pct(sum(v or 0 for k, v in ss.items() if k not in ("1", "2")))}.')

    vr = _get(r, 'price', 'variance_ratio_vs_1s', '60s')
    rev = _get(r, 'price', 'event_time', 'reversal_prob')
    if vr is not None:
        kind = 'mean-reverting' if vr < 0.8 else 'trending' if vr > 1.2 else 'close to a random walk'
        out.append(f'Variance ratio 60s/1s = {vr:.2f}: price is {kind} at minute scale; mid-change reversal prob {_pct(rev)}.')

    t = _get(r, 'price', 'trend', 't_stat_60s_returns')
    drift = _get(r, 'price', 'trend', 'drift_ticks_per_hour')
    if t is not None and drift is not None:
        out.append(f'Drift {drift:+.1f} ticks/hour (t = {t:+.2f} on 60 s returns){"; significant trend" if abs(t) > 2 else ""}.')

    mk = _get(r, 'maker', 'markout_ticks', default={})
    if mk.get('0ms') is not None:
        parts = ', '.join(f'{h} {mk[h]:+.3f}' for h in ('0ms', '100ms', '1s', '5s', '60s') if mk.get(h) is not None)
        verdict = 'passive fills keep an edge' if (mk.get('5s') or 0) > 0 else 'adverse selection eats the half-spread'
        out.append(f'Maker markout per lot (ticks): {parts}; at 5 s {verdict}.')

    mp = _get(r, 'max_pnl', default={})
    if _get(mp, 'theoretical_max', 'pnl') is not None:
        out.append(
            f'Theoretical max PnL {mp["theoretical_max"]["pnl"]:,.0f} ({_get(mp, "theoretical_max", "pnl_per_hour", default=0):,.0f}/hour); '
            f'making only {_get(mp, "making_only", "pnl", default=0):,.0f}, taking only {_get(mp, "taking_only", "pnl", default=0):,.0f}; '
            f'theoretical min {_get(mp, "theoretical_min", "pnl", default=0):,.0f}.'
        )

    best = None
    for name, row in _get(r, 'signals', 'directional', default={}).items():
        sp = _get(row, '1s', 'spearman')
        if sp is not None and (best is None or abs(sp) > abs(best[1])):
            best = (name, sp, _get(row, '1s', 'hit_rate'))
    if best:
        out.append(f'Strongest 1 s predictor: {best[0]} (Spearman {best[1]:+.3f}, hit rate {_pct(best[2])}).')
    oos = _get(r, 'signals', 'linear_combination', '1s', 'r2_out_of_sample')
    if oos is not None:
        out.append(f'Linear combo of directional features: out-of-sample R^2 at 1 s = {oos:.3f}.')

    rk = _get(r, 'indicators', 'ranking_by_abs_spearman_5s', default=[])
    if rk:
        top = ', '.join(f'{x["indicator"]} {x["spearman_5s"]:+.3f}' for x in rk[:3] if x['spearman_5s'] is not None)
        out.append(f'Best technical indicators for the 5 s move (Spearman): {top}.')

    lat = _get(r, 'book', 'latency_5ms', 'event_weighted', 'touch_price_moved_share')
    if lat is not None:
        out.append(f'Touch price changes during the 5 ms latency after {_pct(lat)} of events (stale-quote risk).')

    cw = _get(r, 'orders', 'cancel_within_share', '1s')
    co = _get(r, 'orders', 'outcome_share', 'canceled')
    if cw is not None:
        out.append(f'{_pct(co)} of orders end canceled; {_pct(cw)} of cancels come within 1 s of insert.')

    sa = _get(r, 'trades', 'sign_autocorr', 'lag1')
    if sa is not None:
        out.append(f'Aggressor sign autocorrelation lag1 = {sa:+.3f} ({"order flow is persistent" if sa > 0.1 else "flow signs are weakly related"}).')

    imp = _get(r, 'trades', 'impact_ticks', '1s', 'mean_ticks')
    if imp is not None:
        out.append(f'Average aggressive order moves mid {imp:+.3f} ticks in its direction after 1 s.')

    an = _get(r, 'tape', 'anomalies', default={})
    if an.get('cancel_not_resting'):
        out.append(
            f'{an["cancel_not_resting"]} cancels hit non-resting orders (after full fill {an.get("cancel_after_full_fill")}, unknown id {an.get("cancel_unknown_id")}).'
        )
    if an.get('duplicate_insert_while_resting'):
        out.append(f'WARNING: {an["duplicate_insert_while_resting"]} inserts reuse a resting id; the simulator would raise on these.')
    return out


def _pct(x) -> str:
    return 'n/a' if x is None else f'{100 * x:.1f}%'


def to_json(report: dict, indent: int | None = 1) -> str:
    return json.dumps(report, indent=indent, allow_nan=False)


def to_text(report: dict, max_list: int = 24) -> str:
    """Indented `key: value` text. Long arrays (series, bins) are summarised, not printed."""
    lines: list[str] = []

    def fmt(v) -> str:
        if isinstance(v, float):
            return f'{v:.6g}'
        if v is None:
            return '-'
        return str(v)

    def walk(obj, indent: int, key: str | None):
        pad = '  ' * indent
        head = f'{pad}{key}:' if key is not None else None
        if isinstance(obj, dict):
            if key in ('series', 'bins') and 'columns' in obj:
                lines.append(f'{head} {obj.get("rows")} rows x step {obj.get("step_s")}s, columns: {", ".join(obj["columns"])}')
                return
            if head:
                lines.append(head)
            for k, v in obj.items():
                walk(v, indent + (1 if head else 0), k)
        elif isinstance(obj, list):
            if obj and all(isinstance(x, dict) for x in obj):
                lines.append(head or '')
                for x in obj:
                    lines.append(f'{pad}  - ' + ', '.join(f'{k}={fmt(v)}' for k, v in x.items()))
            elif obj and all(isinstance(x, str) for x in obj):
                lines.append(head or '')
                for x in obj:
                    lines.append(f'{pad}  - {x}')
            else:
                shown = ', '.join(fmt(x) for x in obj[:max_list])
                more = f', ... (+{len(obj) - max_list})' if len(obj) > max_list else ''
                lines.append(f'{head} [{shown}{more}]')
        else:
            lines.append(f'{head} {fmt(obj)}')

    walk(report, 0, None)
    return '\n'.join(lines) + '\n'
