"""Is a constructed validation twin usable? Compare it against real validation submissions (anchors).

An anchor is a folder with the submitted strategy.py plus the arena's result.json and pnl.csv (download them with
scripts/arena_mcp.py: /evaluations/{id} gives strategyFile and resultFileZip; data/validation_r3/ holds round 3's).
Every anchor's exact file is backtested on every twin tape, and the twin is checked against the real run:

  T1 price path   probes (strategies that hold a fixed position) must reproduce their real PnL curve: RMS error of the
                  1 s curve, final mark price. Fails = the S path or the spread around it is off.
  T2 outcome      PnL, fills, traded volume, |inventory| quantiles: twin / real ratio per anchor (target within 15%).
  T3 timing       correlation of 60 s PnL increments, twin vs real (target >= 0.7): profit in the right places.
  T4 ranking      Spearman of twin vs real PnL across the non-probe anchors (needs >= 3); the test that matters for
                  tuning on the twin.
  T5 seed spread  with several twin seeds: is the real PnL inside the range of the twins?

uv run python scripts/twin_check.py data/validation_r3 data/r3_eval_twin.csv [more twin seeds ...] [-j 6]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent


def run(strategy: Path, tape: Path, out: Path) -> Path:
    if not (out / 'result.json').exists():
        subprocess.run(['uv', 'run', 'exchange-simulator', '--strategy', str(strategy), '--backtest', str(tape),
                        '--log-dir', str(out), '--no-logs'], check=True, capture_output=True, cwd=ROOT)
    return out


def curve(d: Path) -> pd.Series:
    c = pd.read_csv(d / 'pnl.csv').drop_duplicates('elapsed_s', keep='last').set_index('elapsed_s').pnl
    return c.reindex(np.arange(0, int(c.index.max()) + 1), method='ffill').fillna(0.0)


def is_probe(res: dict) -> bool:
    return res.get('fills', 0) <= 5  # holds a fixed position: a handful of entry fills


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('anchors', type=Path)
    ap.add_argument('twins', type=Path, nargs='+')
    ap.add_argument('-j', type=int, default=6)
    ap.add_argument('--out', type=Path, default=Path(tempfile.gettempdir()) / 'twin_check')
    a = ap.parse_args()

    anchors = sorted(d for d in a.anchors.iterdir() if (d / 'strategy.py').exists() and (d / 'result.json').exists())
    jobs = [(d, t, a.out / t.stem / d.name) for d in anchors for t in a.twins]
    with ThreadPoolExecutor(a.j) as ex:
        list(ex.map(lambda j: run(j[0] / 'strategy.py', j[1], j[2]), jobs))

    rows = []
    for d in anchors:
        real = json.loads((d / 'result.json').read_text())
        rc = curve(d)
        for t in a.twins:
            od = a.out / t.stem / d.name
            tw = json.loads((od / 'result.json').read_text())
            tc = curve(od)
            n = min(len(rc), len(tc))
            inc_r = np.diff(rc.to_numpy()[:n][::60])
            inc_t = np.diff(tc.to_numpy()[:n][::60])
            corr = float(np.corrcoef(inc_r, inc_t)[0, 1]) if inc_r.std() > 0 and inc_t.std() > 0 else np.nan
            rms = float(np.sqrt(np.mean((rc.to_numpy()[:n] - tc.to_numpy()[:n]) ** 2)))
            ratio = lambda k: tw[k] / real[k] if real.get(k) else np.nan  # noqa: E731
            rows.append({'anchor': d.name, 'twin': t.stem, 'probe': is_probe(real), 'real_pnl': real['pnl'], 'twin_pnl': tw['pnl'],
                         'pnl_ratio': ratio('pnl'), 'fills_ratio': ratio('fills'), 'volume_ratio': ratio('traded_volume'),
                         'invq50_real': real['abs_inventory_q50'], 'invq50_twin': tw['abs_inventory_q50'],
                         'mark_real': real['mark_price'], 'mark_twin': tw['mark_price'], 'inc60_corr': corr, 'curve_rms': rms,
                         'penalty_real': real['breach_penalty'], 'penalty_twin': tw['breach_penalty']})
    R = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    print(R.round(3).to_string(index=False))

    print('\nverdicts:')
    pr = R[R.probe]
    if len(pr):
        print(f'  T1 price path (probes): curve RMS median {pr.curve_rms.median():,.0f}, 60 s increment corr median {pr.inc60_corr.median():.2f}, '
              f'mark price gap median {np.median(np.abs(pr.mark_twin - pr.mark_real)):.1f} ticks')
    st = R[~R.probe]
    if len(st):
        print(f'  T2 outcome (strategies): PnL ratio {st.pnl_ratio.min():.2f}..{st.pnl_ratio.max():.2f}, fills ratio '
              f'{st.fills_ratio.min():.2f}..{st.fills_ratio.max():.2f}, volume ratio {st.volume_ratio.min():.2f}..{st.volume_ratio.max():.2f} '
              f'(target 0.85..1.15)')
        print(f'  T3 timing: 60 s PnL increment corr {st.inc60_corr.min():.2f}..{st.inc60_corr.max():.2f} (target >= 0.7)')
        per = st.groupby('anchor').agg(real=('real_pnl', 'first'), twin=('twin_pnl', 'mean'), lo=('twin_pnl', 'min'), hi=('twin_pnl', 'max'))
        if len(per) >= 3:
            print(f'  T4 ranking: Spearman twin vs real over {len(per)} strategies {spearmanr(per.real, per.twin).statistic:+.2f}')
        else:
            print(f'  T4 ranking: needs >= 3 non-probe anchors (have {len(per)}); submit more distinct strategies')
        if len(a.twins) > 1:
            inside = ((per.real >= per.lo) & (per.real <= per.hi)).mean()
            print(f'  T5 seed spread: real PnL inside the twin range for {inside:.0%} of strategies')
        else:
            print('  T5 seed spread: needs several twin seeds')


if __name__ == '__main__':
    main()
