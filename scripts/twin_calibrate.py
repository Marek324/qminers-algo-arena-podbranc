"""Calibrate a validation twin's order flow against real validation submissions (anchors), with a leave-one-out test.

The twin's price path comes from the long and short hold-probes (scripts/path_twin.py); its order flow comes from the
generator (analysis/generator.py). Here the flow parameters are varied on a small grid; for every candidate twin each
anchor's exact strategy.py is backtested and compared with its real arena result (scripts/twin_check.py's anchor
layout, e.g. data/validation_round3/ from arena_download). Loss per anchor: |log PnL ratio| (skipped for |PnL| < 2k),
|log fills ratio| and |log volume ratio|, plus (1 - 60 s PnL increment correlation) for non-probe anchors.

Leave-one-out: for each anchor, the candidate with the lowest loss on the OTHER anchors predicts it. A twin is usable
for tuning only if those held-out predictions are within about 15% (PnL) and keep the anchors' order.

uv run python scripts/twin_calibrate.py data/validation_round3 --params round3_public [-j 7]
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from analysis.generator import PARAM_DIR, GenParams, generate, write_tape  # noqa: E402
from estimate_model import with_gamma_x  # noqa: E402
from path_twin import mid_from_probes, regimes_from_path  # noqa: E402

OUT = ROOT / 'logs' / 'twin_cal2'


def anchors(d: Path) -> list[Path]:
    return sorted(x for x in d.iterdir() if (x / 'strategy.py').exists() and (x / 'result.json').exists())


def find_probe(anchs: list[Path], marker: str) -> Path:
    for a in anchs:
        if marker in (a / 'strategy.py').read_text()[:400]:
            return a
    raise SystemExit(f'no anchor whose strategy docstring contains {marker!r}')


def backtest(strategy: Path, tape: Path, out: Path) -> dict:
    if not (out / 'result.json').exists():
        subprocess.run(['uv', 'run', 'exchange-simulator', '--strategy', str(strategy), '--backtest', str(tape),
                        '--log-dir', str(out), '--no-logs'], check=True, capture_output=True, cwd=ROOT)
    return json.loads((out / 'result.json').read_text())


def curve(d: Path) -> np.ndarray:
    c = pd.read_csv(d / 'pnl.csv').drop_duplicates('elapsed_s', keep='last').set_index('elapsed_s').pnl
    return c.reindex(np.arange(0, int(c.index.max()) + 1), method='ffill').fillna(0.0).to_numpy()


def score(anchor: Path, run_dir: Path) -> dict:
    real = json.loads((anchor / 'result.json').read_text())
    tw = json.loads((run_dir / 'result.json').read_text())
    probe = real.get('fills', 0) <= 5
    lr = lambda k: abs(np.log(max(tw[k], 1e-9) / max(real[k], 1e-9))) if real.get(k, 0) > 0 and tw.get(k, 0) > 0 else 1.0  # noqa: E731
    loss = lr('fills') + lr('traded_volume')
    pnl_ratio = tw['pnl'] / real['pnl'] if abs(real['pnl']) > 0 else np.nan
    if abs(real['pnl']) >= 2000:
        loss += abs(np.log(pnl_ratio)) if pnl_ratio > 0 else 2.0
    corr = np.nan
    if not probe:
        rc, tc = curve(anchor), curve(run_dir)
        n = min(len(rc), len(tc))
        ir, it = np.diff(rc[:n][::60]), np.diff(tc[:n][::60])
        corr = float(np.corrcoef(ir, it)[0, 1]) if ir.std() > 0 and it.std() > 0 else 0.0
        loss += 1.0 - corr
    return {'loss': loss, 'pnl_real': real['pnl'], 'pnl_twin': tw['pnl'], 'pnl_ratio': pnl_ratio,
            'fills_ratio': tw['fills'] / max(real['fills'], 1), 'corr': corr, 'probe': probe}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('anchors', type=Path)
    ap.add_argument('--params', default='round3_public')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('-j', type=int, default=7)
    ap.add_argument('--cfgs', default='', help='comma-separated config indices to run here (split across machines)')
    ap.add_argument('--merge', type=Path, nargs='*', default=None, help='results.csv files to merge and report on, no runs')
    a = ap.parse_args()

    anchs = anchors(a.anchors)
    long_dir = find_probe(anchs, 'Price probe')
    short_dir = find_probe(anchs, '-10 lots')
    t, s = mid_from_probes(long_dir, short_dir)
    base = GenParams.load(PARAM_DIR / f'{a.params}_params.json')
    reg = regimes_from_path(t, s, base, 20.0) if base.storm_dur_s > 0 else None
    print(f'{len(anchs)} anchors; path {t[-1]:.0f} s from {long_dir.name} / {short_dir.name}', flush=True)

    grid = {
        'ioc_couple': (0.0, 0.15, 0.3),
        'ioc_rate': (1.0, 1.3),
        'x_mean': (0.8, 1.0),
    }
    combos = [dict(zip(grid, v, strict=True)) for v in itertools.product(*grid.values())]
    OUT.mkdir(parents=True, exist_ok=True)
    if a.merge is not None:
        R = pd.concat([pd.read_csv(f) for f in a.merge], ignore_index=True).drop_duplicates(['cfg', 'anchor'])
        report(R, combos, anchs)
        return
    todo = [int(x) for x in a.cfgs.split(',')] if a.cfgs else list(range(len(combos)))

    def build(i_c):
        i, c = i_c
        tape = OUT / f'twin_{i}.csv'
        if not tape.exists():
            p = replace(base, ioc_rate=base.ioc_rate * c['ioc_rate'], ioc_couple=c['ioc_couple'],
                        ioc_rate_storm=(base.ioc_rate_storm or base.ioc_rate) * c['ioc_rate'], impact_k=base.impact_k * 0.6)
            p = with_gamma_x(p, base.x_mean * c['x_mean'], base.x_std)  # X is drawn from the gamma shape, not x_mean
            df, _, _ = generate(p, seed=a.seed, duration_s=float(t[-1]), s_path=(t, s), jump_ticks=20.0, regimes=reg)
            write_tape(df, tape)
        return tape

    with ThreadPoolExecutor(max(1, a.j // 3)) as ex:
        tapes = dict(zip(todo, ex.map(build, [(i, combos[i]) for i in todo]), strict=True))
    jobs = [(i, an) for i in todo for an in anchs]
    with ThreadPoolExecutor(a.j) as ex:
        list(ex.map(lambda j: backtest(j[1] / 'strategy.py', tapes[j[0]], OUT / f'run_{j[0]}' / j[1].name), jobs))

    rows = []
    for i in todo:
        for an in anchs:
            rows.append({'cfg': i, **combos[i], 'anchor': an.name, **score(an, OUT / f'run_{i}' / an.name)})
    R = pd.DataFrame(rows)
    tag = ('_' + a.cfgs.replace(',', '-')) if a.cfgs else ''
    R.to_csv(OUT / f'results{tag}.csv', index=False)
    report(R, combos, anchs)


def report(R: pd.DataFrame, combos: list, anchs: list[Path]) -> None:
    tot = R.groupby('cfg').loss.sum().sort_values()
    best = int(tot.index[0])
    print('\nbest configs (total loss over all anchors):')
    print(pd.DataFrame([dict(cfg=i, loss=round(tot[i], 2), **combos[i]) for i in tot.index[:6]]).to_string(index=False))
    print(f'\nbest config {best} per anchor:')
    print(R[R.cfg == best][['anchor', 'probe', 'pnl_real', 'pnl_twin', 'pnl_ratio', 'fills_ratio', 'corr', 'loss']].round(3).to_string(index=False))

    print('\nleave-one-out (calibrated without the anchor, then predicting it):')
    loo = []
    for an in anchs:
        others = R[R.anchor != an.name].groupby('cfg').loss.sum()
        c = int(others.idxmin())
        r = R[(R.cfg == c) & (R.anchor == an.name)].iloc[0]
        loo.append((an.name, c, r['pnl_real'], r['pnl_twin'], r['pnl_ratio'], r['fills_ratio'], r['corr']))
        print(f"  {an.name}: cfg {c}, real {r['pnl_real']:,.0f}, predicted {r['pnl_twin']:,.0f} (ratio {r['pnl_ratio']:.2f}), "
              f"fills ratio {r['fills_ratio']:.2f}, timing corr {r['corr']:.2f}")
    strat = [x for x in loo if abs(x[2]) >= 2000]
    if len(strat) >= 3:
        from scipy.stats import spearmanr
        print(f'  held-out ranking Spearman over {len(strat)} anchors: {spearmanr([x[2] for x in strat], [x[3] for x in strat]).statistic:+.2f}')
    ok = all(0.85 <= x[4] <= 1.15 for x in strat if not np.isnan(x[4]))
    print(f'  verdict: {"USABLE for tuning" if ok else "NOT usable for tuning (held-out PnL off by more than 15%)"}')


if __name__ == '__main__':
    main()
