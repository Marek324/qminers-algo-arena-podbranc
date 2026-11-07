"""Does the generator rank strategies the way the real tape does? The test that matters for a market model: a variant
that wins on generated sessions must also win on the real tape (round1's model failed it).

usage:
    uv run python scripts/transfer_check.py round2 data/round2_public.csv \\
        strategies/adaptive_mm_plus.py strategies/jump_guard_mm.py:JUMP_MIN=5,K_MOM=0 strategies/jump_guard_mm.py:JUMP_MIN=2 ...

Each variant is a strategy file, optionally followed by `:NAME=value,NAME=value` overrides of its top-level constants.
Every variant runs on the real tape and on the same N generated sessions (common random numbers, so differences
between variants are not session luck). Reports PnL per hour on both, the Spearman rank correlation, the share of
variant pairs ordered the same way, and the real rank of the variant the generated sessions like best.
Output: logs/generator/<name>_transfer.json and .md.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.generator import PARAM_DIR, GenParams, generate, write_tape  # noqa: E402


def make_variant(spec: str, tmp: Path, i: int) -> tuple[str, Path]:
    path, _, ov = spec.partition(':')
    text = Path(path).read_text()
    for kv in filter(None, ov.split(',')):
        name, value = kv.split('=', 1)
        # top-level constants and class attributes (indented) alike
        text, n = re.subn(rf'^(\s*){name} = [^#\n]*', rf'\g<1>{name} = {value}  ', text, count=1, flags=re.M)
        if n == 0:
            raise SystemExit(f'{name} not found in {path}')
    out = tmp / f'variant{i}.py'
    out.write_text(text)
    return spec, out


def backtest(strategy: Path, tape: Path, out: Path) -> dict:
    subprocess.run(['uv', 'run', 'exchange-simulator', '--strategy', str(strategy), '--backtest', str(tape),
                    '--log-dir', str(out), '--no-logs'], check=True, capture_output=True, cwd=ROOT)
    return json.loads((out / 'result.json').read_text())


def tape_hours(tape: Path) -> float:
    with open(tape, 'rb') as f:
        f.seek(-200, os.SEEK_END)
        last = f.read().decode().strip().splitlines()[-1]
    return float(last.split(',')[0]) / 3600


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('name', help='params name: logs/generator/<name>_params.json')
    ap.add_argument('real_tape', type=Path)
    ap.add_argument('variants', nargs='+')
    ap.add_argument('--sessions', type=int, default=6)
    ap.add_argument('--minutes', type=float, default=20)
    ap.add_argument('--seed0', type=int, default=1000)
    ap.add_argument('-j', '--jobs', type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    p = GenParams.load(PARAM_DIR / f'{args.name}_params.json')
    tmp = Path(tempfile.mkdtemp(prefix='transfer_'))
    tapes = []
    for k in range(args.sessions):
        df, _, _ = generate(p, seed=args.seed0 + k, duration_s=args.minutes * 60)
        tapes.append(write_tape(df, tmp / f'session{k}.csv'))
    variants = [make_variant(s, tmp, i) for i, s in enumerate(args.variants)]
    real_h = tape_hours(args.real_tape)

    jobs = [(vi, -1, path, args.real_tape) for vi, (_, path) in enumerate(variants)]
    jobs += [(vi, k, path, t) for vi, (_, path) in enumerate(variants) for k, t in enumerate(tapes)]
    res: dict = {}
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(backtest, path, t, tmp / f'out_{vi}_{k}'): (vi, k) for vi, k, path, t in jobs}
        for f, key in futs.items():
            res[key] = f.result()

    rows = []
    for vi, (spec, _) in enumerate(variants):
        real = res[(vi, -1)]
        syn = [res[(vi, k)]['pnl'] / (args.minutes / 60) for k in range(args.sessions)]
        rows.append({'variant': spec, 'real_per_h': real['pnl'] / real_h, 'real_penalty': real['breach_penalty'],
                     'synth_per_h': float(np.mean(syn)), 'synth_sd': float(np.std(syn)),
                     'synth_penalty': float(np.mean([res[(vi, k)]['breach_penalty'] for k in range(args.sessions)]))})
    real = np.array([r['real_per_h'] for r in rows])
    syn = np.array([r['synth_per_h'] for r in rows])
    pairs = list(itertools.combinations(range(len(rows)), 2))
    agree = np.mean([np.sign(real[a] - real[b]) == np.sign(syn[a] - syn[b]) for a, b in pairs]) if pairs else float('nan')
    best_syn = int(np.argmax(syn))
    real_rank = int((real > real[best_syn]).sum()) + 1
    summary = {'spearman': spearman(real, syn) if len(rows) > 2 else float('nan'), 'pair_agreement': float(agree),
               'synth_best': rows[best_syn]['variant'], 'synth_best_real_rank': real_rank, 'n_variants': len(rows),
               'sessions': args.sessions, 'minutes': args.minutes}
    out = {'name': args.name, 'real_tape': str(args.real_tape), 'summary': summary, 'rows': rows}
    (PARAM_DIR / f'{args.name}_transfer.json').write_text(json.dumps(out, indent=1))
    md = [f'# Transfer check: {args.name} vs {args.real_tape.name}', '',
          f'Spearman {summary["spearman"]:+.2f}, pairs ordered the same way {agree:.0%}, the generated favourite '
          f'(`{summary["synth_best"]}`) ranks {real_rank} of {len(rows)} on the real tape. '
          f'{args.sessions} generated sessions of {args.minutes:g} min.', '',
          '| variant | real /h | synth /h (sd) | real penalty | synth penalty |', '|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: -r['real_per_h']):
        md.append(f'| `{r["variant"]}` | {r["real_per_h"]:,.0f} | {r["synth_per_h"]:,.0f} ({r["synth_sd"]:,.0f}) | '
                  f'{r["real_penalty"]:,.0f} | {r["synth_penalty"]:,.0f} |')
    (PARAM_DIR / f'{args.name}_transfer.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
