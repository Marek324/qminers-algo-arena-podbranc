"""CLI: uv run analysis.py data/<file>.csv [more files] [options]."""

import argparse
from pathlib import Path

from .report import analyze, to_json, to_text


def main() -> None:
    p = argparse.ArgumentParser(prog='analysis.py', description='Analyze market tape CSVs into a structured report.')
    p.add_argument('tapes', nargs='+', type=Path, help='tape CSV files (.csv or .csv.gz)')
    p.add_argument('--out-dir', type=Path, default=Path('logs/analysis'), help='where <name>.json / <name>.txt go')
    p.add_argument('--format', choices=('json', 'text', 'both'), default='both')
    p.add_argument('--series-step', type=float, default=1.0, help='seconds per row of the fine series')
    p.add_argument('--bin-step', type=float, default=60.0, help='seconds per row of the bins table')
    p.add_argument('--compact', action='store_true', help='JSON without indentation')
    p.add_argument('-q', '--quiet', action='store_true')
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.tapes:
        if not args.quiet:
            print(f'{path}:', flush=True)
        report = analyze(path, args.series_step, args.bin_step, progress=not args.quiet)
        stem = path.name.removesuffix('.gz').removesuffix('.csv')
        if args.format in ('json', 'both'):
            out = args.out_dir / f'{stem}.json'
            out.write_text(to_json(report, None if args.compact else 1))
            print(f'  -> {out}')
        if args.format in ('text', 'both'):
            out = args.out_dir / f'{stem}.txt'
            out.write_text(to_text(report))
            print(f'  -> {out}')
        if not args.quiet:
            for h in report['highlights']:
                print(f'  * {h}')


if __name__ == '__main__':
    main()
