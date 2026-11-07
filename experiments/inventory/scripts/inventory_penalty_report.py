"""Report inventory-breach costs from backtest ``result.json`` files as CSV.

Each positional DIR may be a simulator log directory or a root recursively
containing log directories. Labels are the result directories, and rows are
sorted by directory. A single --baseline result is reused for every row.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


RESULT_FIELDS = (
    "pnl",
    "breach_penalty",
    "cash",
    "inventory",
    "traded_volume",
    "fills",
    "abs_inventory_q50",
    "abs_inventory_q68",
    "abs_inventory_q95",
    "abs_inventory_q997",
)
OUTPUT_FIELDS = (
    "label",
    "pnl",
    "breach_penalty",
    "gross_pnl",
    "breach_seconds",
    "cash",
    "inventory",
    "traded_volume",
    "fills",
    "abs_inventory_q50",
    "abs_inventory_q68",
    "abs_inventory_q95",
    "abs_inventory_q997",
)
DELTA_FIELDS = ("delta_net", "delta_gross", "gross_covers_penalty")


class ReportError(Exception):
    """An input directory or result file cannot be reported."""


def read_result(result_path):
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {result_path}: {exc}") from exc

    if not isinstance(result, dict):
        raise ReportError(f"{result_path}: expected a JSON object")
    missing = [field for field in RESULT_FIELDS if field not in result]
    if missing:
        raise ReportError(f"{result_path}: missing fields: {', '.join(missing)}")
    invalid = [
        field
        for field in RESULT_FIELDS
        if isinstance(result[field], bool)
        or not isinstance(result[field], (int, float))
        or (isinstance(result[field], float) and not math.isfinite(result[field]))
    ]
    if invalid:
        raise ReportError(
            f"{result_path}: fields must be finite numbers: {', '.join(invalid)}"
        )
    return result


def discover_results(root):
    if not root.is_dir():
        raise ReportError(f"not a directory: {root}")

    direct_result = root / "result.json"
    if direct_result.is_file():
        directories = [root]
    else:
        try:
            directories = sorted(
                (path.parent for path in root.rglob("result.json") if path.is_file()),
                key=str,
            )
        except OSError as exc:
            raise ReportError(f"cannot search {root}: {exc}") from exc

    if not directories:
        raise ReportError(f"no result.json files found under {root}")
    return [(directory, read_result(directory / "result.json")) for directory in directories]


def make_row(directory, result, baseline=None):
    row = {"label": str(directory)}
    row.update({field: result[field] for field in RESULT_FIELDS})
    row["gross_pnl"] = result["pnl"] + result["breach_penalty"]
    row["breach_seconds"] = result["breach_penalty"] / 10000

    if baseline is not None:
        baseline_gross = baseline["pnl"] + baseline["breach_penalty"]
        row["delta_net"] = result["pnl"] - baseline["pnl"]
        row["delta_gross"] = row["gross_pnl"] - baseline_gross
        row["gross_covers_penalty"] = (
            row["delta_gross"] > result["breach_penalty"]
        )
    return row


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Print a CSV inventory-penalty report for backtest log directories "
            "or roots recursively containing result.json files."
        )
    )
    parser.add_argument(
        "directories",
        metavar="DIR",
        type=Path,
        nargs="+",
        help="backtest log directory or root containing result.json files",
    )
    parser.add_argument(
        "--baseline",
        metavar="DIR",
        type=Path,
        help=(
            "one baseline log directory (or root containing exactly one "
            "result.json) used for every delta column"
        ),
    )
    args = parser.parse_args()

    try:
        runs = [
            run
            for directory in args.directories
            for run in discover_results(directory)
        ]
        runs.sort(key=lambda run: str(run[0]))
        baseline = None
        if args.baseline is not None:
            baseline_runs = discover_results(args.baseline)
            if len(baseline_runs) != 1:
                raise ReportError(
                    f"--baseline must contain exactly one result.json; "
                    f"found {len(baseline_runs)} under {args.baseline}"
                )
            baseline = baseline_runs[0][1]
    except ReportError as exc:
        parser.error(str(exc))

    fields = OUTPUT_FIELDS + (DELTA_FIELDS if baseline is not None else ())
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for directory, result in runs:
        writer.writerow(make_row(directory, result, baseline))


if __name__ == "__main__":
    main()
