"""CLI entry point: replay a backtest against a strategy file and write the results."""

from __future__ import annotations

import argparse
import csv
import getpass
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.strategy import BaseStrategy

ORDER_ACCEPT_DELAY_MS = 5
MAX_POSITION = 100
MAX_POTENTIAL_POSITION = 1000
POSITION_BREACH_PENALTY_PER_SECOND = 10_000.0


def load_strategy_from_file(file_path: str | Path) -> BaseStrategy:
    """Import a Python file and return an instance of the `BaseStrategy` subclass it defines.

    `dir()` is sorted, so with several subclasses the alphabetically first one is used.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f'Strategy file not found: {path}')

    spec = importlib.util.spec_from_file_location('user_strategy_module', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to load module spec from {path}')

    module = importlib.util.module_from_spec(spec)
    sys.modules['user_strategy_module'] = module
    spec.loader.exec_module(module)

    for name in dir(module):
        candidate = getattr(module, name)
        if isinstance(candidate, type) and issubclass(candidate, BaseStrategy) and candidate is not BaseStrategy:
            return candidate()

    raise ValueError(
        f'No BaseStrategy subclass found in {path}. Please define a class that inherits from BaseStrategy.'
    )


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description='Run exchange simulation backtest')
    parser.add_argument(
        '--backtest',
        type=Path,
        required=True,
        help='Path to backtest CSV',
    )
    parser.add_argument(
        '--strategy',
        type=Path,
        required=True,
        help='Path to Python file containing your strategy (must define a BaseStrategy subclass)',
    )
    parser.add_argument(
        '--public-depth',
        type=int,
        default=20,
        help='Number of price ticks per side visible in public feed, counted from the best price',
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        required=True,
        help="Directory to write all logs (created if it doesn't exist)",
    )
    parser.add_argument(
        '--run-log-level',
        choices=['info', 'debug'],
        default='info',
        help='Run log verbosity; debug adds a full book snapshot after every event',
    )
    parser.add_argument(
        '--no-logs',
        action='store_true',
        help='Write only metadata.json, result.json and pnl.csv to --log-dir, skipping the run and CSV logs',
    )
    parser.add_argument(
        '--progress',
        action='store_true',
        help='Show replay progress (a bar on a terminal, JSON lines otherwise)',
    )
    parser.add_argument(
        '--progress-fd',
        type=int,
        default=2,
        help='File descriptor to write --progress to (default 2, stderr)',
    )
    return parser.parse_args()


def main() -> None:
    """Run the replay and write metadata.json, result.json and pnl.csv (plus the logs) to --log-dir."""
    args = parse_args()
    log_dir: Path = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    write_logs = not args.no_logs
    config = SimulationConfig(
        public_depth=args.public_depth,
        run_log_path=log_dir / 'run.jsonl' if write_logs else None,
        book_delta_log_path=log_dir / 'book_deltas.csv' if write_logs else None,
        event_log_path=log_dir / 'events.csv' if write_logs else None,
        trade_log_path=log_dir / 'trades.csv' if write_logs else None,
        min_log_level=args.run_log_level.upper(),
        max_position=MAX_POSITION,
        max_potential_position=MAX_POTENTIAL_POSITION,
        position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
        order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS,
        show_progress=args.progress,
        progress_fd=args.progress_fd,
    )
    strategy = load_strategy_from_file(args.strategy)
    simulator = ExchangeSimulator(strategy=strategy, config=config)
    result = simulator.run(args.backtest)

    metadata = {
        'trader': getpass.getuser(),
        'trader_class': type(strategy).__name__,
        'data_file': str(args.backtest.resolve()),
        'latency': config.order_accept_delay_ms / 1000,
        'position_limit': config.max_position,
        'max_potential_position': config.max_potential_position,
        'position_breach_penalty_per_second': config.position_breach_penalty_per_second,
        'run_timestamp': datetime.now().isoformat(),
    }
    with open(log_dir / 'metadata.json', 'w') as file:
        json.dump(metadata, file, indent=2)
    with open(log_dir / 'result.json', 'w') as file:
        json.dump(result.model_dump(mode='json'), file, indent=2)
    points = simulator.pnl_curve(result.pnl)
    start_us = points[0][0]
    with open(log_dir / 'pnl.csv', 'w', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(['elapsed_s', 'pnl'])
        for timestamp_us, pnl in points:
            writer.writerow([(timestamp_us - start_us) / 1_000_000, pnl])

    print('Simulation complete')
    print(f'fills={result.fills}')
    print(f'traded_volume={result.traded_volume}')
    print(f'inventory={result.inventory}')
    print(f'cash={result.cash:.2f}')
    print(f'mark_price={result.mark_price:.2f}')
    print(f'breach_penalty={result.breach_penalty:.2f}')
    print(f'pnl={result.pnl:.2f}')
    print(f'pnl_se={"none" if result.pnl_se is None else f"{result.pnl_se:.2f}"}')
    print(f'abs_inventory_q50={result.abs_inventory_q50}')
    print(f'abs_inventory_q68={result.abs_inventory_q68}')
    print(f'abs_inventory_q95={result.abs_inventory_q95}')
    print(f'abs_inventory_q997={result.abs_inventory_q997}')
    print(f'log_dir={log_dir}')


if __name__ == '__main__':
    main()
