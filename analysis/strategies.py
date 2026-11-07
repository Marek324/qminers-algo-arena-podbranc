"""Loading a strategy file with its constants overridden in memory, and finding which constants can be tuned.

Shared by the dashboard (strategy replay) and scripts/robustness.py. The strategy file is never written: a fresh copy
of the module is loaded for every instance, and overrides are applied before the strategy object is created.
"""

from __future__ import annotations

import importlib.util
import re
import time
from dataclasses import dataclass
from pathlib import Path

from simulator.main import MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS, POSITION_BREACH_PENALTY_PER_SECOND
from simulator.simulator import SimulationConfig
from simulator.strategy import BaseStrategy


def scoring_config(**kw) -> SimulationConfig:
    """The simulator config `exchange-simulator` scores with (5 ms latency, position limits and penalty)."""
    return SimulationConfig(
        max_position=MAX_POSITION,
        max_potential_position=MAX_POTENTIAL_POSITION,
        position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
        order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS,
        **kw,
    )


@dataclass
class Param:
    name: str
    scope: str  # 'module' (NAME = v at top level) or 'class' (NAME = v inside the strategy class)
    default: bool | int | float
    comment: str


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f'dash_strategy_{abs(hash((str(path), time.time_ns())))}', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strategy_class(module, path: Path) -> type:
    """The class the simulator would use: first BaseStrategy subclass in sorted dir() order."""
    for name in dir(module):
        cand = getattr(module, name)
        if isinstance(cand, type) and issubclass(cand, BaseStrategy) and cand is not BaseStrategy:
            return cand
    raise ImportError(f'No BaseStrategy subclass found in {path}')


def _is_param(name: str, value) -> bool:
    return name.isupper() and not name.startswith('_') and isinstance(value, (bool, int, float))


def discover_params(path: Path) -> tuple[str, list[Param]]:
    """(strategy class name, tunable constants): UPPERCASE numbers/booleans at module level or in the class body."""
    module = _load_module(path)
    cls = _strategy_class(module, path)
    source = path.read_text()
    comments = {}
    for m in re.finditer(r'^\s*([A-Z_][A-Z0-9_]*)\s*(?::[^=]+)?=[^#\n]*#\s*(.+)$', source, flags=re.M):
        comments.setdefault(m.group(1), m.group(2).strip())
    defined = set(re.findall(r'^\s*([A-Z_][A-Z0-9_]*)\s*(?::[^=]+)?=', source, flags=re.M))
    params = []
    for name, value in vars(cls).items():
        if _is_param(name, value) and name in defined:
            params.append(Param(name, 'class', value, comments.get(name, '')))
    class_names = {p.name for p in params}
    for name, value in vars(module).items():
        if _is_param(name, value) and name in defined and name not in class_names:
            params.append(Param(name, 'module', value, comments.get(name, '')))
    return cls.__name__, params


def load_strategy(path: Path, overrides: dict[str, tuple[str, object]] | None = None) -> BaseStrategy:
    """A fresh instance of the strategy in `path`, with constants overridden in memory ({name: (scope, value)}).

    The file is never written. A fresh module is loaded every time, so overrides never leak between runs.
    """
    module = _load_module(path)
    cls = _strategy_class(module, path)
    for name, (scope, value) in (overrides or {}).items():
        if scope == 'class':
            setattr(cls, name, value)
        else:
            setattr(module, name, value)
    return cls()
