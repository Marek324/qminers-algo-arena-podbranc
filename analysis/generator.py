"""Market model and session simulator: our fitted market model for sample_v2 and round1 (docs/DATA_ANALYSIS.md).

Model:
- A hidden fair price S follows a Gaussian random walk (sigma_s ticks per sqrt second).
- Limit orders arrive as a Poisson stream (limit_rate per second), side a fair coin. A buy is priced floor(S - X), a sell
  ceil(S + X), clamped so it never crosses the book (the generator sees the book after matching). X follows x_dist:
  'normal' (x_mean, x_std, clipped at 0), 'lognormal' (log-mean x_a, log-sd x_b) or 'gamma' (shape x_a, scale x_b,
  shift x_loc). Sizes are drawn from the real tape's limit sizes.
- Every limit order gets an exponential cancel timer (mean life_mean_s). round1 writes the cancel row even when the order
  already filled; sample_v2 only cancels orders still resting (cancel_filled).
- IOCs arrive as a Poisson stream (ioc_rate), side a fair coin, sizes drawn from the real tape's IOC sizes. Priced at
  round(S + 1) for a buy and round(S - 1) for a sell (ioc_mode 'S', found in the tape analysis), or relative to the
  opposite touch with the measured offsets (ioc_mode 'touch': 0 = at the touch, 1 = one tick through, -1 = one short).
- Every IOC permanently pushes S in its direction by impact_k * size ** impact_beta ticks (round1: beta about 0.65,
  between square-root and linear; sample_v2 has no push). The IOC side itself is independent of S.

The book is kept while generating (IOC prices and the sample_v2 cancel rule need it), using the dashboard's matching
engine, which mirrors simulator/simulator.py. Output is a tape in the simulator's CSV format plus the S path.
"""

from __future__ import annotations

import heapq
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARAM_DIR = ROOT / 'logs' / 'generator'
sys.path.insert(0, str(ROOT / 'dashboard'))
from engine import Book  # noqa: E402


@dataclass
class GenParams:
    name: str
    duration_s: float
    limit_rate: float
    ioc_rate: float
    sigma_s: float  # ticks per sqrt(second)
    x_mean: float
    x_std: float
    life_mean_s: float
    cancel_filled: bool
    ioc_offset: dict = field(default_factory=dict)  # {ticks through the opposite touch: probability}
    limit_sizes_file: str = ''
    ioc_sizes_file: str = ''
    impact_k: float = 0.0  # permanent S move per IOC: impact_k * size ** impact_beta ticks
    impact_beta: float = 0.5
    x_dist: str = 'normal'  # 'normal' | 'lognormal' | 'gamma'
    x_a: float = 0.0
    x_b: float = 0.0
    x_loc: float = 0.0
    ioc_mode: str = 'touch'  # 'touch' | 'S'
    clamp: bool = False  # limit orders never cross the book
    s0: float = 10000.0

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path) -> GenParams:
        d = json.loads(Path(path).read_text())
        d['ioc_offset'] = {int(k): v for k, v in d['ioc_offset'].items()}
        return cls(**d)


def generate(p: GenParams, seed: int = 0, duration_s: float | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """(tape rows, S sample times in seconds, S values) for one session."""
    rng = np.random.default_rng(seed)
    T = duration_s or p.duration_s
    lim_sizes = np.load(PARAM_DIR / p.limit_sizes_file) if p.limit_sizes_file else np.full(1, 30)
    ioc_sizes = np.load(PARAM_DIR / p.ioc_sizes_file) if p.ioc_sizes_file else np.full(1, 40)

    n_lim = rng.poisson(p.limit_rate * T)
    n_ioc = rng.poisson(p.ioc_rate * T)
    t_lim = np.sort(rng.uniform(0, T, n_lim))
    t_ioc = np.sort(rng.uniform(0, T, n_ioc))
    ioc_side = rng.integers(0, 2, n_ioc)
    ioc_qty = rng.choice(ioc_sizes, n_ioc)
    # S at every insert time: one random walk sampled at the merged insert times, plus each IOC's permanent push,
    # which takes effect right after that IOC (the IOC itself is priced off the book, not S)
    t_all = np.concatenate([t_lim, t_ioc])
    order = np.argsort(t_all, kind='stable')
    ts_sorted = t_all[order]
    dS = rng.standard_normal(len(ts_sorted)) * p.sigma_s * np.sqrt(np.diff(np.r_[0.0, ts_sorted]))
    push = np.zeros(len(t_all))
    push[n_lim:] = p.impact_k * np.where(ioc_side == 0, 1.0, -1.0) * np.power(ioc_qty.astype(float), p.impact_beta)
    push_sorted = push[order]
    S_sorted = p.s0 + np.cumsum(dS) + np.cumsum(push_sorted) - push_sorted
    S_all = np.empty_like(S_sorted)
    S_all[order] = S_sorted
    S_lim, S_ioc = S_all[:n_lim], S_all[n_lim:]

    lim_side = rng.integers(0, 2, n_lim)  # 0 buy, 1 sell
    if p.x_dist == 'lognormal':
        X = rng.lognormal(p.x_a, p.x_b, n_lim)
    elif p.x_dist == 'gamma':
        X = p.x_loc + rng.gamma(p.x_a, p.x_b, n_lim)
    else:
        X = np.clip(rng.normal(p.x_mean, p.x_std, n_lim), 0, None)
    lim_px = np.where(lim_side == 0, np.floor(S_lim - X), np.ceil(S_lim + X)).astype(np.int64)
    lim_qty = rng.choice(lim_sizes, n_lim)
    life = rng.exponential(p.life_mean_s, n_lim)
    offs = np.array(list(p.ioc_offset.keys()) or [0])
    probs = np.array(list(p.ioc_offset.values()) or [1.0])
    ioc_off = rng.choice(offs, n_ioc, p=probs / probs.sum())

    book = Book()
    rows_t: list[float] = []
    rows_op: list[int] = []  # 0 insert G, 1 insert F, 2 cancel
    rows_id: list[int] = []
    rows_side: list[int] = []
    rows_q: list[int] = []
    rows_px: list[int] = []
    cancels: list[tuple[float, int]] = []
    next_id = 1_000_001
    i = j = 0
    tl, ti = t_lim.tolist(), t_ioc.tolist()
    while i < n_lim or j < n_ioc or cancels:
        nxt_l = tl[i] if i < n_lim else np.inf
        nxt_i = ti[j] if j < n_ioc else np.inf
        nxt_c = cancels[0][0] if cancels else np.inf
        if nxt_c <= nxt_l and nxt_c <= nxt_i:
            tc, oid = heapq.heappop(cancels)
            if tc > T:
                continue
            removed = book.cancel(oid)
            if removed or p.cancel_filled:
                rows_t.append(tc); rows_op.append(2); rows_id.append(oid); rows_side.append(-1); rows_q.append(0); rows_px.append(0)  # noqa: E702
        elif nxt_l <= nxt_i:
            oid = next_id
            next_id += 1
            s, px, q = int(lim_side[i]), int(lim_px[i]), int(lim_qty[i])
            if p.clamp:
                if s == 0 and book.ap and px >= book.ap[0]:
                    px = book.ap[0] - 1
                elif s == 1 and book.bp and px <= book.bp[-1]:
                    px = book.bp[-1] + 1
            book.insert(oid, s, px, q, 0)
            rows_t.append(nxt_l); rows_op.append(0); rows_id.append(oid); rows_side.append(s); rows_q.append(q); rows_px.append(px)  # noqa: E702
            heapq.heappush(cancels, (nxt_l + float(life[i]), oid))
            i += 1
        else:
            s, q = int(ioc_side[j]), int(ioc_qty[j])
            if p.ioc_mode == 'S':
                px = int(round(S_ioc[j] + 1)) if s == 0 else int(round(S_ioc[j] - 1))
            elif s == 0:
                ref = book.ap[0] if book.ap else (book.bp[-1] + 1 if book.bp else int(round(S_ioc[j])))
                px = ref + int(ioc_off[j])
            else:
                ref = book.bp[-1] if book.bp else (book.ap[0] - 1 if book.ap else int(round(S_ioc[j])))
                px = ref - int(ioc_off[j])
            oid = next_id
            next_id += 1
            book.insert(oid, s, px, q, 1)
            rows_t.append(nxt_i); rows_op.append(1); rows_id.append(oid); rows_side.append(s); rows_q.append(q); rows_px.append(px)  # noqa: E702
            j += 1

    op = np.array(rows_op)
    side = np.array(rows_side)
    df = pd.DataFrame({
        'Time': np.array(rows_t),
        'Operation': np.where(op == 2, 'Cancel', 'Insert'),
        'OrderId': np.array(rows_id),
        'Side': np.where(side == 0, 'B', np.where(side == 1, 'A', '')),
        'Volume': np.where(op == 2, None, np.array(rows_q, dtype=object)),
        'Price': np.where(op == 2, None, np.array(rows_px, dtype=object)),
        'Lifespan': np.where(op == 2, '', np.where(op == 1, 'F', 'G')),
    })
    return df, ts_sorted, S_sorted


def write_tape(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out['Time'] = out['Time'].map(lambda v: f'{v:.6f}')
    out.to_csv(path, index=False)
    return path
