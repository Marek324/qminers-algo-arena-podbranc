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
- Jumps (round2): S jumps at Poisson times (jump_rate per second) by a size drawn from the real tape's detected jump
  sizes, up with probability jump_up. The book follows within about half a second only because every resting order
  expires by its timer; no special mechanism. After a jump of size J, S gets extra variance exc_var * |J| *
  exp(-t / exc_tau) ticks^2 per second, and extra IOCs arrive at ioc_boost * exp(-t / ioc_tau) per second.
  round1 and sample_v2 have no jumps (all four default to 0).
- Regimes (round2): a hidden calm/storm state switches as a continuous-time Markov chain (mean durations calm_dur_s
  and storm_dur_s). sigma_s, jump_rate and ioc_rate are the calm values; in a storm they are sigma_storm,
  jump_rate_storm and ioc_rate_storm. storm_dur_s = 0 turns regimes off (round1, sample_v2).
- Small jumps: the detected jumps (>= jump_detect ticks) follow a power law; with jump_min > 0 it is extended below the
  detection threshold, adding jumps of jump_min..jump_detect ticks (Pareto, exponent jump_alpha) at the rate the same law
  implies. They have no excitation (it was only measured after detected jumps).

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
    jump_rate: float = 0.0  # S jumps per second
    jump_sizes_file: str = ''  # |jump| sizes in ticks, sampled with replacement
    jump_up: float = 0.5  # probability a jump is upward
    exc_var: float = 0.0  # extra S variance after a jump, ticks^2/s per tick of |jump|, decaying over exc_tau
    exc_tau: float = 1.0
    ioc_boost: float = 0.0  # extra IOCs per second right after a jump, decaying over ioc_tau
    ioc_tau: float = 2.0
    jump_min: float = 0.0  # extend the jump-size power law down to this many ticks (0: off)
    jump_alpha: float = 2.5  # Pareto exponent of |jump|
    jump_detect: float = 5.0  # the threshold the detected jumps (jump_sizes_file, jump_rate) were found with
    calm_dur_s: float = 0.0  # mean calm stretch (s)
    storm_dur_s: float = 0.0  # mean storm stretch (s); 0 = no regimes
    sigma_storm: float = 0.0
    jump_rate_storm: float = 0.0
    ioc_rate_storm: float = 0.0
    life_storm_s: float = 0.0  # mean order life in a storm (0: life_mean_s everywhere)
    clamp: bool = False  # limit orders never cross the book
    cross_mode: str = 'clamp'  # with clamp: 'clamp' moves a crossing limit to one tick inside the opposite best, 'drop' skips it
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
    t_lim = np.sort(rng.uniform(0, T, n_lim))
    # regime path: alternating calm / storm stretches (one calm stretch covering everything without regimes)
    if p.storm_dur_s > 0:
        bounds, states, t, z = [0.0], [], 0.0, int(rng.random() < p.storm_dur_s / (p.calm_dur_s + p.storm_dur_s))
        while t < T:
            states.append(z)
            t += rng.exponential(p.storm_dur_s if z else p.calm_dur_s)
            bounds.append(min(t, T))
            z = 1 - z
    else:
        bounds, states = [0.0, T], [0]
    bounds = np.array(bounds)
    states = np.array(states)

    def stream(rate_calm: float, rate_storm: float) -> np.ndarray:
        """Poisson arrival times with the rate of the regime in force."""
        out = []
        for a, b, z in zip(bounds[:-1], bounds[1:], states, strict=True):
            r = rate_storm if z else rate_calm
            if r > 0 and b > a:
                out.append(rng.uniform(a, b, rng.poisson(r * (b - a))))
        return np.sort(np.concatenate(out)) if out else np.zeros(0)

    t_ioc = stream(p.ioc_rate, p.ioc_rate_storm)
    n_ioc = len(t_ioc)
    # jumps in S, and the burst of extra IOCs after each one
    t_jump = stream(p.jump_rate, p.jump_rate_storm) if (p.jump_rate > 0 or p.jump_rate_storm > 0) else np.zeros(0)
    if len(t_jump):
        sizes = np.load(PARAM_DIR / p.jump_sizes_file) if p.jump_sizes_file else np.full(1, 10.0)
        j_size = rng.choice(sizes, len(t_jump)) * np.where(rng.random(len(t_jump)) < p.jump_up, 1.0, -1.0)
        if p.ioc_boost > 0:
            extra = [tj + rng.exponential(p.ioc_tau, rng.poisson(p.ioc_boost * p.ioc_tau)) for tj in t_jump]
            extra = np.concatenate(extra)
            t_ioc = np.r_[t_ioc, extra[extra < T]]
            n_ioc = len(t_ioc)
    else:
        j_size = np.zeros(0)
    t_ioc = np.sort(t_ioc)
    if p.jump_rate > 0 and 0 < p.jump_min < p.jump_detect and p.storm_dur_s == 0:
        r = (p.jump_min / p.jump_detect) ** p.jump_alpha
        n_small = rng.poisson(p.jump_rate * (1 / r - 1) * T)
        u = rng.random(n_small)
        small = p.jump_min * (1 - u * (1 - r)) ** (-1 / p.jump_alpha)  # Pareto truncated to [jump_min, jump_detect)
        t_small = rng.uniform(0, T, n_small)
        big_t, big_s = t_jump, j_size
        t_jump = np.r_[t_jump, t_small]
        j_size = np.r_[j_size, small * np.where(rng.random(n_small) < p.jump_up, 1.0, -1.0)]
        o = np.argsort(t_jump)
        t_jump, j_size = t_jump[o], j_size[o]
    else:
        big_t, big_s = t_jump, j_size
    n_ioc = len(t_ioc)
    ioc_side = rng.integers(0, 2, n_ioc)
    ioc_qty = rng.choice(ioc_sizes, n_ioc)
    # S at every insert time: one random walk sampled at the merged insert times, plus each IOC's permanent push,
    # which takes effect right after that IOC (the IOC itself is priced off the book, not S)
    t_all = np.concatenate([t_lim, t_ioc])
    order = np.argsort(t_all, kind='stable')
    ts_sorted = t_all[order]
    sig = p.sigma_s
    if p.storm_dur_s > 0:
        z_ev = states[np.clip(np.searchsorted(bounds, ts_sorted, 'right') - 1, 0, len(states) - 1)]
        sig = np.where(z_ev == 1, p.sigma_storm, p.sigma_s)
    var = sig**2 * np.diff(np.r_[0.0, ts_sorted])
    if len(t_jump) and p.exc_var > 0:
        # extra variance per event interval: integral of exc_var * |J| * exp(-(u - t_j) / tau) over the interval
        tau = p.exc_tau
        prev = np.r_[0.0, ts_sorted[:-1]]
        for tj, js in zip(big_t, big_s, strict=True):
            a, b = np.searchsorted(ts_sorted, [tj, tj + 10 * tau], 'right')
            if a >= len(ts_sorted):
                continue
            lo = np.maximum(prev[a:b + 1], tj) - tj
            hi = ts_sorted[a:b + 1] - tj
            var[a:b + 1] += p.exc_var * abs(js) * tau * (np.exp(-lo / tau) - np.exp(-hi / tau))
    dS = rng.standard_normal(len(ts_sorted)) * np.sqrt(var)
    push = np.zeros(len(t_all))
    push[n_lim:] = p.impact_k * np.where(ioc_side == 0, 1.0, -1.0) * np.power(ioc_qty.astype(float), p.impact_beta)
    push_sorted = push[order]
    S_sorted = p.s0 + np.cumsum(dS) + np.cumsum(push_sorted) - push_sorted
    if len(t_jump):
        S_sorted += np.r_[0.0, np.cumsum(j_size)][np.searchsorted(t_jump, ts_sorted, 'right')]
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
    if p.storm_dur_s > 0 and p.life_storm_s > 0:
        z_lim = states[np.clip(np.searchsorted(bounds, t_lim, 'right') - 1, 0, len(states) - 1)]
        life = np.where(z_lim == 1, rng.exponential(p.life_storm_s, n_lim), life)
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
            s, px, q = int(lim_side[i]), int(lim_px[i]), int(lim_qty[i])
            if p.clamp:
                crossing = (s == 0 and book.ap and px >= book.ap[0]) or (s == 1 and book.bp and px <= book.bp[-1])
                if crossing and p.cross_mode == 'drop':
                    i += 1
                    continue
                if s == 0 and book.ap and px >= book.ap[0]:
                    px = book.ap[0] - 1
                elif s == 1 and book.bp and px <= book.bp[-1]:
                    px = book.bp[-1] + 1
            oid = next_id
            next_id += 1
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
