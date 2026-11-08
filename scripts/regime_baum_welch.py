"""Offline Baum-Welch fit of the regime HMM our strategies run online, on one tape.

Observation = n for each new market limit order: half-ticks between the order and the mid beyond the first half tick
(the same quantity `_observe` feeds the online HMM, uncapped). Two states with geometric n (means nq < nv); the
switch probabilities are per order. Prints the fitted means, switch probabilities, storm share, regime durations in
seconds (from the smoothed posterior), and the same fit on each half of the tape (stationarity check). The fitted
values are the starting points H_N_Q0, H_N_V0, H_PQV, H_PVQ for a new round.

usage: .venv/bin/python scripts/regime_baum_welch.py data/round4_public.csv [--iters 20]
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np

from simulator.main import (MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS,
                            POSITION_BREACH_PENALTY_PER_SECOND)
from simulator.simulator import ExchangeSimulator, SimulationConfig

BASE = Path('strategies/round3/hmm_deep_clear_ks.py')


def record(tape: Path):
    spec = importlib.util.spec_from_file_location('base_mod', BASE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base = mod.HmmDeepClearKsStrategy

    class Rec(base):
        MAX_POS = 0        # observe only, never trade
        MIN_NEW = 10 ** 9

        def __init__(self):
            super().__init__()
            self.obs_t = []
            self.obs_n = []
            self.obs_pv = []

        def _observe(self, sign, price, depth, now):
            self.obs_t.append(now)
            self.obs_n.append(max(0, round(2 * (sign * (price - self.mid) - 0.5))))
            super()._observe(sign, price, depth, now)
            self.obs_pv.append(self.pv)

    s = Rec()
    cfg = SimulationConfig(public_depth=20, max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION,
                           position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND,
                           order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
    ExchangeSimulator(strategy=s, config=cfg).run(tape)
    return np.array(s.obs_t) / 1e6, np.array(s.obs_n, float), np.array(s.obs_pv)


def geom_lik(n, m):
    r = m / (1.0 + m)
    return (1.0 - r) * r ** n


def baum_welch(n, iters=20, nq=1.0, nv=3.5, pqv=3e-5, pvq=3e-4):
    """Scaled forward-backward with scalar loops (two states, so plain floats beat numpy per step)."""
    T = len(n)
    nl = n.tolist()
    g = np.empty(T)
    ll = 0.0
    for _ in range(iters):
        lq = geom_lik(n, nq).tolist()
        lv = geom_lik(n, nv).tolist()
        aq_ = [0.0] * T
        av_ = [0.0] * T
        cs = [0.0] * T
        a0, a1 = 0.9 * lq[0], 0.1 * lv[0]
        for t in range(T):
            if t:
                p0, p1 = aq_[t - 1], av_[t - 1]
                a0 = (p0 * (1 - pqv) + p1 * pvq) * lq[t]
                a1 = (p0 * pqv + p1 * (1 - pvq)) * lv[t]
            c = a0 + a1
            cs[t] = c
            aq_[t] = a0 / c
            av_[t] = a1 / c
        b0 = b1 = 1.0
        xqv = xvq = 0.0
        gl = [0.0] * T
        for t in range(T - 1, -1, -1):
            q0, q1 = aq_[t] * b0, av_[t] * b1
            gl[t] = q1 / (q0 + q1)
            if t:
                e0, e1 = lq[t] * b0, lv[t] * b1
                c = cs[t]
                xqv += aq_[t - 1] * pqv * e1 / c
                xvq += av_[t - 1] * pvq * e0 / c
                b0, b1 = ((1 - pqv) * e0 + pqv * e1) / c, (pvq * e0 + (1 - pvq) * e1) / c
        g = np.array(gl)
        nq = float(((1 - g) * n).sum() / (1 - g).sum())
        nv = float((g * n).sum() / g.sum())
        pqv = float(xqv / (1 - g[:-1]).sum())
        pvq = float(xvq / g[:-1].sum())
        ll = float(np.log(np.array(cs)).sum())
    del nl
    return nq, nv, pqv, pvq, g, ll


def durations(t, g):
    s = g > 0.5
    edges = np.flatnonzero(np.diff(s.astype(int))) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(s)]
    out = {True: [], False: []}
    for a, b in zip(starts, ends):
        out[bool(s[a])].append(t[b - 1] - t[a])
    return {k: np.array(v) for k, v in out.items()}


def report(name, t, n, iters):
    nq, nv, pqv, pvq, g, ll = baum_welch(n, iters)
    d = durations(t, g)
    rate = len(n) / (t[-1] - t[0])
    print(f'\n[{name}] {len(n):,} orders, {rate:.1f}/s, loglik {ll:,.0f}')
    print(f'  H_N_Q0 {nq:.3f}  H_N_V0 {nv:.3f}  H_PQV {pqv:.2e}  H_PVQ {pvq:.2e}')
    print(f'  storm share (orders) {g.mean():.3f}, time {d[True].sum() / (t[-1] - t[0]):.3f}; '
          f'expected durations quiet {1 / pqv / rate:.1f} s, storm {1 / pvq / rate:.1f} s')
    for k, lab in ((False, 'quiet'), (True, 'storm')):
        v = d[k]
        if len(v):
            q = np.quantile(v, [0.1, 0.5, 0.9])
            cv = v.std() / v.mean() if v.mean() else 0
            print(f'  {lab}: {len(v)} spells, mean {v.mean():.1f} s, p10/50/90 {q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f} s, '
                  f'CV {cv:.2f} (1 = geometric)')
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tape', type=Path)
    ap.add_argument('--iters', type=int, default=20)
    a = ap.parse_args()
    t, n, pv = record(a.tape)
    print(f'n distribution: ' + ', '.join(f'{k}:{np.mean(n == k):.3f}' for k in range(8)) + f', >=8:{np.mean(n >= 8):.3f}')
    g = report('full tape', t, n, a.iters)
    print(f'  online ks HMM (round-3 constants): share pv>=0.3 {np.mean(pv >= 0.3):.3f}, '
          f'agreement with offline {np.mean((pv >= 0.3) == (g > 0.5)):.3f}')
    h = len(n) // 2
    report('first half', t[:h], n[:h], a.iters)
    report('second half', t[h:], n[h:], a.iters)


if __name__ == '__main__':
    main()
