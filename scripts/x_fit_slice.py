"""scripts/x_fit_slice.py (used by scripts/fit_x.py)

HMM Maximum Likelihood Estimation of continuous fair-value diffusion sigma_S
and placement offset X distribution (Shifted Gamma and LogNormal).

Evaluates the exact 1D continuous-state forward grid filter likelihood on a
slice of round1 limit inserts. Designed for 1-node Slurm jobs
(36 CPUs per node, <= 25 min wall-clock limit).
"""
import os
import sys
import time
import math
import argparse
import json
from pathlib import Path
from multiprocessing import Pool
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats, optimize


def build_grid(prices: np.ndarray, res: float = 0.05, padding: float = 4.0) -> Tuple[np.ndarray, int]:
    p_min = np.min(prices) - padding
    p_max = np.max(prices) + padding
    s_grid = np.arange(p_min, p_max + res * 0.5, res)
    return s_grid, len(s_grid)


def filter_loglik_gamma(
    params: Tuple[float, float, float, float],
    ts: np.ndarray,
    sides: np.ndarray,
    prices: np.ndarray,
    s_grid: np.ndarray,
    unique_prices: np.ndarray,
    res: float = 0.05,
) -> float:
    sig, k, loc, th = params
    if sig <= 0.15 or k <= 0.1 or loc < -0.20 or th <= 0.01:
        return -1e12

    # Precompute observation likelihood tables for unique prices
    lik_b = {}
    lik_a = {}
    for p in unique_prices:
        u_b = s_grid - p
        lik_b[p] = np.maximum(
            1e-15,
            stats.gamma.cdf(u_b, a=k, loc=loc, scale=th) - stats.gamma.cdf(np.maximum(0.0, u_b - 1.0), a=k, loc=loc, scale=th)
        )
        u_a = p - s_grid
        lik_a[p] = np.maximum(
            1e-15,
            stats.gamma.cdf(u_a, a=k, loc=loc, scale=th) - stats.gamma.cdf(np.maximum(0.0, u_a - 1.0), a=k, loc=loc, scale=th)
        )

    n = len(ts)
    n_grid = len(s_grid)
    init_s = prices[0] + (1.5 if sides[0] else -1.5)
    probs = np.exp(-0.5 * ((s_grid - init_s) / 1.0) ** 2)
    sum_p = np.sum(probs)
    if sum_p > 0:
        probs /= sum_p

    total_loglik = 0.0

    for i in range(1, n):
        dt = ts[i] - ts[i - 1]
        if dt > 1e-6:
            std_dt = sig * math.sqrt(dt)
            k_cells = max(1, int(math.ceil(3.0 * std_dt / res)))
            k_grid = np.arange(-k_cells, k_cells + 1) * res
            kernel = np.exp(-0.5 * (k_grid / std_dt) ** 2)
            kernel /= np.sum(kernel)
            probs = np.convolve(probs, kernel, mode='same')

        p = prices[i]
        lik = lik_b[p] if sides[i] else lik_a[p]
        post = probs * lik
        sp = np.sum(post)
        if sp > 0:
            total_loglik += math.log(sp)
            probs = post / sp
        else:
            probs = np.ones(n_grid) / n_grid

    return total_loglik


def filter_loglik_lognorm(
    params: Tuple[float, float, float],
    ts: np.ndarray,
    sides: np.ndarray,
    prices: np.ndarray,
    s_grid: np.ndarray,
    unique_prices: np.ndarray,
    res: float = 0.05,
) -> float:
    sig, mu, sigma_ln = params
    if sig <= 0.15 or sigma_ln <= 0.02:
        return -1e12

    scale_ln = math.exp(mu)
    lik_b = {}
    lik_a = {}
    for p in unique_prices:
        u_b = s_grid - p
        lik_b[p] = np.maximum(
            1e-15,
            stats.lognorm.cdf(u_b, s=sigma_ln, scale=scale_ln) - stats.lognorm.cdf(np.maximum(1e-6, u_b - 1.0), s=sigma_ln, scale=scale_ln)
        )
        u_a = p - s_grid
        lik_a[p] = np.maximum(
            1e-15,
            stats.lognorm.cdf(u_a, s=sigma_ln, scale=scale_ln) - stats.lognorm.cdf(np.maximum(1e-6, u_a - 1.0), s=sigma_ln, scale=scale_ln)
        )

    n = len(ts)
    n_grid = len(s_grid)
    init_s = prices[0] + (1.5 if sides[0] else -1.5)
    probs = np.exp(-0.5 * ((s_grid - init_s) / 1.0) ** 2)
    sum_p = np.sum(probs)
    if sum_p > 0:
        probs /= sum_p

    total_loglik = 0.0

    for i in range(1, n):
        dt = ts[i] - ts[i - 1]
        if dt > 1e-6:
            std_dt = sig * math.sqrt(dt)
            k_cells = max(1, int(math.ceil(3.0 * std_dt / res)))
            k_grid = np.arange(-k_cells, k_cells + 1) * res
            kernel = np.exp(-0.5 * (k_grid / std_dt) ** 2)
            kernel /= np.sum(kernel)
            probs = np.convolve(probs, kernel, mode='same')

        p = prices[i]
        lik = lik_b[p] if sides[i] else lik_a[p]
        post = probs * lik
        sp = np.sum(post)
        if sp > 0:
            total_loglik += math.log(sp)
            probs = post / sp
        else:
            probs = np.ones(n_grid) / n_grid

    return total_loglik


_HMM_SHARED: Dict[str, Any] = {}


def _init_hmm_worker(ts: np.ndarray, sides: np.ndarray, prices: np.ndarray, s_grid: np.ndarray, unique_prices: np.ndarray, res: float):
    _HMM_SHARED['ts'] = ts
    _HMM_SHARED['sides'] = sides
    _HMM_SHARED['prices'] = prices
    _HMM_SHARED['s_grid'] = s_grid
    _HMM_SHARED['unique_prices'] = unique_prices
    _HMM_SHARED['res'] = res


def _eval_gamma_worker(cand: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float, float, float], float]:
    ll = filter_loglik_gamma(
        cand,
        _HMM_SHARED['ts'],
        _HMM_SHARED['sides'],
        _HMM_SHARED['prices'],
        _HMM_SHARED['s_grid'],
        _HMM_SHARED['unique_prices'],
        _HMM_SHARED['res'],
    )
    return cand, ll


def _eval_ln_worker(cand: Tuple[float, float, float]) -> Tuple[Tuple[float, float, float], float]:
    ll = filter_loglik_lognorm(
        cand,
        _HMM_SHARED['ts'],
        _HMM_SHARED['sides'],
        _HMM_SHARED['prices'],
        _HMM_SHARED['s_grid'],
        _HMM_SHARED['unique_prices'],
        _HMM_SHARED['res'],
    )
    return cand, ll


def numerical_hessian(fn, x0: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Compute numerical Hessian for asymptotic standard errors."""
    n = len(x0)
    H = np.zeros((n, n))
    f0 = fn(x0)
    h = np.maximum(eps, np.abs(x0) * eps)

    for i in range(n):
        x_plus = np.copy(x0)
        x_plus[i] += h[i]
        f_plus = fn(x_plus)

        x_minus = np.copy(x0)
        x_minus[i] -= h[i]
        f_minus = fn(x_minus)

        H[i, i] = (f_plus - 2 * f0 + f_minus) / (h[i] ** 2)

        for j in range(i + 1, n):
            x_pp = np.copy(x0)
            x_pp[i] += h[i]
            x_pp[j] += h[j]

            x_pm = np.copy(x0)
            x_pm[i] += h[i]
            x_pm[j] -= h[j]

            x_mp = np.copy(x0)
            x_mp[i] -= h[i]
            x_mp[j] += h[j]

            x_mm = np.copy(x0)
            x_mm[i] -= h[i]
            x_mm[j] -= h[j]

            d2f = (fn(x_pp) - fn(x_pm) - fn(x_mp) + fn(x_mm)) / (4.0 * h[i] * h[j])
            H[i, j] = d2f
            H[j, i] = d2f

    return H


def fit_slice(
    slice_idx: int,
    n_slices: int,
    tape_path: str,
    out_dir: str,
    subsample: int = 1,
    max_iter: int = 120,
    n_workers: int = 36,
) -> Dict[str, Any]:
    print(f"Loading tape from {tape_path}...")
    # Load only necessary columns
    df = pd.read_csv(tape_path, usecols=['Time', 'Operation', 'Side', 'Price', 'Lifespan'])
    mask = (df['Operation'] == 'Insert') & (df['Lifespan'] == 'G')
    limits = df[mask].reset_index(drop=True)

    n_total = len(limits)
    chunk_size = n_total // n_slices
    start_idx = slice_idx * chunk_size
    end_idx = n_total if slice_idx == n_slices - 1 else (slice_idx + 1) * chunk_size

    slice_df = limits.iloc[start_idx:end_idx]
    if subsample > 1:
        slice_df = slice_df.iloc[::subsample]

    ts = slice_df['Time'].to_numpy(dtype=float)
    sides = (slice_df['Side'] == 'B').to_numpy(dtype=bool)
    prices = slice_df['Price'].to_numpy(dtype=int)
    n_events = len(ts)

    t_start = float(ts[0])
    t_end = float(ts[-1])
    print(f"Slice {slice_idx}/{n_slices}: {n_events:,} events, span {t_start:.1f}s .. {t_end:.1f}s ({t_end - t_start:.1f}s)")

    s_grid, n_grid = build_grid(prices, res=0.05, padding=4.0)
    unique_prices = np.unique(prices)

    t0_fit = time.time()

    # -------------------------------------------------------------
    # 1. SHIFTED GAMMA MODEL
    # -------------------------------------------------------------
    print("\n--- Fitting Shifted Gamma Model ---")
    # Parallel grid search for initial starting points
    gamma_candidates = [
        (sig, k, loc, th)
        for sig in [0.75, 0.92, 1.10]
        for k in [1.3, 1.7, 2.2]
        for loc in [0.0, 0.05]
        for th in [0.25, 0.33, 0.45]
    ]

    # Evaluate grid
    if n_workers > 1:
        with Pool(min(n_workers, len(gamma_candidates)), initializer=_init_hmm_worker, initargs=(ts, sides, prices, s_grid, unique_prices, 0.05)) as p:
            gamma_scores = p.map(_eval_gamma_worker, gamma_candidates)
    else:
        _init_hmm_worker(ts, sides, prices, s_grid, unique_prices, 0.05)
        gamma_scores = [_eval_gamma_worker(c) for c in gamma_candidates]

    best_cand_gamma, best_cand_ll = max(gamma_scores, key=lambda x: x[1])
    print(f"Best grid start Gamma: sig={best_cand_gamma[0]}, k={best_cand_gamma[1]}, loc={best_cand_gamma[2]}, th={best_cand_gamma[3]} (LL={best_cand_ll:.1f})")

    # Local optimization via Nelder-Mead
    def nll_gamma(p):
        ll = filter_loglik_gamma(p, ts, sides, prices, s_grid, unique_prices, res=0.05)
        return -ll

    opt_gamma = optimize.minimize(
        nll_gamma,
        best_cand_gamma,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-3, 'fatol': 1e-2, 'disp': False}
    )
    sig_g, k_g, loc_g, th_g = opt_gamma.x
    ll_gamma = -opt_gamma.fun
    aic_gamma = 2 * 4 - 2 * ll_gamma
    bic_gamma = 4 * math.log(n_events) - 2 * ll_gamma

    # Compute SE via numerical Hessian
    try:
        H_g = numerical_hessian(nll_gamma, opt_gamma.x, eps=1e-2)
        cov_g = np.linalg.pinv(H_g)
        se_gamma = np.sqrt(np.maximum(0.0, np.diag(cov_g)))
    except Exception as e:
        print(f"Hessian warning (Gamma): {e}")
        se_gamma = np.zeros(4)

    print(f"Shifted Gamma MLE: sigma_S={sig_g:.4f}±{se_gamma[0]:.4f}, k={k_g:.4f}±{se_gamma[1]:.4f}, loc={loc_g:.4f}±{se_gamma[2]:.4f}, theta={th_g:.4f}±{se_gamma[3]:.4f}")
    print(f"Gamma LogLik: {ll_gamma:.2f}, AIC: {aic_gamma:.2f}")

    # Checkpoint the gamma fit now, so a time-out during the lognormal fit still leaves a result.
    partial = {
        'slice_idx': slice_idx, 'n_slices': n_slices, 'n_events': n_events,
        't_start_s': t_start, 't_end_s': t_end, 'partial': True,
        'shifted_gamma': {
            'sigma_S': float(sig_g), 'sigma_S_se': float(se_gamma[0]),
            'k': float(k_g), 'k_se': float(se_gamma[1]),
            'loc': float(loc_g), 'loc_se': float(se_gamma[2]),
            'theta': float(th_g), 'theta_se': float(se_gamma[3]),
            'log_likelihood': float(ll_gamma), 'aic': float(aic_gamma), 'bic': float(bic_gamma),
        },
    }
    ckpt_path = Path(out_dir) / f"slice_{slice_idx:03d}_gamma.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt_path, 'w') as f:
        json.dump(partial, f, indent=2)

    # -------------------------------------------------------------
    # 2. LOGNORMAL MODEL
    # -------------------------------------------------------------
    print("\n--- Fitting LogNormal Model ---")
    lognorm_candidates = [
        (sig, mu, sigma_ln)
        for sig in [0.75, 0.92, 1.10]
        for mu in [0.15, 0.25, 0.35]
        for sigma_ln in [0.25, 0.35, 0.50]
    ]

    if n_workers > 1:
        with Pool(min(n_workers, len(lognorm_candidates)), initializer=_init_hmm_worker, initargs=(ts, sides, prices, s_grid, unique_prices, 0.05)) as p:
            ln_scores = p.map(_eval_ln_worker, lognorm_candidates)
    else:
        _init_hmm_worker(ts, sides, prices, s_grid, unique_prices, 0.05)
        ln_scores = [_eval_ln_worker(c) for c in lognorm_candidates]

    best_cand_ln, best_cand_ln_ll = max(ln_scores, key=lambda x: x[1])
    print(f"Best grid start LogNorm: sig={best_cand_ln[0]}, mu={best_cand_ln[1]}, sig_ln={best_cand_ln[2]} (LL={best_cand_ln_ll:.1f})")

    def nll_lognorm(p):
        ll = filter_loglik_lognorm(p, ts, sides, prices, s_grid, unique_prices, res=0.05)
        return -ll

    opt_ln = optimize.minimize(
        nll_lognorm,
        best_cand_ln,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-3, 'fatol': 1e-2, 'disp': False}
    )
    sig_ln, mu_ln, sig_ln_val = opt_ln.x
    ll_ln = -opt_ln.fun
    aic_ln = 2 * 3 - 2 * ll_ln
    bic_ln = 3 * math.log(n_events) - 2 * ll_ln

    try:
        H_ln = numerical_hessian(nll_lognorm, opt_ln.x, eps=1e-2)
        cov_ln = np.linalg.pinv(H_ln)
        se_ln = np.sqrt(np.maximum(0.0, np.diag(cov_ln)))
    except Exception as e:
        print(f"Hessian warning (LogNormal): {e}")
        se_ln = np.zeros(3)

    print(f"LogNormal MLE: sigma_S={sig_ln:.4f}±{se_ln[0]:.4f}, mu={mu_ln:.4f}±{se_ln[1]:.4f}, sigma={sig_ln_val:.4f}±{se_ln[2]:.4f}")
    print(f"LogNorm LogLik: {ll_ln:.2f}, AIC: {aic_ln:.2f}")

    elapsed = time.time() - t0_fit

    # Model comparison
    delta_aic = aic_ln - aic_gamma
    preferred = 'shifted_gamma' if delta_aic > 0 else 'lognormal'
    print(f"\nModel Comparison: ΔAIC = {delta_aic:.2f} in favor of {preferred.upper()} (Fit time: {elapsed:.1f}s)")

    result = {
        'slice_idx': slice_idx,
        'n_slices': n_slices,
        'tape': tape_path,
        'n_events': n_events,
        't_start_s': t_start,
        't_end_s': t_end,
        'duration_s': t_end - t_start,
        'fit_time_s': elapsed,
        'shifted_gamma': {
            'sigma_S': float(sig_g),
            'sigma_S_se': float(se_gamma[0]),
            'k': float(k_g),
            'k_se': float(se_gamma[1]),
            'loc': float(loc_g),
            'loc_se': float(se_gamma[2]),
            'theta': float(th_g),
            'theta_se': float(se_gamma[3]),
            'log_likelihood': float(ll_gamma),
            'aic': float(aic_gamma),
            'bic': float(bic_gamma),
        },
        'lognormal': {
            'sigma_S': float(sig_ln),
            'sigma_S_se': float(se_ln[0]),
            'mu': float(mu_ln),
            'mu_se': float(se_ln[1]),
            'sigma': float(sig_ln_val),
            'sigma_se': float(se_ln[2]),
            'log_likelihood': float(ll_ln),
            'aic': float(aic_ln),
            'bic': float(bic_ln),
        },
        'comparison': {
            'preferred': preferred,
            'delta_aic': float(delta_aic),
            'loglik_diff': float(ll_gamma - ll_ln),
        }
    }

    out_path = Path(out_dir) / f"slice_{slice_idx:03d}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Slice result saved to {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="HMM MLE for latent S_t and X distribution per slice.")
    parser.add_argument('--slice', type=int, default=0, help="Slice index (0 to N-1)")
    parser.add_argument('--n-slices', type=int, default=32, help="Total number of slices")
    parser.add_argument('--tape', type=str, default='data/round1_train.csv', help="Path to tape file")
    parser.add_argument('--out-dir', type=str, default='logs/generator/xfit', help="Output directory")
    parser.add_argument('--subsample', type=int, default=1, help="Subsample step (for smoke tests)")
    parser.add_argument('--max-iter', type=int, default=120, help="Nelder-Mead max iterations")
    parser.add_argument('--workers', type=int, default=36, help="Parallel workers for grid search")

    args = parser.parse_args()

    # Fallback to uncompressed round1.csv if round1.csv.gz not found
    tape_path = args.tape
    if not os.path.exists(tape_path):
        alt = tape_path.replace('.gz', '')
        if os.path.exists(alt):
            tape_path = alt

    fit_slice(
        slice_idx=args.slice,
        n_slices=args.n_slices,
        tape_path=tape_path,
        out_dir=args.out_dir,
        subsample=args.subsample,
        max_iter=args.max_iter,
        n_workers=args.workers,
    )


if __name__ == '__main__':
    main()
