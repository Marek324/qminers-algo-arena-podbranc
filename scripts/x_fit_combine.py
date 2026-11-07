"""scripts/x_fit_combine.py (used by scripts/fit_x.py)

Combine per-slice HMM MLE estimates across Slurm array tasks.
Computes:
1. Inverse-variance weighted parameter estimates and pooled SEs.
2. Cross-slice empirical mean and standard error of the mean.
3. Pooled total log-likelihood, AIC, and BIC.
4. Definitive model selection (Shifted Gamma vs LogNormal).
"""
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np


def combine_estimates(results_dir: str, out_file: str = "logs/generator/xfit_combined.json"):
    files = sorted(f for f in glob.glob(f"{results_dir}/slice_*.json") if not f.endswith("_gamma.json"))
    if not files:
        print(f"No slice files found in {results_dir}!")
        return

    slices = []
    for fp in files:
        with open(fp, 'r') as f:
            slices.append(json.load(f))

    n_slices = len(slices)
    tot_events = sum(s['n_events'] for s in slices)
    tot_duration = sum(s['duration_s'] for s in slices)

    print(f"Loaded {n_slices} completed slices ({tot_events:,} total limit orders, {tot_duration:.1f}s tape time)")

    def aggregate_model(model_key: str, param_names: List[str]):
        out = {}
        # Pooled LL
        tot_ll = sum(s[model_key]['log_likelihood'] for s in slices)
        tot_aic = sum(s[model_key]['aic'] for s in slices)
        tot_bic = sum(s[model_key]['bic'] for s in slices)
        out['pooled_log_likelihood'] = tot_ll
        out['pooled_aic'] = tot_aic
        out['pooled_bic'] = tot_bic

        for p in param_names:
            vals = np.array([s[model_key][p] for s in slices])
            ses = np.array([s[model_key].get(f"{p}_se", 0.0) for s in slices])

            # Inverse-variance weights
            valid_se = ses > 1e-8
            if np.any(valid_se):
                w = np.zeros_like(ses)
                w[valid_se] = 1.0 / (ses[valid_se] ** 2)
                inv_var_mean = float(np.sum(w * vals) / np.sum(w))
                pooled_se = float(1.0 / np.sqrt(np.sum(w)))
            else:
                inv_var_mean = float(np.mean(vals))
                pooled_se = float(np.std(vals) / np.sqrt(n_slices))

            emp_mean = float(np.mean(vals))
            emp_se = float(np.std(vals, ddof=1) / np.sqrt(n_slices)) if n_slices > 1 else float(ses[0])

            out[p] = {
                'inv_var_mean': inv_var_mean,
                'pooled_se': pooled_se,
                'empirical_mean': emp_mean,
                'empirical_se': emp_se,
                'slice_min': float(np.min(vals)),
                'slice_max': float(np.max(vals)),
            }
        return out

    gamma_agg = aggregate_model('shifted_gamma', ['sigma_S', 'k', 'loc', 'theta'])
    lognorm_agg = aggregate_model('lognormal', ['sigma_S', 'mu', 'sigma'])

    delta_aic = lognorm_agg['pooled_aic'] - gamma_agg['pooled_aic']
    preferred = 'shifted_gamma' if delta_aic > 0 else 'lognormal'

    combined = {
        'n_slices_combined': n_slices,
        'total_events': tot_events,
        'total_duration_s': tot_duration,
        'shifted_gamma': gamma_agg,
        'lognormal': lognorm_agg,
        'model_comparison': {
            'preferred': preferred,
            'delta_pooled_aic': float(delta_aic),
            'pooled_loglik_diff': float(gamma_agg['pooled_log_likelihood'] - lognorm_agg['pooled_log_likelihood']),
            'gamma_slice_wins': sum(1 for s in slices if s['comparison']['preferred'] == 'shifted_gamma'),
            'lognormal_slice_wins': sum(1 for s in slices if s['comparison']['preferred'] == 'lognormal'),
        }
    }

    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, 'w') as f:
        json.dump(combined, f, indent=2)

    print("\n" + "=" * 80)
    print("                     HMM MLE COMBINED RESULTS")
    print("=" * 80)
    print(f"Shifted Gamma Pooled LogLik: {gamma_agg['pooled_log_likelihood']:15.1f} | AIC: {gamma_agg['pooled_aic']:15.1f}")
    print(f"LogNormal     Pooled LogLik: {lognorm_agg['pooled_log_likelihood']:15.1f} | AIC: {lognorm_agg['pooled_aic']:15.1f}")
    print(f"Delta AIC: {delta_aic:15.1f} -> {preferred.upper()} decisively preferred")
    print("-" * 80)
    print("PARAMETER ESTIMATES (Shifted Gamma):")
    for p in ['sigma_S', 'k', 'loc', 'theta']:
        info = gamma_agg[p]
        print(f"  {p:<10}: {info['inv_var_mean']:8.4f} ± {info['pooled_se']:8.4f} (emp: {info['empirical_mean']:8.4f} ± {info['empirical_se']:8.4f})")
    print("-" * 80)
    print("PARAMETER ESTIMATES (LogNormal):")
    for p in ['sigma_S', 'mu', 'sigma']:
        info = lognorm_agg[p]
        print(f"  {p:<10}: {info['inv_var_mean']:8.4f} ± {info['pooled_se']:8.4f} (emp: {info['empirical_mean']:8.4f} ± {info['empirical_se']:8.4f})")
    print("=" * 80)
    print(f"Combined results written to {out_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Combine HMM MLE slice outputs.")
    parser.add_argument('--results-dir', type=str, default='logs/generator/xfit', help="Directory containing slice JSON files")
    parser.add_argument('--out', type=str, default='logs/generator/xfit_combined.json', help="Output JSON path")
    args = parser.parse_args()

    combine_estimates(args.results_dir, args.out)


if __name__ == '__main__':
    main()
