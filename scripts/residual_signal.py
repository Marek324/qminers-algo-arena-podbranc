"""Which features still predict the mid after the strategy's own fair value? (residual-signal scan)

Input: the `L,` lines an instrumented strategy prints every 100 ms (see the f-string in the logging variant):
t, bb, ba, bq, aq, s_hat, p_vol, imb, fx, inv, f1, f10, far_bid, far_ask, depth.
Target per horizon h: mid(t + h) - fair(t), fair rebuilt with the quiet / volatile coefficients given below.
Prints the correlation of each candidate feature with that residual, per regime and per half of the tape: a feature
is a usable signal only if both halves agree in sign and size (an overfitting guard on a single 6,000 s tape).

usage: .venv/bin/python scripts/residual_signal.py logs/siglog.csv [--k-s 0.75 --k-imb 0.15 --k-flow 1.0 ...]
"""
import argparse

import numpy as np
import pandas as pd

COLS = ['t', 'bb', 'ba', 'bq', 'aq', 's', 'pv', 'imb', 'fx', 'inv', 'f1', 'f10', 'farb', 'fara', 'depth']
HORIZONS = {'0.3s': 3, '1s': 10, '5s': 50, '30s': 300}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--k-s', type=float, default=0.75)
    ap.add_argument('--k-imb', type=float, default=0.15)
    ap.add_argument('--k-flow', type=float, default=1.0)
    ap.add_argument('--v-k-s', type=float, default=0.8)
    ap.add_argument('--v-k-flow', type=float, default=1.2)
    ap.add_argument('--p-on', type=float, default=0.5)
    a = ap.parse_args()
    d = pd.read_csv(a.log, header=None, names=['tag'] + COLS).drop(columns='tag')
    d['mid'] = (d.bb + d.ba) / 2
    vol = d.pv >= a.p_on
    fq = d.mid + a.k_s * (d.s - d.mid) + a.k_imb * d.imb + a.k_flow * d.fx
    fv = d.mid + a.v_k_s * (d.s - d.mid) + a.v_k_flow * d.fx
    d['fair'] = np.where(vol, fv, fq)
    ema = d.mid.ewm(halflife=300).mean()      # ~30 s half-life at 100 ms rows
    feats = {
        's_minus_mid': d.s - d.mid,
        'imb': d.imb,
        'fx': d.fx,
        'f1_depth': d.f1 / d.depth,
        'f10_depth': d.f10 / d.depth,
        'far_net': d.farb - d.fara,          # far-placed new bids minus asks (S moved down / up)
        'far_tot': d.farb + d.fara,
        'spread': d.ba - d.bb,
        'inv': d.inv.astype(float),
        'trend30': d.mid - ema,
        'mom10': d.mid - d.mid.shift(100),
        'mom1': d.mid - d.mid.shift(10),
        'fair_minus_mid': d.fair - d.mid,
    }
    n = len(d)
    half = np.arange(n) < n // 2
    print(f'{n} rows, {vol.mean():.1%} volatile; residual = mid(t+h) - fair(t)\n')
    for reg_name, reg in (('quiet', ~vol), ('volatile', vol)):
        for h, k in HORIZONS.items():
            res = d.mid.shift(-k) - d.fair
            print(f'== {reg_name} h={h}: residual sd {res[reg].std():.3f} t')
            rows = []
            for name, x in feats.items():
                c = []
                for part in (half, ~half):
                    m = reg & part & res.notna() & x.notna()
                    c.append(np.corrcoef(x[m], res[m])[0, 1] if m.sum() > 100 and x[m].std() > 0 else np.nan)
                stable = np.sign(c[0]) == np.sign(c[1]) and min(abs(c[0]), abs(c[1])) >= 0.03
                rows.append((name, c[0], c[1], stable))
            rows.sort(key=lambda r: -min(abs(r[1]), abs(r[2])) if r[3] else 0)
            for name, c0, c1, st in rows:
                print(f'   {name:<16}{c0:>+8.3f}{c1:>+8.3f}  {"STABLE" if st else ""}')
        print()


if __name__ == '__main__':
    main()
