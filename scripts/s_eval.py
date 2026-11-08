"""Score S estimates against the true S read from market IOC prices (IOC price = round(S +- 1)).

usage: .venv/bin/python scripts/s_eval.py logs/features/r3_probe620.npz logs/features/round3_events.npz [name=file.npy ...]
Extra estimators: .npy arrays aligned with the probe's t (one value per callback).
"""
import sys

import numpy as np

p = np.load(sys.argv[1])
z = np.load(sys.argv[2])
ioc = z['ev_kind'] == 1
ti = z['t'][ioc]
strue = z['ev_px'][ioc] - z['ev_side'][ioc]
t = p['t']
j = np.searchsorted(t, ti, side='left') - 1   # last callback strictly before the IOC
ok = j >= 0
j, ti, strue = j[ok], ti[ok], strue[ok]
pv = p['pv'][j]
mid = p['mid'][j]
ests = {'mid': p['mid'], 's_hat': p['s']}
for a in sys.argv[3:]:
    k, f = a.split('=')
    ests[k] = np.load(f)
# future mid 250 ms after the IOC (what the oracle trades on)
jf = np.minimum(np.searchsorted(t, ti + 0.25), len(t) - 1)
fut = p['mid'][jf] - mid
print(f'IOCs {len(ti)}  volatile share {np.mean(pv >= 0.5):.2f}   (rounding noise alone: rmse 0.29)')
for k, e in ests.items():
    err = e[j] - strue
    row = [f'{k:10s}']
    for name, m in (('all', np.ones_like(pv, bool)), ('quiet', pv < 0.5), ('vol', pv >= 0.5)):
        row.append(f'{name} rmse {np.sqrt(np.mean(err[m] ** 2)):5.2f} mae {np.median(np.abs(err[m])):4.2f} bias {np.mean(err[m]):+5.2f}')
    d = e[j] - mid
    c = np.corrcoef(d, fut)[0, 1] if np.std(d) > 0 else 0
    row.append(f'corr(est-mid, mid+250ms - mid) {c:+.3f}')
    print('  '.join(row))
c = np.corrcoef(strue - mid, fut)[0, 1]
print(f'true S: corr(S-mid, mid+250ms - mid) {c:+.3f}')
