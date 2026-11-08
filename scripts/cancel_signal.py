"""Do cancels or order sizes carry information about the hidden price that the strategy does not use?

1. Cancel age vs the order's distance from the mid at cancel time: if the generator cancels orders S has left behind,
   the age falls with distance (memoryless random cancels give the same age at every distance).
2. Signed near-touch cancel flow (bid cancels minus ask cancels within NEAR ticks of the mid, decayed over 1 s) and
   the size of new orders vs their distance from the mid, against the residual mid(t + h) - fair(t) from an
   instrumented-strategy log (scripts/residual_signal.py format), per tape half.

usage: .venv/bin/python scripts/cancel_signal.py data/round3_public.csv logs/siglog.csv
"""
import sys

import numpy as np
import pandas as pd

NEAR = 2.0
DECAY_SUM = 1.0 / (1.0 - 0.5 ** (1 / 7))  # ewm mean (halflife 7 bins) -> decayed sum, as the strategy keeps it
COLS = ['tag', 't', 'bb', 'ba', 'bq', 'aq', 's', 'pv', 'imb', 'fx', 'inv', 'f1', 'f10', 'farb', 'fara', 'depth']


def main() -> None:
    tape, log = sys.argv[1], sys.argv[2]
    d = pd.read_csv(log, header=None, names=COLS)
    d['mid'] = (d.bb + d.ba) / 2
    d['ts'] = d.t / 1e6
    vol = d.pv >= 0.5
    fq = d.mid + 0.75 * (d.s - d.mid) + 0.15 * d.imb + 1.0 * d.fx
    fv = d.mid + 0.8 * (d.s - d.mid) + 1.2 * d.fx
    d['fair'] = np.where(vol, fv, fq)
    grid_t = d.ts.to_numpy()
    grid_mid = d.mid.to_numpy()

    tp = pd.read_csv(tape, dtype={'OrderId': np.int64})
    ins = tp[tp.Operation == 'Insert']
    ins = ins[ins.Lifespan == 'G']
    info = pd.DataFrame({'t0': ins.Time.to_numpy(), 'side': ins.Side.to_numpy(), 'px': ins.Price.to_numpy(),
                         'vol': ins.Volume.to_numpy()}, index=ins.OrderId.to_numpy())
    can = tp[tp.Operation == 'Cancel']
    c = info.reindex(can.OrderId.to_numpy())
    c['t'] = can.Time.to_numpy()
    c = c.dropna(subset=['t0'])
    gi = np.clip(np.searchsorted(grid_t, c.t.to_numpy()) - 1, 0, len(grid_t) - 1)
    mid_c = grid_mid[gi]
    sgn = np.where(c.side == 'B', 1.0, -1.0)
    c['dist'] = sgn * (mid_c - c.px)          # ticks behind the mid (positive = on its own side of the mid)
    c['age'] = c.t - c.t0
    print(f'{len(c):,} cancels of resting orders; age median {c.age.median():.2f} s')
    print('\n1. cancel age by distance from the mid at cancel time (memoryless => flat)')
    bins = [-50, 0, 1, 2, 3, 5, 8, 15, 1e9]
    g = c.groupby(pd.cut(c.dist, bins))['age']
    print(pd.DataFrame({'n': g.size(), 'median_age_s': g.median(), 'mean_age_s': g.mean()}).to_string())
    behind = c.dist < 0
    print(f'   cancels of orders already on the wrong side of the mid: {behind.mean():.1%}')

    # 2. near-touch signed cancel flow on the 100 ms grid
    near = c[(c.dist >= 0) & (c.dist <= NEAR)]
    w = np.where(near.side == 'B', 1.0, -1.0)
    gi2 = np.clip(np.searchsorted(grid_t, near.t.to_numpy()), 0, len(grid_t) - 1)
    raw = np.bincount(gi2, weights=w, minlength=len(grid_t))
    cnet = pd.Series(raw).ewm(halflife=7).mean()      # ~1 s
    # new-order size vs distance from the mid at insert
    gi3 = np.clip(np.searchsorted(grid_t, info.t0.to_numpy()) - 1, 0, len(grid_t) - 1)
    sg3 = np.where(info.side == 'B', 1.0, -1.0)
    idist = sg3 * (grid_mid[gi3] - info.px.to_numpy())
    print('\n   new-order size by distance from the mid at insert:')
    gb = pd.Series(info.vol.to_numpy()).groupby(pd.cut(idist, bins))
    print(pd.DataFrame({'n': gb.size(), 'mean_vol': gb.mean()}).to_string())
    # size-weighted insert imbalance near the touch (bids minus asks, volume), ~1 s
    nmask = (idist >= -0.5) & (idist <= NEAR)
    wv = np.where(sg3 > 0, 1.0, -1.0) * info.vol.to_numpy()
    raw2 = np.bincount(gi3[nmask], weights=wv[nmask], minlength=len(grid_t))
    inet = pd.Series(raw2).ewm(halflife=7).mean()

    n = len(d)
    half = np.arange(n) < n // 2
    print('\n2. correlation with residual mid(t+h) - fair(t), halves 1 / 2')
    for reg_name, reg in (('quiet', ~vol), ('volatile', vol)):
        for h, k in (('0.3s', 3), ('1s', 10), ('5s', 50)):
            res = d.mid.shift(-k) - d.fair
            out = []
            for name, x in (('cancel_net', cnet), ('insert_vol_net', inet)):
                cc = []
                for part in (half, ~half):
                    m = (reg & part & res.notna()).to_numpy()
                    cc.append(np.corrcoef(x.to_numpy()[m], res.to_numpy()[m])[0, 1])
                st = np.sign(cc[0]) == np.sign(cc[1]) and min(map(abs, cc)) >= 0.03
                out.append(f'{name} {cc[0]:+.3f} {cc[1]:+.3f}{" STABLE" if st else ""}')
            sl = []
            for part in (half, ~half):
                m = (reg & part & res.notna()).to_numpy()
                sl.append(np.polyfit(cnet.to_numpy()[m] * DECAY_SUM, res.to_numpy()[m], 1)[0])
            out.append(f'slope per decayed cancel (tau 1 s) {sl[0]:+.4f} {sl[1]:+.4f}')
            print(f'   {reg_name:<9}{h:>5}   ' + '   '.join(out))


if __name__ == '__main__':
    main()
