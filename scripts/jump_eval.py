"""Ensemble jump predictor vs the IdGapJump rule, out of sample, scored by tradable value per fire."""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
a = np.load(sys.argv[1])
L = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
t, mid = a[:, 0], a[:, 1]
gaps, clu, cld, na, nb, far, sw, sp, dsp, lqa, lqb, pv, P, gap_s, ds, jv = a[:, 2:].T
n = len(t)
H = 60   # 300 ms ahead on the 5 ms grid
fwd = np.r_[mid[H:] - mid[:-H], np.zeros(H)]
fwd5 = np.r_[mid[H:] - mid[1:n - H + 1], np.zeros(H)]   # from the next row (order lands 5 ms later)
def feats(m):   # m = +1 up-view, -1 mirrored
    if m > 0:
        return np.c_[gaps, clu, cld, na, nb, far, sw, sp, dsp, lqa, lqb, pv, np.log1p(P), gap_s, ds, jv]
    return np.c_[gaps, cld, clu, nb, na, -far, -sw, sp, dsp, lqb, lqa, pv, np.log1p(P), -gap_s, -ds, -jv]
Xu, Xd = feats(1), feats(-1)
yu, yd = (fwd >= L).astype(int), (fwd <= -L).astype(int)
half = t <= np.median(t)
def fires(score_u, score_d, thr, mask):
    out = []
    last = -1e18
    for i in np.flatnonzero(mask & ((score_u >= thr) | (score_d >= thr))):
        if t[i] - last < 1_000_000:
            continue
        d = 1 if score_u[i] >= score_d[i] else -1
        out.append((i, d)); last = t[i]
    return out
def value(fl):
    v = [d * fwd5[i] - sp[i] / 2 for i, d in fl]
    return len(v), sum(v), sum(1 for x in v if x > 5)
# IdGapJump rule (3 missing ids within 25 ms, clamps on one side, no visible new order on the far side)
rule_u = ((gaps >= 3) & (clu > 0) & (na == 0)).astype(float)
rule_d = ((gaps >= 3) & (cld > 0) & (nb == 0)).astype(float)
for name, tr, te in (('train H1 -> test H2', half, ~half), ('train H2 -> test H1', ~half, half)):
    X = np.r_[Xu[tr], Xd[tr]]; y = np.r_[yu[tr], yd[tr]]
    sc = StandardScaler().fit(X)
    mdl = LogisticRegression(C=0.05, max_iter=2000, class_weight='balanced').fit(sc.transform(X), y)
    su, sd = mdl.predict_proba(sc.transform(Xu))[:, 1], mdl.predict_proba(sc.transform(Xd))[:, 1]
    nr, vr, gr = value(fires(rule_u, rule_d, 0.5, te))
    print(f'\n{name}: rule fires {nr}, value {vr:+.0f} ticks/lot total, good (>5 ticks) {gr}')
    for q in (0.9999, 0.99995, 0.99998, 0.99999):
        thr = np.quantile(np.r_[su[tr], sd[tr]], q)
        nf, vf, gf = value(fires(su, sd, thr, te))
        print(f'  ensemble @q{q}: fires {nf}, value {vf:+.0f}, good {gf}')
    combo_u = np.maximum(rule_u, (su >= np.quantile(np.r_[su[tr], sd[tr]], 0.99998)).astype(float))
    combo_d = np.maximum(rule_d, (sd >= np.quantile(np.r_[su[tr], sd[tr]], 0.99998)).astype(float))
    nc, vc, gc = value(fires(combo_u, combo_d, 0.5, te))
    print(f'  rule OR ensemble@q0.99998: fires {nc}, value {vc:+.0f}, good {gc}')
    names = ['gaps', 'cl_same', 'cl_opp', 'new_same', 'new_opp', 'far', 'sweep', 'spread', 'dspread', 'depth_same', 'depth_opp', 'pv', 'logP', 'S-mid', 'dS', 'jv']
    co = mdl.coef_[0]
    print('  coefs:', ', '.join(f'{k} {v:+.2f}' for k, v in sorted(zip(names, co), key=lambda z: -abs(z[1]))[:8]))
