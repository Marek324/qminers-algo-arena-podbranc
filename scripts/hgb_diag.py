"""Diagnose the HGB gate failure vs the HMM gate: where each fires, and what forward move follows.

Streams round3_public.csv (same machinery as hgb_export.py), scores every 100 ms slot with the
exported trees, and compares the HMM gate (slot pv_max >= 0.3) against the HGB gate (prob >= THR):
duty cycles, overlap, the lag from each HGB fire to the nearest HMM fire, and the mean forward
2 s |mid move| in slots only the HMM catches, only the HGB catches, both, and neither.
"""
import math
import numpy as np

TAPE = "data/round3_train.csv"
SLOT = 0.1
HORIZON = 2.0
H_NMAX = 5
H_N_Q0, H_N_V0 = 1.0, 3.5
H_PQV, H_PVQ, H_LR = 0.00003, 0.0003, 0.0003
P_ON = 0.3
THR = 0.5304          # ~10% duty threshold from the calibration printout


def stream(tape):
    live_b = {}
    live_a = {}
    vol_at = {}
    agg_b, agg_a = {}, {}
    with open(tape) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(",")
            t = float(parts[0])
            op = parts[1]
            if op == "Insert":
                oid, side, vol, price = parts[2], parts[3], int(parts[4]), int(parts[5])
                life = parts[6] if len(parts) > 6 else "G"
                if life == "F":
                    if side == "B" and agg_a:
                        agg_a[price] = agg_a.get(price, 0) - vol
                        if agg_a[price] <= 0:
                            del agg_a[price]
                    elif side == "A" and agg_b:
                        agg_b[price] = agg_b.get(price, 0) - vol
                        if agg_b[price] <= 0:
                            del agg_b[price]
                    yield t, "Fill", oid, side, vol, price, agg_b, agg_a
                    continue
                if side == "B":
                    live_b[oid] = vol
                    agg_b[price] = agg_b.get(price, 0) + vol
                else:
                    live_a[oid] = vol
                    agg_a[price] = agg_a.get(price, 0) + vol
                vol_at[oid] = (side, price)
                yield t, op, oid, side, vol, price, agg_b, agg_a
            elif op == "Cancel":
                oid = parts[2]
                rec = vol_at.get(oid)
                if rec:
                    side, price = rec
                    if side == "B":
                        v = live_b.pop(oid, 0)
                        if price in agg_b:
                            agg_b[price] -= v
                            if agg_b[price] <= 0:
                                del agg_b[price]
                    else:
                        v = live_a.pop(oid, 0)
                        if price in agg_a:
                            agg_a[price] -= v
                            if agg_a[price] <= 0:
                                del agg_a[price]
                    del vol_at[oid]


def run():
    exec(open("/tmp/hgb_data.py").read(), globals())   # HGB_BAS, HGB_TREES

    def tree_prob(x):
        raw = HGB_BAS
        for t in HGB_TREES:
            i = 0
            while True:
                nd = t[i]
                f = nd[0]
                if f < 0:
                    raw += nd[4]
                    break
                if x[f] <= nd[1]:
                    i = nd[2]
                else:
                    i = nd[3]
        return 1.0 / (1.0 + math.exp(-raw)) if -35.0 < raw < 35.0 else (1.0 if raw > 0 else 0.0)

    feats = []
    mids = []
    t0s = []
    pv_maxs = []
    pv = 0.0
    nq, nv = H_N_Q0, H_N_V0
    slot_i = -1
    cnt = ins_vol = 0.0
    nsum = nmax = 0.0
    far = 0
    n_f = f_vol = f_signed = 0.0
    pv_max = 0.0

    r_hist = []

    def flush(mid, t0):
        nonlocal cnt, ins_vol, nsum, nmax, far, n_f, f_vol, f_signed, pv_max
        h = r_hist
        nn = len(h)
        feats.append([cnt, ins_vol, nsum / cnt if cnt else 0.0, nmax, far / cnt if cnt else 0.0,
                      n_f, f_signed, f_vol,
                      mid - h[-1] if nn >= 1 else 0.0,
                      mid - h[-5] if nn >= 5 else 0.0,
                      mid - h[-20] if nn >= 20 else 0.0])
        h.append(mid)
        if len(h) > 20:
            del h[0]
        mids.append(mid)
        t0s.append(t0)
        pv_maxs.append(pv_max)
        cnt = ins_vol = 0.0
        nsum = nmax = far = 0
        n_f = f_vol = f_signed = 0.0
        pv_max = 0.0

    last_mid = None
    slot_t0 = 0.0
    for row in stream(TAPE):
        t, op, oid, side, vol, price, agg_b, agg_a = row
        si = int(t / SLOT)
        bb = max(agg_b) if agg_b else None
        ba = min(agg_a) if agg_a else None
        if si != slot_i:
            if slot_i >= 0 and last_mid is not None:
                flush(last_mid, slot_t0)
            slot_i = si
            slot_t0 = t
        if op == "Insert":
            if bb is not None and ba is not None:
                mid = (bb + ba) / 2
                last_mid = mid
                sign = 1 if side == "B" else -1
                n = min(H_NMAX, max(0, round(2 * (sign * (price - mid) - 0.5))))
                rq = nq / (1.0 + nq)
                rv = nv / (1.0 + nv)
                lq = (1.0 - rq) * rq ** n
                lv = (1.0 - rv) * rv ** n
                p = pv * (1.0 - H_PVQ) + (1.0 - pv) * H_PQV
                a = p * lv
                pv = a / (a + (1.0 - p) * lq)
                nq += H_LR * (1.0 - pv) * (n - nq)
                nv += H_LR * pv * (n - nv)
                if nv < nq + 0.5:
                    nv = nq + 0.5
                cnt += 1
                ins_vol += vol
                nsum += n
                nmax = max(nmax, n)
                if n >= H_NMAX:
                    far += 1
                pv_max = max(pv_max, pv)
        elif op == "Fill":
            n_f += 1
            f_vol += vol
            f_signed += vol if side == "B" else -vol
        last_bb, last_ba = bb, ba
    if slot_i >= 0 and last_mid is not None:
        flush(last_mid, slot_t0)

    T = len(feats)
    mid = np.array(mids)
    H = int(HORIZON / SLOT)
    fwd = np.zeros(T)
    fwd[:T - H] = np.abs(mid[H:] - mid[:T - H])

    probs = np.array([tree_prob([float(v) for v in f]) for f in feats])
    pvmax = np.array(pv_maxs)
    fire_h = pvmax >= P_ON
    fire_g = probs >= THR
    print(f"slots {T}  HMM fire {fire_h.mean():.4f}  HGB fire (thr {THR}) {fire_g.mean():.4f}")
    print(f"overlap: P(HGB|HMM) {float((fire_h & fire_g).sum() / max(fire_h.sum(), 1)):.2f}"
          f"  P(HMM|HGB) {float((fire_h & fire_g).sum() / max(fire_g.sum(), 1)):.2f}")

    both = fire_h & fire_g
    ho = fire_h & ~fire_g
    go = ~fire_h & fire_g
    no = ~fire_h & ~fire_g
    for nm, m in (("both", both), ("hmm only", ho), ("hgb only", go), ("neither", no)):
        if m.sum():
            print(f"  {nm:<9} n {m.sum():>6}  mean fwd |move| {fwd[m].mean():6.2f} ticks"
                  f"  P(move>=5t) {(fwd[m] >= 5).mean():.3f}")

    # lag from each HGB fire to the nearest HMM fire (slots; + = HGB fires later)
    hi = np.where(fire_h)[0]
    gi = np.where(fire_g)[0]
    if len(hi) and len(gi):
        pos = np.searchsorted(hi, gi)
        lags = []
        for k, g in enumerate(gi):
            cands = []
            if pos[k] > 0:
                cands.append(g - hi[pos[k] - 1])
            if pos[k] < len(hi):
                cands.append(g - hi[pos[k]])
            lags.append(min(cands, key=abs))
        lags = np.array(lags)
        print("nearest HMM fire relative to HGB fire (slots, + = HGB later):")
        for lo, lab in ((-10 ** 9, "<=-10"), (-9, "-9..-5"), (-4, "-4..-1"), (0, "0 (same)"),
                        (1, "+1..+2"), (3, "+3..+5"), (6, "+6..+10"), (11, "+11..+30"), (31, ">30")):
            if lo == -10 ** 9:
                c = (lags <= -10).sum()
            elif lo == 31:
                c = (lags > 30).sum()
            else:
                hi_b = {-9: -5, -4: -1, 0: 0, 1: 2, 3: 5, 6: 10, 11: 30}[lo]
                c = ((labs == 0) if False else ((lags >= lo) & (lags <= hi_b)).sum())
            print(f"    {lab:>10}: {c:>5}  ({c / len(lags):.2f})")


if __name__ == "__main__":
    run()
