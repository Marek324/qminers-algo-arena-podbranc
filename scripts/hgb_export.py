"""Train the HistGB storm classifier on round3_public (first half) and export hard-coded trees.

Features per 100 ms slot (identical to hmm_vs_ml.py, HMM outputs excluded):
  [cnt, ins_vol, n_mean, n_max, far, n_f, f_signed, f_vol, r1, r5, r20]
Label: |mid move| >= 5 ticks over the next 2 s. Train: first half of the tape, test: second half.
Also runs the ks HMM (H_N_V0 3.5) per slot to calibrate HGB_ON: the threshold matching the fraction
of slots the ks gate (p_vol >= 0.3) spends in volatile mode.
Writes /tmp/hgb_data.py (HGB_BAS, HGB_ON, HGB_TREES) for direct embedding into a strategy file,
after verifying the exported traversal reproduces predict_proba exactly.
"""
import math
import numpy as np

TAPE = "data/round3_public.csv"
SLOT = 0.1
HORIZON = 2.0
LABEL_TICKS = 5.0
H_NMAX = 5
H_N_Q0, H_N_V0 = 1.0, 3.5          # ks constants
H_PQV, H_PVQ, H_LR = 0.00003, 0.0003, 0.0003
P_ON = 0.3                          # ks gate


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
    feats = []
    mids = []
    pv = 0.0
    nq, nv = H_N_Q0, H_N_V0
    slot_i = -1
    cnt = ins_vol = 0.0
    nsum = nmax = 0.0
    far = 0
    pv_sum = 0.0
    slot_pvs = 0
    n_f = f_vol = f_signed = 0.0
    pv_on = 0    # inserts in this slot with instantaneous pv >= P_ON (true gate duty)

    def flush(mid):
        nonlocal cnt, ins_vol, nsum, nmax, far, pv_sum, slot_pvs, n_f, f_vol, f_signed, pv_on
        feats.append([cnt, ins_vol, nsum / cnt if cnt else 0.0, nmax, far / cnt if cnt else 0.0,
                      n_f, f_signed, f_vol, pv_sum / slot_pvs if slot_pvs else 0.0, pv_on])
        mids.append(mid)
        cnt = ins_vol = nsum = nmax = far = pv_sum = 0.0
        slot_pvs = 0
        n_f = f_vol = f_signed = 0.0
        pv_on = 0

    last_bb = last_ba = None
    for row in stream(TAPE):
        t, op, oid, side, vol, price, agg_b, agg_a = row
        si = int(t / SLOT)
        bb = max(agg_b) if agg_b else None
        ba = min(agg_a) if agg_a else None
        if si != slot_i:
            if slot_i >= 0:
                mid = (last_bb + last_ba) / 2 if last_bb is not None and last_ba is not None else (mids[-1] if mids else 10000.0)
                flush(mid)
            slot_i = si
        if op == "Insert":
            if bb is not None and ba is not None:
                mid = (bb + ba) / 2
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
                pv_sum += pv
                slot_pvs += 1
                if pv >= P_ON:
                    pv_on += 1
        elif op == "Fill":
            n_f += 1
            f_vol += vol
            f_signed += vol if side == "B" else -vol
        last_bb, last_ba = bb, ba
    if slot_i >= 0:
        flush((last_bb + last_ba) / 2 if last_bb is not None and last_ba is not None else mids[-1])

    X = np.array(feats)              # cols: 0 cnt 1 ins_vol 2 n_mean 3 n_max 4 far 5 n_f 6 f_signed 7 f_vol 8 pvol_mean
    mid = np.array(mids)
    T = len(X)
    H = int(HORIZON / SLOT)
    fwd = np.full(T, np.nan)
    fwd[:T - H] = np.abs(mid[H:] - mid[:T - H])
    y = (fwd >= LABEL_TICKS).astype(int)
    valid = ~np.isnan(fwd)

    for k in (1, 5, 20):
        r = np.zeros(T)
        r[k:] = mid[k:] - mid[:-k]
        X = np.column_stack([X, r])
    Xr = np.delete(X, [8, 9], axis=1)   # drop pvol_mean + pv_on -> 11 features

    half = T // 2
    tr = valid.copy(); tr[half:] = False
    te = valid.copy(); te[:half] = False

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=4,
                                         early_stopping=False, random_state=0)
    clf.fit(Xr[tr], y[tr])
    probs = clf.predict_proba(Xr)[:, 1]
    print(f"slots {T}  train {tr.sum()}  test {te.sum()}")
    print(f"HGB  AUC(test) {roc_auc_score(y[te], probs[te]):.3f}   base rate {y[te].mean():.3f}")

    # duty-cycle calibration vs the ks HMM gate (insert-weighted instantaneous pv >= P_ON)
    cnts = X[tr, 0]
    frac_pv = float(X[tr, 9].sum() / cnts.sum()) if cnts.sum() else 0.0
    hgb_on = float(np.quantile(probs[tr], 1.0 - frac_pv))
    print(f"ks gate TRUE duty cycle (train, insert-weighted pv >= {P_ON}): {frac_pv:.3f}")
    print(f"  (test-half duty: {float(X[te, 9].sum() / X[te, 0].sum()):.3f})")
    print(f"HGB_ON (same quantile of train probs): {hgb_on:.4f}")
    for nm, m in (("train", tr), ("test", te)):
        hit = float(y[m][probs[m] >= hgb_on].mean()) if (probs[m] >= hgb_on).any() else float("nan")
        print(f"  {nm}: HGB-on {float((probs[m] >= hgb_on).mean()):.3f}  precision {hit:.3f}")
    # candidate thresholds around the HMM duty for the strategy sweep
    for duty in (0.05, 0.10, 0.15, 0.20, 0.30):
        thr = float(np.quantile(probs[tr], 1.0 - duty))
        row = f"  duty {duty:.2f}: thr {thr:.4f}"
        for nm, m in (("train", tr), ("test", te)):
            on = probs[m] >= thr
            hit = float(y[m][on].mean()) if on.any() else float("nan")
            row += f"  {nm} on {float(on.mean()):.3f} prec {hit:.3f}"
        print(row)

    # ---- export ----
    bas = float(np.ravel(clf._baseline_prediction)[0])
    tree_strs = []
    for pred in clf._predictors:
        nodes = pred[0].nodes
        parts = []
        for nd in nodes:
            if nd["is_leaf"]:
                parts.append("[-1,0.0,0,0," + repr(float(nd["value"])) + "]")
            else:
                parts.append("[" + str(int(nd["feature_idx"])) + "," + repr(float(nd["num_threshold"]))
                             + "," + str(int(nd["left"])) + "," + str(int(nd["right"])) + ",0.0]")
        tree_strs.append("[" + ",".join(parts) + "]")
    hdr = (f"# HGB storm model, trained on first half of round3_public (q1+q2), 100 ms slots.\n"
           f"# features: cnt ins_vol n_mean n_max far n_f f_signed f_vol r1 r5 r20\n"
           f"HGB_BAS = {bas!r}\nHGB_ON = {hgb_on!r}\n")
    body = "HGB_TREES = [\n " + ",\n ".join(tree_strs) + "\n]\n"
    with open("/tmp/hgb_data.py", "w") as f:
        f.write(hdr + body)
    print(f"wrote /tmp/hgb_data.py  ({len(hdr) + len(body)} bytes, {len(tree_strs)} trees)")

    # ---- self-check: exported traversal must reproduce predict_proba ----
    def my_prob(x):
        raw = bas
        for pred in clf._predictors:
            nds = pred[0].nodes
            i = 0
            while True:
                nd = nds[i]
                if nd["is_leaf"]:
                    raw += float(nd["value"])
                    break
                if x[int(nd["feature_idx"])] <= nd["num_threshold"]:
                    i = int(nd["left"])
                else:
                    i = int(nd["right"])
        return 1.0 / (1.0 + math.exp(-raw)) if -35.0 < raw < 35.0 else (1.0 if raw > 0 else 0.0)

    idx = np.where(valid)[0][::max(1, T // 900)]
    diffs = [abs(my_prob([float(v) for v in Xr[i]]) - probs[i]) for i in idx]
    print(f"self-check on {len(idx)} slots: max |my_prob - predict_proba| = {max(diffs):.2e}")
    assert max(diffs) < 1e-9, "export mismatch!"
    print("EXPORT OK")


if __name__ == "__main__":
    run()
