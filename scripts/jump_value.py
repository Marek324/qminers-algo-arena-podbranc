"""What would a jump predictor be worth to a strategy? From a full-logs run and its tape:

1. Timing: PnL of our fills in the window before each jump (they are the ones a perfect timing predictor would avoid),
   split passive / taking, marked out 30 s later.
2. Capture: for each jump, the stale liquidity traded in the 2 s after it (trades at prices the new level makes
   stale), how much of it we took vs market takers, and how fast the first stale trade came.
3. Direction: a cross-validated logistic model of the jump sign on pre-jump features (last 0.5 / 2 / 10 s).

uv run python scripts/jump_value.py logs/<full-logs run> data/<tape>.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from analysis.replay import replay  # noqa: E402
from analysis.tape import load_tape  # noqa: E402
from estimate_model import detect_jumps  # noqa: E402

US = 1_000_000


def main(run: str, tape: str) -> None:
    rp = replay(load_tape(Path(tape)), 10**9)
    J = detect_jumps(rp)
    jt, js = J['t_us'], J['size']
    mid = pd.Series(rp.mid).ffill().bfill().to_numpy()
    mid_at = lambda t: mid[np.clip(np.searchsorted(rp.ev_ts, t, 'right') - 1, 0, len(mid) - 1)]  # noqa: E731
    tr = pd.read_csv(Path(run) / 'trades.csv')
    me = ((set(tr.taker_owner) | set(tr.maker_owner)) - {'market'}).pop()
    tb = tr.taker_side.astype(str).str.upper().str.startswith('B').to_numpy()
    ours_t = (tr.taker_owner == me).to_numpy()
    ours_m = (tr.maker_owner == me).to_numpy()
    ts = tr.timestamp_us.to_numpy()
    px = tr.price.to_numpy().astype(float)
    q = tr.quantity.to_numpy()
    sign_ours = np.where(ours_t, np.where(tb, 1, -1), np.where(tb, -1, 1))
    print(f'{run}: {len(jt)} jumps; big (>= 15 ticks): {(np.abs(js) >= 15).sum()}')

    # 1. our fills just before jumps
    ours = ours_t | ours_m
    mo = sign_ours * (mid_at(ts + 30 * US) - px) * q
    for w in (0.5, 2.0):
        k = np.searchsorted(jt, ts, 'left')
        nxt = np.where(k < len(jt), jt[np.minimum(k, len(jt) - 1)], np.inf)
        pre = ours & ((nxt - ts) <= w * US)
        for name, m in (('passive', pre & ours_m), ('taking', pre & ours_t)):
            print(f'  fills in the {w:g} s before a jump, {name}: {q[m].sum():,} lots, 30 s markout {mo[m].sum():,.0f}')
    print(f'  all our fills, 30 s markout {mo[ours].sum():,.0f}')

    # 2. stale capture after big jumps
    rows = []
    for t0, s in zip(jt, js):
        if abs(s) < 15:
            continue
        new = mid_at(t0 + 2 * US)
        w = (ts >= t0 - 200_000) & (ts < t0 + 2 * US)
        # stale: buys at prices below the new level - 1 (jump up), sells above new + 1 (jump down)
        stale = w & ((tb & (px < new - 1)) if s > 0 else (~tb & (px > new + 1)))
        edge = np.abs(new - px) * q
        first = ts[stale].min() - t0 if stale.any() else np.nan
        rows.append((t0, s, q[stale].sum(), q[stale & ours_t].sum(), edge[stale].sum(), edge[stale & ours_t].sum(), first / 1000))
    R = pd.DataFrame(rows, columns=['t', 'size', 'stale_lots', 'our_lots', 'stale_edge', 'our_edge', 'first_ms'])
    print(f'\n  big jumps {len(R)}: stale lots traded {R.stale_lots.sum():,} (ours {R.our_lots.sum():,}, '
          f'{R.our_lots.sum() / max(R.stale_lots.sum(), 1):.0%}); stale edge {R.stale_edge.sum():,.0f} (ours {R.our_edge.sum():,.0f}, '
          f'{R.our_edge.sum() / max(R.stale_edge.sum(), 1):.0%}); first stale trade median {R.first_ms.median():.0f} ms after the jump bin')

    # 3. direction model; LAG shifts every feature window back, to tell prediction from early detection
    lim, ioc = ~rp.od_ioc, rp.od_ioc
    lag = int(float(sys.argv[3]) * US) if len(sys.argv) > 3 else 0
    feats = []
    for t0 in jt - lag:
        f = {}
        for w in (0.5, 2.0, 10.0):
            a = t0 - int(w * US)
            m_ioc = ioc & (rp.od_ts >= a) & (rp.od_ts < t0)
            m_lim = lim & (rp.od_ts >= a) & (rp.od_ts < t0)
            f[f'ioc_net_{w}'] = float((rp.od_side[m_ioc] * rp.od_qty[m_ioc]).sum())
            f[f'lim_side_{w}'] = float(rp.od_side[m_lim].mean()) if m_lim.any() else 0.0
            f[f'drift_{w}'] = float(mid_at(t0) - mid_at(a))
            g = rp.od_cross_gap[m_lim]
            f[f'far_signed_{w}'] = float((-rp.od_side[m_lim][np.isfinite(g) & (g <= -5)]).sum())
        k = np.searchsorted(rp.ev_ts, t0, 'right') - 1
        bq, aq = rp.bid_qty[k], rp.ask_qty[k]
        f['imbalance'] = float((bq - aq) / max(bq + aq, 1))
        feats.append(f)
    X = pd.DataFrame(feats).to_numpy()
    y = (js > 0).astype(int)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    acc = cross_val_score(LogisticRegression(C=0.3, max_iter=2000), X, y, cv=5, scoring='accuracy')
    print(f'\n  jump direction (features end {lag / 1000:.0f} ms before the jump), logistic on {X.shape[1]} features, '
          f'5-fold CV accuracy {acc.mean():.2f} +- {acc.std():.2f} (base rate {max(y.mean(), 1 - y.mean()):.2f}, n {len(y)})')
    names = list(pd.DataFrame(feats).columns)
    for n_, c in sorted(zip(names, X.T), key=lambda z: -abs(np.corrcoef(z[1], y)[0, 1]))[:5]:
        print(f'    {n_:<16} corr with sign {np.corrcoef(c, y)[0, 1]:+.2f}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
