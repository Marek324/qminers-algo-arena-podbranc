"""Which stale-book opportunities does a strategy take, miss with room, or miss because its inventory is capped?

Opportunity (strategy-agnostic, hindsight): on a 100 ms grid, the best ask is >= MARGIN ticks below the mid 2 s later
(a buy) or the best bid that far above it (a sell). Value = edge x touch quantity (capped at CAP lots). Consecutive
grid points in one direction form a burst. For each burst, from a full-logs run: our inventory at its start, lots we
took in its direction during it (+0.5 s), and the class: TOOK (>= 20 lots), BLOCKED (room in that direction < 20 lots
under MAX_POS), MISSED (room left but nothing taken). Also splits by regime.

uv run python scripts/missed_opps.py logs/<full-logs run> data/<tape>.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from analysis.replay import replay  # noqa: E402
from analysis.tape import load_tape  # noqa: E402
from estimate_model import REGIME_WINDOW_S, detect_jumps, detect_regimes  # noqa: E402

US = 1_000_000
STEP = 100_000
MARGIN = 2.0
CAP = 200
MAX_POS = 100


def main(run: str, tape: str) -> None:
    run = Path(run)
    rp = replay(load_tape(Path(tape)), 10**9)
    reg = detect_regimes(rp, detect_jumps(rp))
    t0, t1 = rp.ev_ts[0], rp.ev_ts[-1]
    g = np.arange(t0, t1 - 3 * US, STEP)
    idx = np.maximum(np.searchsorted(rp.ev_ts, g, 'right') - 1, 0)
    mid = pd.Series(rp.mid).ffill().bfill().to_numpy()
    m2 = mid[np.maximum(np.searchsorted(rp.ev_ts, g + 2 * US, 'right') - 1, 0)]
    ba, bb = rp.best_ask[idx], rp.best_bid[idx]
    aq, bq = rp.ask_qty[idx].astype(float), rp.bid_qty[idx].astype(float)
    buy_edge = np.nan_to_num(m2 - ba)
    sell_edge = np.nan_to_num(bb - m2)
    opp_dir = np.where(buy_edge >= MARGIN, 1, np.where(sell_edge >= MARGIN, -1, 0))
    value = np.where(opp_dir == 1, buy_edge * np.minimum(aq, CAP), np.where(opp_dir == -1, sell_edge * np.minimum(bq, CAP), 0.0))

    tr = pd.read_csv(run / 'trades.csv')
    owners = (set(tr.taker_owner) | set(tr.maker_owner)) - {'market'}
    me = owners.pop()
    ours = tr[(tr.taker_owner == me) | (tr.maker_owner == me)].copy()
    taking = (ours.taker_owner == me).to_numpy()
    tb = ours.taker_side.astype(str).str.upper().str.startswith('B').to_numpy()
    sign = np.where(taking, np.where(tb, 1, -1), np.where(tb, -1, 1))
    q = ours.quantity.to_numpy()
    fts = ours.timestamp_us.to_numpy()
    inv = np.cumsum(sign * q)
    storm_prob = np.array(reg['storm_prob']) if reg else np.zeros(1)

    bursts = []
    i = 0
    n = len(g)
    while i < n:
        if opp_dir[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and opp_dir[j + 1] == opp_dir[i]:
            j += 1
        d = opp_dir[i]
        a, b = g[i], g[j] + STEP + 500_000
        k = np.searchsorted(fts, a, 'right') - 1
        inv0 = int(inv[k]) if k >= 0 else 0
        m = (fts >= a) & (fts < b) & taking & (sign == d)
        took = int(q[m].sum())
        room = MAX_POS - d * inv0
        cls = 'TOOK' if took >= 20 else ('BLOCKED' if room < 20 else 'MISSED')
        storm = storm_prob[min(int((a - t0) // (REGIME_WINDOW_S * US)), len(storm_prob) - 1)] > 0.5
        bursts.append((a, d, float(value[i:j + 1].max()), (j - i + 1) * STEP / 1000, inv0, took, cls, storm))
        i = j + 1
    B = pd.DataFrame(bursts, columns=['t', 'dir', 'value', 'dur_ms', 'inv0', 'took', 'class', 'storm'])
    if len(sys.argv) > 3:  # optional debug log of the strategy: DBG,ts,p_vol,s_hat,mid,inv every 100 ms
        D = pd.read_csv(sys.argv[3], header=None, names=['tag', 'ts', 'pv', 's', 'mid', 'inv'])
        k = np.clip(np.searchsorted(D.ts.to_numpy(), B.t.to_numpy(), 'right') - 1, 0, len(D) - 1)
        B['pv'] = D.pv.to_numpy()[k]
        B['s_minus_mid'] = (D.s - D.mid).to_numpy()[k] * B.dir
        for cls in ('TOOK', 'MISSED', 'BLOCKED'):
            x = B[(B['class'] == cls) & B.storm & (B.value >= 1000)]
            if len(x):
                print(f'storm {cls} (value >= 1000, n {len(x)}): p_vol at start median {x.pv.median():.2f}, share p_vol < 0.5 '
                      f'{(x.pv < 0.5).mean():.0%}; S_hat - mid toward the opportunity median {x.s_minus_mid.median():.2f} ticks')
        x = B[(B['class'] == 'MISSED') & B.storm & (B.value >= 1000)].sort_values('value', ascending=False)
        print(x.head(16).round(2).to_string(index=False))
    dur_h = (t1 - t0) / US / 3600
    print(f'{run}: {len(B)} opportunity bursts (edge >= {MARGIN} vs the mid 2 s later), value capped at {CAP} lots')
    s = B.groupby(['storm', 'class']).agg(n=('value', 'size'), value=('value', 'sum'), took_lots=('took', 'sum'))
    s['value_per_h'] = s.value / dur_h
    print(s.round(0).to_string())
    big = B[B.value >= 1000]
    print(f'\nbursts worth >= 1000: {len(big)}')
    print(big.groupby(['storm', 'class']).value.agg(['size', 'sum']).round(0).to_string())
    blocked = B[B['class'] == 'BLOCKED']
    if len(blocked):
        print(f'\nBLOCKED bursts: our inventory was already {blocked.inv0.abs().mean():.0f} lots in the opportunity direction on average; '
              f'{(blocked.inv0 * blocked.dir > 0).mean():.0%} were long when a buy opportunity came (or short for a sell)')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
