"""Run a strategy on a tape and break its PnL down by fill type (IOC vs resting) and by markout horizon.

Wraps the strategy's send_order / on_private_order_update to log every fill (time, sign, price, qty,
order type). Markouts use the market mid from scripts/tape_events.py output.

usage: uv run python scripts/fill_audit.py strategies/round2/lag_taker.py data/round2_train.csv \
           logs/features/round2_events.npz [out.npz]
"""
import sys
from pathlib import Path

import numpy as np

from simulator.main import load_strategy_from_file
from simulator.models import OrderStatus, OrderType
from simulator.simulator import ExchangeSimulator, SimulationConfig


def main() -> None:
    strat_path, tape, ev = sys.argv[1], sys.argv[2], sys.argv[3]
    strat = load_strategy_from_file(strat_path)
    kinds: dict[str, int] = {}
    sides: dict[str, int] = {}
    fills: list[tuple] = []
    orig_send = strat.send_order
    orig_upd = strat.on_private_order_update

    def send(req):
        oid = orig_send(req)
        kinds[oid] = 1 if req.order_type == OrderType.IOC else 0
        sides[oid] = 1 if req.side.value == 'B' else -1
        return oid

    def upd(event):
        if event.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            fills.append((strat.now_us() / 1e6, sides.get(event.order_id, 0), event.fill_price,
                          event.filled_quantity, kinds.get(event.order_id, -1)))
        orig_upd(event)

    strat.send_order = send
    strat.on_private_order_update = upd
    sim = ExchangeSimulator(strat, SimulationConfig())
    res = sim.run(Path(tape))
    print('pnl', getattr(res, 'pnl', res))
    f = np.array(fills)
    if len(sys.argv) > 4:
        np.save(sys.argv[4], f)
    z = np.load(ev)
    t = z['t']
    bb, ba = z['bb'].copy(), z['ba'].copy()
    n = len(t)
    for arr in (bb, ba):
        idx = np.where(arr > 0, np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)
        arr[:] = arr[idx]
    mid = (bb + ba) / 2
    ft, fs, fp, fq, fk = f.T
    for k, name in ((1, 'IOC'), (0, 'LIMIT')):
        m = fk == k
        row = [f'{name}: fills {m.sum()} lots {fq[m].sum():.0f}']
        for h in (0, 0.1, 1, 5, 30):
            j = np.minimum(np.searchsorted(t, ft[m] + h), n - 1)
            mo = (mid[j] - fp[m]) * fs[m]
            row.append(f'@{h}s {np.sum(mo * fq[m]) / max(1, fq[m].sum()):+.2f}/lot ({np.sum(mo * fq[m]):+.0f})')
        print('  '.join(row))


if __name__ == '__main__':
    main()
