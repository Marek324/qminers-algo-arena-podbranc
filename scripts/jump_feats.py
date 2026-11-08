"""Record jump-detector features every 5 ms (no trading) for an ensemble jump predictor."""
import importlib.util, sys
from pathlib import Path
from collections import deque
import numpy as np
from simulator.main import MAX_POSITION, MAX_POTENTIAL_POSITION, ORDER_ACCEPT_DELAY_MS, POSITION_BREACH_PENALTY_PER_SECOND
from simulator.simulator import ExchangeSimulator, SimulationConfig
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
base = [v for k, v in vars(m).items() if k.startswith('R4') and k.endswith('Strategy')][0]
base.IDG_SNIPE = 0; base.SNIPE_FAR = 0; base.IDG_P = 1e-12; base.MAX_POS = 0; base.MIN_NEW = 10**9
W = 25_000
class R(base):
    def __init__(self):
        super().__init__()
        self.rows = []; self.nxt = 0
        self.farq = deque(); self.swq = deque(); self.spq = deque()
        self.qa_ema = self.qb_ema = 0.0; self.sp_ema = 1.0
    def _observe(self, sign, price, depth, now):
        if self.mid:
            dist = sign * (price - self.mid)
            if dist > 3:
                self.farq.append((now, sign, dist))
        super()._observe(sign, price, depth, now)
    def on_order_book_change(self, snap, trades):
        s_before = self.s
        super().on_order_book_change(snap, trades)
        now = snap.timestamp_us
        bids, asks = snap.bids, snap.asks
        if not bids or not asks or not self.mid:
            return
        lv = {}
        for tr in trades:
            if tr.taker_owner == 'market':
                e = lv.setdefault(tr.taker_order_id, [1 if tr.book_side == 'ask' else -1, set()]); e[1].add(tr.price)
        for d0, ps in lv.values():
            if len(ps) >= 2:
                self.swq.append((now, d0 * (len(ps) - 1)))
        qa = sum(o.quantity for o in asks if o.owner == 'market'); qb = sum(o.quantity for o in bids if o.owner == 'market')
        if not self.qa_ema:
            self.qa_ema, self.qb_ema = float(qa), float(qb)
        sp = self.ba - self.bb if self.ba and self.bb else 1
        for q in (self.farq, self.swq):
            while q and now - q[0][0] > W:
                q.popleft()
        if now >= self.nxt:
            self.nxt = now + 5000
            g = self.idg; g.check(now)
            far_s = sum(sg * min(d, 30) for _, sg, d in self.farq)
            sw_s = sum(v for _, v in self.swq)
            self.rows.append((now, self.mid, len(g.gaps), len(g.cl_u), len(g.cl_d), len(g.na), len(g.nb), far_s, sw_s,
                              sp, sp - self.sp_ema, np.log((qa + 1) / (self.qa_ema + 1)), np.log((qb + 1) / (self.qb_ema + 1)),
                              self.pv, self.P, self.s - self.mid, self.s - s_before, self.jv))
        self.qa_ema += 0.002 * (qa - self.qa_ema); self.qb_ema += 0.002 * (qb - self.qb_ema)
        self.sp_ema += 0.002 * (sp - self.sp_ema)
s = R()
cfg = SimulationConfig(public_depth=20, max_position=MAX_POSITION, max_potential_position=MAX_POTENTIAL_POSITION, position_breach_penalty_per_second=POSITION_BREACH_PENALTY_PER_SECOND, order_accept_delay_ms=ORDER_ACCEPT_DELAY_MS)
ExchangeSimulator(strategy=s, config=cfg).run(Path(sys.argv[2]))
a = np.array(s.rows, float)
np.save(sys.argv[3], a)
print('rows', a.shape)
