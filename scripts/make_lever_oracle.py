"""Write strategies/round3/oracle_lever_b2_q<Q>.py: hmm_imm_b2 plus the hindsight leverage schedule of
scripts/lever_oracle.py --dump. Local ceiling test only: it reads the future. Never submit.

.venv/bin/python scripts/make_lever_oracle.py logs/lever_schedule_round3_public_q900.json 900
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src = (ROOT / 'strategies/round3/hmm_imm_b2.py').read_text()
q = int(sys.argv[2])
min_net = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
sched = [tuple(r[:3]) for r in json.loads(Path(sys.argv[1]).read_text()) if r[3] >= min_net]
tag = f'q{q}' + (f'_n{int(min_net)}' if min_net else '')
out = ROOT / f'strategies/round3/oracle_lever_b2_{tag}.py'

header = (f'"""ORACLE, never submit: hmm_imm_b2 plus {len(sched)} hindsight leverage excursions of up to {q} lots on\n'
          'round3_public (scripts/lever_oracle.py --dump). The lever lots are tracked apart from b2\'s own inventory,\n'
          'so b2 quotes exactly as before; entry and exit are IOCs sent 5 ms ahead of the scheduled times.\n"""\n\n')
src = src.split('"""', 2)[2].lstrip('\n')
src = src.replace('class HmmImmB2Strategy(BaseStrategy):', 'class OracleLeverB2Strategy(BaseStrategy):', 1)
assert 'OracleLeverB2Strategy' in src

src = src.replace('''        self.inv = 0
''', f'''        self.inv = 0
        self.lev = 0                        # oracle lever inventory (not in self.inv)
        self.lev_orders: dict = {{}}          # order_id -> sign
        self.lev_i = 0
        self.lev_exit = 0                   # send time of the exit, 0 when no excursion
        self.lev_sched = LEV_SCHED
''', 1)

src = src.replace('''        if bb is None or ba is None:
            return
        mid = (bb + ba) / 2
        self.mid = mid
        now = snapshot.timestamp_us
''', '''        if bb is None or ba is None:
            return
        mid = (bb + ba) / 2
        self.mid = mid
        now = snapshot.timestamp_us
        self._lever(now, bb, ba)
''', 1)

src = src.replace('''    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        oid = event.order_id
''', f'''    def _lever(self, now: int, bb: int, ba: int) -> None:
        sch = self.lev_sched
        while self.lev_i < len(sch) and sch[self.lev_i][0] - 5_000 <= now:
            t, sign, hold = sch[self.lev_i]
            self.lev_i += 1
            if self.lev_exit or self.lev or self.lev_orders:
                continue
            resting = sum(o[2] for o in self.orders.values() if o[0] == sign)
            q = min({q}, 1000 - sign * (self.inv + self.lev) - resting)
            if q < 10:
                continue
            px = ba + 30 if sign > 0 else max(1, bb - 30)
            side = Side.BUY if sign > 0 else Side.SELL
            oid = self.send_order(OrderRequest(side=side, price=px, quantity=q, order_type=OrderType.IOC))
            self.lev_orders[oid] = sign
            self.lev_exit = t + hold - 5_000
        if self.lev_exit and now >= self.lev_exit and not self.lev_orders:
            if self.lev == 0:
                self.lev_exit = 0
                return
            sign = -1 if self.lev > 0 else 1
            px = ba + 30 if sign > 0 else max(1, bb - 30)
            side = Side.BUY if sign > 0 else Side.SELL
            oid = self.send_order(OrderRequest(side=side, price=px, quantity=abs(self.lev), order_type=OrderType.IOC))
            self.lev_orders[oid] = sign

    def on_private_order_update(self, event: PrivateOrderEvent) -> None:
        oid = event.order_id
        ls = self.lev_orders.get(oid)
        if ls is not None:
            st = event.status
            if st == OrderStatus.FILLED or st == OrderStatus.PARTIALLY_FILLED:
                self.lev += ls * event.filled_quantity
            if st == OrderStatus.FILLED or st == OrderStatus.CANCELED or st == OrderStatus.REJECTED:
                self.lev_orders.pop(oid, None)
            return
''', 1)
src += f'\n\nLEV_SCHED = {sched!r}\n'
out.write_text(header + src)
print(out, len(sched))
