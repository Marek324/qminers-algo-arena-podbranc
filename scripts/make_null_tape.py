"""Write a synthetic control tape with NO exploitable structure, to measure a strategy's downside.

Usage: python3 -I scripts/make_null_tape.py data/null_<seed>.csv [seed] [aggr_follow]

Shaped like sample.csv: one resting order per side, re-quoted about every 0.25 s, spread 2 ticks,
plus an aggressive IOC every few seconds that takes part of the touch. Here, though, the mid is a
pure random walk that is independent of the aggressive orders, so there is no trend to follow.

aggr_follow (default 0.0) is the probability that the mid steps one tick in the aggressor's
direction after each aggressive order. Raise it toward 1 to put back a sample-like trend
and see how the strategy's PnL responds.
"""
import random
import sys

out = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
follow = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
rng = random.Random(seed)

rows = ['Time,Operation,OrderId,Side,Volume,Price,Lifespan']
oid = 1_000_000
mid = 10_000  # bid = mid - 1, ask = mid + 1
t = 0.0
bid_id = ask_id = None
next_aggr = rng.uniform(2, 6)
drift = 0  # pending ticks of move from aggressive follow-through


def nid():
    global oid
    oid += 1
    return oid


while t < 3600:
    # cancel and re-insert both quotes around the current mid
    if bid_id is not None:
        rows.append(f'{t:.6f},Cancel,{bid_id},,,,')
        rows.append(f'{t + 1e-6:.6f},Cancel,{ask_id},,,,')
    t += 2e-6
    bid_id, ask_id = nid(), nid()
    rows.append(f'{t:.6f},Insert,{bid_id},B,{rng.randint(10, 40)},{mid - 1},G')
    rows.append(f'{t + 1e-6:.6f},Insert,{ask_id},A,{rng.randint(10, 40)},{mid + 1},G')

    t += rng.uniform(0.15, 0.35)
    if t >= next_aggr:
        side = rng.choice('BA')
        price = mid + 1 if side == 'B' else mid - 1
        rows.append(f'{t:.6f},Insert,{nid()},{side},{rng.randint(5, 20)},{price},F')
        if rng.random() < follow:
            drift += 1 if side == 'B' else -1
        next_aggr = t + rng.uniform(2, 6)
        t += 1e-3
    # mid moves: random-walk noise plus any follow-through
    step = rng.choice((-1, 0, 0, 1))
    if drift:
        step += 1 if drift > 0 else -1
        drift -= 1 if drift > 0 else -1
    mid += step

with open(out, 'w') as f:
    f.write('\n'.join(rows) + '\n')
print(out, len(rows) - 1, 'rows')
