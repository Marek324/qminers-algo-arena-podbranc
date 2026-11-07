"""Split a tape into N equal-time chunks for out-of-sample checks.

Usage: python3 -I scripts/split_tape.py data/sample.csv 2   ->  data/sample_p1of2.csv, data/sample_p2of2.csv

Each chunk starts with an empty book (cancels of orders from earlier chunks are ignored by the simulator).
"""
import sys
from pathlib import Path

src = Path(sys.argv[1])
n = int(sys.argv[2])
lines = src.read_text().splitlines()
header, rows = lines[0], lines[1:]
end = float(rows[-1].split(',', 1)[0])
outs = [[header] for _ in range(n)]
for row in rows:
    k = min(n - 1, int(float(row.split(',', 1)[0]) / end * n))
    outs[k].append(row)
for k, out in enumerate(outs):
    path = src.with_name(f'{src.stem}_p{k + 1}of{n}.csv')
    path.write_text('\n'.join(out) + '\n')
    print(path, len(out) - 1, 'rows')
