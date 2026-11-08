"""Write a variant of a strategy with class constants overridden: vt_var.py base.py out.py NAME=val ..."""
import re
import sys

src = open(sys.argv[1]).read()
for kv in sys.argv[3:]:
    k, v = kv.split('=', 1)
    src, n = re.subn(rf'^(\s*){k} = [^#\n]*', rf'\g<1>{k} = {v}  ', src, count=1, flags=re.M)
    assert n == 1, k
open(sys.argv[2], 'w').write(src)
