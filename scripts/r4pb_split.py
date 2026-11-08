"""Split a fill_audit .npy (t, sign, price, qty, kind) into take / passive x calm / storm with markouts.

usage: .venv/bin/python scripts/r4pb_split.py logs/r4pb/audit_x.npz.npy logs/features/round4_events.npz
"""
import sys
import numpy as np
f=np.load(sys.argv[1]); d=np.load(sys.argv[2]); t=d['t']; bb=d['bb']; ba=d['ba']
ok=(bb>0)&(ba>0)&(ba<1e9); t,bb,ba=t[ok],bb[ok],ba[ok]; mid=(bb+ba)/2
def m(x): return mid[np.clip(np.searchsorted(t,x,side='right')-1,0,len(t)-1)]
ft,sg,px,q=f[:,0],f[:,1],f[:,2],f[:,3]
m0=m(ft-1e-6); take=sg*(px-m0)>0
vol=np.abs(m(ft)-m(ft-1.0))
for name,sel in (('take',take),('passive',~take)):
  for vn,vs in (('calm',vol<=2),('storm',vol>2)):
    s=sel&vs; row=[f'{name:7} {vn:5} fills {s.sum():>6} lots {q[s].sum():>9,.0f}']
    for h in (0.1,1,5,30):
      mk=sg[s]*(m(ft[s]+h)-px[s])*q[s]; row.append(f'@{h}s {mk.sum()/max(q[s].sum(),1):+.2f} ({mk.sum()/1e3:+,.0f}k)')
    print('  '.join(row))
