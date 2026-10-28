"""master_20261018.py -- run the two decisive overfit experiments back to back.

1) overfit_single.py   two rows, 300 steps: can the model memorise ANYTHING (CE -> ~0)?
2) overfit100_bos.py   the 7.111 run repeated with --use_bos: how much of the free-sampling collapse
                       was the BOS bug, and how much is exposure bias?

Sequential on purpose: both runs are memory-heavy (~1.9 GB RSS, 5 threads) and must not overlap.
Each stage writes its own log; this master log records boundaries so a partial run is interpretable.

Run (background): python -u master_20261018.py
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

import os
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

ROOT = r'D:\work\bitsandbytes-CPU'
LOG = os.path.join(ROOT, 'master_20261018.log')
os.chdir(ROOT)


def say(msg):
    line = time.strftime('%H:%M:%S') + '  ' + msg
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)


with open(LOG, 'w', encoding='utf-8') as f:
    f.write('master start ' + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')

say('stage 1/2: overfit_single (2 rows, 300 steps, ~25 min)')
t0 = time.time()
with open(os.path.join(ROOT, 'master_single.log'), 'w', encoding='utf-8') as f:
    rc = subprocess.call([sys.executable, '-u', 'overfit_single.py'], stdout=f, stderr=subprocess.STDOUT)
say('stage 1 exit %d  (%.0f s)' % (rc, time.time() - t0))

say('stage 2/2: overfit100 with BOS (2000 steps = 20 epochs, ~2.7 h)')
t1 = time.time()
with open(os.path.join(ROOT, 'master_bos.log'), 'w', encoding='utf-8') as f:
    rc2 = subprocess.call([sys.executable, '-u', 'overfit100_bos.py'], stdout=f, stderr=subprocess.STDOUT)
say('stage 2 exit %d  (%.0f s)' % (rc2, time.time() - t1))

say('all done ' + time.strftime('%Y-%m-%d %H:%M:%S'))
