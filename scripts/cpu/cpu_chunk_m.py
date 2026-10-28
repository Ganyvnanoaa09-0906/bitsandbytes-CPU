"""Chunk the batch dimension, not the output dimension. A different cache story.

Chunking N was tested in 7.52 and withdrawn in 7.54: it lost to a preallocated full call, because
whatever it saved on cache it paid in dispatch. But chunking N keeps nothing resident -- both B's
slice and C's slice change each iteration, so every chunk refetches the same A.

Chunking M is different. With C[i:i+m] = A[i:i+m] @ B, the whole of B stays fixed for the entire loop,
so it can live in L2 or L3 while only A's slice streams through. For the model's MLP at
2048 x 512 x 2736, B is 5.6 MB against an 8 MB L3, and the output is 22.4 MB which does not fit at
all -- but a row block of it can.

This has never been measured. Given that the same kernel at n=384 where the working set fits L2 runs
364 GFLOPS and at n=2048 only 235, a 1.55x swing decided purely by residency, a chunking scheme that
makes the working set resident is the most promising CPU lever left.

Measured against a preallocated control, which is the mistake 7.54 caught: allocation cost must not be
part of the comparison.

Decision rule, fixed in advance: chunking M must beat the preallocated full call by more than 5% for
it to stand, the same bar the N version failed.
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import statistics
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


from scripts.lib.measure_guard import assert_idle, stamp

assert_idle()
stamp('切 M 维：让 B 驻留')
torch.set_num_threads(6)
print()

SHAPES = [(2048, 2736, 512), (2048, 512, 2736), (2048, 512, 2048), (2048, 2048, 512)]


def bench(fn, reps=4, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


print('  %-18s %12s %13s %10s %14s' % ('形状 MxNxK', '全量预分配', '切M最优', '倍数', '最优 m'))
print('  ' + '-' * 72)
for (M, N, K) in SHAPES:
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    C = torch.empty(M, N)
    fl = 2.0 * M * N * K
    base = bench(lambda: torch.mm(A, B, out=C))
    best = (0.0, None)
    for m in (128, 256, 512, 1024):
        if m >= M:
            continue
        def chunked(m=m):
            for i in range(0, M, m):
                torch.mm(A[i:i + m], B, out=C[i:i + m])
        t = bench(chunked)
        gf = fl / t / 1e9
        if gf > best[0]:
            best = (gf, m)
    print('  %-18s %11.1f %12.1f %9.3fx %13s'
          % ('%dx%dx%d' % (M, N, K), fl / base / 1e9, best[0], best[0] / (fl / base / 1e9),
             best[1]))
print()
print('  判据（事先固定）: 切 M 比【预分配全量】快 > 5% ⇒ 站得住；否则与切 N 一样作废')
print('  参考: 7.54 里切 N 对预分配的比值是 0.87~1.006×（全部作废）')
