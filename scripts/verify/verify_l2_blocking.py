"""Is the L2 blocking speedup real, or just avoided allocation?

7.52.1 measured a column-chunked GEMM at 1.359x the full call, and the obvious reading is that
chunking keeps the working set in cache. But the two calls were not otherwise identical: torch.mm(A, B)
allocates a fresh 22.4 MB output on every invocation, while the chunked version allocated one buffer
and wrote slices into it. A 22.4 MB allocation per call is not free, and in a training loop the
output is allocated anyway, so the comparison may be measuring the allocator rather than the cache.

Three variants, to separate them:

  full, fresh output      torch.mm(A, B)                    what the baseline did
  full, preallocated      torch.mm(A, B, out=C)             same work, no allocation
  chunked, preallocated   C[:, j:j+k] = torch.mm(A, B[:, j:j+k], out=C[:, j:j+k])

If chunking still wins against the preallocated full call, the effect is cache blocking and worth
implementing. If the gap closes, the 1.359x was allocation and the recommendation in 7.52.1 is wrong.

That distinction matters more than the number, because the trainer's MLP always needs a fresh output
for autograd. A win that only exists when the output is reused does not transfer.

Decision rule, fixed in advance: chunking must beat the preallocated full call by more than 5% for
the recommendation to stand.
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
stamp('L2 分块：混淆排查（分配 vs 缓存）')
torch.set_num_threads(6)
print()

SHAPES = [(2048, 2736, 512), (2048, 512, 2736), (2048, 2048, 512)]


def bench(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


print('  %-16s %10s %12s %12s %12s' % ('形状', '全量新分配', '全量预分配', '分块预分配', '分块/预分配'))
print('  ' + '-' * 68)
for (M, N, K) in SHAPES:
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    C = torch.empty(M, N)
    fl = 2.0 * M * N * K

    t1 = bench(lambda: torch.mm(A, B))
    t2 = bench(lambda: torch.mm(A, B, out=C))

    def chunked(k=512):
        for j in range(0, N, k):
            torch.mm(A, B[:, j:j + k], out=C[:, j:j + k])
    t3 = bench(chunked)

    print('  %-16s %9.1f %11.1f %11.1f %11.3fx'
          % ('%dx%dx%d' % (M, N, K), fl / t1 / 1e9, fl / t2 / 1e9, fl / t3 / 1e9, t2 / t3))
print()
print('  判据（事先固定）: 分块 vs 预分配全量 > 5% ⇒ 是缓存分块，值得实现；')
print('                    否则 1.359× 来自分配，7.52.1 的建议作废。')
print()
print('  参考: 7.52.1 记的是 1.359×（分块 vs 【新分配】全量）')
