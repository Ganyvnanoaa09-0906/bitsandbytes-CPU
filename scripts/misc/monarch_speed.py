"""Does Monarch actually beat a dense matmul on this CPU? The self-check passed, so now time it.

The theory says operations fall from mn to roughly b(m+n), a 10-28x reduction at the model's shapes
as the parameter counts show. But parameter count is not speed on a CPU: the blockwise einsum runs
many small matmuls instead of one large one, and small matmuls are exactly where this machine is
weak. Earlier in this project the batched Newton-Schulz was 11% slower than a serial loop for the
same total work, and the iGPU was 1.55x slower than the CPU on small matrices. So the reduction in
FLOPs may not survive contact with the BLAS dispatch overhead.

Measured at the two shapes that matter, 512->2048 and 2048->512 being the MLP's two layers with w12
at width 2736, forward only and forward plus backward. A Monarch layer that is not faster than
nn.Linear is not worth wiring into the model however good its parameter count looks.

Decision rule, fixed in advance: the layered version must be at least 2x faster in wall clock at the
model's batch size for the replacement to be worth building into the trainer. Below 2x the FLOP
reduction is being eaten by dispatch, and the effort belongs elsewhere.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import statistics
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


from measure_guard import assert_idle, stamp  # noqa: E402
from monarch_layer_v2 import MonarchLinear  # noqa: E402

assert_idle()
stamp('Monarch 对稠密：CPU 上真的更快吗')
torch.set_num_threads(6)
print()

# the model's real shapes: batch 2 x 1024 tokens = 2048 rows
CASES = [(2048, 512, 512), (2048, 512, 2048), (2048, 2048, 512), (2048, 512, 2736), (2048, 2736, 512)]


def timeit(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


print('  %-16s %12s %12s %10s %12s' % ('批次×n->m', '稠密 ms', 'Monarch ms', '倍数', 'FLOPs 比'))
print('  ' + '-' * 68)
verdict = []
for (rows, n, m) in CASES:
    x = torch.randn(rows, n)
    dense = nn.Linear(n, m)
    mon = MonarchLinear(n, m)
    with torch.no_grad():
        td = timeit(lambda: dense(x))
        tm = timeit(lambda: mon(x))
    fl_d = rows * n * m
    fl_m = rows * (n * mon.b_r + m * mon.b_l)
    print('  %-16s %12.3f %12.3f %9.2fx %11.1fx'
          % ('%dx%d->%d' % (rows, n, m), td * 1e3, tm * 1e3, td / tm, fl_d / fl_m))
    verdict.append((('%dx%d->%d' % (rows, n, m)), td / tm))
print()
print('  判据（事先固定）: 壁钟 ≥ 2× 才值得接进训练器；否则 FLOPs 削减被调度开销吃掉')
ok = all(v >= 2.0 for _, v in verdict)
print('  ⇒ %s' % ('【值得】✓' if ok else '【不值得】✗ 最好的一档是 %.2fx' % max(v for _, v in verdict)))
print()
print('  各形状:' + ' · '.join('%s %.2fx' % (k, v) for k, v in verdict))
