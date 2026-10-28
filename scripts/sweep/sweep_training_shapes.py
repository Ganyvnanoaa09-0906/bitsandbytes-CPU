"""sweep_training_shapes.py -- why does the model's own matmul run at 115 GFLOPS?

The cache-resident peak on this machine is 269 GFLOPS and training sustains 114-198
depending on batch. The profile says aten::mm is 55.78% of a step, and a step is about
1.5 TFLOP (30.5M parameters x 2 x 8192 tokens x 3 for forward and backward), so the mm
operators themselves run near 115 GFLOPS -- the shortfall is inside the matmuls, not in
the surrounding operators. That rules out the attention and activation overhead as the
explanation.

What differs between the two cases is shape and size:

    peak measurement   768 x 768 x 768      all three dimensions equal, 6.9 MB resident
    training           M = B*T = 8192, K = 512, N = 512..2048

M is large and the output is 8192 x 2048 x 4 = 67 MB for the widest layer, which does not
fit in the 8 MB L3 and must be written to memory. Arithmetic intensity is not the issue --
2MNK/(MK+KN+MN) is near 1000 for these shapes -- so if something is limiting, it is the
cost of streaming large operands and results rather than a lack of reuse.

Note also that batch 2 sustained 198.5 while batch 8 sustains 114, i.e. smaller M is
faster per FLOP. That is the opposite of what a compute-bound kernel does and points at
the same thing.

This sweeps the model's actual shapes, plus M and N varied independently, so the cause can
be separated from the coincidence of a square peak benchmark.
"""
from __future__ import annotations
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

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

torch.set_num_threads(6)


def bench(M, N, K, reps=None):
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    c = torch.empty(M, N)
    torch.mm(a, b, out=c)
    if reps is None:
        reps = max(3, int(2e9 / (2.0 * M * N * K)))
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps):
            torch.mm(a, b, out=c)
        ts.append((time.perf_counter() - t0) / reps)
    dt = statistics.median(ts)
    gf = 2.0 * M * N * K / dt / 1e9
    mb = (M * K + K * N + M * N) * 4 / 2**20
    return dt, gf, mb


print('=== 模型的真实形状（batch 8, T=1024 ⇒ M=8192） ===')
print('  %-22s %10s %10s %10s' % ('形状', '毫秒', 'GFLOPS', '总字节 MB'))
print('  ' + '-' * 56)
for (M, N, K, label) in [
        (8192, 512, 512, 'q/k/v/o proj'),
        (8192, 2048, 512, 'MLP up'),
        (8192, 512, 2048, 'MLP down'),
        (8192, 1024, 512, 'head'),
]:
    dt, gf, mb = bench(M, N, K)
    print('  %-22s %10.2f %10.1f %10.1f' % (label, dt * 1e3, gf, mb))

print()
print('=== batch 2（M=2048，训练实测 198.5 的那个） ===')
for (M, N, K, label) in [
        (2048, 512, 512, 'q/k/v/o proj'),
        (2048, 2048, 512, 'MLP up'),
        (2048, 512, 2048, 'MLP down'),
]:
    dt, gf, mb = bench(M, N, K)
    print('  %-22s %10.2f %10.1f %10.1f' % (label, dt * 1e3, gf, mb))

print()
print('=== 单独变 M：K=N=512 固定 ⇒ 看 M 的影响 ===')
print('  %-12s %10s %10s %10s' % ('M', '毫秒', 'GFLOPS', 'MB'))
for M in (256, 512, 1024, 2048, 4096, 8192, 16384):
    dt, gf, mb = bench(M, 512, 512)
    print('  %-12d %10.3f %10.1f %10.1f' % (M, dt * 1e3, gf, mb))

print()
print('=== 单独变 K：M=8192, N=512 固定 ⇒ K 小是算术强度低，K 大是缓存压力 ===')
print('  %-12s %10s %10s %10s' % ('K', '毫秒', 'GFLOPS', 'MB'))
for K in (128, 256, 512, 1024, 2048, 4096):
    dt, gf, mb = bench(8192, 512, K)
    print('  %-12d %10.3f %10.1f %10.1f' % (K, dt * 1e3, gf, mb))

print()
print('=== 单独变 N：M=8192, K=512 固定 ===')
print('  %-12s %10s %10s %10s' % ('N', '毫秒', 'GFLOPS', 'MB'))
for N in (128, 256, 512, 1024, 2048, 4096):
    dt, gf, mb = bench(8192, N, 512)
    print('  %-12d %10.3f %10.1f %10.1f' % (N, dt * 1e3, gf, mb))

print()
print('参照: cache 内峰值 269 GFLOPS（768 立方）；训练实测 114（batch 8）/ 198.5（batch 2）')
