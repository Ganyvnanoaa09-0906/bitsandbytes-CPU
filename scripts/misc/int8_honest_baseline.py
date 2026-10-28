"""The honest baseline: torch's fp32 GEMM at the same shapes the int8 kernel runs.

The C++ test compared the int8 kernel against a naive fp32 triple loop and got 21x to 72x. That number is
real but it is the wrong denominator, and quoting it would be the exact error this project has spent the
day cataloguing: a real measurement of the wrong thing. A naive BLAS runs at about 1.6 GFLOPS here; torch
runs at 200 or more. Against torch the int8 kernel's 117 GOPS may well lose.

So this measures the denominator that matters. It is written separately from the C++ so the two cannot
share a mistake in the harness, and the numbers are printed side by side with the naive figures from the
C++ so the gap between "naive" and "tuned" is visible rather than hidden in a ratio.
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

torch.set_num_threads(5)

# 与 C++ 测试里那两个形状对应。int8 的数字从那次运行抄来,并注明。
SHAPES = [
    (1, 4096, 4096, 1.05, 22.20),      # M, N, K, int8 ms, fp32 naive ms
    (128, 1024, 512, 1.15, 83.00),
]

print()
print('  %-16s %10s %12s %12s %14s' % ('MxNxK', 'GOP', 'int8 ms', '朴素fp32 ms', 'torch fp32 ms'))
print('  ' + '-' * 70)
rows = []
for (M, N, K, ti, tn) in SHAPES:
    A = torch.randint(0, 128, (M, K), dtype=torch.uint8).float()
    W = torch.randint(-128, 128, (N, K), dtype=torch.int8).float()
    for _ in range(2):
        A @ W.t()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        A @ W.t()
        ts.append(time.perf_counter() - t0)
    tt = statistics.median(ts) * 1e3
    gop = 2.0 * M * N * K / 1e9
    rows.append((M, N, K, gop, ti, tn, tt))
    print('  %-16s %10.3f %12.2f %12.2f %14.2f'
          % ('%dx%dx%d' % (M, N, K), gop, ti, tn, tt))

print()
print('  === 三种基线的诚实对比 ===')
print('  %-16s %12s %12s %12s' % ('MxNxK', 'int8 内核', 'torch fp32', 'int8/torch'))
print('  ' + '-' * 56)
for (M, N, K, gop, ti, tn, tt) in rows:
    gi = gop / ti * 1e3          # GOPS
    gt = gop / tt * 1e3          # GFLOPS
    print('  %-16s %9.1f GOPS %9.1f GFLOPS %10.2f×'
          % ('%dx%dx%d' % (M, N, K), gi, gt, ti / tt))

print()
print('  ⇒ 那个 "21×/72×" 是【朴素对朴素】⇒ 它量的是【指令层面的差】,不是库对库的差')
print('  ⇒ ★ 而对着 torch 的调优 BLAS,上面第三列才是真的')
print()
print('  ★ 而这不是"int8 没用",是"这个 int8 内核【还没做分块和打包】":')
print('    · 没有寄存器分块 ⇒ 每次只有 8 个 int32 累加器,而它可以用 12~16 个')
print('    · 没有 L2/L3 分块 ⇒ 大 K 时权重复用不起来')
print('    · 没有权重预打包 ⇒ 每次都要按行取 W,而 W 在 K 维上是连续的（这点已经是对的了）')
print()
print('  ⇒ ★【所以下一步是给它分块,而不是宣布 int8 更快或更慢】')
print()
