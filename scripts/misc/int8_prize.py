"""How big is the INT8 prize? Measure torch._int_mm's AVX2 fallback against what it should cost.

ops.py notes that without AVX-512 torch._int_mm uses a scalar fallback. That is the gap the AVX2
integer dot-product instructions would fill. Before writing a kernel, measure what the fallback
actually costs and what the ceiling would be, because the prize could be small.

Method: time torch._int_mm at shapes the model would use, and compare against an fp32 matmul of the
same shape. An int8 kernel moves a quarter of the bytes of fp32 and can do 2-4 dot products per
instruction, so if the scalar fallback is only slightly slower than fp32 there is little to win; if it
is far slower, the gap is real.

Also check whether anything in our training path even reaches _int_mm. If nothing does, then improving
it helps other people's models and not ours, which changes whether it is worth this project's time.

Criterion, fixed in advance: if the fallback is more than 2x slower than fp32 at these shapes, the
gap is worth a kernel. If it is comparable, then the AVX2 instructions would give roughly the fp32
rate at a quarter of the memory, which is a memory win rather than a speed one.
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

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


from measure_guard import assert_idle, stamp  # noqa: E402

assert_idle()
stamp('INT8 的奖金有多大：_int_mm 的 AVX2 回退实测')
torch.set_num_threads(6)
print()

print('  torch %s · CPU: %s' % (torch.__version__, torch.backends.cpu.get_cpu_capability()))
print()

SHAPES = [(2048, 2736, 512), (2048, 512, 2048), (1, 4096, 4096), (8, 4096, 4096)]


def bench(fn, reps=4, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


print('  %-18s %13s %13s %10s %12s' % ('形状 MxNxK', 'int8 ms', 'fp32 ms', 'int8/fp32', 'int8 GOPS'))
print('  ' + '-' * 72)
rows = []
for (M, N, K) in SHAPES:
    a8 = torch.randint(-127, 127, (M, K), dtype=torch.int8)
    b8 = torch.randint(-127, 127, (K, N), dtype=torch.int8)
    af = torch.randn(M, K)
    bf = torch.randn(K, N)
    try:
        _ = torch._int_mm(a8, b8)
        t8 = bench(lambda: torch._int_mm(a8, b8))
    except Exception as e:
        print('  %-18s _int_mm 不可用: %s' % ('%dx%dx%d' % (M, N, K), str(e)[:40]))
        continue
    tf = bench(lambda: torch.mm(af, bf))
    ops = 2.0 * M * N * K
    rows.append((M, N, K, t8, tf))
    print('  %-18s %12.3f %12.3f %9.2fx %11.2f'
          % ('%dx%dx%d' % (M, N, K), t8 * 1e3, tf * 1e3, t8 / tf, ops / t8 / 1e9))
print()
if rows:
    worst = max(r[3] / r[4] for r in rows)
    best = min(r[3] / r[4] for r in rows)
    print('  int8/fp32 的比值范围: %.2f× ~ %.2f×' % (best, worst))
    print()
    print('  判据（事先固定）: 回退比 fp32 慢 >2× ⇒ 值得写内核')
    print('  ⇒ %s' % ('【值得】✓' if best > 2.0 else
                      '【不值得 —— 差距不大。而这意味着 AVX2 的整数指令即使在，'
                      '也只是把 int8 拉到 fp32 的水平，收益是【内存】而不是速度】✗'))
print()
print('  参考: AVX2 的 _mm256_maddubs_epi16 每条做 32 个乘加（两个 int16 结果），')
print('        配合 _mm256_madd_epi16 归约 ⇒ 理论上是 fp32 FMA 的 2~4 倍吞吐')
