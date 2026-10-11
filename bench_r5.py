# -*- coding: utf-8 -*-
"""bench_r5.py — R5-4500U 的两个硬件级瓶颈探测。

1) 内存带宽（我们的负载是带宽敏感的：batch 加大吞吐不涨就是证据）
   - 双通道 DDR4-2667 理论约 42.6 GB/s
   - 若实测远低于此，或在 3200 条子上只跑 2667，就是可优化空间
2) 持续 AVX2 负载下是否降频（15W TDP 的笔记本芯片，6核满载 AVX2 很可能掉到基频）
   - 连续跑 sgemm，每 10 秒报告一次 GFLOPS，看是否随时间衰减
"""
import time
import torch

torch.set_num_threads(6)


def bench(fn, iters, warm=2):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


print('=' * 64, flush=True)
print('R5-4500U 硬件瓶颈探测', flush=True)
print('=' * 64, flush=True)

# ---------------- 1. 内存带宽 ----------------
print()
print('--- 内存带宽 ---', flush=True)
print('  双通道 DDR4-2667 理论 ≈ 42.6 GB/s；DDR4-3200 理论 ≈ 51.2 GB/s', flush=True)

N = 64 * 1024 * 1024          # 256 MB float32
a = torch.empty(N, dtype=torch.float32)
b = torch.empty(N, dtype=torch.float32)
a.fill_(1.0)

# 读+写（copy）：流量 = 2 × N × 4 字节
dt = bench(lambda: b.copy_(a), iters=8)
gbps = 2 * N * 4 / dt / 1e9
print('  copy_   256MB: %6.1f GB/s   (%.3f s)' % (gbps, dt), flush=True)

# 纯读（sum 归约）：流量 ≈ N × 4
dt = bench(lambda: a.sum(), iters=8)
gbps_r = N * 4 / dt / 1e9
print('  sum     256MB: %6.1f GB/s   (%.3f s)' % (gbps_r, dt), flush=True)

# 三输入运算：流量 ≈ 3 × N × 4
c = torch.empty(N, dtype=torch.float32)
dt = bench(lambda: torch.add(a, b, out=c), iters=8)
gbps_a = 3 * N * 4 / dt / 1e9
print('  add     256MB: %6.1f GB/s   (%.3f s)' % (gbps_a, dt), flush=True)

del a, b, c

# ---------------- 2. 降频探测 ----------------
print()
print('--- 持续 AVX2 负载降频探测（连续 90 秒 sgemm）---', flush=True)
n = 1024
A = torch.randn(n, n)
B = torch.randn(n, n)
A @ B
print('  %8s %12s %10s' % ('时刻', 'GFLOP/s', '相对'), flush=True)
first = None
t_start = time.perf_counter()
for i in range(9):
    t0 = time.perf_counter()
    reps = 0
    while time.perf_counter() - t0 < 10.0:
        A @ B
        reps += 1
    el = time.perf_counter() - t0
    gf = reps * 2 * n ** 3 / el / 1e9
    if first is None:
        first = gf
    print('  %6.0fs %12.2f %9.1f%%' % (time.perf_counter() - t_start, gf, 100 * gf / first), flush=True)

print()
print('  若"相对"持续下滑 → 功耗墙降频，那是硬件限制，软件无解', flush=True)
print('  若基本持平 → 没有降频问题', flush=True)
print()
print('DONE', flush=True)
