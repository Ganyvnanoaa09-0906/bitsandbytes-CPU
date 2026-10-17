"""Is there headroom in the GEMM blocking that torch already does?

The measured facts: 371 GFLOPS at n=384 where the working set fits L2, 230 at n=2048 where it does
not. That gap is real and the obvious reading is that blocking for L2 residency would recover it.

Two things argue against that reading. The largest output matrix in the model is [2048, 2736] at
22.4 MB, against 8 MB of L3 -- the output alone does not fit, so no amount of input blocking keeps
the working set on chip. And torch's GEMM goes through MKL or oneDNN, both of which already block
for L2 and L3 internally; the 240 GFLOPS at our shapes is the blocked result.

What remains is an empirical question rather than an argument: can a different blocking do better
than torch's? Split the output columns, so B's chunk and C's chunk are both small, and sweep the
chunk size. If chunking beats the full call, there is headroom. If it does not, torch is already at
the ceiling for these shapes and the direction is closed with a measurement rather than a prediction.

Also tested: whether the 259 ns pointer-chase latency has any bearing on this at all. That figure
came from a latency-bound microbenchmark, and applying it to a compute-bound GEMM is exactly the
kind of mismatch this project keeps making, so the arithmetic intensity is printed alongside the
timings to make the distinction explicit.

Shapes are the model's real ones. Decision rule fixed in advance: chunking must beat the full call by
more than 5% for this to be worth pursuing.
"""
from __future__ import annotations

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
stamp('L2 驻留分块：能否胜过 torch 自己的分块')
torch.set_num_threads(6)
print()

SHAPES = [(2048, 512, 512), (2048, 2736, 512), (2048, 512, 2736), (2048, 2048, 512)]

print('%-18s %10s %12s %10s' % ('形状 MxNxK', '输出 MB', '算术强度', '全量 GFLOPS'))
print('-' * 56)
for (M, N, K) in SHAPES:
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    out_mb = M * N * 4 / 1024 ** 2
    ai = 2.0 * M * N * K / ((M * K + K * N + M * N) * 4)
    for _ in range(2):
        torch.mm(A, B)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        torch.mm(A, B)
        ts.append(time.perf_counter() - t0)
    gf = 2.0 * M * N * K / statistics.median(ts) / 1e9
    print('%-18s %9.1fM %11.0f %10.1f' % ('%dx%dx%d' % (M, N, K), out_mb, ai, gf))
print()
print('  ⇒ 带宽受限门槛 = 240 GFLOPS / 27 GB/s ≈ 9 FLOP/字节')
print('    四个形状都远在其上 ⇒ 【计算受限】,所以那 259 ns 的延迟数字与此无关')
print()

M, N, K = 2048, 2736, 512
A = torch.randn(M, K)
B = torch.randn(K, N)
for _ in range(2):
    torch.mm(A, B)
ts = []
for _ in range(5):
    t0 = time.perf_counter()
    C = torch.mm(A, B)
    ts.append(time.perf_counter() - t0)
base_t = statistics.median(ts)
base = 2.0 * M * N * K / base_t / 1e9
print('=== 分块累加扫描（%dx%dx%d）===' % (M, N, K))
print('  基线 torch.mm: 【%.1f GFLOPS】' % base)
print()
print('  %-14s %12s %10s %8s' % ('分块 k', '每块 MB', 'GFLOPS', '相对'))
print('  ' + '-' * 48)
best = base
for k in (64, 128, 256, 342, 512, 684, 1024):
    if k > N:
        continue
    per_mb = (K * k + M * k) * 4 / 1024 ** 2
    try:
        for _ in range(1):
            Cc = torch.empty(M, N)
            for j in range(0, N, k):
                Cc[:, j:j + k] = torch.mm(A, B[:, j:j + k])
        ts = []
        for _ in range(3):
            Cc = torch.empty(M, N)
            t0 = time.perf_counter()
            for j in range(0, N, k):
                Cc[:, j:j + k] = torch.mm(A, B[:, j:j + k])
            ts.append(time.perf_counter() - t0)
        gf = 2.0 * M * N * K / statistics.median(ts) / 1e9
        best = max(best, gf)
        print('  %-14d %11.2fM %10.1f %7.3fx' % (k, per_mb, gf, gf / base))
    except Exception as e:
        print('  %-14d 失败 %s' % (k, str(e)[:34]))
print()
print('  判据（事先固定）: 分块比全量快 > 5% ⇒ 有空间；否则 torch 已是上限，方向关闭')
print('  ⇒ 最好结果 %.3fx ⇒ %s' % (best / base, '【有空间】✓' if best / base > 1.05 else '【关闭】✗'))
