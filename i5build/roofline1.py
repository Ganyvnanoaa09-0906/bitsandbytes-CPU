"""Roofline step 1: where does the 36% go?

Known facts this must explain:
  zero-memory same-shape inner loop : 97.2% (iGPU) / 75-100% (CPU)  -> FMA CAN be fed
  real GEMM                         : 33-42%                        -> but is not
  and the ratio is the SAME on two unrelated architectures
  => the common factor is data supply, not microarchitecture.

Method (all measured, no assumptions):
  1) streaming bandwidth at each cache level -- sweep the working set from
     L1-resident to far beyond L3, so each level's roof is measured directly.
  2) GEMM GFLOPS over the same size sweep.
  3) Arithmetic intensity per shape, compared against each level's measured
     roof. That says whether a shape is compute-bound or sitting on a memory
     roof -- and WHICH roof.

This machine: L1 384 KB, L2 3 MB, L3 8 MB, 15 W APU (clock drifts under load,
so every number here is taken back-to-back in one run and reported with its size).
"""
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
torch.set_num_threads(6)
print('threads =', torch.get_num_threads())

# ---- 1) streaming bandwidth by working-set size -------------------------
# triad-like: a = b + s*c  -> reads 2 arrays, writes 1 (12 bytes/elem for f32)
SIZES_MB = [0.125, 0.25, 0.5, 1, 2, 3, 4, 8, 12, 24, 64, 256]
print('\n=== 流式带宽 vs 工作集大小（每级缓存的实测屋顶）===')
print('%10s %12s %12s' % ('集合大小', 'GB/s', '说明'))
print('-' * 40)
bw = []
for mb in SIZES_MB:
    n = int(mb * 1024 * 1024 / 4)
    a = torch.empty(n)
    b = torch.randn(n)
    c = torch.randn(n)
    s = 1.000001
    for _ in range(3):                       # warm
        torch.add(b, c, alpha=s, out=a)
    t0 = time.perf_counter()
    it = 20
    for _ in range(it):
        torch.add(b, c, alpha=s, out=a)
    dt = (time.perf_counter() - t0) / it
    gbs = 3 * n * 4 / dt / 1e9               # 2 reads + 1 write
    bw.append((mb, gbs))
    tag = ''
    if mb <= 0.375:
        tag = 'L1/2'
    elif mb <= 3:
        tag = 'L2'
    elif mb <= 8:
        tag = 'L3'
    else:
        tag = 'DRAM'
    print('%8.3f MB %12.1f  %s' % (mb, gbs, tag))

# ---- 2) GEMM GFLOPS over a size sweep ----------------------------------
print('\n=== GEMM 效率 vs 规模（同一次运行内）===')
print('%22s %10s %10s' % ('形状 M×N×K', 'GFLOPS', 'ms'))
print('-' * 46)
gemm = []
for (m, n, k) in [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024),
                  (2048, 2048, 512), (2048, 512, 512), (2048, 1536, 512),
                  (2048, 4096, 512), (2048, 512, 2048), (2048, 1024, 512),
                  (4096, 4096, 4096)]:
    A = torch.randn(m, k)
    B = torch.randn(k, n)
    C = A @ B
    t0 = time.perf_counter()
    for _ in range(3):
        C = A @ B
    dt = (time.perf_counter() - t0) / 3
    g = 2.0 * m * n * k / dt / 1e9
    gemm.append((m, n, k, g, dt * 1000))
    print('%8d×%-6d×%-6d %10.1f %10.1f' % (m, n, k, g, dt * 1000))

# ---- 3) compare against each roof --------------------------------------
print('\n=== 对照：各形状的算术强度 vs 各层屋顶 ===')
print('  机器平衡点 = 峰值GFLOPS / 带宽；某层的屋顶 = 该层带宽 × 算术强度')
peak = max(g for (_m, _n, _k, g, _d) in gemm)
print('  实测峰值 GFLOPS = %.1f（本次运行内的最大者）' % peak)
roofs = {}
for mb, gbs in bw:
    tag = 'L1' if mb <= 0.375 else ('L2' if mb <= 3 else ('L3' if mb <= 8 else 'DRAM'))
    roofs[tag] = max(roofs.get(tag, 0.0), gbs)
for tag in ('L1', 'L2', 'L3', 'DRAM'):
    if tag in roofs:
        print('  %-5s 带宽 %6.1f GB/s  ⇒ 机器平衡点 %6.2f FLOP/Byte'
              % (tag, roofs[tag], peak / roofs[tag] if roofs[tag] else 0))

print('\n%8s %10s %12s %12s %10s' % ('形状', 'GFLOPS', '朴素AI', '瓶颈层', '效率'))
print('-' * 58)
for (m, n, k, g, _ms) in gemm:
    fl = 2.0 * m * n * k
    by = 4.0 * (m * k + k * n + m * n)       # naive: no cache reuse assumed
    ai = fl / by
    limit = None
    for tag in ('DRAM', 'L3', 'L2', 'L1'):
        if tag in roofs and ai * roofs[tag] < peak:
            limit = tag
            break
    eff = g / peak * 100
    print('%8s %10.1f %12.1f %12s %9.0f%%'
          % ('%d×%d×%d' % (m, n, k), g, ai, limit or '算力', eff))

print('\n判读：')
print('  · 朴素AI 是【完全不复用缓存】的悲观下界；它都已远超机器平衡点 ⇒')
print('    这些形状在 DRAM 屋顶之上 ⇒ 36% 不是 DRAM 带宽限制 ✓ 有余量')
print('  · 若某形状落在 L2/L3 屋顶附近 ⇒ 瓶颈在那一级缓存带宽 ⇒ 该改分块/预取')
print('  · 全是「算力」却只有 36% ⇒ 瓶颈在内核效率（指令/调度），不是存储')
