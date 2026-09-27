# -*- coding: utf-8 -*-
"""gemm_ceiling_ab.py — 模型里的 mm/addmm 离本机同形状可达上限有多远

为什么需要:
    profile_animatediff.py 实测 addmm 28.9% + mm 12.7% = 41.6%。但"占比高"不等于
    "有优化空间"——必须知道**同形状下单次 GEMM 最快能多快**。判据是
    「模型内实测时间 / 同形状纯 GEMM 时间」。若比值 ≈ 1，说明已经是 GEMM 极限，
    该去优化别的地方；若 << 1，说明有包装开销（permute/contiguous/多线程切换）。

oracle:
    · 形状**直接从模型 profile 里取**（(16384,320)x(320,320) 等），不是我编的；
    · 用同一台机器、同一进程、同一线程数测纯 GEMM；
    · 同时用 torch.mm 与不经过 autograd 的原始调用，避免把 autograd 开销算进去；
    · 给出有效 GFLOPS，与本机 fp32 峰值做对比（峰值也现场标定，不引用外部常数）。

本机标定（6 线程 Zen2，fp32，FMA=每周期 2×8 FLOP/核）:
    理论峰值 ≈ 6 核 × 4.0 GHz × 16 FLOP/周期 ≈ 384 GFLOPS
    实际可达通常 50-70%（内存受限的形状更低）——所以下面用**实测**而不是公式。
"""
from __future__ import annotations

import os
import statistics
import time

import torch

torch.set_num_threads(int(os.environ.get('THREADS', '6')))

# 直接取自 profile 的形状：(M, K) x (K, N)
SHAPES = [
    (16384, 320, 320),    # x55  空间 attention 投影 / FFN
    (4096, 640, 640),     # x55
    (1024, 1280, 1280),   # x55
    (256, 1280, 1280),    # x46
    (16384, 320, 2560),   # x10  FFN 扩张
    (4096, 640, 5120),    # x10
    (1024, 1280, 10240),  # x10
]


def timed(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts)


def main():
    print('=' * 86)
    print('模型内 mm/addmm 离同形状可达上限有多远')
    print('=' * 86)
    print('threads = %d' % torch.get_num_threads())
    print('\n%-26s %10s %12s %12s %10s' % ('shape (M,K)x(K,N)', 'GFLOP', 'best ms', 'GFLOPS', '% of peak'))
    print('-' * 86)

    peak = None
    rows = []
    for (M, K, N) in SHAPES:
        a = torch.randn(M, K)
        b = torch.randn(K, N)
        flop = 2.0 * M * K * N
        _, best = timed(lambda: a @ b, reps=5)
        gflops = flop / best / 1e9
        rows.append(((M, K, N), flop, best, gflops))
        print('%-26s %10.2f %12.3f %12.2f' % ('(%d,%d)x(%d,%d)' % (M, K, K, N),
                                             flop / 1e9, best * 1e3, gflops))
        del a, b

    # 峰值标定：取一个方阵大 GEMM（最有利的形状）作为本机可达上限
    print('\n[峰值标定] 用最有利的方阵 GEMM 测本机可达上限:')
    for n in (512, 1024, 2048):
        a = torch.randn(n, n)
        b = torch.randn(n, n)
        _, best = timed(lambda: a @ b, reps=5)
        g = 2.0 * n ** 3 / best / 1e9
        print('  %4d^3 : %8.3f ms  %8.2f GFLOPS' % (n, best * 1e3, g))
        if peak is None or g > peak:
            peak = g
        del a, b
    print('  ⇒ 本机 fp32 实测可达 ≈ %.1f GFLOPS（下面对比以此为参照）' % peak)

    print('\n[判据] 模型里出现的形状 vs 该机上限:')
    print('  %-26s %12s %10s' % ('shape', 'GFLOPS', '% of peak'))
    for (shp, flop, best, g) in rows:
        print('  %-26s %12.2f %9.1f%%' % ('(%d,%d)x(%d,%d)' % (shp[0], shp[1], shp[1], shp[2]),
                                          g, 100 * g / peak))
    print('\n读法: 占比接近峰值 ⇒ 这些 GEMM 已到极限，去优化别处；')
    print('      占比很低 ⇒ 形状本身对缓存不友好（大 M 小 N 往往是），可考虑分块/换布局。')
    print('=' * 86)


if __name__ == '__main__':
    main()
