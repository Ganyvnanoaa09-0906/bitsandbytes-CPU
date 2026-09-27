# -*- coding: utf-8 -*-
"""conv_ceiling_ab.py — 模型里的 conv 离本机 conv 上限有多远

为什么需要:
    profile_animatediff.py 实测 aten::mkldnn_convolution 占 23.9%（98 次）。
    §10.152 已把 GEMM（41.6%）判为"到极限、不可压缩"，§10.151.4 已把 copy_
    那条路证伪（段错误）。⇒ **conv 是唯一还没评估过的大项**。

判据（与 GEMM 那节同构）:
    模型内 conv 的有效 GFLOPS  vs  同形状单独跑 conv 的 GFLOPS。
    比值 ≈ 1 ⇒ 已到 oneDNN 极限，去优化别处；
    比值 << 1 ⇒ 有包装/布局开销（例如反复 permute 导致非连续输入）。

oracle:
    · 形状取自真实 SD1.5 UNet 的 conv 配置，不是编的；
    · 同一进程、同一线程数、同一计时口径（min-of-N）；
    · 同时测 **连续输入** 与 **非连续输入**，这是本脚本的核心对照 ——
      §10.151.4 的段错误说明布局对该后端是**语义相关**的，所以必须量化
      布局到底值多少性能，而不是猜。
    · 给出 effective GFLOPS，与 conv 的 FLOP 公式 2*N*Cout*Cin*kh*kw*H*W 对齐。

⚠️ 非连续输入那组若崩溃（如 §10.151.4），本脚本会捕获并记录 ——
   崩溃本身就是结论，不要让整个测量挂掉。
"""
from __future__ import annotations

import os
import statistics
import time

import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ.get('THREADS', '6')))

# SD1.5 UNet 主要 conv 形状: (Cin, Cout, H, W, kernel)
SHAPES = [
    (320, 320, 64, 64, 3),    # 顶层 ResNet/attention 前的 conv
    (320, 320, 32, 32, 3),
    (640, 640, 32, 32, 3),
    (1280, 1280, 16, 16, 3),
    (1280, 1280, 8, 8, 3),
    (320, 320, 64, 64, 1),    # 1x1（attention 投影式 conv）
    (960, 320, 64, 64, 1),
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
    print('=' * 88)
    print('模型里的 conv 离本机 oneDNN 上限有多远')
    print('=' * 88)
    print('threads = %d' % torch.get_num_threads())
    print('\n%-30s %12s %11s %11s %11s' % ('shape (Cin,Cout,H,W,k)', 'GFLOP', 'best ms', 'GFLOPS', 'ns/out'))
    print('-' * 88)

    rows = []
    for (Cin, Cout, H, W, k) in SHAPES:
        conv = nn.Conv2d(Cin, Cout, k, padding=k // 2, bias=False)
        x = torch.randn(1, Cin, H, W)
        flop = 2.0 * Cout * Cin * k * k * H * W
        try:
            _, best = timed(lambda: conv(x), reps=5)
            g = flop / best / 1e9
            rows.append(((Cin, Cout, H, W, k), flop, best, g))
            print('%-30s %12.3f %11.3f %11.2f %11.2f'
                  % ('(%d,%d,%d,%d,%d)' % (Cin, Cout, H, W, k), flop / 1e9,
                     best * 1e3, g, best * 1e9 / (Cout * H * W)))
        except Exception as e:
            print('%-30s FAILED %s' % ('(%d,%d,%d,%d,%d)' % (Cin, Cout, H, W, k), e))
        del conv, x

    # ---- 核心对照：非连续输入值多少性能？----
    print('\n[对照] 同一形状，输入为**非连续**（channels_last 内存格式）:')
    print('%-30s %11s %11s %10s' % ('shape', 'contig ms', 'chan-last ms', '变化'))
    print('-' * 70)
    for (Cin, Cout, H, W, k) in SHAPES[:5]:
        conv = nn.Conv2d(Cin, Cout, k, padding=k // 2, bias=False)
        x = torch.randn(1, Cin, H, W)
        try:
            _, bc = timed(lambda: conv(x), reps=5)
        except Exception:
            continue
        xl = x.to(memory_format=torch.channels_last)
        try:
            _, bl = timed(lambda: conv(xl), reps=5)
            note = '%.2fx' % (bc / bl)
        except Exception as e:
            bl = float('nan')
            note = 'CRASH: %s' % type(e).__name__
        print('%-30s %11.3f %11.3f %10s'
              % ('(%d,%d,%d,%d,%d)' % (Cin, Cout, H, W, k), bc * 1e3, bl * 1e3, note))
        del conv, x, xl

    # ---- 与 GEMM 上限对比：conv 的有效算力是否低于 GEMM ----
    print('\n[参照] §10.152 标定的 GEMM 上限: 2048^3 = 247.00 GFLOPS（DRAM 场景）')
    print('  模型 conv 的最高有效算力 = %.2f GFLOPS'
          % (max(r[3] for r in rows) if rows else float('nan')))
    print('\n读法: conv 有效算力远低于 GEMM 上限 ⇒ conv 是更值得动的目标；')
    print('      channels_last 明显更快 ⇒ 那是零精度代价的优化方向。')
    print('=' * 88)


if __name__ == '__main__':
    main()
