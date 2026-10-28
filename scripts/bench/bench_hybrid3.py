# -*- coding: utf-8 -*-
"""bench_hybrid3.py — 同条件（持续负载）对比 CPU vs 核显，在真实 FFN 形状上。

上一版错误：拿 CPU 的【爆发】比核显，而 CPU 持续只有爆发的 37%。
本版两边都跑 40 秒，只统计后 20 秒稳态。

修 bug：ffn_tasks 循环变量 k 覆盖了注意力张量 k（改名 kk）。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import threading
import time
import numpy as np
import pyopencl as cl
import torch
import torch.nn.functional as F

torch.set_num_threads(6)
platform = cl.get_platforms()[0]
dev = platform.get_devices()[0]
ctx = cl.Context([dev])
queue = cl.CommandQueue(ctx)

print('=' * 78)
print('同条件（持续稳态）对比  |  核显:', dev.name, '| CPU 6 线程')
print('=' * 78)

KERNEL = open('bench_hybrid2.py', encoding='utf-8').read()
i0 = KERNEL.index("KERNEL = r'''") + len("KERNEL = r'''")
i1 = KERNEL.index("'''", i0)
prg = cl.Program(ctx, KERNEL[i0:i1]).build()
K8 = cl.Kernel(prg, 'sgemm8')


def cl_setup(M, N, Kd):
    A = np.random.randn(M, Kd).astype(np.float32)
    B = np.random.randn(Kd, N).astype(np.float32)
    mf = cl.mem_flags
    ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    cb = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    gsz = ((N + 63) // 64 * 8, (M + 63) // 64 * 8)
    return (np.int32(M), np.int32(N), np.int32(Kd), ab, bb, cb), gsz


def sustained_cpu(M, N, Kd, secs=40, tail=20):
    A = torch.randn(M, Kd); B = torch.randn(Kd, N)
    A @ B
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < secs - tail:
        A @ B
    reps = 0
    t1 = time.perf_counter()
    while time.perf_counter() - t1 < tail:
        A @ B
        reps += 1
    el = time.perf_counter() - t1
    return reps * 2 * M * N * Kd / el / 1e9


def sustained_gpu(args, gsz, M, N, Kd, secs=40, tail=20):
    K8(queue, gsz, (8, 8), *args); queue.finish()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < secs - tail:
        K8(queue, gsz, (8, 8), *args)
    queue.finish()
    reps = 0
    t1 = time.perf_counter()
    while time.perf_counter() - t1 < tail:
        K8(queue, gsz, (8, 8), *args)
        reps += 1
    queue.finish()
    el = time.perf_counter() - t1
    return reps * 2 * M * N * Kd / el / 1e9


B_SZ, T = 8, 256
M = B_SZ * T

print()
print('  %-30s %14s %14s %10s' % ('FFN 形状', 'CPU 稳态', '核显稳态', '核显/CPU'), flush=True)
for tag, mm, nn, kk in (('w1  %dx512 @ 512x2048' % M, M, 2048, 512),
                        ('w2  %dx2048 @ 2048x512' % M, M, 512, 2048)):
    c = sustained_cpu(mm, nn, kk)
    args, gsz = cl_setup(mm, nn, kk)
    g = sustained_gpu(args, gsz, mm, nn, kk)
    print('  %-30s %11.1f GF   %11.1f GF  %8.2fx  %s'
          % (tag, c, g, g / c, '核显赢' if g > c * 1.05 else ('CPU 赢' if c > g * 1.05 else '平手')),
          flush=True)
    time.sleep(10)

print()
print('DONE')
