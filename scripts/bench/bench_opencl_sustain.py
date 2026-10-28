# -*- coding: utf-8 -*-
"""bench_opencl_sustain.py — 决定性实验：核显在【持续负载】下会不会降频。

背景：
  CPU 持续 90 秒 sgemm 实测从 131.7 → 106.9 GFLOPS（撞 15W 功耗墙，-19%）
  核显短时 245 GFLOPS（我的 OpenCL） / 294.5（DML）
  但核显和 CPU 共享同一个封装功耗预算 —— 它可能降得更狠

如果核显持续能保住 200+，那它相对 CPU 的持续 107 就是 2 倍以上，
值得为它写一个自定义后端。
"""
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
import time
import numpy as np
import pyopencl as cl

platform = cl.get_platforms()[0]
dev = platform.get_devices()[0]
ctx = cl.Context([dev])
queue = cl.CommandQueue(ctx)

SRC = open('bench_opencl2.py', encoding='utf-8').read()
# 从 bench_opencl2.py 里抠出 kernel 源码
i0 = SRC.index("SRC = r'''") + len("SRC = r'''")
i1 = SRC.index("'''", i0)
KERNEL = SRC[i0:i1]
prg = cl.Program(ctx, KERNEL).build()
K8 = cl.Kernel(prg, 'sgemm8')

print('=' * 70)
print('核显持续负载降频探测  |  %s  %d CU' % (dev.name, dev.max_compute_units))
print('=' * 70)
print('对照: CPU 持续 90s 从 131.7 → 106.9 GFLOPS (-19%)')
print()

n = 2048
A = np.random.randn(n, n).astype(np.float32)
B = np.random.randn(n, n).astype(np.float32)
mf = cl.mem_flags
a_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
b_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
c_b = cl.Buffer(ctx, mf.WRITE_ONLY, n * n * 4)
gsz = (n // 64 * 8, n // 64 * 8)
args = (np.int32(n), np.int32(n), np.int32(n), a_b, b_b, c_b)

K8(queue, gsz, (8, 8), *args)
queue.finish()

print('  %8s %12s %10s' % ('时刻', 'GFLOPS', '相对'))
first = None
t_start = time.perf_counter()
for i in range(9):
    t0 = time.perf_counter()
    reps = 0
    while time.perf_counter() - t0 < 10.0:
        K8(queue, gsz, (8, 8), *args)
        reps += 1
    queue.finish()
    el = time.perf_counter() - t0
    gf = reps * 2 * n ** 3 / el / 1e9
    if first is None:
        first = gf
    print('  %6.0fs %12.2f %9.1f%%' % (time.perf_counter() - t_start, gf, 100 * gf / first))

print()
print('  若稳定 -> 核显可长期跑，相对 CPU 持续 107 是 2 倍以上')
print('  若持续下滑 -> 共享功耗墙，和 CPU 一样受限')
print('DONE')
