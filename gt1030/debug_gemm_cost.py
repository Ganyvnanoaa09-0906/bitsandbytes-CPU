"""Measure gemm_nn in isolation at executor shapes, and with interleaved numpy allocations,
to find why the executor's GEMM time (2.4 s/step) is 4.7x the schedule benchmark (0.515 s)."""
from __future__ import annotations

import ctypes
import statistics
import sys
import time

import numpy as np
import pyopencl as cl

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.ar_step_executor as ex

ctx, queue = ex.ctx, ex.queue
print('device:', ctx.devices[0].name)


def bench(m, n, k, iters=5, interleave=False):
    a = np.zeros((m, k), dtype=np.float16)
    b = np.zeros((k, n), dtype=np.float16)
    # warm
    ex.gemm_nn(a, b, m, n, k)
    queue.finish()
    samples = []
    trash = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ex.gemm_nn(a, b, m, n, k)
        queue.finish()
        samples.append(time.perf_counter() - t0)
        if interleave:
            trash.append(np.zeros((m, k), dtype=np.float32))
    return statistics.median(samples)


shapes = [(2048, 1024, 512), (2048, 1368, 512), (512, 1024, 2048), (512, 1368, 2048),
          (2048, 512, 1368), (2048, 1024, 512)]
print('%-18s %8s %8s %10s' % ('shape', 'plain', 'interleave', 'GFLOPS'))
for (m, n, k) in shapes:
    p = bench(m, n, k)
    i = bench(m, n, k, interleave=True)
    print('%-18s %7.2fms %7.2fms %9.1f'
          % ('%dx%dx%d' % (m, n, k), p * 1e3, i * 1e3, 2 * m * n * k / p / 1e9))
