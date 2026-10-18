"""Where does the executor's 8-11 ms per GEMM go? Attribute it: A copy, B copy, kernel, D2H.

The schedule benchmark (buffers resident, no per-call copies) measured 3-6 ms per GEMM at the same
shapes. The executor measures 6.5-11.5 ms. This isolates the difference.
"""
from __future__ import annotations

import ctypes
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.ar_step_executor as ex
from gt1030.ar_step_executor import ONE, ZERO, LAYOUT_COL, TRANSPOSE_NO, bl, queue

import pyopencl as cl

SHAPES = [(2048, 1024, 512), (2048, 1368, 512), (512, 1024, 2048), (2048, 512, 1368)]


def timed(fn, iters=10):
    fn()
    queue.finish()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        queue.finish()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def copy_to(dev, host):
    cl.enqueue_copy(queue, dev, host)


def copy_from(host, dev):
    cl.enqueue_copy(queue, host, dev)


def kernel(ab, bb, cb, m, n, k):
    return bl.CLBlastHgemm(
        LAYOUT_COL, TRANSPOSE_NO, TRANSPOSE_NO, n, m, k, ONE,
        ctypes.c_void_p(int(bb.int_ptr)), 0, n,
        ctypes.c_void_p(int(ab.int_ptr)), 0, k, ZERO,
        ctypes.c_void_p(int(cb.int_ptr)), 0, n,
        ctypes.byref(ex.QP), None)


print('%-18s %10s %10s %10s %10s %10s' % ('shape', 'A copy', 'B copy', 'kernel', 'D2H', 'total'))
for (m, n, k) in SHAPES:
    a = np.zeros((m, k), dtype=np.float16)
    b = np.zeros((k, n), dtype=np.float16)
    ab = ex._buf('a', m * k * 2)
    bb = ex._buf('b', k * n * 2)
    cb = ex._buf('c', m * n * 2)
    out = np.empty((m, n), dtype=np.float16)

    t_a = timed(lambda: copy_to(ab, a))
    t_b = timed(lambda: copy_to(bb, b))
    t_k = timed(lambda: kernel(ab, bb, cb, m, n, k))
    t_d = timed(lambda: copy_from(out, cb))
    print('%-18s %8.2fms %8.2fms %8.2fms %8.2fms %8.2fms'
          % ('%dx%dx%d' % (m, n, k), t_a * 1e3, t_b * 1e3, t_k * 1e3, t_d * 1e3,
             (t_a + t_b + t_k + t_d) * 1e3))
