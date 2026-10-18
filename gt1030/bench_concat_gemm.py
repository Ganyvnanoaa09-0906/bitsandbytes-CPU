"""Would concatenating W1/W3 (and da|db) into one wider GEMM pay off?

The v2 attribution showed 11 GEMMs = 41.5 ms (623 GFLOPS) of the 54.7 ms block. Two of the GEMM pairs
share both operands: the two SwiGLU up-projections (h @ W1, h @ W3) and the two dW products
(hT @ da, hT @ db). Concatenating the weight/activation matrices turns each pair into one wider GEMM
with N=2*FF -- fewer launches and a better tile shape. This measures the difference on device buffers
before touching the full step.
"""
from __future__ import annotations

import ctypes
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.full_gpu_step_v2 as v2
from gt1030.full_gpu_step_v2 import ctx, mf, queue, bl, QP, ONE, ZERO, LAYOUT_COL, NO, M, D, FF


def hgemm(a, b, c, m, n, k):
    rc = bl.CLBlastHgemm(
        LAYOUT_COL, NO, NO, n, m, k, ONE,
        ctypes.c_void_p(int(b.int_ptr)), 0, n,
        ctypes.c_void_p(int(a.int_ptr)), 0, k, ZERO,
        ctypes.c_void_p(int(c.int_ptr)), 0, n,
        ctypes.byref(QP), None)
    assert rc == 0, rc


def buf(nbytes):
    return cl.Buffer(ctx, mf.READ_WRITE, nbytes)


import pyopencl as cl


def med(fn, n=8):
    fn(); queue.finish()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); queue.finish(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts[2:])


h = buf(M * D * 2)
w1 = buf(D * FF * 2); w3 = buf(D * FF * 2); w13 = buf(D * 2 * FF * 2)
a = buf(M * FF * 2); b = buf(M * FF * 2); ab = buf(M * 2 * FF * 2)
hT = buf(D * M * 2)
da = buf(M * FF * 2); db = buf(M * FF * 2); dab = buf(M * 2 * FF * 2)
dw1 = buf(D * FF * 2); dw3 = buf(D * FF * 2); dw13 = buf(D * 2 * FF * 2)

t_two_fwd = med(lambda: (hgemm(h, w1, a, M, FF, D), hgemm(h, w3, b, M, FF, D)))
t_one_fwd = med(lambda: hgemm(h, w13, ab, M, 2 * FF, D))
print('up projections  : two x (2048x1368x512) %.2f ms | one x (2048x2736x512) %.2f ms  (%.1f%% faster)'
      % (t_two_fwd * 1e3, t_one_fwd * 1e3, (t_two_fwd / t_one_fwd - 1) * 100))

t_two_dw = med(lambda: (hgemm(hT, da, dw1, D, FF, M), hgemm(hT, db, dw3, D, FF, M)))
t_one_dw = med(lambda: hgemm(hT, dab, dw13, D, 2 * FF, M))
print('dW products     : two x (512x1368x2048)  %.2f ms | one x (512x2736x2048)  %.2f ms  (%.1f%% faster)'
      % (t_two_dw * 1e3, t_one_dw * 1e3, (t_two_dw / t_one_dw - 1) * 100))
