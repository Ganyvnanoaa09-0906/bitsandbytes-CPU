"""check_hgemm_ns.py — upper bound for Muon NS on the iGPU with fp16 CLBlast.

Same 60 matrices / NS5 structure as muon_ns_microbench.py, but GEMMs are CLBlast Hgemm (fp16
storage). This is an upper bound: real NS in fp16 would also need fp16 elementwise kernels and
has precision risk, so treat the number as "best case if everything were fp16".
"""
from __future__ import annotations

import ctypes
import sys
import time

import numpy as np
import pyopencl as cl

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
ROW, NO, YES = 101, 111, 112
HALF_ONE = 0x3C00
HALF_ZERO = 0x0000
A_, B_, C_ = 3.4445, -4.7750, 2.0315
FAMILIES = [((512, 512), 20), ((256, 512), 20), ((2736, 512), 10), ((512, 1368), 10)]

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
print('device:', ctx.devices[0].name)

bl = ctypes.CDLL(DLL)
bl.CLBlastHgemm.restype = ctypes.c_int
bl.CLBlastHgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
QP = ctypes.c_void_p(int(queue.int_ptr))


def half_bits(x):
    return int(np.float16(x).view(np.uint16))


def hgemm(a, b, c, m, n, k, tb=False, alpha=HALF_ONE, beta=HALF_ZERO):
    lda = k
    ldb = k if tb else n
    rc = bl.CLBlastHgemm(ROW, NO, YES if tb else NO, m, n, k, ctypes.c_uint16(alpha),
                         ctypes.c_void_p(int(a.int_ptr)), 0, lda,
                         ctypes.c_void_p(int(b.int_ptr)), 0, ldb, ctypes.c_uint16(beta),
                         ctypes.c_void_p(int(c.int_ptr)), 0, n,
                         ctypes.byref(QP), None)
    if rc != 0:
        raise RuntimeError('Hgemm rc=%d' % rc)


def buf(shape):
    x = (np.random.randn(*shape) * 0.02).astype(np.float16)
    return cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)


def bench_shape(m, n):
    """One NS5 iteration for shape (m,n): 3 GEMMs, matching the fp32 version."""
    X = buf((m, n))
    A = buf((m, m))
    Bm = buf((m, m))
    X2 = buf((m, n))
    alpha_c = half_bits(C_)
    alpha_a = half_bits(A_)

    state = {'X': X, 'X2': X2}

    def one_iter():
        x, x2 = state['X'], state['X2']
        hgemm(x, x, A, m, m, n, tb=True)                  # A = X X^T
        hgemm(A, A, Bm, m, m, m, alpha=alpha_c)           # Bm = c A A (b A folded in real run)
        hgemm(Bm, x, x2, m, n, m, alpha=HALF_ONE, beta=HALF_ZERO)  # X2 = Bm X
        state['X'], state['X2'] = x2, x

    one_iter()
    queue.finish()
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        one_iter()
    queue.finish()
    return (time.perf_counter() - t0) / reps * 1000


total = 0.0
for (o, i), count in FAMILIES:
    m, n = min(o, i), max(o, i)
    per_iter = bench_shape(m, n)
    per_matrix = per_iter * 5  # NS5
    fam = per_matrix * count
    total += fam
    print('family %4dx%-4d x%2d: %.2f ms/iter -> %.1f ms/matrix, %.2f s total'
          % (m, n, count, per_iter, per_matrix, fam / 1000))
total_s = total / 1000.0
print('\nfp16 Hgemm-only NS5 upper bound: %.3f s (fp32 CLBlast measured 1.751 s)' % total_s)
print('projected Muon step if NS were fp16: %.2f s (CPU Muon 6.42, AdamW 3.95)'
      % (6.42 - 2.42 + total_s))
