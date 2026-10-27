# -*- coding: utf-8 -*-
"""探针 43：attention 的 strided-batched GEMM 换成 16 次单 GEMM（带偏移）会不会更快？
   如果快，就值得把 attention 的 6 个 batched 调用全换掉。"""
from __future__ import annotations
import ctypes
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.attention_gpu as A

import gt1030.ar_v2_multi as _MM   # 它会设 A.B=2 -> A.BH=16
T, HD, BH = 1024, 64, 16
bl = A.bl
bl.CLBlastHgemm.restype = ctypes.c_int
bl.CLBlastHgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
QP = ctypes.c_void_p(int(A.queue.int_ptr))
mf = cl.mem_flags
rng = np.random.default_rng(0)
Q = cl.Buffer(A.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
              hostbuf=(rng.standard_normal((BH, T, HD)) * 0.5).astype(np.float16))
Kb = cl.Buffer(A.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
               hostbuf=(rng.standard_normal((BH, T, HD)) * 0.5).astype(np.float16))
S1 = cl.Buffer(A.ctx, mf.READ_WRITE, BH * T * T * 2)
S2 = cl.Buffer(A.ctx, mf.READ_WRITE, BH * T * T * 2)


def gemm_off(a, b, c, m, n, k, off_a, off_b, off_c, ta, tb, lda, ldb, ldc):
    rc = bl.CLBlastHgemm(A.ROW, ta, tb, m, n, k, A.ONE,
                         ctypes.c_void_p(int(a.int_ptr)), off_a, lda,
                         ctypes.c_void_p(int(b.int_ptr)), off_b, ldb, A.ZERO,
                         ctypes.c_void_p(int(c.int_ptr)), off_c, ldc,
                         ctypes.byref(QP), None)
    if rc != 0:
        raise RuntimeError('rc=%d' % rc)


def batched():
    A.hgemm_batched(Q, Kb, S1, T, T, HD, A.NO, A.YES, HD, HD, T,
                    T * HD, T * HD, T * T)


def loopcall():
    for i in range(BH):
        gemm_off(Q, Kb, S2, T, T, HD, i * T * HD, i * T * HD, i * T * T,
                 A.NO, A.YES, HD, HD, T)


def bench(fn, n=10):
    fn(); A.queue.finish()
    t0 = time.time()
    for _ in range(n):
        fn()
    A.queue.finish()
    return (time.time() - t0) / n


batched(); A.queue.finish()
loopcall(); A.queue.finish()
a1 = np.empty(BH * T * T, np.float16); cl.enqueue_copy(A.queue, a1, S1)
a2 = np.empty(BH * T * T, np.float16); cl.enqueue_copy(A.queue, a2, S2)
A.queue.finish()
print('数值一致: %s   max|diff|=%.3e' % (np.array_equal(a1, a2),
                                          float(np.abs(a1.astype(np.float32) -
                                                       a2.astype(np.float32)).max())))
tb = bench(batched)
tl = bench(loopcall)
gf = 2 * BH * T * T * HD
print('batched : %7.2f ms  %6.1f GFLOPS' % (tb * 1000, gf / tb / 1e9))
print('16 次单GEMM: %7.2f ms  %6.1f GFLOPS  (%.2fx)' % (tl * 1000, gf / tl / 1e9, tb / tl))
