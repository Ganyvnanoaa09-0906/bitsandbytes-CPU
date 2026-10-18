"""Benchmark fp16 GEMM (CLBlast Hgemm) at the AR model's real shapes, as the first P1 datapoint.

Target: the whole training step needs ~631 GFLOPS effective to tie a GT 1030 (report 7.118).
fp32 CLBlast at these shapes measured 385-505 GFLOPS; fp16 halves the operand traffic, so this
answers whether the storage format alone can get us to the target before spending time on a custom
kernel. Accumulation precision of CLBlast Hgemm is whatever its kernel does; correctness is checked
against a float64 reference of the same fp16 inputs (only accumulation error is measured).
"""
from __future__ import annotations

import ctypes
import statistics
import sys
import time

import numpy as np
import pyopencl as cl

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
LAYOUT_COL, TRANSPOSE_NO = 0, 111

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
bl = ctypes.CDLL(DLL)
bl.CLBlastHgemm.restype = ctypes.c_int
bl.CLBlastHgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
qp = ctypes.c_void_p(int(queue.int_ptr))
ONE_FP16 = 0x3C00
ZERO_FP16 = 0x0000

SHAPES = [(2048, 1536, 512), (2048, 512, 512), (2048, 2048, 512), (2048, 512, 2048)]
print('device: %s (%d CU)' % (dev.name, dev.max_compute_units))
print('%5s %5s %5s  %9s  %9s  %8s' % ('M', 'N', 'K', 'GFLOPS', 'ms', 'max|err|'))

for m, n, k in SHAPES:
    a = (np.random.randn(m, k) * 0.1).astype(np.float16)
    b = (np.random.randn(k, n) * 0.1).astype(np.float16)
    c = np.zeros((m, n), dtype=np.float16)
    abuf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    bbuf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    cbuf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=c)

    def call():
        # C_row = A_row @ B_row via column-major API: C' = B^T @ A^T
        return bl.CLBlastHgemm(
            LAYOUT_COL, TRANSPOSE_NO, TRANSPOSE_NO, n, m, k, ONE_FP16,
            ctypes.c_void_p(int(bbuf.int_ptr)), 0, n,
            ctypes.c_void_p(int(abuf.int_ptr)), 0, k, ZERO_FP16,
            ctypes.c_void_p(int(cbuf.int_ptr)), 0, n,
            ctypes.byref(qp), None)

    rc = call()
    queue.finish()
    samples = []
    for _ in range(7):
        t0 = time.perf_counter()
        call()
        queue.finish()
        samples.append(time.perf_counter() - t0)
    sec = statistics.median(samples)
    gflops = 2.0 * m * n * k / sec / 1e9
    cl.enqueue_copy(queue, c, cbuf)
    queue.finish()
    ref = a.astype(np.float64) @ b.astype(np.float64)
    err = float(np.max(np.abs(c.astype(np.float64) - ref)))
    print('%5d %5d %5d  %9.1f  %9.2f  %8.3f  (rc=%d)' % (m, n, k, gflops, sec * 1e3, err, rc))

print()
print('target for GT 1030 equivalence on the AR step: ~631 GFLOPS effective')
