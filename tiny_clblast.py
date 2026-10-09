"""tiny_clblast.py -- a 4x3x2 GEMM against numpy, to separate three causes.

Every cubic shape works and every non-cubic shape fails, with three different
allocation error codes. In a cubic GEMM all three leading dimensions coincide, so
a wrong leading-dimension convention is undetectable there and would only show up
once M, N and K differ -- which is exactly the observed split.

A tiny case separates the possibilities cleanly:

  * fails to launch          -> not about the leading dimensions
  * runs but numbers differ  -> the call convention is wrong
  * runs and matches numpy   -> the failure is size-dependent, not structural

Leading dimensions follow the row-major convention: A is MxK with ld_a = K,
B is KxN with ld_b = N, C is MxN with ld_c = N.
"""
from __future__ import annotations

import ctypes
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'

bl = ctypes.CDLL(DLL)
bl.CLBlastSgemm.restype = ctypes.c_int
bl.CLBlastSgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
]
LAYOUT_ROW, TRANSPOSE_NO = 1, 111

import pyopencl as cl  # noqa: E402

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
qp = ctypes.c_void_p(int(queue.int_ptr))

CODES = {-1011: 'CL_MEM_OBJECT_ALLOCATION_FAILURE',
         -1015: 'CL_OUT_OF_HOST_MEMORY',
         -1016: 'CL_OUT_OF_RESOURCES'}


def gemm(m, n, k):
    A = (np.arange(m * k, dtype=np.float32).reshape(m, k) % 7) - 3
    B = (np.arange(k * n, dtype=np.float32).reshape(k, n) % 5) - 2
    ref = A @ B
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    bc = cl.Buffer(ctx, mf.READ_WRITE, m * n * 4)
    out = np.zeros((m, n), dtype=np.float32)
    cl.enqueue_copy(queue, bc, out).wait()
    rc = bl.CLBlastSgemm(LAYOUT_ROW, TRANSPOSE_NO, TRANSPOSE_NO, m, n, k,
                         ctypes.c_float(1.0),
                         ctypes.c_void_p(int(ba.int_ptr)), 0, k,
                         ctypes.c_void_p(int(bb.int_ptr)), 0, n,
                         ctypes.c_float(0.0),
                         ctypes.c_void_p(int(bc.int_ptr)), 0, n,
                         ctypes.byref(qp), None)
    queue.finish()
    if rc != 0:
        return rc, None, ref
    cl.enqueue_copy(queue, out, bc).wait()
    return 0, out, ref


for (m, n, k) in [(4, 3, 2), (8, 8, 8), (16, 12, 8), (64, 64, 64), (128, 96, 64),
                  (256, 256, 128), (512, 384, 256)]:
    rc, got, ref = gemm(m, n, k)
    tag = '%dx%dx%d' % (m, n, k)
    if rc != 0:
        print('  %-14s 【启动失败】 %d %s' % (tag, rc, CODES.get(rc, '')))
    else:
        ok = np.allclose(got, ref, atol=1e-2, rtol=1e-3)
        diff = float(np.abs(got - ref).max())
        print('  %-14s 【跑通 ✓】 numpy 一致=%s  最大差=%.4g' % (tag, '是 ✓' if ok else '否 ✗', diff))
