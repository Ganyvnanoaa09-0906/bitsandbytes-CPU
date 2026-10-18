"""Probe row-major transpose conventions for CLBlastHgemm (layout=101).

Needed for the attention backward pass: dWo = O^T dOut, dO = dOut Wo^T, dV = P^T dO, dK = dS^T Q,
dWqkv = x^T dqkv, dx = dqkv Wqkv^T. Each is a transposed GEMM in row-major layout.
"""
from __future__ import annotations

import ctypes
import sys

import numpy as np
import pyopencl as cl

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
ONE, ZERO = 0x3C00, 0x0000
ROW, NO, YES = 101, 111, 112

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
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
rng = np.random.default_rng(0)


def call_ta(ta, tb, m, n, k, a, lda, b, ldb, c, ldc):
    """C(m,n) = op(A) @ op(B), row-major, A is (k,m) if ta else (m,k); B is (n,k) if tb else (k,n)."""
    ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    cb = cl.Buffer(ctx, mf.WRITE_ONLY, m * n * 2)
    rc = bl.CLBlastHgemm(ROW, ta, tb, m, n, k, ONE,
                         ctypes.c_void_p(int(ab.int_ptr)), 0, lda,
                         ctypes.c_void_p(int(bb.int_ptr)), 0, ldb, ZERO,
                         ctypes.c_void_p(int(cb.int_ptr)), 0, ldc,
                         ctypes.byref(QP), None)
    queue.finish()
    out = np.empty((m, n), dtype=np.float16)
    cl.enqueue_copy(queue, out, cb)
    queue.finish()
    return rc, out


M, K, N = 9, 7, 5

# dX = dY @ W^T : A=(M,K) no-trans, B=(N,K) trans  -> C=(M,N)
dy = (rng.standard_normal((M, K)).astype(np.float32)).astype(np.float16)
w = (rng.standard_normal((N, K)).astype(np.float32)).astype(np.float16)   # B stored (N,K) for transB
rc, c = call_ta(NO, YES, M, N, K, dy, K, w, K, None, N)
ref = dy.astype(np.float32) @ w.astype(np.float32).T
print('transB (dX=dY@W^T): rc=%d err=%.4f shape=%s' % (rc, np.abs(c.astype(np.float32) - ref).max(), c.shape))

# dW = X^T @ dY : A=(M,K)?? want C=(K,N) = A^T B with A=(M,K) stored, B=(M,N)
x = (rng.standard_normal((M, K)).astype(np.float32)).astype(np.float16)
dy2 = (rng.standard_normal((M, N)).astype(np.float32)).astype(np.float16)
rc, c = call_ta(YES, NO, K, N, M, x, K, dy2, N, None, N)
ref = x.astype(np.float32).T @ dy2.astype(np.float32)
print('transA (dW=X^T@dY): rc=%d err=%.4f shape=%s' % (rc, np.abs(c.astype(np.float32) - ref).max(), c.shape))
