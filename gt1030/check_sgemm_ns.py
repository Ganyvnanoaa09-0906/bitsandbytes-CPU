"""Sanity check the fp32 CLBlast Sgemm timing used by the Muon NS microbenchmark."""
from __future__ import annotations

import ctypes
import time

import numpy as np
import pyopencl as cl

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
ROW, NO, YES = 101, 111, 112

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
dev = ctx.devices[0]
print('device:', dev.name, dev.global_mem_size / 1e9, 'GB')

bl = ctypes.CDLL(DLL)
bl.CLBlastSgemm.restype = ctypes.c_int
bl.CLBlastSgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
QP = ctypes.c_void_p(int(queue.int_ptr))


def sgemm(a, b, c, m, n, k, ta=False, tb=False, alpha=1.0, beta=0.0):
    lda = m if ta else k
    ldb = k if tb else n
    rc = bl.CLBlastSgemm(ROW, YES if ta else NO, YES if tb else NO, m, n, k,
                         ctypes.c_float(alpha),
                         ctypes.c_void_p(int(a.int_ptr)), 0, lda,
                         ctypes.c_void_p(int(b.int_ptr)), 0, ldb, ctypes.c_float(beta),
                         ctypes.c_void_p(int(c.int_ptr)), 0, n,
                         ctypes.byref(QP), None)
    if rc != 0:
        raise RuntimeError('Sgemm rc=%d' % rc)


# small correctness check: C = A @ B^T with A (4,8), B (5,8)
A = np.random.randn(4, 8).astype(np.float32)
B = np.random.randn(5, 8).astype(np.float32)
C = np.zeros((4, 5), dtype=np.float32)
a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
c = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C)
sgemm(a, b, c, 4, 5, 8, tb=True)
queue.finish()
out = np.empty((4, 5), dtype=np.float32)
cl.enqueue_copy(queue, out, c)
queue.finish()
ref = A @ B.T
print('small GEMM max abs err:', np.abs(out - ref).max())

# big timing: m=n=512,k=2736 and the two NS-iteration shapes
for (m, n, k, tb) in [(512, 512, 2736, True), (512, 2736, 512, False),
                      (512, 512, 512, False), (256, 512, 256, True)]:
    A2 = np.random.randn(m, k).astype(np.float32)
    B2 = np.random.randn(n, k).astype(np.float32) if tb else np.random.randn(k, n).astype(np.float32)
    C2 = np.zeros((m, n), dtype=np.float32)
    ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A2)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B2)
    cb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C2)
    sgemm(ab, bb, cb, m, n, k, tb=tb)
    queue.finish()
    n_iter = 20
    t0 = time.perf_counter()
    for _ in range(n_iter):
        sgemm(ab, bb, cb, m, n, k, tb=tb)
    queue.finish()
    dt = (time.perf_counter() - t0) / n_iter
    gflop = 2.0 * m * n * k / 1e9
    print('GEMM m=%d n=%d k=%d tb=%s: %.2f ms -> %.1f GFLOPS'
          % (m, n, k, tb, dt * 1000, gflop / dt))
