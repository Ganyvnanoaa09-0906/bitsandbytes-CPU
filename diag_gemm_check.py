"""diag_gemm_check.py -- A diagonal with distinct values makes a transposed A visible.

identity_gemm_check used A = I and concluded the kernel was correct. That was a bad
test: every row and every column of I is identical, so no matter how A is
mis-read, C still comes out as B. It proves B is read correctly and nothing about A.
This is the same mistake as benchmarking only cubic shapes, where all three
leading dimensions coincide and a wrong one cannot show -- made twice in a row.

Take A = diag(1, 2, ..., n). Then C[i,:] = (i+1) * B[i,:], and if A is being read
transposed, C[i,:] becomes (i+1) * B[:,i] instead -- a completely different matrix,
and one with the wrong shape unless B is square. Distinct diagonal entries also rule
out a permutation of A's entries hiding the error.

Also runs a non-diagonal asymmetric case, since that is what the real model needs.
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
import pyopencl as cl  # noqa: E402

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
qp = ctypes.c_void_p(int(queue.int_ptr))
ROW, NO = 1, 111


def gemm(A, B, lda=None, ldb=None, ldc=None):
    m, k = A.shape
    k2, n = B.shape
    lda = k if lda is None else lda
    ldb = n if ldb is None else ldb
    ldc = n if ldc is None else ldc
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    bc = cl.Buffer(ctx, mf.READ_WRITE, m * n * 4)
    cl.enqueue_copy(queue, bc, np.zeros((m, n), dtype=np.float32)).wait()
    rc = bl.CLBlastSgemm(ROW, NO, NO, m, n, k2, ctypes.c_float(1.0),
                         ctypes.c_void_p(int(ba.int_ptr)), 0, lda,
                         ctypes.c_void_p(int(bb.int_ptr)), 0, ldb,
                         ctypes.c_float(0.0),
                         ctypes.c_void_p(int(bc.int_ptr)), 0, ldc,
                         ctypes.byref(qp), None)
    queue.finish()
    out = np.zeros((m, n), dtype=np.float32)
    if rc == 0:
        cl.enqueue_copy(queue, out, bc).wait()
    return rc, out


n = 128
B = (np.arange(n * n, dtype=np.float32).reshape(n, n) % 89)
D = np.diag(np.arange(1, n + 1).astype(np.float32))     # 对角元全不同
rc, out = gemm(D, B)
exp = D @ B
print('A = diag(1..%d),  B[i,j]=(i*%d+j)%%89' % (n, n))
print('  返回码 %d' % rc)
if rc == 0:
    print('  与 D@B   最大差 %.4g' % np.abs(out - exp).max())
    print('  与 B@D   最大差 %.4g' % np.abs(out - B @ D).max() if (B @ D).shape == out.shape else '')
    print('  C[0,:4] =', np.round(out[0, :4], 1), ' 期望', np.round(exp[0, :4], 1))
    print('  C[1,:4] =', np.round(out[1, :4], 1), ' 期望', np.round(exp[1, :4], 1))
    print('  ⇒ %s' % ('C == D@B ✓ 内核正确' if np.abs(out - exp).max() < 1e-2
                      else 'C != D@B ✗✗ 内核错'))
print()

# 真正的目标形状：非对角、非对称
for (m, k, nn) in [(64, 32, 64), (128, 64, 128), (256, 128, 256), (256, 256, 128)]:
    rng = np.random.default_rng(11)
    A = rng.standard_normal((m, k)).astype(np.float32)
    Bb = rng.standard_normal((k, nn)).astype(np.float32)
    rc, out = gemm(A, Bb)
    if rc != 0:
        print('  %-18s 返回码 %d ⇒ 启动失败 ✗' % ('%dx%dx%d' % (m, nn, k), rc))
        continue
    d = float(np.abs(out - A @ Bb).max())
    s = float(np.abs(A @ Bb).max()) or 1.0
    print('  %-18s 最大差 %-10.4g 相对 %.1e %s'
          % ('%dx%dx%d' % (m, nn, k), d, d / s, '✓' if d / s < 1e-4 else '✗✗'))
