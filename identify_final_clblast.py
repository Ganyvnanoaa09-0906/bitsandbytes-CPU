"""identify_final_clblast.py -- with the working ld set, what does it compute?

The launch problem is solved: on M=256, N=512, K=128 the only accepted leading
dimensions are (m, k, m) = (256, 128, 256), and the reason every other choice fails
is now known from CLBlast's own status codes -- -1016 kInvalidLeadDimA, -1010
kInsufficientMemoryB, -1009 kInsufficientMemoryC. Those are argument validation, so
the earlier non-cubic failures were never about resources.

With that ld set it launches but the output differs from A@B by 2.69e4. The three
accepted values say CLBlast is reading A as (K,M), B as (N,K) and C as (N,M), i.e.
the column-major interpretation of row-major data. So identify which permutation of
the intended product the output actually equals, then invert it.

Candidates, all well-defined for these shapes:
    A @ B            (M,N) from (M,K)@(K,N)
    A.T @ B          needs K==M, skip here
    (B.T @ A.T)      = (A @ B).T
    A.T @ B.T        (K,N) shape, skip
    and the products obtained by feeding transposed operands into the same call.
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

m, n, k = 256, 512, 128
rng = np.random.default_rng(3)
A = rng.standard_normal((m, k)).astype(np.float32)
B = rng.standard_normal((k, n)).astype(np.float32)
ref = A @ B


def run(layout, ta, tb, lda, ldb, ldc, Amat, Bmat, om, on):
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(Amat))
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(Bmat))
    bc = cl.Buffer(ctx, mf.READ_WRITE, om * on * 4)
    cl.enqueue_copy(queue, bc, np.zeros((om, on), dtype=np.float32)).wait()
    rc = bl.CLBlastSgemm(layout, ta, tb, m, n, k, ctypes.c_float(1.0),
                         ctypes.c_void_p(int(ba.int_ptr)), 0, lda,
                         ctypes.c_void_p(int(bb.int_ptr)), 0, ldb,
                         ctypes.c_float(0.0),
                         ctypes.c_void_p(int(bc.int_ptr)), 0, ldc,
                         ctypes.byref(qp), None)
    queue.finish()
    out = None
    if rc == 0:
        out = np.zeros((om, on), dtype=np.float32)
        cl.enqueue_copy(queue, out, bc).wait()
    del ba, bb, bc
    return rc, out


print('M=%d N=%d K=%d' % (m, n, k))
print('ref = A@B, shape %s, max|ref| = %.3g' % (ref.shape, np.abs(ref).max()))
print()

cases = [
    ('原样 A,B  ld=m,k,m', A, B, ROW, m, k, m, (m, n)),
    ('原样 A,B  ld=k,n,n', A, B, ROW, k, n, n, (m, n)),
    ('传 A.T,B.T ld=m,k,m', A.T, B.T, ROW, m, k, m, (k, n)),
    ('传 B,A  ld=k,n,n', B, A, ROW, k, n, n, (k, m)),
    ('传 B,A  ld=n,m,m', B, A, ROW, n, m, m, (k, m)),
]
for label, Am, Bm, lay, lda, ldb, ldc, oshape in cases:
    rc, out = run(lay, NO, NO, lda, ldb, ldc, Am, Bm, oshape[0], oshape[1])
    if rc != 0:
        print('  %-24s 返回码 %d ✗' % (label, rc))
        continue
    cands = {'A@B': ref, '(A@B).T': ref.T}
    best, bd = None, 1e18
    for nm, c in cands.items():
        if c.shape != out.shape:
            continue
        d = float(np.abs(out - c).max()) / (float(np.abs(c).max()) or 1.0)
        if d < bd:
            best, bd = nm, d
    print('  %-24s shape=%s  最接近 %s（相对 %.3g）%s'
          % (label, out.shape, best, bd, '✓ 正确' if bd < 1e-4 else ''))
