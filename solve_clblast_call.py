"""solve_clblast_call.py -- find the call that works on non-cubic shapes.

The error codes were the whole problem, and I had been reading them as OpenCL
codes for hours. They are CLBlast's own:

    -1016  kInvalidLeadDimA          LD of A is smaller than the matrix's first dimension
    -1015  kInvalidLeadDimB          LD of B is smaller than the matrix's first dimension
    -1011  kInsufficientMemoryA      Matrix A's OpenCL buffer is too small
    -1010  kInsufficientMemoryB      Matrix B's OpenCL buffer is too small

So every failure was argument validation, not resource exhaustion. That also
explains the shape pattern: 128x128x128 passes because k == m, and 256x256x128
returns kInvalidLeadDimA because k=128 is smaller than m=256.

Under a row-major layout CLBlast wants the leading dimension of A to be at least
the first dimension of A, which here is M, not K -- consistent with the separate
finding that this call convention returns B@A rather than A@B. Both are symptoms
of one convention mismatch.

So sweep layout, transposition and all leading-dimension choices on a NON-CUBIC
shape, and require a correct result rather than merely a successful launch.
Correctness is checked against both A@B and B@A, since the convention is what is
under test.
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
ROW, COL, NO = 1, 2, 111

CODES = {-1016: 'kInvalidLeadDimA', -1015: 'kInvalidLeadDimB', -1014: 'kInvalidLeadDimC',
         -1011: 'kInsufficientMemoryA', -1010: 'kInsufficientMemoryB',
         -1009: 'kInsufficientMemoryC', -1017: 'kInvalidDimension',
         -1022: 'kInvalidMatrixA', -1021: 'kInvalidMatrixB', -1020: 'kInvalidMatrixC'}

m, n, k = 256, 512, 128          # 非立方，正是模型里那种形状
A = (np.arange(m * k, dtype=np.float32).reshape(m, k) % 61)
B = (np.arange(k * n, dtype=np.float32).reshape(k, n) % 53)
ref_ab = A @ B
ref_ba = B.T @ A.T if False else None   # B(128,512) @ A(256,128) 不合法；只比 A@B
print('形状 M=%d N=%d K=%d（非立方）' % (m, n, k))
print()


def attempt(layout, ta, tb, lda, ldb, ldc):
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    bc = cl.Buffer(ctx, mf.READ_WRITE, m * n * 4)
    cl.enqueue_copy(queue, bc, np.zeros((m, n), dtype=np.float32)).wait()
    rc = bl.CLBlastSgemm(layout, ta, tb, m, n, k, ctypes.c_float(1.0),
                         ctypes.c_void_p(int(ba.int_ptr)), 0, lda,
                         ctypes.c_void_p(int(bb.int_ptr)), 0, ldb,
                         ctypes.c_float(0.0),
                         ctypes.c_void_p(int(bc.int_ptr)), 0, ldc,
                         ctypes.byref(qp), None)
    queue.finish()
    out = None
    if rc == 0:
        out = np.zeros((m, n), dtype=np.float32)
        cl.enqueue_copy(queue, out, bc).wait()
    del ba, bb, bc
    return rc, out


print('%-5s %-4s %-4s %-14s %-8s %s' % ('布局', 'tA', 'tB', 'ld(a,b,c)', '返回码', '正确性'))
print('-' * 68)
hits = []
for layout, lname in ((ROW, 'row'), (COL, 'col')):
    for ta, tan in ((NO, 'N'),):
        for tb, tbn in ((NO, 'N'),):
            for lda in (k, m):
                for ldb in (n, k):
                    for ldc in (n, m):
                        rc, out = attempt(layout, ta, tb, lda, ldb, ldc)
                        if rc != 0:
                            tag = '%d %s' % (rc, CODES.get(rc, ''))
                            verdict = '-'
                        else:
                            d = float(np.abs(out - ref_ab).max())
                            s = float(np.abs(ref_ab).max()) or 1.0
                            if d / s < 1e-4:
                                verdict = '✓ A@B 正确'
                                hits.append((lname, lda, ldb, ldc))
                            else:
                                verdict = '差 %.3g ✗' % d
                            tag = '0'
                        print('%-5s %-4s %-4s %-14s %-8s %s'
                              % (lname, tan, tbn, '%d,%d,%d' % (lda, ldb, ldc), tag, verdict))
print()
if hits:
    print('  ⇒ 可用组合: %s' % hits)
else:
    print('  ⇒ 该形状下没有组合能正确计算（可能 K 仍不满足校验）')
