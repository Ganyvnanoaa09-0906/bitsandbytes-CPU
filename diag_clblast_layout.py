"""diag_clblast_layout.py -- the cubic results are wrong; find out how.

8x8x8 and 64x64x64 launch fine and produce numbers that differ from numpy by 28
and 32. That is not rounding: the kernel is computing something else. The obvious
candidate is the layout/transpose convention, and a cubic shape hides it because
every leading dimension coincides.

Two things to settle:

  1. Which convention actually produces A@B? Try row-major, then column-major,
     then compare the output against A@B, A.T@B, A@B.T, B.T@A.T and so on, to
     identify what the kernel is really computing.
  2. Whether that also fixes the non-cubic launches.

This matters beyond tidiness. bench_clblast.py times the kernel without ever
reading the result back, so its 644.9 GFLOPS could have been measured on a kernel
that computes garbage. If so, section 7.19's conclusion rests on an invalid
measurement and has to be retracted.
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


def run(layout, m, n, k, A, B, ld_a, ld_b, ld_c):
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(A, dtype=np.float32))
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(B, dtype=np.float32))
    bc = cl.Buffer(ctx, mf.READ_WRITE, m * n * 4)
    out = np.zeros((m, n), dtype=np.float32)
    cl.enqueue_copy(queue, bc, out).wait()
    rc = bl.CLBlastSgemm(layout, NO, NO, m, n, k, ctypes.c_float(1.0),
                         ctypes.c_void_p(int(ba.int_ptr)), 0, ld_a,
                         ctypes.c_void_p(int(bb.int_ptr)), 0, ld_b,
                         ctypes.c_float(0.0),
                         ctypes.c_void_p(int(bc.int_ptr)), 0, ld_c,
                         ctypes.byref(qp), None)
    queue.finish()
    if rc != 0:
        return rc, None
    cl.enqueue_copy(queue, out, bc).wait()
    return 0, out


m, n, k = 8, 8, 8
A = (np.arange(m * k, dtype=np.float32).reshape(m, k) % 7) - 3
B = (np.arange(k * n, dtype=np.float32).reshape(k, n) % 5) - 2
ref = A @ B
print('A = (m=%d, k=%d),  B = (k=%d, n=%d),  参考 A@B' % (m, k, k, n))
print('A[0] =', A[0])
print('B[0] =', B[0])
print()

for label, layout, ld_a, ld_b, ld_c in [
    ('row-major, ld=K,N,N', ROW, k, n, n),
    ('col-major, ld=M,K,M', COL, m, k, m),
    ('row-major, ld=M,K,M', ROW, m, k, m),
    ('col-major, ld=K,N,N', COL, k, n, n),
]:
    rc, out = run(layout, m, n, k, A, B, ld_a, ld_b, ld_c)
    if rc != 0:
        print('%-24s 启动失败 %d' % (label, rc))
        continue
    cands = {'A@B': ref, 'A.T@B': A.T @ B, 'A@B.T': A @ B.T, 'B.T@A.T': B.T @ A.T,
             'B@A': B @ A if B.shape == A.shape else None}
    best, bd = None, 1e9
    for nm, c in cands.items():
        if c is None or c.shape != out.shape:
            continue
        d = float(np.abs(out - c).max())
        if d < bd:
            best, bd = nm, d
    print('%-24s 最大差 %.4g  ⇒ 最接近 %s%s'
          % (label, bd, best, '  ✓ 一致' if bd < 1e-2 else '  ✗ 都不一致'))
    if best and bd > 1e-2:
        print('%-24s   输出[0]=%s' % ('', np.round(out[0], 2)))
