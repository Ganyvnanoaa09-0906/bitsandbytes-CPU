"""final_sweep_clblast.py -- both operand orders, both shape assignments, all lds.

CLBlast's own validation says ld_a must be at least A's first dimension, which for
row-major data is the row count M -- not the standard row-major convention, where
the leading dimension is the column count K. That, together with this call
convention returning B@A rather than A@B, says its row-major path assigns the two
operands the opposite roles to what the parameter names suggest.

The earlier attempt at swapping operands used leading dimensions from the family
that had already been shown to fail validation. This sweeps the product of:

  * operand order      (A,B) or (B,A)
  * which shape goes to (m,n)  matching that order
  * every leading dimension drawn from {m, n, k}

and requires the output to match the intended product, so a launch is not enough.
"""
from __future__ import annotations

import ctypes
import itertools
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

m, n, k = 128, 256, 64          # 全非立方，小一点以便多试
rng = np.random.default_rng(9)
A = rng.standard_normal((m, k)).astype(np.float32)
B = rng.standard_normal((k, n)).astype(np.float32)
ref = A @ B


def attempt(layout, Am, Bm, mm, nn, kk, lda, ldb, ldc, om, on):
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(Am))
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(Bm))
    bc = cl.Buffer(ctx, mf.READ_WRITE, om * on * 4)
    cl.enqueue_copy(queue, bc, np.zeros((om, on), dtype=np.float32)).wait()
    rc = bl.CLBlastSgemm(layout, NO, NO, mm, nn, kk, ctypes.c_float(1.0),
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


print('目标 A@B: A%s @ B%s = %s' % (A.shape, B.shape, ref.shape))
print()
hits = []
tried = 0
for layout in (ROW, COL):
    for swap in (False, True):
        Am, Bm = (B, A) if swap else (A, B)
        mm, nn, kk = (n, m, k) if swap else (m, n, k)
        om, on = (mm, kk), (kk, nn) if False else (mm, nn)
        oshape = (mm, nn)
        for lda, ldb, ldc in itertools.product((mm, nn, kk), (mm, nn, kk), (mm, nn)):
            tried += 1
            rc, out = attempt(layout, Am, Bm, mm, nn, kk, lda, ldb, ldc, *oshape)
            if rc != 0 or out is None:
                continue
            d = float(np.abs(out - ref).max()) / (float(np.abs(ref).max()) or 1.0)
            if d < 1e-4:
                tag = ('布局=%s 顺序=%s m,n,k=%d,%d,%d ld=%d,%d,%d'
                       % ('row' if layout == ROW else 'col',
                          'B,A' if swap else 'A,B', mm, nn, kk, lda, ldb, ldc))
                hits.append(tag)
                print('  ✓ %s' % tag)
print()
print('  试了 %d 种组合，命中 %d 种' % (tried, len(hits)))
if not hits:
    print('  ⇒ 没有组合能正确计算该非立方形状')
