"""sweep_identity_gemm.py -- find the size where the GEMM stops being correct.

n=64 with A=identity: C equals B exactly, element for element. That rules out a
layout or convention error, and it also rules out my earlier comparison being
wrong in some blanket way -- at 64 the kernel is right.

But verify_clblast_correct.py measured relative error 1.30-1.60 at 256, 512, 1024
and 2048. Both cannot be true unless correctness depends on size. A=identity makes
the expected result unambiguous at every size, so sweep the size and find where it
breaks. That also explains the non-cubic launch failures if there is a single
resource limit being crossed.

Reported per size: whether it launches, whether C == B, and the worst element.
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

print('%-8s %-10s %-14s %-14s %s' % ('n', '返回码', 'max|C-B|', 'max|C-B.T|', '结论'))
print('-' * 70)

for n in (64, 96, 128, 192, 256, 320, 384, 448, 512, 768, 1024):
    B = (np.arange(n * n, dtype=np.float32).reshape(n, n) % 97)
    I = np.eye(n, dtype=np.float32)
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=I)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    bc = cl.Buffer(ctx, mf.READ_WRITE, n * n * 4)
    cl.enqueue_copy(queue, bc, np.zeros((n, n), dtype=np.float32)).wait()
    rc = bl.CLBlastSgemm(ROW, NO, NO, n, n, n, ctypes.c_float(1.0),
                         ctypes.c_void_p(int(ba.int_ptr)), 0, n,
                         ctypes.c_void_p(int(bb.int_ptr)), 0, n,
                         ctypes.c_float(0.0),
                         ctypes.c_void_p(int(bc.int_ptr)), 0, n,
                         ctypes.byref(qp), None)
    queue.finish()
    if rc != 0:
        print('%-8d %-10d %-14s %-14s %s' % (n, rc, '-', '-', '启动失败 ✗'))
        del ba, bb, bc
        continue
    out = np.zeros((n, n), dtype=np.float32)
    cl.enqueue_copy(queue, out, bc).wait()
    d_b = float(np.abs(out - B).max())
    d_bt = float(np.abs(out - B.T).max())
    verdict = 'C == B ✓' if d_b < 1e-3 else ('C == B.T ⇒ 转置 ✗' if d_bt < 1e-3 else '既非 B 也非 B.T ✗✗')
    print('%-8d %-10d %-14.4g %-14.4g %s' % (n, rc, d_b, d_bt, verdict))
    del ba, bb, bc, out
