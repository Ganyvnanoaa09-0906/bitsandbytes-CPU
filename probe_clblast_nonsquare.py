"""probe_clblast_nonsquare.py -- which dimension breaks CLBlast SGEMM here?

bench_clblast.py succeeds on every square shape (256/512/1024/2048) and fails with
-1016 the first time it tries a non-square one (2048x2048@512). My own harness
failed on ALL seven of the model's real shapes, all non-square. That is too
consistent for a memory problem: 1024x512x512 needs 2 MB per buffer, not 6 GB.

So the question is which of M, N, K may differ from the others. Vary one at a time
from a known-good square and see where it stops working. Everything else -- layout,
transposes, leading dimensions, buffer wrapping -- is copied verbatim from
bench_clblast.py, which is known to work, so a failure here points at the shape
and not at the call.
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


def run(m, n, k):
    A = np.random.randn(m, k).astype(np.float32)
    B = np.random.randn(k, n).astype(np.float32)
    a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c = cl.Buffer(ctx, mf.WRITE_ONLY, m * n * 4)
    args = (LAYOUT_ROW, TRANSPOSE_NO, TRANSPOSE_NO,
            m, n, k, ctypes.c_float(1.0),
            ctypes.c_void_p(int(a.int_ptr)), 0, k,
            ctypes.c_void_p(int(b.int_ptr)), 0, n,
            ctypes.c_float(0.0),
            ctypes.c_void_p(int(c.int_ptr)), 0, n,
            ctypes.byref(qp), None)
    rc = bl.CLBlastSgemm(*args)
    queue.finish()
    return rc


N = 1024
cases = [
    ('方形对照     ', N, N, N),
    ('只变 K       ', N, N, N // 2),
    ('只变 N       ', N, N // 2, N),
    ('只变 M       ', N // 2, N, N),
    ('K 更小 (512)', N, N, 512),
    ('M=N=512,K=512', 512, 512, 512),
    ('512x512x256 ', 512, 512, 256),
    ('1024x1024x512', 1024, 1024, 512),
]
print('%-15s %-18s %-8s %s' % ('用例', 'M x N x K', '返回码', '含义'))
print('-' * 72)
for label, m, n, k in cases:
    try:
        rc = run(m, n, k)
    except Exception as e:
        rc = 'EXC:%s' % type(e).__name__
    ok = 'OK ✓' if rc == 0 else str(rc)
    mean = CODES.get(rc, '') if isinstance(rc, int) else ''
    print('%-15s %-18s %-8s %s' % (label, '%d x %d x %d' % (m, n, k), ok, mean))
