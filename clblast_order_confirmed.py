"""clblast_order_confirmed.py -- verify CLBlast against B@A, not A@B.

diag_gemm_check established with a distinct-entry diagonal (identity cannot expose
an error in A, since all its rows and columns are identical) that CLBlast returns
B@A for the call convention in use: C matched B@D to 0 while differing from D@B by
1.0e4.

verify_clblast_correct.py compared against A@B and reported relative error
1.30-1.60. For random square operands A@B and B@A are different matrices that agree
only to O(1) relative, so those numbers are exactly what the wrong comparison would
produce. If so, CLBlast is correct, the throughput is real, and the only remaining
obstacle to using the iGPU is that non-cubic shapes do not launch.

Test both orders at the sizes that mattered, on non-symmetric operands, and time
the same calls so correctness and throughput come from one run.
"""
from __future__ import annotations

import ctypes
import sys
import time

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

print('%-8s %10s %14s %14s %s' % ('n', 'GFLOPS', 'vs A@B', 'vs B@A', '结论'))
print('-' * 74)
for n in (128, 256, 512, 1024, 2048):
    rng = np.random.default_rng(5)
    A = rng.standard_normal((n, n)).astype(np.float32)
    B = rng.standard_normal((n, n)).astype(np.float32)
    # make them clearly non-symmetric so A@B != B@A
    A = A + np.diag(np.arange(1, n + 1, dtype=np.float32))
    B = B + np.diag(np.arange(n, 0, -1, dtype=np.float32))
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    bc = cl.Buffer(ctx, mf.READ_WRITE, n * n * 4)
    cl.enqueue_copy(queue, bc, np.zeros((n, n), dtype=np.float32)).wait()

    def call():
        return bl.CLBlastSgemm(ROW, NO, NO, n, n, n, ctypes.c_float(1.0),
                               ctypes.c_void_p(int(ba.int_ptr)), 0, n,
                               ctypes.c_void_p(int(bb.int_ptr)), 0, n,
                               ctypes.c_float(0.0),
                               ctypes.c_void_p(int(bc.int_ptr)), 0, n,
                               ctypes.byref(qp), None)

    rc = call()
    queue.finish()
    if rc != 0:
        print('%-8d %10s %14s %14s 启动失败 %d' % (n, '-', '-', '-', rc))
        del ba, bb, bc
        continue
    out = np.zeros((n, n), dtype=np.float32)
    cl.enqueue_copy(queue, out, bc).wait()
    d_ab = float(np.abs(out - (A @ B)).max())
    d_ba = float(np.abs(out - (B @ A)).max())
    scale = float(np.abs(B @ A).max()) or 1.0

    call()
    queue.finish()
    t0 = time.perf_counter()
    for _ in range(3):
        call()
    queue.finish()
    dt = (time.perf_counter() - t0) / 3
    gf = 2.0 * n ** 3 / dt / 1e9

    verdict = ('C == B@A ✓ 内核正确' if d_ba / scale < 1e-4 else
               ('C == A@B ✓' if d_ab / scale < 1e-4 else '两者都不吻合 ✗'))
    print('%-8d %10.1f %14.4g %14.4g %s' % (n, gf, d_ab, d_ba, verdict))
    del ba, bb, bc, out
