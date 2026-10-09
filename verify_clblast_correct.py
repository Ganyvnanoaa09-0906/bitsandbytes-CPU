"""verify_clblast_correct.py -- is the 644.9 GFLOPS computed correctly?

bench_clblast.py times CLBlastSgemm without ever reading the result buffer back,
so a kernel that launches fast and computes garbage would report exactly the same
GFLOPS. Section 7.19 rests on that number, so it has to be checked rather than
assumed -- the more so because a naive test of mine reported cubic results as
wrong by 28, and a careful re-test of the same shape reported a maximum difference
of 0. One of those two is itself broken, so the question is live.

Method: for each cubic size, allocate A and B from a reproducible pattern, run the
kernel, copy C back, and compare against numpy. Report both the maximum difference
and the GFLOPS, so correctness and speed are read off the same run.

If the results are correct, 7.19 stands. If they are not, 7.19's numbers are
invalid and the section has to be retracted.
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

print('%-10s %12s %12s %14s  %s' % ('形状', 'GFLOPS', '最大差', '相对误差', '结论'))
print('-' * 74)

for nn in (256, 512, 1024, 2048):
    rng = np.random.default_rng(1234)
    A = rng.standard_normal((nn, nn)).astype(np.float32)
    B = rng.standard_normal((nn, nn)).astype(np.float32)
    ref = A @ B
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    bc = cl.Buffer(ctx, mf.READ_WRITE, nn * nn * 4)
    out = np.zeros((nn, nn), dtype=np.float32)
    cl.enqueue_copy(queue, bc, out).wait()

    def call():
        return bl.CLBlastSgemm(ROW, NO, NO, nn, nn, nn, ctypes.c_float(1.0),
                               ctypes.c_void_p(int(ba.int_ptr)), 0, nn,
                               ctypes.c_void_p(int(bb.int_ptr)), 0, nn,
                               ctypes.c_float(0.0),
                               ctypes.c_void_p(int(bc.int_ptr)), 0, nn,
                               ctypes.byref(qp), None)

    rc = call()
    queue.finish()
    if rc != 0:
        print('%-10s %12s %12s %14s  %s' % ('%d^3' % nn, '-', '-', '-', '启动失败 %d' % rc))
        del ba, bb, bc
        continue

    cl.enqueue_copy(queue, out, bc).wait()
    diff = float(np.abs(out - ref).max())
    scale = float(np.abs(ref).max())
    rel = diff / scale if scale else 0.0

    call()
    queue.finish()
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        call()
    queue.finish()
    dt = (time.perf_counter() - t0) / reps
    gf = 2.0 * nn ** 3 / dt / 1e9

    ok = rel < 1e-4
    print('%-10s %12.1f %12.4g %14.2e  %s'
          % ('%d^3' % nn, gf, diff, rel, '结果正确 ✓' if ok else '结果错误 ✗✗'))
    del ba, bb, bc
