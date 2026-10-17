"""Does a CPU-heavy phase before the iGPU benchmark slow the iGPU down, or is the in-context GEMM
slowdown structural? The step benchmark measured 38.9 ms for a 2048^3 GEMM (442 GFLOPS) while the
standalone benchmark in the same session measured 26.2 ms (655 GFLOPS). Everything else about the two
calls is identical; the difference is that the step script runs the CPU arm (12 x ~200 ms of 6-thread
torch) before the device arm. So measure the same GEMM before and after a CPU-heavy phase.
"""
from __future__ import annotations

import ctypes
import statistics as st
import sys
import time

import numpy as np
import pyopencl as cl
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

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
LAYOUT_COL, TRANSPOSE_NO = 0, 111
ctx = cl.create_some_context(interactive=False)
q = cl.CommandQueue(ctx)
mf = cl.mem_flags
qp = ctypes.c_void_p(int(q.int_ptr))
N = 2048
A = cl.Buffer(ctx, mf.READ_ONLY, N * N * 4)
B = cl.Buffer(ctx, mf.READ_ONLY, N * N * 4)
C = cl.Buffer(ctx, mf.WRITE_ONLY, N * N * 4)
zero = np.zeros(N * N, np.float32)
cl.enqueue_copy(q, A, zero).wait()
cl.enqueue_copy(q, B, zero).wait()


def one():
    rc = bl.CLBlastSgemm(LAYOUT_COL, TRANSPOSE_NO, TRANSPOSE_NO, N, N, N, ctypes.c_float(1.0),
                         ctypes.c_void_p(int(B.int_ptr)), 0, N,
                         ctypes.c_void_p(int(A.int_ptr)), 0, N,
                         ctypes.c_float(0.0),
                         ctypes.c_void_p(int(C.int_ptr)), 0, N,
                         ctypes.byref(qp), None)
    if rc != 0:
        raise RuntimeError('rc=%d' % rc)
    q.finish()


t0 = time.perf_counter()
while time.perf_counter() - t0 < 15:
    one()
print('warm-up done')


def measure(label, iters=8):
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        one()
        ts.append(time.perf_counter() - t0)
    med = st.median(ts)
    print('  %-28s %7.2f ms  %6.1f GFLOPS' % (label, med * 1e3, 2.0 * N ** 3 / med / 1e9))
    return med


print()
print('=== cube 2048, before/after a CPU-heavy phase ===')
measure('iGPU alone (before)')
print('  CPU heavy phase: 10 s of torch matmul on 6 threads')
torch.set_num_threads(6)
a = torch.randn(2048, 2048)
b = torch.randn(2048, 2048)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 10:
    a @ b
measure('iGPU right after CPU load')
time.sleep(5)
measure('iGPU 5 s after CPU load')
time.sleep(10)
measure('iGPU 15 s after CPU load')
