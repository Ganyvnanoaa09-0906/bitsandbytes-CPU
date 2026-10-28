"""Two measurements in one session:

1. Corrected pure-compute peaks. The first version of this benchmark gave 2253 GFLOPS fp32 — 196% of the
   rated peak — which is impossible and therefore a measurement error by the report's own rule. Cause:
   the eight accumulators had identical initial values and identical recurrences, so the compiler can CSE
   them into one chain and multiply by eight. Fix: distinct initial values AND data-dependent constants,
   so no two chains are the same expression.

2. fp16 GEMM over sizes, looking for the largest honest number: the target is the GT 1030's theoretical
   FP32 value of 1127 GFLOPS. At the fp16 peak (~2x fp32) that needs a kernel efficiency of about 50%,
   which the 2048-4096 shapes did not yet give (best 998 GFLOPS at 4096^3).
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import ctypes
import statistics as st
import sys
import time

import numpy as np
import pyopencl as cl

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
print('device: %s | %d CU' % (dev.name, dev.max_compute_units))

SRC = r'''
__kernel void peak_f32(__global float* out, const int iters) {
    int g = get_global_id(0);
    float a = 1.000001f + 1e-7f * (g & 7);
    float b = 0.999999f - 1e-7f * ((g >> 3) & 7);
    float c0 = a * 1.1f, c1 = a * 1.2f, c2 = a * 1.3f, c3 = a * 1.4f;
    float c4 = a * 1.5f, c5 = a * 1.6f, c6 = a * 1.7f, c7 = a * 1.8f;
    for (int i = 0; i < iters; i++) {
        c0 = fma(a, c0, b); c1 = fma(a, c1, b); c2 = fma(a, c2, b); c3 = fma(a, c3, b);
        c4 = fma(a, c4, b); c5 = fma(a, c5, b); c6 = fma(a, c6, b); c7 = fma(a, c7, b);
    }
    out[g] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
}

__kernel void peak_f16(__global float* out, const int iters) {
    int g = get_global_id(0);
    half2 a = (half2)(1.0001h + 1e-5h * (g & 7), 1.0002h - 1e-5h * (g & 7));
    half2 b = (half2)(0.9999h - 1e-5h * ((g >> 3) & 7), 0.9998h + 1e-5h * ((g >> 3) & 7));
    half2 c0 = a * 1.1h, c1 = a * 1.2h, c2 = a * 1.3h, c3 = a * 1.4h;
    half2 c4 = a * 1.5h, c5 = a * 1.6h, c6 = a * 1.7h, c7 = a * 1.8h;
    for (int i = 0; i < iters; i++) {
        c0 = fma(a, c0, b); c1 = fma(a, c1, b); c2 = fma(a, c2, b); c3 = fma(a, c3, b);
        c4 = fma(a, c4, b); c5 = fma(a, c5, b); c6 = fma(a, c6, b); c7 = fma(a, c7, b);
    }
    float2 s = convert_float2(c0) + convert_float2(c1) + convert_float2(c2) + convert_float2(c3)
             + convert_float2(c4) + convert_float2(c5) + convert_float2(c6) + convert_float2(c7);
    out[g] = s.x + s.y;
}
'''
prog = cl.Program(ctx, SRC).build()
LANES = 6 * 64
N = LANES * 64
iters = 20000
out = cl.Buffer(ctx, mf.WRITE_ONLY, N * 4)


def bench(kern, ops_per_lane):
    k = cl.Kernel(prog, kern)
    for _ in range(2):
        k(queue, (N,), (256,), out, np.int32(iters))
    queue.finish()
    t0 = time.perf_counter()
    for _ in range(3):
        k(queue, (N,), (256,), out, np.int32(iters))
    queue.finish()
    dt = (time.perf_counter() - t0) / 3.0
    return N * 8 * iters * ops_per_lane / dt / 1e9


print()
print('=== corrected pure-compute peaks (distinct chains, data-dependent constants) ===')
g32 = bench('peak_f32', 2)
g16 = bench('peak_f16', 4)
print('  fp32: %8.1f GFLOPS  (%.1f%% of 1150)' % (g32, 100 * g32 / 1150))
print('  fp16: %8.1f GFLOPS  (%.1f%% of 2300)  ratio %.2fx' % (g16, 100 * g16 / 2300, g16 / g32))

# ---------------- fp16 GEMM sweep ----------------
DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
bl = ctypes.CDLL(DLL)
bl.CLBlastHgemm.restype = ctypes.c_int
bl.CLBlastHgemm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                            ctypes.c_uint16,
                            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                            ctypes.c_uint16,
                            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
LAYOUT_COL, TRANSPOSE_NO = 0, 111
qp = ctypes.c_void_p(int(queue.int_ptr))


def hgemm(m, n, k, A, B, C):
    rc = bl.CLBlastHgemm(LAYOUT_COL, TRANSPOSE_NO, TRANSPOSE_NO, n, m, k, ctypes.c_uint16(0x3C00),
                         ctypes.c_void_p(int(B.int_ptr)), 0, n,
                         ctypes.c_void_p(int(A.int_ptr)), 0, k,
                         ctypes.c_uint16(0),
                         ctypes.c_void_p(int(C.int_ptr)), 0, n,
                         ctypes.byref(qp), None)
    if rc != 0:
        raise RuntimeError('Hgemm rc=%d on %s' % (rc, (m, n, k)))


print()
print('=== warm-up + fp16 GEMM sweep (target: 1127 GFLOPS) ===')
a = np.random.randn(1024, 1024).astype(np.float16)
b = np.random.randn(1024, 1024).astype(np.float16)
A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
C0 = cl.Buffer(ctx, mf.READ_WRITE, 1024 * 1024 * 2)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 15:
    hgemm(1024, 1024, 1024, A, B, C0)
    queue.finish()

print('  %-10s %10s %8s %8s' % ('size', 'ms', 'GFLOPS', '% of 1127'))
best = 0.0
for size in (2048, 3072, 4096, 6144):
    m = n = k = size
    a = np.random.randn(m, k).astype(np.float16)
    b = np.random.randn(k, n).astype(np.float16)
    A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    C = cl.Buffer(ctx, mf.READ_WRITE, m * n * 2)
    hgemm(m, n, k, A, B, C)
    queue.finish()
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        hgemm(m, n, k, A, B, C)
        queue.finish()
        ts.append(time.perf_counter() - t0)
    dt = st.median(ts)
    gf = 2.0 * m * n * k / dt / 1e9
    best = max(best, gf)
    print('  %-10s %10.2f %8.1f %7.1f%%' % ('%d^3' % size, dt * 1e3, gf, 100 * gf / 1127))
    del a, b, A, B, C
    time.sleep(1)

print()
print('  best fp16 GEMM: %.1f GFLOPS = %.3fx of the GT 1030 FP32 theoretical value (1127)'
      % (best, best / 1127))
if best >= 1127:
    print('  => the measured GEMM throughput REACHES the GT 1030 theoretical FP32 value (at fp16).')
else:
    print('  => %.1f%% of the target; needs a better kernel, not a bigger shape.' % (100 * best / 1127))
