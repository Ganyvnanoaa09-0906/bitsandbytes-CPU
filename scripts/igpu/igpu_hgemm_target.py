"""Push the iGPU's measured GEMM throughput to the GT 1030's theoretical FP32 number (1127 GFLOPS).

Why this can be done at all: the iGPU (Vega 6) has the same peak FP32 as a GT 1030 (1150 vs 1127 GFLOPS),
and its FP32 GEMM already measures 655 GFLOPS. The remaining route to 1127 is precision: Vega 6 has
packed half2 math (2x FP32 rate) while the GT 1030 has FP16 at 1/64 of FP32 -- so an HGEMM is the one
test this hardware can run that a GT 1030 cannot, and it can exceed 1127 GFLOPS legitimately.

This script measures:
  * the same shapes in fp32 (CLBlastSgemm) and fp16 (CLBlastHgemm), CPU included for fp32,
  * correctness against a float64 reference of the *same fp16 inputs* (so the only error measured is the
    kernel's accumulation error, not the input rounding),
  * and states plainly whether the measured number reaches the GT 1030's theoretical value.

A number above 1127 here means "this kernel, at fp16, outruns the arithmetic the GT 1030 has at fp32".
It is not an fp32 equivalence and the report must not present it as one.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

import ctypes
import os
import statistics as st
import sys
import time

import numpy as np
import pyopencl as cl

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
bl = ctypes.CDLL(DLL)
_common = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
           ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
_tail = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
         ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
         None,  # alpha, replaced below
         ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
         ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]

bl.CLBlastSgemm.restype = ctypes.c_int
bl.CLBlastSgemm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                            ctypes.c_float,
                            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                            ctypes.c_float,
                            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
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
HALF_ONE = 0x3C00     # fp16 1.0
HALF_ZERO = 0x0000

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
qp = ctypes.c_void_p(int(queue.int_ptr))
print('device: %s | %d CU' % (dev.name, dev.max_compute_units))


def gemm32(m, n, k, A, B, C):
    rc = bl.CLBlastSgemm(LAYOUT_COL, TRANSPOSE_NO, TRANSPOSE_NO, n, m, k, ctypes.c_float(1.0),
                         ctypes.c_void_p(int(B.int_ptr)), 0, n,
                         ctypes.c_void_p(int(A.int_ptr)), 0, k,
                         ctypes.c_float(0.0),
                         ctypes.c_void_p(int(C.int_ptr)), 0, n,
                         ctypes.byref(qp), None)
    if rc != 0:
        raise RuntimeError('Sgemm rc=%d on %s' % (rc, (m, n, k)))


def gemm16(m, n, k, A, B, C):
    rc = bl.CLBlastHgemm(LAYOUT_COL, TRANSPOSE_NO, TRANSPOSE_NO, n, m, k, ctypes.c_uint16(HALF_ONE),
                         ctypes.c_void_p(int(B.int_ptr)), 0, n,
                         ctypes.c_void_p(int(A.int_ptr)), 0, k,
                         ctypes.c_uint16(HALF_ZERO),
                         ctypes.c_void_p(int(C.int_ptr)), 0, n,
                         ctypes.byref(qp), None)
    if rc != 0:
        raise RuntimeError('Hgemm rc=%d on %s' % (rc, (m, n, k)))


def alloc(dtype, count):
    return cl.Buffer(ctx, mf.READ_WRITE, count * np.dtype(dtype).itemsize)


def hbuf(m, k):
    a = np.random.randn(m, k).astype(np.float16)
    b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    return a, b


def check(m, n, k):
    """Correctness of Hgemm against a float64 matmul of the SAME fp16 inputs."""
    a, A = hbuf(m, k)
    b, B = hbuf(k, n)
    C = alloc(np.float16, m * n)
    gemm16(m, n, k, A, B, C)
    queue.finish()
    out = np.empty((m, n), np.float16)
    cl.enqueue_copy(queue, out, C).wait()
    ref = a.astype(np.float64) @ b.astype(np.float64)
    num = np.abs(out.astype(np.float64) - ref)
    denom = np.abs(ref).max() + 1e-30
    rel = float(num.max() / denom)
    print('  Hgemm correctness %-20s max_abs=%.4f  rel=%.2e  %s'
          % ('%dx%dx%d' % (m, n, k), float(num.max()), rel,
             'OK(fp16 accumulation)' if rel < 5e-2 else 'WRONG'))
    return rel


def bench_fp32(m, n, k, iters=6, warm=2):
    a, A = hbuf(m, k)          # reuse fp16 buffers for both? no: fp32 needs 4 bytes
    a32 = a.astype(np.float32)
    b, B = hbuf(k, n)
    b32 = b.astype(np.float32)
    A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a32)
    B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b32)
    C = alloc(np.float32, m * n)

    def once():
        gemm32(m, n, k, A, B, C)
        queue.finish()
    for _ in range(warm):
        once()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); once(); ts.append(time.perf_counter() - t0)
    dt = st.median(ts)
    return 2.0 * m * n * k / dt / 1e9, dt


def bench_fp16(m, n, k, iters=6, warm=2, check_first=False):
    a, A = hbuf(m, k)
    b, B = hbuf(k, n)
    C = alloc(np.float16, m * n)

    def once():
        gemm16(m, n, k, A, B, C)
        queue.finish()
    for _ in range(warm):
        once()
    if check_first:
        out = np.empty((m, n), np.float16)
        cl.enqueue_copy(queue, out, C).wait()
        ref = a.astype(np.float64) @ b.astype(np.float64)
        rel = float(np.abs(out.astype(np.float64) - ref).max() / (np.abs(ref).max() + 1e-30))
        print('  Hgemm correctness %-20s rel=%.2e  %s'
              % ('%dx%dx%d' % (m, n, k), rel,
                 'expected-fp16-accum' if rel < 5e-2 else 'WRONG'))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); once(); ts.append(time.perf_counter() - t0)
    dt = st.median(ts)
    return 2.0 * m * n * k / dt / 1e9, dt


print()
print('=== correctness first (fp16 inputs, fp16 accumulation, vs float64 of the same inputs) ===')
for shape in ((17, 23, 31), (128, 96, 64), (256, 256, 256)):
    check(*shape)

print()
print('=== warm-up (15 s) ===')
a = np.random.randn(1024, 1024).astype(np.float16)
b = np.random.randn(1024, 1024).astype(np.float16)
A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
C = alloc(np.float16, 1024 * 1024)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 15:
    gemm16(1024, 1024, 1024, A, B, C)
    queue.finish()
print('  done')

GT1030_FP32_PEAK = 1127.0
SHAPES = [
    ('2048^3                ', 2048, 2048, 2048),
    ('4096^3                ', 4096, 4096, 4096),
    ('MLP up   2048x2048x512', 2048, 2048, 512),
    ('MLP down 2048x512x2048', 2048, 512, 2048),
    ('QKV      2048x1536x512', 2048, 1536, 512),
]
print()
print('=== GEMM throughput: fp32 vs fp16 (CLBlast, same shapes) ===')
print('  %-22s %12s %12s %9s' % ('shape', 'fp32 GFLOPS', 'fp16 GFLOPS', 'fp16/fp32'))
rows = []
for label, m, n, k in SHAPES:
    g32, _ = bench_fp32(m, n, k)
    g16, d16 = bench_fp16(m, n, k, check_first=(label.startswith('2048^3')))
    rows.append((label, m, n, k, g32, g16, d16))
    print('  %-22s %12.1f %12.1f %8.2fx' % (label, g32, g16, g16 / g32))

print()
print('=== the target: GT 1030 theoretical FP32 = %.0f GFLOPS ===' % GT1030_FP32_PEAK)
best = max(r[5] for r in rows)
print('  best measured fp16: %.1f GFLOPS = %.2fx the GT 1030 theoretical value' % (best, best / GT1030_FP32_PEAK))
print('  best measured fp32: %.1f GFLOPS = %.2fx' % (max(r[4] for r in rows),
                                                     max(r[4] for r in rows) / GT1030_FP32_PEAK))
print()
if best >= GT1030_FP32_PEAK:
    print('  => measured throughput REACHES the GT 1030 theoretical FP32 value, at fp16.')
    print('     Not an fp32 equivalence: the GT 1030 runs fp16 at 1/64 of fp32, Vega 6 at 2x.')
else:
    print('  => still below; the fp16 headroom is in the kernel, not in the hardware.')
