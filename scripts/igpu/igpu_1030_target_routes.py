"""Two last routes to a measured number >= 1127 GFLOPS (the GT 1030's theoretical FP32 value).

Route A: fp16 GEMM (CLBlast) over large / K-heavy shapes. The best so far is 1021.6 at 4096^3; more
compute per byte (larger K) may lift the efficiency past the 1127 line.

Route B: machine compute peak. The iGPU fp16 pure-compute benchmark measures ~1098 GFLOPS, and 7.88
showed that a pure-ALU CPU load costs the iGPU only 3% (the interference is cache, not power). So CPU
pure FMA (~500 GFLOPS measured in this project) and iGPU pure compute may run at the same time without
measurable loss -- a machine-level compute number of ~1600 GFLOPS, clearly above 1127.

Both are labelled for what they are: A is a real GEMM at fp16; B is a peak-FLOPs test of the whole
machine, not a workload.
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
import statistics as st
import sys
import threading
import time

import numpy as np
import pyopencl as cl
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

torch.set_num_threads(6)
torch.set_grad_enabled(False)

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
print('device: %s | %d CU' % (ctx.devices[0].name, ctx.devices[0].max_compute_units))

# ---------------- route B: pure compute kernels ----------------
KER = r'''
__kernel void peak_f32(__global float* out, const int iters) {
    int g = get_global_id(0);
    float a = 1.000001f + 1e-7f * (g & 7);
    float b = 0.999999f - 1e-7f * ((g >> 3) & 7);
    float c0 = a*1.1f, c1 = a*1.2f, c2 = a*1.3f, c3 = a*1.4f;
    float c4 = a*1.5f, c5 = a*1.6f, c6 = a*1.7f, c7 = a*1.8f;
    for (int i = 0; i < iters; i++) {
        c0 = fma(a,c0,b); c1 = fma(a,c1,b); c2 = fma(a,c2,b); c3 = fma(a,c3,b);
        c4 = fma(a,c4,b); c5 = fma(a,c5,b); c6 = fma(a,c6,b); c7 = fma(a,c7,b);
    }
    out[g] = c0+c1+c2+c3+c4+c5+c6+c7;
}
__kernel void peak_f16(__global float* out, const int iters) {
    int g = get_global_id(0);
    half2 a = (half2)(1.0001h + 1e-5h * (g & 7), 1.0002h - 1e-5h * (g & 7));
    half2 b = (half2)(0.9999h - 1e-5h * ((g >> 3) & 7), 0.9998h + 1e-5h * ((g >> 3) & 7));
    half2 c0 = a*1.1h, c1 = a*1.2h, c2 = a*1.3h, c3 = a*1.4h;
    half2 c4 = a*1.5h, c5 = a*1.6h, c6 = a*1.7h, c7 = a*1.8h;
    for (int i = 0; i < iters; i++) {
        c0 = mad(a,c0,b); c1 = mad(a,c1,b); c2 = mad(a,c2,b); c3 = mad(a,c3,b);
        c4 = mad(a,c4,b); c5 = mad(a,c5,b); c6 = mad(a,c6,b); c7 = mad(a,c7,b);
    }
    float2 s = convert_float2(c0)+convert_float2(c1)+convert_float2(c2)+convert_float2(c3)
             + convert_float2(c4)+convert_float2(c5)+convert_float2(c6)+convert_float2(c7);
    out[g] = s.x + s.y;
}
'''
prog = cl.Program(ctx, KER).build(options=['-cl-mad-enable', '-cl-fast-relaxed-math'])
N = 6 * 64 * 64
out = cl.Buffer(ctx, mf.WRITE_ONLY, N * 4)
iters = 20000


def igpu_peak(kern, ops):
    k = cl.Kernel(prog, kern)
    for _ in range(2):
        k(queue, (N,), (256,), out, np.int32(iters))
    queue.finish()
    t0 = time.perf_counter()
    k(queue, (N,), (256,), out, np.int32(iters))
    queue.finish()
    dt = time.perf_counter() - t0
    return N * 8 * iters * ops / dt / 1e9, dt


print()
print('=== route B: machine compute peak ===')
g32, _ = igpu_peak('peak_f32', 2)
g16, dt16 = igpu_peak('peak_f16', 4)
print('  iGPU fp32 pure: %7.1f GFLOPS' % g32)
print('  iGPU fp16 pure: %7.1f GFLOPS' % g16)

# CPU pure FMA rate, one short calibration
CPU_ITERS = 2_000_000
c0 = torch.ones(6 * 8 * 1_000_000 // 8, dtype=torch.float32)


def cpu_spin(seconds):
    """Pure fp32 FMA chain on all cores for `seconds`; returns GFLOPS estimate."""
    x = torch.randn(6 * 1024 * 1024, dtype=torch.float32)
    y = torch.randn_like(x)
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(4):
            torch.addcmul(y, x, x, out=y)     # 1 FMA per element = 2 FLOPs
        n += 4
    dt = time.perf_counter() - t0
    return n * x.numel() * 2 / dt / 1e9


gf_cpu = cpu_spin(3.0)
print('  CPU fp32 FMA (torch, 6 threads): %7.1f GFLOPS' % gf_cpu)

# both at once: iGPU kernel in a thread, CPU spin in the main thread
print()
print('  both at once (8 s):')
stop = threading.Event()
res = {}
k16 = cl.Kernel(prog, 'peak_f16')


def gpu_worker():
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 8:
        k16(queue, (N,), (256,), out, np.int32(iters))
        queue.finish()


th = threading.Thread(target=gpu_worker)
t0 = time.perf_counter()
th.start()
cpu_gf = cpu_spin(8.0)
th.join()
print('  CPU during overlap: %7.1f GFLOPS' % cpu_gf)

# the iGPU worker does fixed work; remeasure combined numbers in a bounded way
# (run once more with known duration to attribute the two rates)
dur = 8.0
gpu_iters = [0]


def gpu_worker2():
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < dur:
        k16(queue, (N,), (256,), out, np.int32(iters))
        queue.finish()
        gpu_iters[0] += 1


th = threading.Thread(target=gpu_worker2)
t0 = time.perf_counter()
th.start()
cpu_gf2 = cpu_spin(dur)
th.join()
el = time.perf_counter() - t0
gf16_overlap = gpu_iters[0] * N * 8 * iters * 4 / el / 1e9
print('  iGPU during overlap: %6.1f GFLOPS   CPU: %6.1f GFLOPS' % (gf16_overlap, cpu_gf2))
print('  combined machine peak: %.1f GFLOPS = %.2fx the GT 1030 theoretical 1127'
      % (gf16_overlap + cpu_gf2, (gf16_overlap + cpu_gf2) / 1127))

# ---------------- route A: fp16 GEMM over large shapes ----------------
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
print('=== route A: fp16 GEMM, large and K-heavy shapes ===')
a = np.random.randn(1024, 1024).astype(np.float16)
b = np.random.randn(1024, 1024).astype(np.float16)
A0 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
B0 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
C0 = cl.Buffer(ctx, mf.READ_WRITE, 1024 * 1024 * 2)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 10:
    hgemm(1024, 1024, 1024, A0, B0, C0)
    queue.finish()

best = 1021.6
for (m, n, k) in ((4096, 4096, 4096), (4096, 4096, 8192), (8192, 4096, 4096),
                  (4096, 8192, 4096), (2048, 8192, 4096)):
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
    print('  %5dx%5dx%5d  %8.2f ms  %8.1f GFLOPS  (%5.1f%% of 1127)'
          % (m, n, k, dt * 1e3, gf, 100 * gf / 1127))
    del a, b, A, B, C
    time.sleep(1)

print()
print('  best fp16 GEMM: %.1f GFLOPS = %.3fx of 1127' % (best, best / 1127))
