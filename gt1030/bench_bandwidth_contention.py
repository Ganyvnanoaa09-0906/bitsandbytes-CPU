"""Is bandwidth the current bottleneck for "GT 1030 equivalence"? Measure, don't assume.

Three questions:
  1. What is this machine's effective memory bandwidth (idle, 1 and 6 threads)?
  2. How much does a busy CPU memory stream slow the iGPU GEMM (shared L3/DRAM)?
  3. What bandwidth does the iGPU GEMM actually imply? If it is far below the ceiling, the GEMM is
     compute/occupancy bound, not bandwidth bound.

All numbers are on the same idle machine; the CPU-side stream uses torch operations that are
bandwidth-limited (sum, copy), and the iGPU side uses the CLBlast Hgemm call already validated in
gt1030/bench_hgemm.py.
"""
from __future__ import annotations

import ctypes
import statistics
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

DLL = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
ONE, ZERO = 0x3C00, 0x0000
LAYOUT_COL, TRANSPOSE_NO = 0, 111

torch.set_num_threads(6)

# ---------- 1. CPU memory bandwidth ----------
N = 256 * 1024 * 1024 // 4          # 256 MB fp32
a = torch.randn(N)
b = torch.empty_like(a)
for threads in (1, 6):
    torch.set_num_threads(threads)
    # warm
    a.sum()
    t0 = time.perf_counter()
    for _ in range(5):
        a.sum()
    sec = (time.perf_counter() - t0) / 5
    read_gb = a.numel() * 4 / 1e9 / sec
    t0 = time.perf_counter()
    for _ in range(5):
        b.copy_(a)
    sec = (time.perf_counter() - t0) / 5
    copy_gb = a.numel() * 4 * 2 / 1e9 / sec
    print('CPU %d thread : read %6.2f GB/s   copy r+w %6.2f GB/s' % (threads, read_gb, copy_gb))
torch.set_num_threads(6)

# ---------- iGPU setup ----------
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
bl = ctypes.CDLL(DLL)
bl.CLBlastHgemm.restype = ctypes.c_int
bl.CLBlastHgemm.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint16,
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
qp = ctypes.c_void_p(int(queue.int_ptr))


def make_hgemm(m, n, k):
    A = (np.random.randn(m, k) * 0.1).astype(np.float16)
    B = (np.random.randn(k, n) * 0.1).astype(np.float16)
    C = np.zeros((m, n), dtype=np.float16)
    ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    cb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=C)

    def call():
        return bl.CLBlastHgemm(
            LAYOUT_COL, TRANSPOSE_NO, TRANSPOSE_NO, n, m, k, ONE,
            ctypes.c_void_p(int(bb.int_ptr)), 0, n,
            ctypes.c_void_p(int(ab.int_ptr)), 0, k, ZERO,
            ctypes.c_void_p(int(cb.int_ptr)), 0, n,
            ctypes.byref(qp), None)

    # bytes actually streamed by one GEMM call: A, B and C once each
    gemm_bytes = A.nbytes + B.nbytes + C.nbytes
    flops = 2.0 * m * n * k
    return call, flops, gemm_bytes


def time_hgemm(call, iters=5):
    call()
    queue.finish()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        call()
        queue.finish()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


SHAPE = (512, 12288, 4096)   # Qwen3-8B mlp-up
call, flops, gemm_bytes = make_hgemm(*SHAPE)
m, n, k = SHAPE
print()
print('iGPU Hgemm %dx%dx%d' % (m, n, k))
alone = time_hgemm(call)
print('  alone          : %7.2f ms  %8.1f GFLOPS  implied bandwidth %6.2f GB/s'
      % (alone * 1e3, flops / alone / 1e9, gemm_bytes / alone / 1e9))


# ---------- 2. contention: CPU memory stream while the iGPU runs ----------
stop = threading.Event()


def cpu_stream():
    # bandwidth-limited: copy a 256 MB buffer back and forth
    x = torch.randn(N)
    y = torch.empty_like(x)
    while not stop.is_set():
        y.copy_(x)


th = threading.Thread(target=cpu_stream, daemon=True)
th.start()
time.sleep(0.3)
during = time_hgemm(call)
stop.set()
th.join(timeout=2)
print('  + CPU mem stream: %7.2f ms  %8.1f GFLOPS  (slowdown %.1f%%)'
      % (during * 1e3, flops / during / 1e9, (during / alone - 1) * 100))

# ---------- 3. arithmetic intensity of the target workloads ----------
print()
print('workload arithmetic intensity vs machine ridge:')
print('  AR-30.5M step    : 375 GFLOP / 0.6 GB   = %8.0f FLOP/byte' % (374.7e9 / 0.6e9))
print('  Qwen3-8B NF4 512t: 16.8 TFLOP / 5.14 GB = %8.0f FLOP/byte  (weights only)'
      % (16.8e12 / 5.14e9))
print('  machine ridge (iGPU fp16 1150 GFLOPS / ~20 GB/s read) ~ %5.0f FLOP/byte'
      % (1150e9 / 20e9))
print('  => both target workloads sit 10-100x above the ridge: compute-bound, not bandwidth-bound.')
