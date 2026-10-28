"""Can the iGPU run the model's GEMM stream while the CPU runs the non-GEMM half of the step?

The AR step profile (appendix A.3, batch 8) is: mm 53.6%, attention 18.8%, elementwise 18.8%,
other 8.6%. Now that CLBlast runs on the real (non-cubic) shapes at 2-3x the CPU's GEMM throughput,
the question changes from "which device is faster" to "can the two halves overlap". Sequential hand-off
would give about 1/(0.536/2.7 + 0.464) = 1.47x; overlap would give about 1/0.464 = 2.15x -- but only if
the iGPU keeps its throughput while the CPU is busy, which is exactly what 7.88 says is at risk (shared
L3, not power).

So measure all three phases with the same clocks:
  A. iGPU alone  -> suite throughput in GFLOPS
  B. CPU alone   -> non-GEMM workload units/s
  C. both        -> retention of each

The CPU workload is a mix of the ops the profile actually names (silu, mul, rms-norm, softmax over the
attention shape), not a synthetic matmul.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import ctypes
import os
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
LAYOUT_COL, TRANSPOSE_NO, TRANSPOSE_YES = 0, 111, 112

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
qp = ctypes.c_void_p(int(queue.int_ptr))
print('device: %s | %d CU' % (dev.name, dev.max_compute_units))


def sgemm_rows(m, n, k, A, B, C):
    """C_row[m,n] = A_row[m,k] @ B_row[k,n] (CLBlast column-major API, no copies)."""
    return bl.CLBlastSgemm(
        LAYOUT_COL, TRANSPOSE_NO, TRANSPOSE_NO,
        n, m, k, ctypes.c_float(1.0),
        ctypes.c_void_p(int(B.int_ptr)), 0, n,
        ctypes.c_void_p(int(A.int_ptr)), 0, k,
        ctypes.c_float(0.0),
        ctypes.c_void_p(int(C.int_ptr)), 0, n,
        ctypes.byref(qp), None)


# One layer's GEMMs at batch*T = 2048 tokens, the AR model's shapes (d=512, ffn=2048, heads 8x64).
# (m, n, k) pairs for forward; the suite is a stand-in for the 10 layers' mm stream.
SUITE = [
    (2048, 2048, 512),   # MLP up
    (2048, 512, 2048),   # MLP down
    (2048, 1536, 512),   # qkv
    (2048, 512, 512),    # o proj
    (2048, 1024, 512),   # head
]


def make_bufs():
    out = []
    for (m, n, k) in SUITE:
        A = cl.Buffer(ctx, mf.READ_ONLY, m * k * 4)
        B = cl.Buffer(ctx, mf.READ_ONLY, k * n * 4)
        C = cl.Buffer(ctx, mf.WRITE_ONLY, m * n * 4)
        # fill once so the buffers are faulted; content is irrelevant for timing
        cl.enqueue_fill_buffer(queue, A, np.array([0.0], np.float32), 0, m * k * 4)
        cl.enqueue_fill_buffer(queue, B, np.array([0.0], np.float32), 0, k * n * 4)
        out.append((A, B, C))
    queue.finish()
    return out


bufs = make_bufs()


def run_suite():
    for (m, n, k), (A, B, C) in zip(SUITE, bufs):
        rc = sgemm_rows(m, n, k, A, B, C)
        if rc != 0:
            raise RuntimeError('CLBlast rc=%d on %s' % (rc, (m, n, k)))
    queue.finish()


suite_flops = sum(2.0 * m * n * k for (m, n, k) in SUITE)


# ---------------- CPU non-GEMM workload: the ops the profile names ----------------
_x = torch.randn(2048, 512)
_s = torch.empty(2048, 512)
_s2 = torch.empty(2048, 512)
_w = torch.randn(512)
_attn = torch.randn(8, 2048, 1024)          # the attention-score tensor for T=1024, 8 heads
_out = torch.empty_like(_attn)


def cpu_unit():
    """silu + scale + rms-norm + softmax(attention shape): the ops the profile names."""
    torch.sigmoid(_x, out=_s)
    torch.mul(_x, _s, out=_s)                 # silu
    torch.mul(_s, _w, out=_s2)
    n = torch.sum(_x * _x, dim=-1, keepdim=True)
    r = torch.rsqrt(n + 1e-6)
    torch.mul(_x, r, out=_s)                  # rms-norm broadcast
    torch.softmax(_attn, dim=-1, out=_out)


def timed(seconds, fn, setup=None):
    """Run fn in a loop for `seconds`, return (iterations, elapsed)."""
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fn()
        n += 1
    return n, time.perf_counter() - t0


print()
print('=== warm-up (iGPU clocks need ~15 s) ===')
run_suite()
t0 = time.perf_counter()
while time.perf_counter() - t0 < 15:
    run_suite()
print('  done')

print()
print('=== A: iGPU alone (8 s) ===')
_n, _t = timed(8, run_suite)
gflops_alone = _n * suite_flops / _t / 1e9
print('  %d suites in %.2f s -> %.1f GFLOPS' % (_n, _t, gflops_alone))

print()
print('=== B: CPU non-GEMM workload alone (8 s) ===')
_n, _t = timed(8, cpu_unit)
units_alone = _n / _t
print('  %d units in %.2f s -> %.1f units/s' % (_n, _t, units_alone))

print()
print('=== C: both at once (8 s) ===')
stop = threading.Event()
cpu_n = [0]


def cpu_worker():
    while not stop.is_set():
        cpu_unit()
        cpu_n[0] += 1


th = threading.Thread(target=cpu_worker, daemon=True)
t0 = time.perf_counter()
th.start()
gpu_n = 0
while time.perf_counter() - t0 < 8:
    run_suite()
    gpu_n += 1
elapsed = time.perf_counter() - t0
stop.set()
th.join(timeout=10)
gflops_both = gpu_n * suite_flops / elapsed / 1e9
units_both = cpu_n[0] / elapsed
print('  iGPU: %d suites in %.2f s -> %.1f GFLOPS   (retention %.0f%%)'
      % (gpu_n, elapsed, gflops_both, 100 * gflops_both / gflops_alone))
print('  CPU : %d units in %.2f s -> %.1f units/s   (retention %.0f%%)'
      % (cpu_n[0], elapsed, units_both, 100 * units_both / units_alone))

print()
print('=== projection onto the real step (AR model, batch 2, T=1024) ===')
# From clblast_colmajor_shapes.py in the same session: CPU suite 73.65 ms, iGPU suite 29.78 ms for
# 15.03 GFLOP -> CPU 204.0 GFLOPS, iGPU 505 GFLOPS, so the iGPU is 2.47x the CPU on this suite.
CPU_SUITE_GFLOPS = 204.0
CPU_STEP_GEMM, CPU_STEP_NONGEMM, CPU_STEP = 1.787, 1.546, 3.333
speedup = gflops_alone / CPU_SUITE_GFLOPS          # >1 means the iGPU is faster
ret_gpu = gflops_both / gflops_alone
ret_cpu = units_both / units_alone
gpu_gemm_alone = CPU_STEP_GEMM / speedup
seq = gpu_gemm_alone + CPU_STEP_NONGEMM
ovl = max(gpu_gemm_alone / ret_gpu, CPU_STEP_NONGEMM / ret_cpu)
print('  measured suite speedup iGPU/CPU: %.2fx' % speedup)
print('  GEMM half moved to the iGPU alone : %.2f s (was %.2f s)' % (gpu_gemm_alone, CPU_STEP_GEMM))
print('  sequential hand-off               : %.2f s/step -> %.2fx' % (seq, CPU_STEP / seq))
print('  overlapped (retention %.0f%% / %.0f%%) : %.2f s/step -> %.2fx'
      % (100 * ret_gpu, 100 * ret_cpu, ovl, CPU_STEP / ovl))
print()
print('  reference: a GT 1030 runs the whole step on one device. Its GEMM half projects to')
print('  620-680 GFLOPS (1127 peak * cuBLAS-class 55-60%%), and its non-GEMM half to GPU rates.')
print('  so the CPU+iGPU ceiling here is a hand-off, not a 1030.')
