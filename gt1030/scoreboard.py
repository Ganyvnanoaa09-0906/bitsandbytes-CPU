"""GT 1030 scoreboard -- grade the CPU/iGPU training modes against the 1030 roofline.

Modes:
  --report-only   use the numbers already measured in the report (no heavy compute; safe to run
                  while other jobs are running).
  --measure       measure fresh GFLOPS on this machine: CPU torch fp32 GEMM and iGPU CLBlast fp32
                  GEMM at the AR model's real shapes. Run this when the machine is idle -- a
                  concurrent training job makes every number meaningless.

Score: fraction = T_1030(W) / T_ours(W); >= 1.0 means "as fast as a GT 1030 on that workload".
The 1030's own time comes from gt1030/ref.py (specs + our same-width GPU's measured efficiency),
not from a card -- see report 7.118.6 for the honesty boundary.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from gt1030.ref import EFF_GEMM, achievable_gflops, fraction  # noqa: E402
from gt1030.workloads import WORKLOADS  # noqa: E402


def report_only() -> None:
    w = WORKLOADS['ar_step']
    r = fraction(w['ours_cpu_s'], w['flops'], w['bytes'])
    print('W1 %s' % w['desc'])
    print('   FLOPs/step %.1f GFLOP, bytes/step %.2f GB'
          % (w['flops'] / 1e9, w['bytes'] / 1e9))
    print('   GT 1030 reference: %.3f s (compute floor %.3f s, memory floor %.3f s)'
          % (r['ref_s'], r['compute_floor_s'], r['memory_floor_s']))
    print('   ours CPU (measured 7.12): %.3f s -> 1030 fraction %.2fx'
          % (r['ours_s'], r['fraction']))
    # iGPU projection: whole step at the real-shape CLBlast throughput measured in 7.109,
    # plus the 3-9% non-GEMM share from 7.12.
    print()
    print('   iGPU projection (whole step at measured real-shape GEMM, +3~9%% non-GEMM):')
    for gflops in (371.0, 480.0, 655.0, 1039.0):
        for overhead in (1.03, 1.09):
            t = w['flops'] / (gflops * 1e9) * overhead
            print('     %6.0f GFLOPS  +%.0f%%  ->  %.3f s  =>  1030 fraction %.2fx'
                  % (gflops, (overhead - 1) * 100, t, r['ref_s'] / t))
    print()
    print('   target for "equivalent": whole-step effective throughput >= %.0f GFLOPS'
          % achievable_gflops())


def measure() -> None:
    import torch

    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    print('CPU: torch %s, fp32, %d threads' % (torch.__version__, torch.get_num_threads()))
    shapes = WORKLOADS['ar_gemm']['shapes']
    for (m, n, k) in shapes:
        a = torch.randn(m, k)
        b = torch.randn(k, n)
        for _ in range(3):
            c = a @ b
        samples = []
        for _ in range(10):
            t0 = time.perf_counter()
            c = a @ b
            samples.append(time.perf_counter() - t0)
        sec = statistics.median(samples)
        gflops = 2.0 * m * n * k / sec / 1e9
        print('   CPU GEMM %5dx%5dx%5d  %7.3f GFLOPS  (median %.2f ms)'
              % (m, n, k, gflops, sec * 1e3))

    # iGPU via CLBlast, column-major trick (report 7.109). Import here so --report-only never
    # needs pyopencl.
    dll = r'D:\work\clblast_build\CLBlast-master\build\clblast.dll'
    if not os.path.exists(dll):
        print('iGPU: CLBlast DLL missing (%s) -- skipped' % dll)
        return
    import ctypes

    import numpy as np
    import pyopencl as cl

    bl = ctypes.CDLL(dll)
    bl.CLBlastSgemm.restype = ctypes.c_int
    bl.CLBlastSgemm.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    qp = ctypes.c_void_p(int(queue.int_ptr))
    print('iGPU: %s (%d CU)' % (ctx.devices[0].name, ctx.devices[0].max_compute_units))
    for (m, n, k) in shapes:
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        c = np.zeros((m, n), dtype=np.float32)
        abuf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        bbuf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        cbuf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)
        # C_row = A_row @ B_row via the column-major API: C' = B^T @ A^T
        args = (0, 111, 111, n, m, k, ctypes.c_float(1.0),
                ctypes.c_void_p(int(bbuf.int_ptr)), 0, n,
                ctypes.c_void_p(int(abuf.int_ptr)), 0, k, ctypes.c_float(0.0),
                ctypes.c_void_p(int(cbuf.int_ptr)), 0, n,
                ctypes.byref(qp), None)
        rc = bl.CLBlastSgemm(*args)
        queue.finish()
        samples = []
        for _ in range(6):
            t0 = time.perf_counter()
            bl.CLBlastSgemm(*args)
            queue.finish()
            samples.append(time.perf_counter() - t0)
        sec = statistics.median(samples)
        gflops = 2.0 * m * n * k / sec / 1e9
        print('   iGPU GEMM %4dx%5dx%5d  %7.3f GFLOPS  (median %.2f ms, rc=%d)'
              % (m, n, k, gflops, sec * 1e3, rc))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--measure', action='store_true', help='run fresh benchmarks (idle machine!)')
    args = ap.parse_args()
    print('GT 1030 scoreboard (achievable reference: %.0f GFLOPS at %.0f%% efficiency)'
          % (achievable_gflops(), EFF_GEMM * 100))
    report_only()
    if args.measure:
        print()
        measure()
    return 0


if __name__ == '__main__':
    sys.exit(main())
