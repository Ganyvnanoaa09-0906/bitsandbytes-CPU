"""sweep_split_ratio.py -- find how much of the work the iGPU should take.

Two measurements disagree, and the disagreement is informative.

  test_cpu_gpu_parallel.py, rerun just now: iGPU takes the left half of the columns,
  CPU the right half, total 0.013 s against 0.018 s serial and 0.022 s for the iGPU
  alone -- 2.80x over serial, 1.61x over the iGPU. Essentially the ideal
  max(0.012, 0.006). Overlap works.

  igpu_mm.py, and today's pinned variant: splitting the same matmul in half gives
  0.3-0.7x, i.e. worse than plain torch.

The difference is the split ratio, not the overlap. In the working case the CPU half
takes 0.006 s and the iGPU half 0.012 s, so the CPU finishes early and the two are not
both saturating memory at once. Split evenly and both run full tilt for the whole
duration -- and this is an APU, where the iGPU has no memory of its own and shares the
same ~32 GB/s bus as the CPU cores. Saturating it from both sides makes each slower than
either alone. That also explains the 0.83x concurrent-training figure in table 69, which
I earlier attributed to per-operator synchronisation; the bus is the simpler explanation.

So the ratio is a real tuning parameter and the optimum is where the two halves finish
together without the bus being the binding constraint. This sweeps it on the model's real
shape and on a larger one, using fixed buffers and a bare OpenCL queue rather than torch
slicing, so the CPU side is not paying for view copies.

Reported per ratio: total time, the split of iGPU-only and CPU-only times, correctness,
and the speedup against the best single device.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import statistics
import sys
import time

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

import pyopencl as cl  # noqa: E402

KERNEL = r'''
#define TS 64
#define TK 8
#define TX 8
#define TY 8
#define TM 8
#define TN 8
// LD is the row stride of B: the full buffer is uploaded once, so the kernel must be
// told the real stride rather than assuming N. Getting this wrong produced 1.38
// relative error in the previous test.
__kernel void gemm(const int M, const int N, const int K, const int LD,
                   const __global float* restrict A,
                   const __global float* restrict B,
                   __global float* restrict C) {
    __local float As[TS][TK + 1];
    __local float Bs[TK][TS + 1];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * TX + tx;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += TK) {
        #pragma unroll
        for (int p = 0; p < TS * TK / 64; p++) {
            int f = p * 64 + lane, r = f / TK, c = f % TK;
            int gr = by + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        #pragma unroll
        for (int p = 0; p < TK * TS / 64; p++) {
            int f = p * 64 + lane, r = f / TS, c = f % TS;
            int gr = k0 + r, gc = bx + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * LD + gc] : 0.f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] += a[i] * b[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gr = by + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gc = bx + tx * TN + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
    }
}
'''

TS = 64
ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
prg = cl.Program(ctx, KERNEL).build()
kern = cl.Kernel(prg, 'gemm')
print('设备 %s | %d CU' % (dev.name, dev.max_compute_units))
print()


def run(M, N, K, ratios=(0.0, 0.15, 0.25, 1 / 3, 0.5), threads=6, reps=5):
    torch.set_num_threads(threads)
    rng = np.random.default_rng(7)
    A = np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float32))
    B = np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float32))
    ref = A @ B
    ba_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    bc_ = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    out_g = np.empty((M, N), dtype=np.float32)
    At = torch.from_numpy(A)
    Bt = torch.from_numpy(B)

    def ng(frac):
        return int(N * frac) // TS * TS

    def timeit(fn, reps=reps):
        fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts)

    def gpu_only(frac):
        n = ng(frac)
        gsz = ((n + TS - 1) // TS * 8, (M + TS - 1) // TS * 8)

        def f():
            kern(queue, gsz, (8, 8), np.int32(M), np.int32(n), np.int32(K),
                 np.int32(N), ba_, bb_, bc_)
            queue.finish()
        return f

    def cpu_only(frac):
        n = ng(frac)
        right = Bt[:, n:].contiguous()

        def f():
            return At @ right
        return f

    def both(frac):
        n = ng(frac)
        gsz = ((n + TS - 1) // TS * 8, (M + TS - 1) // TS * 8)
        right = Bt[:, n:].contiguous()

        def f():
            # n == 0 (ratio 0.0) must not launch: a zero global work size is invalid
            if n > 0:
                kern(queue, gsz, (8, 8), np.int32(M), np.int32(n), np.int32(K),
                     np.int32(N), ba_, bb_, bc_)
            cpu = At @ right               # runs while the kernel does
            if n > 0:
                cl.enqueue_copy(queue, out_g, bc_)   # pageable, so this waits
            queue.finish()
            return cpu
        return f

    t_cpu_full = timeit(cpu_only(0.0))
    ser = {}
    par = {}
    for r in ratios:
        n = ng(r)
        if n > 0:
            ser[r] = timeit(gpu_only(r)) + timeit(cpu_only(r))
        else:
            ser[r] = t_cpu_full
        par[r] = timeit(both(r))

    # correctness of the parallel path at the chosen ratio
    r = ratios[-1]
    n = ng(r)
    gsz = ((n + TS - 1) // TS * 8, (M + TS - 1) // TS * 8)
    kern(queue, gsz, (8, 8), np.int32(M), np.int32(n), np.int32(K),
         np.int32(N), ba_, bb_, bc_)
    cpu = At @ Bt[:, n:].contiguous()
    cl.enqueue_copy(queue, out_g, bc_).wait()
    got = np.concatenate([out_g[:, :n], cpu.numpy()], axis=1)
    err = float(np.abs(got - ref).max()) / (float(np.abs(ref).max()) or 1.0)

    print('  %dx%dx%d  （CPU 线程 %d）  纯 CPU %.3f s' % (M, N, K, threads, t_cpu_full))
    print('    %6s %10s %10s %10s %9s' % ('核显占比', '串行', '并行', '对纯CPU', '对串行'))
    best = None
    for r in ratios:
        sp = t_cpu_full / par[r]
        if best is None or par[r] < best[1]:
            best = (r, par[r], sp)
        print('    %6.2f %10.3f %10.3f %9.2fx %8.2fx'
              % (r, ser[r], par[r], sp, ser[r] / par[r]))
    print('    ⇒ 最优占比 %.2f，%.3f s，相对纯 CPU %.2fx   正确性 %.2e %s'
          % (best[0], best[1], best[2], err, '✓' if err < 1e-5 else '✗'))
    print()
    return best


print('扫描核显承担的列比例（CPU 线程 6）：')
print()
run(1024, 2736, 512)
print('同样扫描，但 CPU 只用 4 线程（减少与核显争抢内存总线）：')
print()
run(1024, 2736, 512, threads=4)
