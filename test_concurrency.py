"""test_concurrency.py -- do two independent workloads actually overlap?

This is a different question from the one the earlier tests asked, and I had been
conflating them.

  splitting: one matmul, cut in half, iGPU takes some columns and CPU the rest. Measured
             2.80x over serial at the bare-call level (test_cpu_gpu_parallel.py) and
             0.07-0.62x through the torch wrapper (igpu_mm.py).
  concurrency: two separate workloads running at the same time, each with its own inputs
             and outputs. This is what table 69's 0.83x for concurrent training refers
             to, and what I claimed did not reproduce.

The hypothesis for concurrency is memory bandwidth. This device is an APU: the iGPU has
no memory of its own and reads the same bus as the CPU cores, measured earlier at about
32 GB/s shared. If that is the constraint, two memory-heavy workloads must slow each
other down, and the amount depends on arithmetic intensity -- a large GEMM reuses each
loaded value many times and should tolerate sharing, while an operator-at-a-time training
step touches memory constantly and should not.

So: time each workload alone, then time them together, and compare against the three
reference points. If together is near max(alone) the bus is not limiting; near the sum,
there is no overlap at all; above max but below sum, there is partial contention, which is
what 0.83x would look like.

Both workloads write to disjoint buffers and are launched without any dependence between
them, so nothing in the API forces them to serialise.
"""
from __future__ import annotations

import statistics
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
import pyopencl as cl  # noqa: E402

KERNEL = r'''
#define TS 64
#define TK 8
#define TX 8
#define TY 8
#define TM 8
#define TN 8
__kernel void gemm(const int M, const int N, const int K,
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
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
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

# a pure streaming kernel, to see whether the bus itself saturates
BANDWIDTH = r'''
__kernel void stream(const int n, const __global float* restrict X,
                     __global float* restrict Y) {
    const int gid = get_global_id(0);
    if (gid < n) Y[gid] = X[gid] * 1.0001f;
}
'''

TS = 64
ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
prg = cl.Program(ctx, KERNEL).build()
kern = cl.Kernel(prg, 'gemm')
bw = cl.Kernel(cl.Program(ctx, BANDWIDTH).build(), 'stream')

print('设备 %s | %d CU' % (dev.name, dev.max_compute_units))
print()


def make(M, N, K, seed):
    rng = np.random.default_rng(seed)
    A = np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float32))
    B = np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float32))
    out = np.empty((M, N), dtype=np.float32)
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    gsz = ((N + TS - 1) // TS * 8, (M + TS - 1) // TS * 8)
    return dict(A=A, B=B, out=out, ba=ba, bb=bb, bc=bc, gsz=gsz, M=M, N=N, K=K)


def gpu_launch(w):
    kern(queue, w['gsz'], (8, 8), np.int32(w['M']), np.int32(w['N']), np.int32(w['K']),
         w['ba'], w['bb'], w['bc'])


def gpu_run(w):
    gpu_launch(w)
    queue.finish()


def timeit(fn, reps=5):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


SHAPE = (1024, 2736, 512)
w1 = make(*SHAPE, seed=1)
w2 = make(*SHAPE, seed=2)
w3 = make(*SHAPE, seed=3)

torch.set_num_threads(6)
At = torch.from_numpy(w1['A'])
Bt = torch.from_numpy(w1['B'])
Ct = torch.empty(w1['M'], w1['N'])
At2 = torch.from_numpy(w2['A'])
Bt2 = torch.from_numpy(w2['B'])

print('形状 %dx%dx%d；两个【互相独立】的完整 GEMM' % SHAPE)
print()

# CPU reference, and the same result written into different output buffers so the two
# workloads genuinely do not share state
t_cpu = timeit(lambda: torch.mm(At, Bt, out=Ct))
t_gpu = timeit(lambda: gpu_run(w1))
print('  CPU 单独（6 线程）      %.3f s' % t_cpu)
print('  iGPU 单独               %.3f s' % t_gpu)
print('  串行（先 CPU 后 iGPU）  %.3f s' % (t_cpu + t_gpu))
print()

# concurrency: launch the iGPU kernel, run the CPU gemm while it executes, then wait.
# w3's own buffers are used for the CPU side so the two are fully independent.
At3 = torch.from_numpy(w3['A'])
Bt3 = torch.from_numpy(w3['B'])
Ct3 = torch.empty(w3['M'], w3['N'])


def concurrent():
    gpu_launch(w2)                        # enqueue, do not wait
    torch.mm(At3, Bt3, out=Ct3)           # CPU works meanwhile
    queue.finish()


t_both = timeit(concurrent)
best = max(t_cpu, t_gpu)
worst = t_cpu + t_gpu
print('  并发（iGPU 启一个 + CPU 跑另一个）  %.3f s' % t_both)
print()
print('  参照: max(单独) = %.3f   sum(单独) = %.3f' % (best, worst))
print('  并发效率: 相对 max = %.2fx   相对 sum = %.2fx'
      % (best / t_both, worst / t_both))
verdict = ('完全并发（总线未饱和）' if t_both < best * 1.1 else
           '部分争抢' if t_both < worst * 0.95 else '完全没有并发')
print('  ⇒ %s' % verdict)
print()

# is the bus the reason?  measure achievable bandwidth with both engines pulling
NB = 8 * 1024 * 1024
X = np.ascontiguousarray(np.random.rand(NB).astype(np.float32))
Y = np.empty_like(X)
bx_ = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
by_ = cl.Buffer(ctx, mf.WRITE_ONLY, X.nbytes)


def bw_gpu():
    bw(queue, (NB,), None, np.int32(NB), bx_, by_)
    queue.finish()


t_bw_gpu = timeit(bw_gpu, reps=3)
gb = 2 * X.nbytes / 1e9
print('  核显拷贝带宽 (%.0f MB 读+写): %.1f GB/s' % (X.nbytes / 2**20, gb / t_bw_gpu))


def bw_cpu():
    Y[:] = X * np.float32(1.0001)


t_bw_cpu = timeit(bw_cpu, reps=3)
print('  CPU 拷贝带宽:  %.1f GB/s' % (gb / t_bw_cpu))


def bw_both():
    bw(queue, (NB,), None, np.int32(NB), bx_, by_)
    Y[:] = X * np.float32(1.0001)
    queue.finish()


t_bw_both = timeit(bw_both, reps=3)
print('  两者同时:      %.1f GB/s（单侧的 %.2fx）'
      % (gb / t_bw_both, (gb / t_bw_both) / (gb / t_bw_gpu)))
print()
print('判据: 并发若接近 max ⇒ 总线不限制；若接近 sum ⇒ 完全串行；')
print('      带宽同时跑若明显低于单侧之和 ⇒ 总线是共享且饱和的。')
