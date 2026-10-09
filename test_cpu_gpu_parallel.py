"""test_cpu_gpu_parallel.py -- does running both actually help?

Report table 69 records concurrency for training at 0.83x, a net loss, and the
report's own note attributes it to per-operation synchronisation rather than
hardware. But that measurement was taken before anyone read the output back, and
the kernel involved computes about a quarter of the matrix correctly -- so the
number is not trustworthy and the question is open again.

It matters because it decides the user's target. The corrected kernel now reaches
300 GFLOPS against the CPU's 198.5; if the two can genuinely work at once, a
22-hour round becomes roughly 8.8 hours, which is the goal. If they contend, it
does not.

Method: one GEMM at the model's real shape, split by columns of B.
  * iGPU alone: full N
  * CPU alone:  full N
  * both:       iGPU takes the first half of the columns, CPU the second

The iGPU call is enqueued without a blocking finish, so the CPU work overlaps the
kernel's execution -- which is the actual question. Synchronising first and then
timing would measure nothing.

Hardware note: this runs for well under a minute. The user has asked not to leave
the machine under sustained load overnight, and this respects that; no training run
is started.
"""
from __future__ import annotations

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
    const int NT = TX * TY, lane = ty * TX + tx;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int p = lane; p < TS * TK / 4; p += NT) {
            int f = p * 4, r = f / TK, c = f % TK;
            int gr = by + r, gc = k0 + c;
            float4 v = (float4)0.f;
            if (gr < M && gc + 3 < K) v = *(const __global float4*)(A + gr * K + gc);
            As[r][c+0]=v.x; As[r][c+1]=v.y; As[r][c+2]=v.z; As[r][c+3]=v.w;
        }
        for (int p = lane; p < TK * TS / 4; p += NT) {
            int f = p * 4, r = f / TS, c = f % TS;
            int gr = k0 + r, gc = bx + c;
            float4 v = (float4)0.f;
            if (gr < K && gc + 3 < N) v = *(const __global float4*)(B + gr * N + gc);
            Bs[r][c+0]=v.x; Bs[r][c+1]=v.y; Bs[r][c+2]=v.z; Bs[r][c+3]=v.w;
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

M, N, K = 1024, 2736, 512
TS = 64
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
prg = cl.Program(ctx, KERNEL).build()
kern = cl.Kernel(prg, 'gemm')

rng = np.random.default_rng(7)
A = rng.standard_normal((M, K)).astype(np.float32)
B = rng.standard_normal((K, N)).astype(np.float32)
torch.set_num_threads(6)
At = torch.from_numpy(A)
Bt = torch.from_numpy(B)


ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(A))


def gpu_gemm(Bmat, Nn, wait=True):
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(Bmat))
    bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * Nn * 4)
    gsz = ((Nn + TS - 1) // TS * 8, (M + TS - 1) // TS * 8)
    # A must be passed! An earlier version of this file passed None here, which
    # produced NaN and ran 23x slower, and the parallelism conclusion drawn from
    # it was therefore meaningless.
    kern(queue, gsz, (8, 8), np.int32(M), np.int32(Nn), np.int32(K), ba, bb, bc)
    if wait:
        queue.finish()
        out = np.zeros((M, Nn), dtype=np.float32)
        cl.enqueue_copy(queue, out, bc).wait()
        return out
    return bb, bc


# 先做一次正确的对照：核显跑全量，与 numpy 比
t0 = time.perf_counter()
out_full = gpu_gemm(B, N)
t_gpu = time.perf_counter() - t0
ref = A @ B
d = float(np.abs(out_full - ref).max()) / float(np.abs(ref).max())
print('核显全量: %.3f s (%.1f GFLOPS)  正确性 相对差 %.2e %s'
      % (t_gpu, 2.0 * M * N * K / t_gpu / 1e9, d, '✓' if d < 1e-4 else '✗'))
del out_full

_ = At @ Bt
_ = At @ Bt
t0 = time.perf_counter()
_ = At @ Bt
t_cpu = time.perf_counter() - t0
print('CPU 全量: %.3f s (%.1f GFLOPS)' % (t_cpu, 2.0 * M * N * K / t_cpu / 1e9))
print()

# 并行：核显跑前一半列（不等待），CPU 跑后一半，最后一起等
half = N // 2
A2 = np.ascontiguousarray(A)
B_first = np.ascontiguousarray(B[:, :half])
B_second = np.ascontiguousarray(B[:, half:])
N2 = N - half

# 先各自单独测那一半
t0 = time.perf_counter(); o1 = gpu_gemm(B_first, half); tg_half = time.perf_counter() - t0
t0 = time.perf_counter(); _ = At @ torch.from_numpy(B_second); tc_half = time.perf_counter() - t0
print('核显跑左半 (N=%d): %.3f s' % (half, tg_half))
print('CPU  跑右半 (N=%d): %.3f s' % (N2, tc_half))
print('串行合计 = %.3f s' % (tg_half + tc_half))
print()

# 真正并行：先入队核显（不阻塞），同时 CPU 算另一半，最后等核显
best = 1e9
for trial in range(3):
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B_first)
    bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * half * 4)
    gsz = ((half + TS - 1) // TS * 8, (M + TS - 1) // TS * 8)
    t0 = time.perf_counter()
    kern(queue, gsz, (8, 8), np.int32(M), np.int32(half), np.int32(K), ba, bb, bc)
    _ = At @ torch.from_numpy(B_second)        # CPU 与核显同时工作
    queue.finish()                             # 一起等
    dt = time.perf_counter() - t0
    best = min(best, dt)
    del bb, bc
print('并行（核显左半 + CPU 右半）: %.3f s' % best)
print()
serial = t_gpu + t_cpu
print('判据:')
print('  各自单独跑全量串行 = %.3f + %.3f = %.3f s' % (t_gpu, t_cpu, serial))
print('  并行完成同样工作量   = %.3f s  ⇒ 相对串行 %.2fx' % (best, serial / best))
print('  相对只用更快的核显   = %.3f s  ⇒ %.2fx' % (t_gpu, t_gpu / best))
if best < t_gpu * 0.95:
    print('  ⇒ 【并行有效 ✓✓ CPU 是纯增益 ✓】')
elif best < t_gpu * 1.05:
    print('  ⇒ 【并行无收益 ✗ 核显已占满共享资源 ✓】')
else:
    print('  ⇒ 【并行有害 ✗✗ 互相拖累 ✓】')
