"""What does a switch cost? The size of the slice is set by the ramp, not by the theory.

Time slicing beats concurrency because the two engines never run at once, so power arbitration is
never triggered and each works at its own full rate. The arithmetic says splitting a fraction f to
the iGPU gives 1/(1-0.513f), so f=0.5 is 1.35x and f=1.0 is 2.05x.

But that assumes switching is free. It is not obviously free: the CPU's death grip on the power
budget has to be released and the iGPU has to come back up, and neither is instant. If the ramp costs
one slow call of 45 ms, a slice must be several calls long to amortise it; if ramp costs 500 ms, the
whole idea needs slices of seconds and the training step, which alternates GEMM and non-GEMM every
layer, cannot provide them.

Measured: after a CPU burst of a given length, run the GPU GEMM repeatedly and time each call
separately. The first call after the burst carries the ramp; the steady-state calls that follow show
what it settles to. Sweep the burst length so the ramp's dependence on how long the CPU was busy is
visible.

Decision rule, fixed in advance: if a slice of five GEMM calls recovers to within 10% of the
idle-CPU rate, slicing is viable at a granularity the training could supply. If recovery takes longer
than that, the slices would have to be larger than a training step contains and the idea fails
regardless of its arithmetic.
"""
from __future__ import annotations

import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


import pyopencl as cl  # noqa: E402
from measure_guard import assert_idle, stamp  # noqa: E402

SRC = r'''
#define TS 64
#define TK 16
#define TM 4
#define TN 4
__kernel void gemm_fast(const int M, const int N, const int K,
                        const __global float* restrict A,
                        const __global float* restrict B,
                        __global float* restrict C) {
    __local float As[TS][TK + 1];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 16 + tx;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += TK) {
        { int f4 = lane; int r = (f4*4)/TK, c = (f4*4)%TK;
          int gr = by + r, gc = k0 + c; float4 v = (float4)0.f;
          if (gr < M && gc + 3 < K) v = *(const __global float4*)(A + gr*K + gc);
          As[r][c+0]=v.x; As[r][c+1]=v.y; As[r][c+2]=v.z; As[r][c+3]=v.w; }
        { int f4 = lane; int k = (f4*4)/TS, n = (f4*4)%TS;
          int gk = k0 + k, gn = bx + n; float4 v = (float4)0.f;
          if (gk < K && gn + 3 < N) v = *(const __global float4*)(B + gk*N + gn);
          #pragma unroll
          for (int q = 0; q < 4; q++) { int cc = n + q;
              Bs[k][(cc % TN) * (TS / TN) + (cc / TN)] =
                  (q==0)?v.x:((q==1)?v.y:((q==2)?v.z:v.w)); } }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[ty*TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][j*(TS/TN) + tx];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] += a[i]*b[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i = 0; i < TM; i++) { int gr = by + ty*TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) { int gc = bx + tx*TN + j;
            if (gr < M && gc < N) C[gr*N + gc] = acc[i][j]; } }
}
'''

_plat = cl.get_platforms()[0]
ctx = cl.Context(devices=[_plat.get_devices()[0]])
q = cl.CommandQueue(ctx)
mf = cl.mem_flags

M, N, K = 2048, 2736, 512
FLOP = 2.0 * M * N * K
rng = np.random.default_rng(5)
A = np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float32))
B = np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float32))
ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
kern = cl.Kernel(cl.Program(ctx, SRC).build(), 'gemm_fast')
gsz = ((N + 63) // 64 * 16, (M + 63) // 64 * 16)
ta, tb = torch.from_numpy(A), torch.from_numpy(B)


def gpu_call():
    kern(q, gsz, (16, 16), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
    q.finish()


def cpu_burst(ms):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < ms / 1000.0:
        torch.mm(ta[:1024], tb[:, :512])


assert_idle()
stamp('切换成本：CPU 忙完之后,核显几秒恢复')
torch.set_num_threads(6)
print()
print('  %dx%dx%d = %.1f GFLOP' % (M, N, K, FLOP / 1e9))
print()

for _ in range(3):
    gpu_call()
gts = []
for _ in range(8):
    t0 = time.perf_counter()
    gpu_call()
    gts.append(time.perf_counter() - t0)
base = statistics.median(gts[2:])
print('  基准（CPU 完全空闲）: %.3f ms ⇒ %.1f GFLOPS' % (base * 1e3, FLOP / base / 1e9))
print()

print('  %-12s %s' % ('CPU 忙多久', '之后连续 6 次核显调用的耗时 ms（依次）'))
print('  ' + '-' * 74)
for burst_ms in (50, 200, 1000):
    print('  %-12s' % ('%d ms' % burst_ms), end='')
    row = []
    for trial in range(2):
        cpu_burst(burst_ms)
        t0 = time.perf_counter()
        for _ in range(6):
            t0c = time.perf_counter()
            gpu_call()
            row.append((time.perf_counter() - t0c) * 1e3)
    # take the second trial to avoid first-run warm effects
    row = row[6:]
    print(' ' + '  '.join('%6.1f' % v for v in row))
print()
print('  判据（事先固定）: 前 5 次调用内恢复到空闲速率的 10% 以内 ⇒ 切片可行')
print('  空闲速率 %.3f ms ⇒ 判据上限 %.3f ms' % (base * 1e3, base * 1e3 * 1.1))
