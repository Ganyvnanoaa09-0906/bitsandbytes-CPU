"""Reserve a core for the driver: does torch's thread count explain the 6.6x?

The measured facts. The same GEMM on the same shape takes 15.3 ms in a tight loop with the CPU idle,
and 101 ms when the CPU has a 200 ms torch burst around it. A 300 ms sleep instead of the burst gives
the same 101 ms, so nothing is sleeping -- the difference tracks whether the host CPU is busy.

The plausible mechanism is that this part has six cores, six threads and no SMT, and torch is
configured to use all six. The OpenCL driver progresses a command from its own threads, and with no
spare core those threads are preempted continuously, so the launch and the synchronisation stall even
though the device itself is idle.

If so the fix is one line: give the driver a core by running torch on five threads. The CPU side
loses a sixth of its throughput and the iGPU side may recover most of the 6.6x, which would change
the combined arithmetic completely.

Sweep the torch thread count from 1 to 6 and measure both sides under each setting, so the trade is
visible rather than assumed.

Decision rule, fixed in advance: if five torch threads bring the iGPU within 2x of its tight-loop
rate while the CPU keeps at least 80% of its throughput, the mechanism is confirmed and the combined
figure is worth computing. If the iGPU stays slow across all thread counts, the mechanism is
something else and reserving a core is not the answer.
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

assert_idle()
stamp('给驱动留一个核：线程数扫描')
print()

ta = torch.from_numpy(A)
tb = torch.from_numpy(B)

# CPU throughput reference at each thread count, measured alone
print('  CPU 侧吞吐（单独测）:')
cpu_ref = {}
for nt in (1, 2, 4, 5, 6):
    torch.set_num_threads(nt)
    torch.mm(ta, tb)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        torch.mm(ta, tb)
        ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    cpu_ref[nt] = med
    print('    %d 线程: %.3f ms  (%.1f GFLOPS)' % (nt, med * 1e3, FLOP / med / 1e9))
print()

print('  核显 GEMM 在同一线程数下（CPU 同时跑 torch burst）:')
print('  %-10s %12s %12s %14s' % ('torch 线程', '核显 ms', '核显 GFLOPS', 'CPU 并发 ms'))
print('  ' + '-' * 54)
rows = []
for nt in (6, 5, 4, 2, 1):
    torch.set_num_threads(nt)
    kern(q, gsz, (16, 16), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
    q.finish()
    gts, cts = [], []
    for _ in range(5):
        t0 = time.perf_counter()
        kern(q, gsz, (16, 16), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
        gts.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        torch.mm(ta, tb)
        cts.append(time.perf_counter() - t0)
        q.finish()
    g = statistics.median(gts[1:]) if len(gts) > 2 else statistics.median(gts)
    c = statistics.median(cts)
    rows.append((nt, g, c))
    print('  %-10d %12.3f %12.1f %14.3f' % (nt, g * 1e3, FLOP / g / 1e9, c * 1e3))
print()
print('  判据（事先固定）: torch 5 线程时核显进入紧循环的 2× 以内,且 CPU 保住 80% ⇒ 机制确认')
best = min(rows, key=lambda r: r[1])
print('  ⇒ 核显最快的一档是 %d 线程: %.1f GFLOPS' % (best[0], FLOP / best[1] / 1e9))
print('  ⇒ 而紧循环基准是 375.4 GFLOPS')
