"""How does the iGPU's kernel time scale with CPU intensity? The other side of the trade.

Everything so far asked whether work can be handed to the iGPU while the CPU does its full training
load. But the iGPU alone runs this GEMM in 14.7 ms against the CPU's 30.2, so it is twice as fast
when the CPU is idle. The interesting question is therefore not "can the CPU share its work" but
"how much CPU activity can coexist before the iGPU stops being the faster engine".

We know two points: CPU idle gives 14.7 ms, six CPU threads give about 59. Nothing in between has
been measured, and the inversion only makes sense if the curve is flat for the first few threads.

Sweep the CPU thread count from 0 to 6 while the iGPU runs the same GEMM, measuring the GPU's
launch-to-finish time and the CPU's own throughput at each setting. Both matter: the crossover is
where the sum stops beating the CPU alone.

Decision rule, fixed in advance: if the iGPU holds within 20% of its idle time while the CPU keeps at
least three threads of useful work, the inversion is viable and the combined figure gets computed.
If the iGPU degrades as soon as any CPU activity appears, the interference is not a matter of degree
and no split works.
"""
from __future__ import annotations

import statistics
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')

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


assert_idle()
stamp('核显为主：内核时间 vs CPU 强度')
print()
print('  %-10s %12s %14s %14s %12s' %
      ('CPU 线程', '核显 ms', '核显 GFLOPS', 'CPU GFLOPS', '合计 GFLOPS'))
print('  ' + '-' * 68)
rows = []
for nt in (0, 1, 2, 3, 4, 6):
    torch.set_num_threads(max(1, nt))
    stop = threading.Event()
    ccount = [0]

    def cpu_load():
        while not stop.is_set():
            torch.mm(ta, tb)
            ccount[0] += 1

    th = None
    if nt > 0:
        th = threading.Thread(target=cpu_load, daemon=True)
        th.start()
        time.sleep(0.8)
    gpu_call()
    gts = []
    for _ in range(5):
        t0 = time.perf_counter()
        gpu_call()
        gts.append(time.perf_counter() - t0)
    g = statistics.median(gts[1:])
    c0 = ccount[0]
    t0 = time.perf_counter()
    time.sleep(1.0)
    cpu_flops = (ccount[0] - c0) * FLOP / (time.perf_counter() - t0)
    stop.set()
    if th:
        th.join(timeout=5)
    rows.append((nt, g, cpu_flops))
    print('  %-10d %12.3f %14.1f %14.1f %12.1f'
          % (nt, g * 1e3, FLOP / g / 1e9, cpu_flops / 1e9,
             (FLOP / g + cpu_flops) / 1e9))

print()
base = rows[0][1]
print('  判据（事先固定）: 核显保持空闲时间的 20% 以内,且 CPU 还能干 3 线程的活 ⇒ 倒置可行')
for nt, g, cf in rows:
    if nt >= 3:
        print('    %d 线程: 核显 %.2f× 空闲 · CPU %.1f GFLOPS · 合计 %.1f GFLOPS'
              % (nt, g / base, cf / 1e9, (FLOP / g + cf) / 1e9))
print()
print('  对照: CPU 单干（6 线程）= 190.0 GFLOPS · 核显单干（CPU 闲）= 389.9 GFLOPS')
