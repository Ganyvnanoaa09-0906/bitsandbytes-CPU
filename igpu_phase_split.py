"""The decisive test: is the cost in the enqueue, in the sync, or in the device?

The previous run measured 0.172 ms for a GEMM that takes 15 ms, because it timed the enqueue only
and called finish() outside both timers. That proves enqueuing is cheap even with six torch threads
saturating the CPU, but it does not say when the kernel actually completes. 33446 GFLOPS is not a
speed, it is a missing stopwatch.

So separate the three phases and time each:

    enqueue    the launch call itself
    device     from launch to the kernel actually finishing, with the CPU idle
    sync       the finish() call, under three conditions -- CPU idle throughout, CPU busy during
               the kernel, and CPU busy only after the kernel has already completed

If finish() returns almost immediately after a busy period, the device ran while the CPU worked and
asynchronous offload is viable. If finish() still costs tens of milliseconds, the device itself is
being held back by host contention and the enqueue cost was a red herring.

Decision rule, fixed in advance: if total time from launch to a completed finish during CPU load
stays under 2x the CPU-idle figure, asynchronous offload works. Beyond that, it does not.
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


def launch():
    kern(q, gsz, (16, 16), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)


def cpu_burst(ms):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < ms / 1000.0:
        torch.mm(ta[:512], tb[:, :512])


assert_idle()
stamp('三段分离：入队 / 设备 / 同步')
torch.set_num_threads(6)
print()
print('  %dx%dx%d = %.1f GFLOP, 6 个 torch 线程' % (M, N, K, FLOP / 1e9))
print()

for _ in range(3):
    launch()
q.finish()

print('  1. 紧循环（CPU 空闲，launch+finish 一起）')
ts = []
for _ in range(8):
    t0 = time.perf_counter()
    launch()
    q.finish()
    ts.append(time.perf_counter() - t0)
tight = statistics.median(ts[2:])
print('     每次 %.3f ms → %.1f GFLOPS' % (tight * 1e3, FLOP / tight / 1e9))
print()

print('  2. 三段分离：CPU 空闲')
la, fi = [], []
for _ in range(8):
    t0 = time.perf_counter()
    launch()
    t1 = time.perf_counter()
    q.finish()
    t2 = time.perf_counter()
    la.append(t1 - t0)
    fi.append(t2 - t1)
print('     入队 %.3f ms · 同步 %.3f ms · 合计 %.3f ms'
      % (statistics.median(la) * 1e3, statistics.median(fi) * 1e3,
         (statistics.median(la) + statistics.median(fi)) * 1e3))
print()

print('  3. 三段分离：CPU 在【内核执行期间】忙')
la, fi, cb = [], [], []
for _ in range(8):
    t0 = time.perf_counter()
    launch()
    t1 = time.perf_counter()
    t0b = time.perf_counter()
    cpu_burst(30)
    cb.append(time.perf_counter() - t0b)
    t2 = time.perf_counter()
    q.finish()
    t3 = time.perf_counter()
    la.append(t1 - t0)
    fi.append(t3 - t2)
print('     入队 %.3f ms · CPU 忙 %.1f ms · 【同步 %.3f ms】· 合计 %.3f ms'
      % (statistics.median(la) * 1e3, statistics.median(cb) * 1e3,
         statistics.median(fi) * 1e3,
         (statistics.median(la) + statistics.median(cb) + statistics.median(fi)) * 1e3))
print()

print('  4. 对照：CPU 忙【之前先等在同步上】（旧模式）')
la, fi = [], []
for _ in range(6):
    t0 = time.perf_counter()
    launch()
    t1 = time.perf_counter()
    q.finish()
    t2 = time.perf_counter()
    t0b = time.perf_counter()
    cpu_burst(30)
    la.append(t1 - t0)
    fi.append(t2 - t1)
print('     入队 %.3f ms · 【同步 %.3f ms】' % (statistics.median(la) * 1e3, statistics.median(fi) * 1e3))
print()
print('  === 判读 ===')
print('  紧循环基准: %.3f ms ⇒ 设备侧真实执行时间约在这个量级' % (tight * 1e3))
print('  第 3 组的同步若 ≈ 0 ⇒ 设备在 CPU 忙时【已经算完了】⇒ 异步可用 ✓')
print('  第 4 组的同步若 ≈ 100 ms ⇒ 那才是"同步跟在内核后"的旧模式代价 ✗')
