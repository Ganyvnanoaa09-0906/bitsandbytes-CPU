"""Is the iGPU's slowness a wake-up cost, and does a keep-alive dummy fix it?

Two measurements of the same kernel on the same shape disagree by 4.5x: a tight loop reports 409
GFLOPS and isolated calls report 90. The kernel is identical, so the difference is what happens
between calls. The hypothesis is power gating -- a laptop APU gates the iGPU when idle and each
isolated call pays a wake-up and clock ramp.

Four conditions on the same shape (mlp w12, 2048x2736x512, about 5.7 GFLOP):

  isolated      one call, then 200 ms of CPU work, then the next -- forces a wake each time
  tight        20 calls back to back
  slept        one call, then 300 ms sleep, then the next
  keep-alive   a tiny register-only FMA kernel launched between real calls, so the iGPU never idles

The keep-alive case is the one that matters practically: if a cheap dummy kernel holds the clock up,
the real calls should approach the tight-loop rate instead of the isolated one. It is also the case
with a cost, because on a 15 W part the dummy's power comes out of the CPU's budget, so the CPU side
is timed in every condition to see whether that shows.

Decision rule, fixed in advance: if keep-alive gets within 20% of the tight-loop rate, the mechanism
is confirmed and the strategy is viable. If it does not, the slowness is not a wake-up cost and the
tighter number came from something else.
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
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


import pyopencl as cl
from scripts.lib.measure_guard import assert_idle, stamp

GEMM = r'''
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

# register-only, no memory traffic: keeps the clock up without touching bandwidth
DUMMY = r'''
__kernel void keepalive(__global float* out, const int iters) {
    float a = (float)(get_global_id(0) % 7) * 0.13f + 1.0f;
    float b = (float)(get_global_id(0) % 5) * 0.17f + 1.0f;
    float c = 0.0f, d = 0.0f, e = 0.0f, f = 0.0f;
    for (int i = 0; i < iters; i++) {
        c = fma(a, b, c); d = fma(b, c, d); e = fma(c, d, e); f = fma(d, e, f);
    }
    if (get_global_id(0) == 999999) out[0] = c + d + e + f;
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
outbuf = cl.Buffer(ctx, mf.WRITE_ONLY, 4)
kern = cl.Kernel(cl.Program(ctx, GEMM).build(), 'gemm_fast')
dummy = cl.Kernel(cl.Program(ctx, DUMMY).build(), 'keepalive')
gsz = ((N + 63) // 64 * 16, (M + 63) // 64 * 16)

assert_idle()
stamp('核显：唤醒假说与 keep-alive')
print()
print('  形状 %dx%dx%d = %.1f GFLOP' % (M, N, K, FLOP / 1e9))
print()

# CPU side: a fixed chunk of torch work, timed, to see whether the dummy steals power
ta = torch.from_numpy(A)
tb = torch.from_numpy(B)


def cpu_burst(ms=200.0):
    torch.mm(ta, tb)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < ms / 1000.0:
        torch.mm(ta[:512], tb[:, :512])
    return time.perf_counter() - t0


def one_gemm():
    kern(q, gsz, (16, 16), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)


kern(q, gsz, (16, 16), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
q.finish()

print('  %-14s %12s %12s %14s' % ('条件', '每次 ms', 'GFLOPS', 'CPU burst ms'))
print('  ' + '-' * 56)


def run_cond(name, gap_s, keepalive, n=8):
    ts = []
    cpus = []
    for i in range(n):
        if keepalive:
            dummy(q, (256,), (256,), outbuf, np.int32(2000))
        t0 = time.perf_counter()
        one_gemm()
        q.finish()
        ts.append(time.perf_counter() - t0)
        if gap_s > 0:
            if gap_s >= 0.2:
                cpus.append(cpu_burst(gap_s * 1000))
            else:
                time.sleep(gap_s)
    med = statistics.median(ts[2:]) if len(ts) > 3 else statistics.median(ts)
    cm = ('%.1f' % (statistics.median(cpus) * 1e3)) if cpus else '—'
    print('  %-14s %12.3f %12.1f %14s' % (name, med * 1e3, FLOP / med / 1e9, cm))
    return med


tight = run_cond('紧循环', 0.0, False)
iso = run_cond('孤立(200ms CPU)', 0.2, False)
slept = run_cond('孤立(300ms sleep)', 0.3, False)
ka = run_cond('keep-alive', 0.2, True)

print()
print('  紧循环 / 孤立 = %.2fx   ⇒ %s' %
      (iso / tight, '唤醒开销显著' if iso / tight > 1.5 else '差异不大'))
print('  keep-alive 对紧循环 = %.2fx  ⇒ %s' %
      (ka / tight, '【假说成立，方案可行】✓' if ka / tight <= 1.2 else '【未达判据】✗'))
print()
print('  判据（事先固定）: keep-alive 进入紧循环的 20% 以内 ⇒ 机制确认、方案可行')
