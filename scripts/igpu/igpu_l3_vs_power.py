"""Is the iGPU's problem the CPU's power, or the CPU's memory footprint? One variable: working-set size.

The previous run found that a single thread spinning on pure ALU work costs the iGPU 3%, while 7.62
measured a single thread running torch.mm costing it 5.1x. Same thread count, same "the CPU is busy"
description, opposite results. What differs is what the thread touches: the ALU spin touches nothing,
and torch.mm at 1024x512 has a 48 MB working set.

If the shared 8 MB L3 is the mechanism, then the variable is the working set and not the thread count
or the power. The iGPU's own GEMM wants A at 4 MB and B at 5.6 MB resident; a CPU workload that streams
48 MB through the same cache evicts them, after which the iGPU reads from DDR4 at a bandwidth it now
shares.

That is testable with one variable: run the same torch.mm, one thread throughout, and change only the
matrix size so the working set moves from well inside L3 to well outside it. Everything else -- thread
count, operation, library, power profile -- stays fixed.

    n=256    2 MB working set      fits in the 8 MB L3 with room for the iGPU's data
    n=512    6 MB                  marginal
    n=1024  24 MB                  exceeds L3
    n=2048  96 MB                  far exceeds it

Criterion, fixed in advance, and it decides between two mechanisms that need opposite responses: if the
iGPU's throughput falls as the working set grows while the thread count is fixed at one, the cause is
cache and memory rather than power, and 7.62's conclusion is wrong. If the iGPU degrades equally at
every size, then merely running torch is enough and power remains the better explanation.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

import statistics
import subprocess
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
from scripts.lib.measure_guard import assert_idle, stamp

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


def gpu_call():
    kern(q, gsz, (16, 16), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
    q.finish()


def gpu_steady(reps=6):
    for _ in range(2):
        gpu_call()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        gpu_call()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


assert_idle()
torch.set_num_threads(1)
stamp('核显被打下来:是 CPU 的功耗,还是它的【工作集】')
print()

idle = gpu_steady()
print('  CPU 空闲基准: %.3f ms ⇒ %.1f GFLOPS' % (idle * 1e3, FLOP / idle / 1e9))
print()
print('  %-22s %12s %10s %12s %12s %10s' %
      ('CPU 负载（1 线程）', 'CPU ms', '工作集', '核显 ms', 'GFLOPS', '相对空闲'))
print('  ' + '-' * 84)

res = []
for n in (256, 512, 1024, 2048):
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    ws = (n * n * 4 * 3) / 1024 ** 2
    stop = [False]
    import threading

    def worker(a=a, b=b, stop=stop):
        while not stop[0]:
            torch.mm(a, b)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(1.5)
    # time one CPU matmul for context
    t0 = time.perf_counter()
    torch.mm(a, b)
    cpu_ms = (time.perf_counter() - t0) * 1e3
    gt = gpu_steady()
    stop[0] = True
    t.join(timeout=2)
    time.sleep(1.5)
    res.append((n, ws, gt))
    print('  %-22s %12.2f %9.1f MB %12.3f %12.1f %9.2f×'
          % ('torch.mm %d³' % n, cpu_ms, ws, gt * 1e3, FLOP / gt / 1e9, gt / idle))

print()
print('  判据（事先固定）:')
print('    核显随工作集增大而变慢（线程数固定为 1）⇒ ★【是缓存/内存,不是功耗】⇒ 7.62 要改')
print('    各尺寸下核显掉得一样多            ⇒ 是"跑 torch"本身 ⇒ 功耗仍是更好的解释')
print()
small = [r for r in res if r[1] < 8]
big = [r for r in res if r[1] > 20]
if small and big:
    s = statistics.mean(r[2] for r in small) / idle
    b = statistics.mean(r[2] for r in big) / idle
    print('  工作集 <8 MB 的平均倍数: %.2f×' % s)
    print('  工作集 >20 MB 的平均倍数: %.2f×' % b)
    print('  ⇒ %s' % ('★【是缓存/内存】✓ —— 同样的线程数,只改了工作集'
                      if b > s * 1.3 else '差异不显著'))
