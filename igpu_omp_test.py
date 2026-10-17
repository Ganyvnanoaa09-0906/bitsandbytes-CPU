"""Is the 25% residual the OpenMP pool spinning? Measure the CPU during the GPU slice.

Time slicing gives 308 GFLOPS in a slice against 387.6 when the CPU has been idle, and the guess is
that torch's OpenMP worker threads keep spinning for a while after a matmul, so the CPU is not really
idle during the GPU phase and power arbitration stays partly engaged.

Two ways to check. First, measure the CPU's actual utilisation during a GPU-only stretch that follows
a CPU burst -- if the pool is spinning, the machine will not be idle even though no work is being
done. Second, drop torch to one thread after the burst and see whether the GPU slice recovers.

The environment variable form, OMP_WAIT_POLICY=PASSIVE, has to be set before the interpreter starts,
so it is tested by re-invoking this script rather than set inside it. The script prints which mode it
is in so the two runs can be told apart.

Decision rule, fixed in advance: if either fixing the pool or PASSIVE brings a slice within 5% of the
idle rate of 14.8 ms, the residual is the pool and the slice figure rises to near 1.96x. If neither
moves it, the residual is something else in the power state and the 1.62x upper bound stands.
"""
from __future__ import annotations

import os
import statistics
import sys
import threading
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


def cpu_burst(ms=200.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < ms / 1000.0:
        torch.mm(ta[:1024], tb[:, :512])


def cpu_pct(duration_s):
    """Fraction of total machine CPU time used during a wall-clock window."""
    import subprocess
    t0 = time.perf_counter()
    a = subprocess.run(['powershell', '-NoProfile', '-Command',
                        '(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage'
                        ' -Average).Average'], capture_output=True, text=True)
    time.sleep(duration_s)
    b = subprocess.run(['powershell', '-NoProfile', '-Command',
                        '(Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage'
                        ' -Average).Average'], capture_output=True, text=True)
    try:
        return float(b.stdout.strip()), float(a.stdout.strip())
    except Exception:
        return -1.0, -1.0


assert_idle()
stamp('残余惩罚是不是 OMP 空转')
print()
print('  模式: OMP_WAIT_POLICY=%s  KMP_BLOCKTIME=%s' %
      (os.environ.get('OMP_WAIT_POLICY', '(未设)'), os.environ.get('KMP_BLOCKTIME', '(未设)')))
print()

for _ in range(3):
    gpu_call()
gts = []
for _ in range(8):
    t0 = time.perf_counter()
    gpu_call()
    gts.append(time.perf_counter() - t0)
base = statistics.median(gts[2:])
print('  空闲基准: %.3f ms ⇒ %.1f GFLOPS' % (base * 1e3, FLOP / base / 1e9))
print()
print('  %-30s %12s %14s' % ('条件', '切片稳态 ms', '切片 GFLOPS'))
print('  ' + '-' * 60)


def slice_steady(post_action=None, label=''):
    cpu_burst(200)
    if post_action:
        post_action()
    for _ in range(2):
        gpu_call()
    ts = []
    for _ in range(6):
        t0 = time.perf_counter()
        gpu_call()
        ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    print('  %-30s %12.3f %14.1f' % (label, med * 1e3, FLOP / med / 1e9))
    return med


torch.set_num_threads(6)
a = slice_steady(None, '6 线程,不做处理（对照）')
b = slice_steady(lambda: torch.set_num_threads(1), '切片前降到 1 线程')
torch.set_num_threads(6)
c = slice_steady(lambda: (torch.set_num_threads(1),
                          [torch.mm(ta[:64], tb[:64, :64]) for _ in range(1)]),
                 '降 1 线程 + 触发一次小 matmul')

print()
print('  判据（事先固定）: 任一条件把切片带进空闲速率的 5% 以内（≤ %.3f ms）⇒ 残余是线程池' %
      (base * 1e3 * 1.05))
best = min(a, b, c)
print('  ⇒ 最好 %.3f ms（%.2f× 空闲）⇒ %s'
      % (best * 1e3, best / base, '【成立】✓' if best <= base * 1.05 else '【不成立】✗'))
