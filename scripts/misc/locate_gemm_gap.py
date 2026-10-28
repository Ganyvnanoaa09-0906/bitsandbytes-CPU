"""locate_gemm_gap.py -- separate the LDS reads from the FMA to find the 3.6x.

Five kernel iterations have not moved throughput: 300 GFLOPS for the fp32 GEMM against
1089 for pure compute on the same device, and the last attempt -- transposing A in local
memory to remove what I believed were bank conflicts -- made it half as fast. Going back
over the reasoning showed the belief was wrong: in the compute loop `a[i] = As[k][ty*TM+i]`
varies only with `i`, and `i` is a per-thread constant within an unrolled iteration, so
all 32 lanes of a warp read the SAME address. That is a broadcast, which has no conflict
at all. There was nothing to fix.

So the gap is unexplained, and further guessing is expensive. This measures the
components instead:

  lds_only    the inner loop reads from local memory but does no arithmetic. Measures
              how fast the reads alone can go.
  fma_only    arithmetic on registers with no local memory traffic beyond the initial
              fill. Should reproduce the ~1089 pure-compute figure.
  both        the real inner loop, which is what currently yields 300.

If lds_only is already far below what the loop needs, reads are the limit and the fix is
to read less -- larger per-thread tiles, vectorised loads. If lds_only is fast and both
is slow, the two interfere, and the answer is more parallelism to hide the latency.

All three write a result that depends on the work, so the compiler cannot delete them.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

import pyopencl as cl  # noqa: E402

SRC = r'''
#define TS 64
#define TK 8
#define TM 8
#define TN 8
#define REPS 64

// LDS reads only: same addressing as the real kernel, no arithmetic on the values.
__kernel void lds_only(const int K, __global float* out) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int lane = ty * 8 + tx;
    for (int p = 0; p < TS * TK / 64; p++) {
        int f = p * 64 + lane;
        As[f / TK][f % TK] = (float)(f & 7);
        Bs[f / TS][f % TS] = (float)(f & 3);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float s = 0.f;
    for (int rep = 0; rep < REPS; rep++) {
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][tx * TN + j];
            // consume without FMA: a cheap dependent sum keeps the reads alive
            #pragma unroll
            for (int i = 0; i < TM; i++) s += a[i];
            #pragma unroll
            for (int j = 0; j < TN; j++) s += b[j];
        }
    }
    out[get_global_id(0)] = s;
}

// FMA only: registers, no local memory in the measured loop.
__kernel void fma_only(const int K, __global float* out) {
    float a[TM], b[TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) a[i] = (float)(i + 1) * 1e-3f;
    #pragma unroll
    for (int j = 0; j < TN; j++) b[j] = (float)(j + 1) * 1e-3f;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    for (int rep = 0; rep < REPS; rep++) {
        for (int k = 0; k < TK; k++) {
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) s += acc[i][j];
    out[get_global_id(0)] = s;
}

// Both: the real inner loop shape.
__kernel void both(const int K, __global float* out) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int lane = ty * 8 + tx;
    for (int p = 0; p < TS * TK / 64; p++) {
        int f = p * 64 + lane;
        As[f / TK][f % TK] = (float)(f & 7);
        Bs[f / TS][f % TS] = (float)(f & 3);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    for (int rep = 0; rep < REPS; rep++) {
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) s += acc[i][j];
    out[get_global_id(0)] = s;
}
'''

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
prg = cl.Program(ctx, SRC).build()

GROUPS = 6 * 8           # 8 workgroups per CU, matching the real kernel's occupancy
G = GROUPS * 64
buf = cl.Buffer(ctx, mf.WRITE_ONLY, G * 4)
REPS, TK, TM, TN = 64, 8, 8, 8

print('设备 %s | %d CU | 工作项 %d（%d 组 x 64 线程）' % (dev.name, dev.max_compute_units, G, GROUPS))
print()
print('%-12s %10s %14s %14s' % ('内核', '秒', 'FMA/秒', '每次 LDS 读的 FMA'))
print('-' * 56)
res = {}
for name in ('fma_only', 'lds_only', 'both'):
    k = cl.Kernel(prg, name)
    k(queue, (G,), (64,), np.int32(0), buf)
    queue.finish()
    t0 = time.perf_counter()
    for _ in range(3):
        k(queue, (G,), (64,), np.int32(0), buf)
    queue.finish()
    dt = (time.perf_counter() - t0) / 3
    fma = G * REPS * TK * TM * TN / dt
    # lds_only performs (TM+TN) reads per k per thread; fma_only none
    reads = 0 if name == 'fma_only' else G * REPS * TK * (TM + TN) / dt
    res[name] = (fma, reads)
    print('%-12s %10.5f %14.3e %14s'
          % (name, dt, fma, '—' if name == 'fma_only' else '%.2f' % (fma / reads)))

fo, _ = res['fma_only']
lo, lr = res['lds_only']
bo, _ = res['both']
print()
print('判据:')
print('  若 lds_only 的读速率 × (TM*TN)/(TM+TN) 低于 both 的 FMA 速率 ⇒ 读是瓶颈')
print('     both 需要 %.2e FMA/秒，每个 FMA 需 %.2f 次读 ⇒ 需 %.2e 读/秒'
      % (bo, (TM + TN) / (TM * TN), bo * (TM + TN) / (TM * TN)))
print('     lds_only 实际提供 %.2e 读/秒  ⇒ %s'
      % (lr, '读不够 ✓ 读是瓶颈' if lr < bo * (TM + TN) / (TM * TN) else '读够 ⇒ 不是读的带宽问题'))
print('  若 lds_only 远快于 both 的需要 ⇒ 两者相互干扰，问题在延迟/调度')
