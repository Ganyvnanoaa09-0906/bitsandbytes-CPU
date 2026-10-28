"""decompose_gemm_time.py -- where the unexplained 4.7 ms goes.

v4 runs at 300 GFLOPS. The arithmetic for the model's real shape (1024x2736x512,
2.87 GFLOP) says:

    compute at the measured 1089 GFLOPS pure-compute rate   2.6 ms
    global traffic (18.6 MB at ~30 GB/s)                    0.6 ms
    LDS traffic (1.44 GB at ~845 GB/s)                      1.7 ms
                                                            ------
                                                            4.9 ms
    measured                                                9.6 ms

so roughly half the time is not accounted for by any of the three. Six kernel variants
and two wrong diagnoses have not found it, which means the next step is not another
variant but a decomposition: same loop structure, same iteration counts, one component
removed at a time.

    full        staging + LDS reads + FMA      (the real kernel, ~300 GFLOPS)
    no_lds      staging + FMA on register constants
    no_fma      staging + LDS reads, values summed instead of multiplied
    no_stage    LDS pre-filled once, then reads + FMA for the same loop count
    empty       the loop structure with neither reads nor staging

All five execute the same number of outer iterations and the same inner k loop, so the
differences are attributable. Every kernel writes a result that depends on the work.
"""
from __future__ import annotations
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
#define KB 64

__kernel void full(const int K, const __global float* restrict A,
                   const __global float* restrict B, __global float* restrict C) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 8 + tx;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    for (int kb = 0; kb < KB; kb++) {
        for (int p = 0; p < TS * TK / 64; p++) {
            int f = p * 64 + lane;
            int r = f / TK, c = f % TK;
            int gr = by + r, gc = (kb * TK + c) % K;
            As[r][c] = A[gr * K + gc];
        }
        for (int p = 0; p < TK * TS / 64; p++) {
            int f = p * 64 + lane;
            int r = f / TS, c = f % TS;
            int gr = (kb * TK + r) % K, gc = bx + c;
            Bs[r][c] = B[gr * (get_global_size(0)) + gc];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
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
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) s += acc[i][j];
    C[get_global_id(0)] = s;
}

__kernel void no_lds(const int K, const __global float* restrict A,
                     const __global float* restrict B, __global float* restrict C) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 8 + tx;
    float a[TM], b[TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) a[i] = (float)i * 1e-3f;
    #pragma unroll
    for (int j = 0; j < TN; j++) b[j] = (float)j * 1e-3f;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    for (int kb = 0; kb < KB; kb++) {
        for (int p = 0; p < TS * TK / 64; p++) {
            int f = p * 64 + lane;
            As[f / TK][f % TK] = A[(by + f / TK) * K + (kb * TK + f % TK) % K];
        }
        for (int p = 0; p < TK * TS / 64; p++) {
            int f = p * 64 + lane;
            Bs[f / TS][f % TS] = B[((kb * TK + f / TS) % K) * get_global_size(0) + bx + f % TS];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TK; k++) {
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float s = 0.f;
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) s += acc[i][j];
    C[get_global_id(0)] = s;
}

__kernel void no_fma(const int K, const __global float* restrict A,
                     const __global float* restrict B, __global float* restrict C) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 8 + tx;
    float s = 0.f;
    for (int kb = 0; kb < KB; kb++) {
        for (int p = 0; p < TS * TK / 64; p++) {
            int f = p * 64 + lane;
            As[f / TK][f % TK] = A[(by + f / TK) * K + (kb * TK + f % TK) % K];
        }
        for (int p = 0; p < TK * TS / 64; p++) {
            int f = p * 64 + lane;
            Bs[f / TS][f % TS] = B[((kb * TK + f / TS) % K) * get_global_size(0) + bx + f % TS];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++) s += a[i];
            #pragma unroll
            for (int j = 0; j < TN; j++) s += b[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[get_global_id(0)] = s;
}

__kernel void no_stage(const int K, const __global float* restrict A,
                       const __global float* restrict B, __global float* restrict C) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int lane = ty * 8 + tx;
    for (int p = 0; p < TS * TK / 64; p++) {
        int f = p * 64 + lane;
        As[f / TK][f % TK] = A[f];
    }
    for (int p = 0; p < TK * TS / 64; p++) {
        int f = p * 64 + lane;
        Bs[f / TS][f % TS] = B[f];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    for (int kb = 0; kb < KB; kb++) {
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
    C[get_global_id(0)] = s;
}
'''

# The kernel constants must be mirrored here. Any value that appears only inside the
# OpenCL source is invisible to Python, and the launch geometry and the reporting both
# need them -- this has now cost three runs, on v5, v6 and this file.
TS, TK, TM, TN = 64, 8, 8, 8
KB = 64

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
prg = cl.Program(ctx, SRC).build()

M, N, K = 1024, 2736, 512
GROUPS = ((N + 63) // 64 * 8, (M + 63) // 64 * 8)
G = GROUPS[0] * GROUPS[1]
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)
ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
bc = cl.Buffer(ctx, mf.WRITE_ONLY, G * 4)
print('设备 %s | %d CU | 组 %d | 每线程 %d 次 FMA x %d 块 = %d FMA'
      % (dev.name, dev.max_compute_units, G, TM * TN, KB, TM * TN * KB))
print()
print('%-12s %10s %14s %10s  %s' % ('内核', '秒', 'FMA/秒', '相对 full', '含哪部分'))
print('-' * 70)
res = {}
for name, parts in (('full', '装入+读+FMA'), ('no_lds', '装入+FMA'),
                    ('no_fma', '装入+读'), ('no_stage', '读+FMA')):
    k = cl.Kernel(prg, name)
    k(queue, GROUPS, (8, 8), np.int32(K), ba, bb, bc)
    queue.finish()
    t0 = time.perf_counter()
    for _ in range(3):
        k(queue, GROUPS, (8, 8), np.int32(K), ba, bb, bc)
    queue.finish()
    dt = (time.perf_counter() - t0) / 3
    fma = G * KB * TM * TN / dt
    res[name] = dt
    base = res.get('full', dt)
    print('%-12s %10.5f %14.3e %9.2fx  %s' % (name, dt, fma, dt / base, parts))
print()
print('读法: full - no_lds ≈ 读的代价；no_stage - full ≈ 装入的代价；')
print('      no_stage 与 no_lds 应分别接近"读+FMA"与"纯FMA"，两者之差才是真正的谜。')
