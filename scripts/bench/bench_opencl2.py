# -*- coding: utf-8 -*-
"""bench_opencl2.py — 调优版 OpenCL GEMM，目标：打赢 DML 的 294.5 GFLOPS。

调优要点（针对 GCN / gfx902，wavefront=64）：
  1) 8x8 寄存器分块 —— 每线程算 64 个输出，算术访存比拉高
  2) float4 向量化全局加载 —— 减少访存指令数
  3) K 方向分块到 local，加 1 列 padding 消除 bank conflict
  4) 64x64 tile，workgroup 8x8=64 线程（正好一个 wavefront）

对照基线：
  CPU  短时 255 / 持续降频后 107
  DML  294.5（docs_cpu/TECH_REPORT.md，已同步真值）
  理论 6 CU x 64 x 2 x 1.5GHz = 1150 GFLOPS
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import time
import numpy as np
import pyopencl as cl

platform = cl.get_platforms()[0]
dev = platform.get_devices()[0]
ctx = cl.Context([dev])
queue = cl.CommandQueue(ctx)

print('=' * 74)
print('OpenCL 调优版 GEMM  |  device:', dev.name, '|', dev.max_compute_units, 'CU')
print('=' * 74)

SRC = r'''
#define TS 64          // tile 边长
#define TK 16          // K 方向每次载入的深度
#define TM 8           // 每线程输出行数
#define TN 8           // 每线程输出列数

__kernel void sgemm8(const int M, const int N, const int K,
                     const __global float* restrict A,
                     const __global float* restrict B,
                     __global float* restrict C) {
    __local float As[TK][TS + 1];
    __local float Bs[TK][TS + 1];

    const int tx = get_local_id(0);       // 0..7
    const int ty = get_local_id(1);       // 0..7
    const int bx = get_group_id(0) * TS;
    const int by = get_group_id(1) * TS;

    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    const int rowBase = by + ty * TM;     // 本线程负责的 8 行起点
    const int colBase = bx + tx * TN;     // 本线程负责的 8 列起点

    for (int k0 = 0; k0 < K; k0 += TK) {
        // ---- 协作载入 TK x TS 的 A 与 B ----
        // A: TK 行 x TS 列；B: TK 行 x TS 列。共 TK*TS*2 个元素，64 线程分担。
        // 用 float4 向量化：每行 TS 个元素 = 16 个 float4
        const int lane = ty * 8 + tx;               // 0..63
        #pragma unroll
        for (int p = 0; p < (TK * TS * 2) / (64 * 4); p++) {
            int flat4 = p * 64 + lane;              // 第 flat4 个 float4
            int total4 = TK * TS / 4;               // 每张 tile 的 float4 数
            if (flat4 < total4) {
                int r = (flat4 * 4) / TS;
                int c = (flat4 * 4) % TS;
                int gr = by + r, gc = k0 + c;
                float4 v = (gr < M && gc + 3 < K)
                    ? *(const __global float4*)(A + gr * K + gc) : (float4)0.f;
                As[r][c + 0] = v.x; As[r][c + 1] = v.y;
                As[r][c + 2] = v.z; As[r][c + 3] = v.w;
            } else {
                int f2 = flat4 - total4;
                int r = (f2 * 4) / TS;
                int c = (f2 * 4) % TS;
                int gr = k0 + r, gc = bx + c;
                float4 v = (gr < K && gc + 3 < N)
                    ? *(const __global float4*)(B + gr * N + gc) : (float4)0.f;
                Bs[r][c + 0] = v.x; Bs[r][c + 1] = v.y;
                Bs[r][c + 2] = v.z; Bs[r][c + 3] = v.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ---- 8x8 寄存器分块累加 ----
        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int r = rowBase + i;
        if (r >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int c = colBase + j;
            if (c < N) C[r * N + c] = acc[i][j];
        }
    }
}
'''

try:
    prg = cl.Program(ctx, SRC).build()
    K8 = cl.Kernel(prg, 'sgemm8')
    print('  kernel 编译 OK')
except Exception as e:
    print('  编译失败:')
    print(str(e)[:2000])
    raise SystemExit(1)

print()
print('  %-8s %12s %12s %10s' % ('n', 'GFLOPS', 'ms', '占理论%'))
best = 0.0
for n in (256, 512, 1024, 2048):
    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)
    C = np.zeros((n, n), dtype=np.float32)
    mf = cl.mem_flags
    a_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c_b = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)
    gsz = ((n + 63) // 64 * 8, (n + 63) // 64 * 8)
    args = (np.int32(n), np.int32(n), np.int32(n), a_b, b_b, c_b)
    try:
        K8(queue, gsz, (8, 8), *args)
        queue.finish()
    except Exception as e:
        print('  %-8d 失败: %s' % (n, str(e)[:90])); continue
    it = 30 if n <= 1024 else 10
    t0 = time.perf_counter()
    for _ in range(it):
        K8(queue, gsz, (8, 8), *args)
    queue.finish()
    dt = (time.perf_counter() - t0) / it
    gf = 2 * n ** 3 / dt / 1e9
    best = max(best, gf)
    print('  %-8d %12.2f %12.3f %9.1f%%' % (n, gf, dt * 1e3, 100 * gf / 1150))
    del a_b, b_b, c_b, A, B, C

print()
print('  最佳 %.1f GFLOPS   vs   DML 294.5   vs   CPU 短时 255 / 持续 107' % best)
print('DONE')
