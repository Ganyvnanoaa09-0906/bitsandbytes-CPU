# -*- coding: utf-8 -*-
"""bench_opencl.py — 自己写 OpenCL kernel 压榨 Vega 6，量硬件真实天花板。

对比基线：
  CPU  sgemm 实测        255 GFLOPS（短时）/ 107（持续降频后）
  DML  2048² 流水线      294.5 GFLOPS（来自 docs_cpu/TECH_REPORT.md，已同步的真值）
  理论（6 CU × 64 × 2 × 1.5GHz）  1.15 TFLOPS fp32 / 2.3 TFLOPS fp16 打包

两个 kernel：
  1) 向量化 copy —— 量可用内存带宽（共享 DDR4，CPU 也在用）
  2) 分块 GEMM    —— TS=32 tile，每线程 4x4 输出寄存器累加
"""
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
import time
import numpy as np
import pyopencl as cl

# ---------------- 选设备 ----------------
platform = cl.get_platforms()[0]
dev = platform.get_devices()[0]
ctx = cl.Context([dev])
queue = cl.CommandQueue(ctx)
print('=' * 74)
print('OpenCL 设备:', dev.name)
print('  CUs=%d  clock=%dMHz  global=%.1fGB  local=%dKB'
      % (dev.max_compute_units, dev.max_clock_frequency,
         dev.global_mem_size / 1e9, dev.local_mem_size / 1024))
print('  fp16:', 'cl_khr_fp16' in dev.extensions, ' subgroups:', 'cl_khr_subgroups' in dev.extensions)
print('=' * 74)

SRC = r'''
// ---------------- 1) 向量化 copy ----------------
__kernel void copy4(const __global float4* restrict a,
                    __global float4* restrict b) {
    const size_t i = get_global_id(0);
    b[i] = a[i];
}

// ---------------- 2) 分块 SGEMM ----------------
// tile 32x32，workgroup 8x8=64 线程，每线程累加 4x4
#define TS 32
#define TM 4
#define TN 4

__kernel void sgemm(const int M, const int N, const int K,
                    const __global float* restrict A,
                    const __global float* restrict B,
                    __global float* restrict C) {
    __local float As[TS][TS + 1];
    __local float Bs[TS][TS + 1];

    const int tx = get_local_id(0);      // 0..7
    const int ty = get_local_id(1);      // 0..7
    const int bx = get_group_id(0) * TS;
    const int by = get_group_id(1) * TS;

    float acc[TM][TN];
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    const int lx = tx * TM;              // 本线程负责的行起点(0..28)
    const int ly = ty * TN;              // 本线程负责的列起点(0..28)

    for (int k0 = 0; k0 < K; k0 += TS) {
        // 协作加载 32x32 的 A / B 到 local（64 线程 × 16 元素）
        for (int t = 0; t < TS * TS / 64; t++) {
            int idx = t * 64 + ty * 8 + tx;
            int r = idx / TS, c = idx % TS;
            int gr = by + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
            gr = k0 + r; gc = bx + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; k++) {
            float a[TM], b[TN];
            for (int i = 0; i < TM; i++) a[i] = As[lx + i][k];
            for (int j = 0; j < TN; j++) b[j] = Bs[k][ly + j];
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) {
            int r = by + lx + i, c = bx + ly + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}
'''

HALF_SRC = r'''
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void hgemm(const int M, const int N, const int K,
                    const __global half* restrict A,
                    const __global half* restrict B,
                    __global float* restrict C) {
    const int r = get_global_id(0);
    const int c = get_global_id(1);
    if (r >= M || c >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; k++)
        acc = fma((float)A[r * K + k], (float)B[k * N + c], acc);
    C[r * N + c] = acc;
}
'''

prg = cl.Program(ctx, SRC).build()
prg_h = None
if 'cl_khr_fp16' in dev.extensions:
    try:
        prg_h = cl.Program(ctx, HALF_SRC).build()
    except Exception as e:
        print('  fp16 kernel 编译失败:', str(e)[:100])


def bench_queue(iters=10, warm=3):
    for _ in range(warm):
        queue.finish()
    t0 = time.perf_counter()
    for _ in range(iters):
        queue.finish()
    return (time.perf_counter() - t0) / iters


# ---------------- 1. 带宽 ----------------
print()
print('--- 内存带宽（向量化 copy）---')
for mb in (64, 256):
    n = mb * 1024 * 1024 // 16          # float4 个数
    a = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, n * 16)
    b = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, n * 16)
    try:
        prg.copy4(queue, (n,), None, a, b)
        queue.finish()
    except Exception as e:
        print('  失败:', str(e)[:80]); continue
    it = 20
    t0 = time.perf_counter()
    for _ in range(it):
        prg.copy4(queue, (n,), None, a, b)
    queue.finish()
    dt = (time.perf_counter() - t0) / it
    gbps = 2 * n * 16 / dt / 1e9
    print('  %4d MB  %7.1f GB/s  (%.3f ms)' % (mb, gbps, dt * 1e3))
    del a, b

# ---------------- 2. GEMM fp32 ----------------
print()
print('--- SGEMM (fp32, 自写分块 kernel) ---')
print('  %-8s %12s %12s' % ('n', 'GFLOPS', 'ms'))
for n in (256, 512, 1024, 2048):
    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)
    C = np.zeros((n, n), dtype=np.float32)
    mf = cl.mem_flags
    a_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    b_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    c_b = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)
    gsz = (n // 32 * 8, n // 32 * 8)
    try:
        prg.sgemm(queue, gsz, (8, 8), np.int32(n), np.int32(n), np.int32(n), a_b, b_b, c_b)
        queue.finish()
    except Exception as e:
        print('  %-8d 失败: %s' % (n, str(e)[:70])); continue
    it = 20 if n <= 1024 else 8
    t0 = time.perf_counter()
    for _ in range(it):
        prg.sgemm(queue, gsz, (8, 8), np.int32(n), np.int32(n), np.int32(n), a_b, b_b, c_b)
    queue.finish()
    dt = (time.perf_counter() - t0) / it
    print('  %-8d %12.2f %12.3f' % (n, 2 * n ** 3 / dt / 1e9, dt * 1e3))
    del a_b, b_b, c_b, A, B, C

# ---------------- 3. GEMM fp16 存储 ----------------
if prg_h is not None:
    print()
    print('--- HGEMM (fp16 输入 / fp32 累加，慢版本只为看 fp16 是否能跑) ---')
    for n in (512, 1024):
        A = np.random.randn(n, n).astype(np.float16)
        B = np.random.randn(n, n).astype(np.float16)
        C = np.zeros((n, n), dtype=np.float32)
        mf = cl.mem_flags
        a_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        b_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        c_b = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)
        try:
            prg_h.hgemm(queue, (n, n), None, np.int32(n), np.int32(n), np.int32(n), a_b, b_b, c_b)
            queue.finish()
            print('  n=%-6d 可运行（fp16 kernel 编译+执行 OK）' % n)
        except Exception as e:
            print('  n=%-6d 失败: %s' % (n, str(e)[:70]))
        del a_b, b_b, c_b

print()
print('DONE')
