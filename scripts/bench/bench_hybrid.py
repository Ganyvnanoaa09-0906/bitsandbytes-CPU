# -*- coding: utf-8 -*-
"""bench_hybrid.py — 决定性实验：CPU 和核显【同时】跑，吞吐能不能叠加。

你文档里的结论是「DML 并发 0.83×」，根因是 DML 的惰性执行
（"入队仅 0.3ms；CPU 空转 800ms 后排空仍需完整执行时间"）。

假设：OpenCL 不是惰性的 —— 命令入队即执行，所以真正的重叠可能成立。

测三种：
  A) 只有 CPU 跑 GEMM
  B) 只有核显跑 GEMM
  C) 两者同时跑
若 C 的总吞吐 > max(A,B)，则并发成立。

注意：核显要驱动桌面，这个测试会让系统有点卡（比满负载轻，用了中等规模）。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import threading
import time
import numpy as np
import pyopencl as cl
import torch

torch.set_num_threads(6)

platform = cl.get_platforms()[0]
dev = platform.get_devices()[0]
ctx = cl.Context([dev])
queue = cl.CommandQueue(ctx)
print('=' * 72)
print('CPU + 核显 并发吞吐测试  |  核显:', dev.name, '| CPU 6 线程')
print('=' * 72)

SRC = r'''
#define TS 64
#define TK 16
#define TM 8
#define TN 8
__kernel void sgemm8(const int M, const int N, const int K,
                     const __global float* restrict A,
                     const __global float* restrict B,
                     __global float* restrict C) {
    __local float As[TK][TS + 1];
    __local float Bs[TK][TS + 1];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    float acc[TM][TN];
    for (int i = 0; i < TM; i++) for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += TK) {
        const int lane = ty * 8 + tx;
        for (int p = 0; p < (TK * TS * 2) / (64 * 4); p++) {
            int flat4 = p * 64 + lane;
            int total4 = TK * TS / 4;
            if (flat4 < total4) {
                int r = (flat4 * 4) / TS, c = (flat4 * 4) % TS;
                int gr = by + r, gc = k0 + c;
                float4 v = (gr < M && gc + 3 < K) ? *(const __global float4*)(A + gr*K + gc) : (float4)0.f;
                As[r][c] = v.x; As[r][c+1] = v.y; As[r][c+2] = v.z; As[r][c+3] = v.w;
            } else {
                int f2 = flat4 - total4;
                int r = (f2 * 4) / TS, c = (f2 * 4) % TS;
                int gr = k0 + r, gc = bx + c;
                float4 v = (gr < K && gc + 3 < N) ? *(const __global float4*)(B + gr*N + gc) : (float4)0.f;
                Bs[r][c] = v.x; Bs[r][c+1] = v.y; Bs[r][c+2] = v.z; Bs[r][c+3] = v.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            for (int i = 0; i < TM; i++) a[i] = As[k][ty*TM + i];
            for (int j = 0; j < TN; j++) b[j] = Bs[k][tx*TN + j];
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = 0; i < TM; i++) {
        int r = by + ty*TM + i;
        if (r >= M) continue;
        for (int j = 0; j < TN; j++) {
            int c = bx + tx*TN + j;
            if (c < N) C[r*N + c] = acc[i][j];
        }
    }
}
'''
prg = cl.Program(ctx, SRC).build()
K8 = cl.Kernel(prg, 'sgemm8')

N = 1024
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)
mf = cl.mem_flags
a_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
b_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
c_b = cl.Buffer(ctx, mf.WRITE_ONLY, N * N * 4)
gsz = (N // 64 * 8, N // 64 * 8)
cl_args = (np.int32(N), np.int32(N), np.int32(N), a_b, b_b, c_b)

At = torch.from_numpy(A)
Bt = torch.from_numpy(B)
FLOPS_PER = 2 * N ** 3


def gpu_worker(stop, count):
    while not stop.is_set():
        K8(queue, gsz, (8, 8), *cl_args)
        count[0] += 1
        if count[0] % 8 == 0:
            queue.finish()


def cpu_worker(stop, count):
    while not stop.is_set():
        At @ Bt
        count[0] += 1


DUR = 12.0

# A) 只有 CPU
K8(queue, gsz, (8, 8), *cl_args); queue.finish()
At @ Bt
stop = threading.Event(); c = [0]
t = threading.Thread(target=cpu_worker, args=(stop, c)); t.start()
time.sleep(DUR); stop.set(); t.join()
cpu_only = c[0] * FLOPS_PER / DUR / 1e9

# B) 只有核显
stop = threading.Event(); c = [0]
t = threading.Thread(target=gpu_worker, args=(stop, c)); t.start()
time.sleep(DUR); stop.set(); t.join(); queue.finish()
gpu_only = c[0] * FLOPS_PER / DUR / 1e9

# C) 同时
stop = threading.Event(); cc = [0]; cg = [0]
tc = threading.Thread(target=cpu_worker, args=(stop, cc))
tg = threading.Thread(target=gpu_worker, args=(stop, cg))
tc.start(); tg.start()
time.sleep(DUR); stop.set(); tc.join(); tg.join(); queue.finish()
cpu_both = cc[0] * FLOPS_PER / DUR / 1e9
gpu_both = cg[0] * FLOPS_PER / DUR / 1e9

print()
print('  %-14s %14s %14s' % ('模式', 'CPU GFLOPS', '核显 GFLOPS'))
print('  ' + '-' * 46)
print('  %-14s %14.2f %14s' % ('A 只有 CPU', cpu_only, '-'))
print('  %-14s %14s %14.2f' % ('B 只有核显', '-', gpu_only))
print('  %-14s %14.2f %14.2f' % ('C 同时跑', cpu_both, gpu_both))
print()
print('  A+B 串行 = %.1f' % (cpu_only + gpu_only))
print('  C  并发 = %.1f' % (cpu_both + gpu_both))
best_single = max(cpu_only, gpu_only)
print('  最好单设备 = %.1f' % best_single)
print()
print('  并发加速比 vs 最好单设备: %.2fx' % ((cpu_both + gpu_both) / best_single))
print('  并发加速比 vs 两者之和串行: %.2fx' % ((cpu_both + gpu_both) / (cpu_only + gpu_only)))
print()
print('  判据: > 1.0 说明真并发成立；< 1.0 说明共享功耗/带宽把它们拖在一起')
print('DONE')
