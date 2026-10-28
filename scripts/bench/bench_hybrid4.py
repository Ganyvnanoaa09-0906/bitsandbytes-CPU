# -*- coding: utf-8 -*-
"""bench_hybrid4.py — 修正版：限流 OpenCL 队列，同条件对比 CPU vs 核显。

上一版的致命 bug：sustained_gpu 里以微秒级速度狂塞 kernel，20 秒后
queue.finish() 要执行掉全部积压（可能是几分钟的 GPU 工作量），所以永远跑不完。
⇒ 之前 bench_opencl_sustain.py 也一样，所以那个"降频曲线"从未拿到数据。

修正：每批只排 BATCH 个 kernel 就 finish 一次，循环到时间预算用尽。
      这样队列深度有界，finish 的等待时间可控。

⚠️ 注意：本地 v3 训练正在跑（约 4-5 个核），所以这里测出的 CPU 数字是【下界】。
   核显不受 v3 影响。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import time
import numpy as np
import pyopencl as cl
import torch

torch.set_num_threads(6)
platform = cl.get_platforms()[0]
dev = platform.get_devices()[0]
ctx = cl.Context([dev])
queue = cl.CommandQueue(ctx)

print('=' * 78, flush=True)
print('修正版：同条件持续对比  |  核显:', dev.name, '| CPU 6 线程', flush=True)
print('⚠️ 本地 v3 训练在跑，CPU 数字是下界', flush=True)
print('=' * 78, flush=True)

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
                As[r][c]=v.x; As[r][c+1]=v.y; As[r][c+2]=v.z; As[r][c+3]=v.w;
            } else {
                int f2 = flat4 - total4;
                int r = (f2 * 4) / TS, c = (f2 * 4) % TS;
                int gr = k0 + r, gc = bx + c;
                float4 v = (gr < K && gc + 3 < N) ? *(const __global float4*)(B + gr*N + gc) : (float4)0.f;
                Bs[r][c]=v.x; Bs[r][c+1]=v.y; Bs[r][c+2]=v.z; Bs[r][c+3]=v.w;
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

BATCH = 8      # 队列深度上限


def gpu_rate(M, N, Kd, secs=30, tail=15):
    A = np.random.randn(M, Kd).astype(np.float32)
    B = np.random.randn(Kd, N).astype(np.float32)
    mf = cl.mem_flags
    ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    cb = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    gsz = ((N + 63) // 64 * 8, (M + 63) // 64 * 8)
    args = (np.int32(M), np.int32(N), np.int32(Kd), ab, bb, cb)

    def run_batch(nb):
        for _ in range(nb):
            K8(queue, gsz, (8, 8), *args)
        queue.finish()

    run_batch(BATCH)

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < secs - tail:
        run_batch(BATCH)

    reps = 0
    t1 = time.perf_counter()
    while time.perf_counter() - t1 < tail:
        run_batch(BATCH)
        reps += BATCH
    el = time.perf_counter() - t1
    return reps * 2 * M * N * Kd / el / 1e9


def cpu_rate(M, N, Kd, secs=30, tail=15):
    A = torch.randn(M, Kd); B = torch.randn(Kd, N)
    A @ B
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < secs - tail:
        A @ B
    reps = 0
    t1 = time.perf_counter()
    while time.perf_counter() - t1 < tail:
        A @ B
        reps += 1
    el = time.perf_counter() - t1
    return reps * 2 * M * N * Kd / el / 1e9


M = 8 * 256       # 2048
print()
print('  %-30s %13s %13s %10s %s'
      % ('形状', 'CPU 稳态', '核显稳态', '核显/CPU', '判定'), flush=True)
for tag, mm, nn, kk in (('w1  %dx512 @ 512x2048' % M, M, 2048, 512),
                        ('w2  %dx2048 @ 2048x512' % M, M, 512, 2048),
                        ('正方 2048x2048 @ 2048x2048', 2048, 2048, 2048)):
    c = cpu_rate(mm, nn, kk)
    g = gpu_rate(mm, nn, kk)
    verdict = '核显赢 %.2fx' % (g / c) if g > c * 1.05 else ('CPU 赢 %.2fx' % (c / g) if c > g * 1.05 else '平手')
    print('  %-30s %10.1f GF %10.1f GF %8.2fx  %s' % (tag, c, g, g / c, verdict), flush=True)
    time.sleep(8)

print()
print('DONE', flush=True)
