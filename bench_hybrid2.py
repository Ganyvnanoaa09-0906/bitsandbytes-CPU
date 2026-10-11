# -*- coding: utf-8 -*-
"""bench_hybrid2.py — 真实分工的 CPU/核显配合测试。

上次 bench_hybrid.py 测出只有 1.07×，但两边跑的是【同一个密集 GEMM】——
那是最大争用情形，不代表真实分工。

我们的负载天然分属：
    核显 → FFN 大 GEMM   (M=2048, K=512, N=2048)   大而规整
    CPU  → 注意力         (S=256, hd=64, 8头)        小而碎

本脚本测：
  1) 核显在【真实 FFN 形状】上的 GEMM 吞吐
  2) CPU 在【真实 FFN 形状】上的吞吐
  3) CPU 在【真实注意力形状】上的耗时
  4) 【核显跑 FFN】+【CPU 跑注意力】并发 → 总时间能不能 < 串行
"""
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

print('=' * 76)
print('真实分工配合测试  |  核显:', dev.name)
print('=' * 76)

KERNEL = r'''
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
prg = cl.Program(ctx, KERNEL).build()
K8 = cl.Kernel(prg, 'sgemm8')


def cl_gemm_bench(M, N, K, iters=10):
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    mf = cl.mem_flags
    ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    cb = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    gsz = ((N + 63) // 64 * 8, (M + 63) // 64 * 8)
    args = (np.int32(M), np.int32(N), np.int32(K), ab, bb, cb)
    K8(queue, gsz, (8, 8), *args)
    queue.finish()
    t0 = time.perf_counter()
    for _ in range(iters):
        K8(queue, gsz, (8, 8), *args)
    queue.finish()
    dt = (time.perf_counter() - t0) / iters
    return dt, 2 * M * N * K


def cpu_gemm_bench(M, N, K, iters=10):
    A = torch.randn(M, K); B = torch.randn(K, N)
    A @ B
    t0 = time.perf_counter()
    for _ in range(iters):
        A @ B
    dt = (time.perf_counter() - t0) / iters
    return dt


B_SZ, T = 8, 256
M = B_SZ * T        # 2048
print()
print('--- 1/2) FFN 真实形状 GEMM ---')
print('  %-28s %10s %10s %12s' % ('形状', 'CPU ms', '核显 ms', '核显/CPU', ), flush=True)
ffn_tasks = [('w1: %dx512 @ 512x2048' % M, M, 2048, 512),
             ('w2: %dx2048 @ 2048x512' % M, M, 512, 2048)]
for tag, m, n, k in ffn_tasks:
    dtc = cpu_gemm_bench(m, n, k)
    dtg, fl = cl_gemm_bench(m, n, k)
    print('  %-28s %10.2f %10.2f %11.2fx' % (tag, dtc * 1e3, dtg * 1e3, dtc / dtg), flush=True)

# ---------------- 3) CPU 注意力形状 ----------------
import torch.nn.functional as F
Nh, HD, S = 8, 64, T
q = torch.randn(B_SZ, Nh, S, HD)
k = torch.randn(B_SZ, Nh, S, HD)
v = torch.randn(B_SZ, Nh, S, HD)
F.scaled_dot_product_attention(q, k, v, is_causal=True)
t0 = time.perf_counter()
ATT_ITERS = 30
for _ in range(ATT_ITERS):
    F.scaled_dot_product_attention(q, k, v, is_causal=True)
att_dt = (time.perf_counter() - t0) / ATT_ITERS
print()
print('--- 3) CPU 注意力（真实形状）---')
print('  B=8 nh=8 S=256 hd=64    %.3f ms/次' % (att_dt * 1e3), flush=True)

# ---------------- 4) 真实分工并发 ----------------
print()
print('--- 4) 【核显跑 FFN】+【CPU 跑注意力】并发 ---')

m, n, k = M, 2048, 512
A = np.random.randn(m, k).astype(np.float32)
B = np.random.randn(k, n).astype(np.float32)
mf = cl.mem_flags
ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
cb = cl.Buffer(ctx, mf.WRITE_ONLY, m * n * 4)
gsz = ((n + 63) // 64 * 8, (m + 63) // 64 * 8)
cl_args = (np.int32(m), np.int32(n), np.int32(k), ab, bb, cb)


def gpu_ffn(stop, cnt):
    while not stop.is_set():
        K8(queue, gsz, (8, 8), *cl_args)
        cnt[0] += 1
        if cnt[0] % 4 == 0:
            queue.finish()


def cpu_attn(stop, cnt):
    while not stop.is_set():
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
        cnt[0] += 1


# 单独基线
K8(queue, gsz, (8, 8), *cl_args); queue.finish()
stop = threading.Event(); c = [0]
t = threading.Thread(target=gpu_ffn, args=(stop, c)); t.start()
time.sleep(10); stop.set(); t.join(); queue.finish()
gpu_solo = c[0] / 10.0

stop = threading.Event(); c = [0]
t = threading.Thread(target=cpu_attn, args=(stop, c)); t.start()
time.sleep(10); stop.set(); t.join()
att_solo = c[0] / 10.0

# 并发
stop = threading.Event(); cg = [0]; ca = [0]
tg = threading.Thread(target=gpu_ffn, args=(stop, cg))
ta = threading.Thread(target=cpu_attn, args=(stop, ca))
tg.start(); ta.start()
time.sleep(10); stop.set(); tg.join(); ta.join(); queue.finish()
gpu_both = cg[0] / 10.0
att_both = ca[0] / 10.0

print('  %-24s %14s %14s' % ('', '核显 FFN 次/秒', 'CPU 注意力 次/秒'), flush=True)
print('  %-24s %14.1f %14.1f' % ('单独跑', gpu_solo, att_solo), flush=True)
print('  %-24s %14.1f %14.1f' % ('并发跑', gpu_both, att_both), flush=True)
print()
print('  核显保持率 %.0f%%   CPU 注意力保持率 %.0f%%'
      % (100 * gpu_both / gpu_solo, 100 * att_both / att_solo), flush=True)

# 折算成"跑完一层 FFN + 注意力"的时间
ffn_solo_t = 2 * (1.0 / gpu_solo)       # w1 + w2 两次 GEMM
layer_serial = ffn_solo_t + (1.0 / att_solo)
layer_par = max(1.0 / gpu_both * 2, 1.0 / att_both)
print()
print('  一层（FFN 两次 GEMM + 一次注意力）:')
print('    核显单独跑 FFN + CPU 串行跑注意力 = %.1f ms' % (layer_serial * 1e3), flush=True)
print('    并发（重叠）                      = %.1f ms' % (layer_par * 1e3), flush=True)
print('  => 重叠收益 %.2fx' % (layer_serial / layer_par), flush=True)
print()
print('DONE')
