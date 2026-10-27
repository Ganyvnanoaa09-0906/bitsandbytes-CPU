# -*- coding: utf-8 -*-
"""向量化 fp16 GEMM（row-major，fp32 累加）——目标是打赢 CLBlast 在我们这些形状上的 124~260 GFLOPS。

形状（都是 C[m,n] = A[m,k] @ B[k,n]，即 hgemm_nn）：
    Wqkv 2048x1024x512   Wo 2048x512x512   W12 2048x2736x512   W3 2048x512x1368

设计要点（针对 gfx902 / 6 CU / 32KB LDS）：
  · 工作组 16x16=256 线程，块 tile 64x64，每线程寄存器 4x4 fp32 累加
  · `__local` 里放 A[TS][BK] 与 B[BK][TS]（各 4KB，BK=32）—— A 按行连续读、B 按列连续读
  · 全局读用 half4（8 字节）向量化，`vload4`
  · 计算循环做外积：每 kk 取 TM+TN=8 个 local 值做 16 次 FMA
  · 边界用谓词保护（我们的 n=2736/1368 不是 tile 的整数倍）

对外：
  gemm_nn(dev, A, B, C, M, N, K, alpha=1.0)   -> A(m,k) @ B(k,n)
  bench harness: python gt1030/gemm_half.py --bench
"""
from __future__ import annotations
import ctypes
import os
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl

TS = 64          # 块 tile 边长
BK = 32          # k 方向分块
TM = 4           # 每线程行数
TN = 4           # 每线程列数
LX = 16          # 线程列数
LY = 16          # 线程行数

KSRC = r"""
#define TS 64
#define BK 32
#define TM 4
#define TN 4
#define NTH (16*16)

__kernel void gemm_nn_h(__global const half* A, __global const half* B, __global half* C,
                        int M, int N, int K, float alpha) {
    __local half Al[TS * BK];
    __local half Bl[BK * TS];
    int lx = get_local_id(0);          // 0..15  -> n 方向
    int ly = get_local_id(1);          // 0..15  -> m 方向
    int mb = get_group_id(1) * TS;     // 块行起点
    int nb = get_group_id(0) * TS;     // 块列起点
    int tid = ly * 16 + lx;
    float acc[TM][TN];
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- A tile: TS 行 x BK 列，沿 k 向量化 ----
        for (int t = tid; t < TS * BK / 4; t += NTH) {
            int r = t / (BK / 4);
            int c4 = t % (BK / 4);
            int gm = mb + r, gk = k0 + c4 * 4;
            half4 v = (half4)(0.f, 0.f, 0.f, 0.f);
            if (gm < M && gk + 3 < K) v = vload4(0, A + (long)gm * K + gk);
            vstore4(v, 0, (__local half*)Al + r * BK + c4 * 4);
        }
        // ---- B tile: BK 行 x TS 列，沿 n 向量化 ----
        for (int t = tid; t < BK * TS / 4; t += NTH) {
            int r = t / (TS / 4);
            int c4 = t % (TS / 4);
            int gk = k0 + r, gn = nb + c4 * 4;
            half4 v = (half4)(0.f, 0.f, 0.f, 0.f);
            if (gk < K && gn + 3 < N) v = vload4(0, B + (long)gk * N + gn);
            vstore4(v, 0, (__local half*)Bl + r * TS + c4 * 4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        __local half* ap = (__local half*)Al + (ly * TM) * BK;
        for (int kk = 0; kk < BK; ++kk) {
            float av[TM], bv[TN];
            for (int i = 0; i < TM; ++i) av[i] = (float)ap[i * BK + kk];
#pragma unroll
            for (int j = 0; j < TN; ++j) bv[j] = (float)Bl[kk * TS + lx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += av[i] * bv[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < TM; ++i) {
        int gm = mb + ly * TM + i;
        if (gm >= M) break;
        for (int j = 0; j < TN; ++j) {
            int gn = nb + lx * TN + j;
            if (gn >= N) break;
            C[(long)gm * N + gn] = (half)(acc[i][j] * alpha);
        }
    }
}
"""


class GemmHalf(object):
    def __init__(self, ctx, queue):
        self.ctx, self.q = ctx, queue
        self.prog = cl.Program(ctx, KSRC).build(options=['-cl-mad-enable'])
        self.k = cl.Kernel(self.prog, 'gemm_nn_h')

    def nn(self, A, B, C, M, N, K, alpha=1.0):
        gx = (N + TS - 1) // TS
        gy = (M + TS - 1) // TS
        self.k(self.q, (gx * LX, gy * LY), (LX, LY), A, B, C,
               np.int32(M), np.int32(N), np.int32(K), np.float32(alpha))


def _bench(fn, n=10):
    fn()
    A.queue.finish() if hasattr(A, 'queue') else None
    return None


def main():
    import gt1030.ar_v2_multi as MM       # 设好 A.B=2/A.BH=16，并给出 ctx/queue
    import gt1030.attention_gpu as A
    import gt1030.full_gpu_layer as L

    g = GemmHalf(A.ctx, A.queue)
    mf = cl.mem_flags
    M = 2048
    shapes = [('Wqkv', 1024, 512), ('Wo', 512, 512), ('W12', 2736, 512), ('W3', 512, 1368),
              ('head', 1024, 512)]
    rng = np.random.default_rng(0)
    print('%-6s %10s %10s %8s   %s' % ('shape', 'CLBlast', '自写', '比值', '数值'))
    for nm, n, k in shapes:
        a = (rng.standard_normal((M, k)) * 0.05).astype(np.float16)
        b = (rng.standard_normal((k, n)) * 0.05).astype(np.float16)
        ab = cl.Buffer(A.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        bb = cl.Buffer(A.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        c1 = cl.Buffer(A.ctx, mf.READ_WRITE, M * n * 2)
        c2 = cl.Buffer(A.ctx, mf.READ_WRITE, M * n * 2)

        def do_blast():
            L.hgemm_nn(ab, bb, c1, M, n, k)

        def do_mine():
            g.nn(ab, bb, c2, M, n, k)

        for fn in (do_blast, do_mine):
            fn()
        A.queue.finish()
        o1 = np.empty((M, n), np.float16); cl.enqueue_copy(A.queue, o1, c1)
        o2 = np.empty((M, n), np.float16); cl.enqueue_copy(A.queue, o2, c2)
        A.queue.finish()
        ref = o1.astype(np.float32)
        rel = float(np.abs(o2.astype(np.float32) - ref).max() / (np.abs(ref).max() + 1e-12))

        def t(fn):
            fn(); A.queue.finish()
            t0 = time.time()
            for _ in range(10):
                fn()
            A.queue.finish()
            return (time.time() - t0) / 10

        t1, t2 = t(do_blast), t(do_mine)
        gf = 2.0 * M * n * k / 1e9
        print('%-6s %8.1f GF %8.1f GF %8.2fx   rel=%.2e (%.1f/%.1f ms)'
              % (nm, gf / t1, gf / t2, t1 / t2, rel, t1 * 1000, t2 * 1000))


if __name__ == '__main__':
    main()
