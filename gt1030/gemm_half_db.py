# -*- coding: utf-8 -*-
"""GEMM 双缓冲版：用 async_work_group_strided_copy 把下一块 A/B 通过 DMA 通道预取，
让"载入"和"计算"真正重叠（现在每个 k 迭代是 载入→屏障→算→屏障，载入期间 ALU 空转）。

配置：MS=128, NS=64, BK=32, 每线程 TM=8/TN=4, 16x16 线程。
local: As[2][128*32]（8KB×2）+ Bs[2][32*64]（2KB×2）≈ 20KB，装得下 32KB。
A 直存（不转置）以便 DMA 直拷，计算时 av 用标量 local 读；B 同样直存，bv 用 vload4。
"""
from __future__ import annotations
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl

SRC = r"""
#define MS 128
#define NS 64
#define BK 32
#define TM 8
#define TN 4

__kernel void gemm_nn_db(__global const half* A, __global const half* B, __global half* C,
                         int M, int N, int K, float alpha) {
    __local half As[2][MS * BK];
    __local half Bs[2][BK * NS];
    int lx = get_local_id(0), ly = get_local_id(1);
    int tid = ly * 16 + lx;
    int nb = get_group_id(0) * NS;
    int mb = get_group_id(1) * MS;
    int m0 = ly * TM, n0 = lx * TN;
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    int nk = (K + BK - 1) / BK;
    // 预取第 0 块（直拷：A 是 (m,k) 行主序、B 是 (k,n) 行主序，都按 stride 拷进 local）
    event_t e0 = async_work_group_strided_copy(As[0], A + (long)mb * K, MS * BK, K, 0);
    event_t e1 = async_work_group_strided_copy(Bs[0], B + nb, BK * NS, N, 0);
    wait_group_events(2, (event_t[]){e0, e1});
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kk0 = 0; kk0 < nk; ++kk0) {
        int cur = kk0 & 1, nxt = 1 - cur;
        event_t ev[2];
        int nev = 0;
        if (kk0 + 1 < nk) {                     // 先把下一块发出去（DMA），再算当前块
            int k1 = (kk0 + 1) * BK;
            ev[nev++] = async_work_group_strided_copy(As[nxt], A + (long)mb * K + k1, MS * BK, K, 0);
            ev[nev++] = async_work_group_strided_copy(Bs[nxt], B + (long)k1 * N + nb, BK * NS, N, 0);
        }
        __local half* ap = (__local half*)As[cur];
        __local half* bp = (__local half*)Bs[cur];
        for (int kk = 0; kk < BK; ++kk) {
            float av[TM], bv[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) av[i] = (float)ap[(m0 + i) * BK + kk];
            half4 b = vload4(0, bp + kk * NS + n0);
            bv[0] = (float)b.x; bv[1] = (float)b.y; bv[2] = (float)b.z; bv[3] = (float)b.w;
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += av[i] * bv[j];
        }
        if (nev) { wait_group_events(nev, ev); }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = 0; i < TM; ++i) {
        int gm = mb + m0 + i;
        if (gm >= M) break;
        for (int j = 0; j < TN; ++j) {
            int gn = nb + n0 + j;
            if (gn >= N) break;
            C[(long)gm * N + gn] = (half)(acc[i][j] * alpha);
        }
    }
}
"""


def main():
    import gt1030.ar_v2_multi as MM
    import gt1030.attention_gpu as A
    import gt1030.full_gpu_layer as L

    ctx, q = A.ctx, A.queue
    mf = cl.mem_flags
    kern = cl.Kernel(cl.Program(ctx, SRC).build(options=['-cl-mad-enable']), 'gemm_nn_db')
    M = 2048
    shapes = [('Wqkv', 1024, 512), ('Wo', 512, 512), ('W12', 2736, 512), ('W3', 512, 1368)]
    rng = np.random.default_rng(0)
    print('%-6s %10s %10s %8s' % ('shape', 'CLBlast', '双缓冲', '比值'))
    tot = 0.0
    for nm, n, k in shapes:
        a = (rng.standard_normal((M, k)) * 0.05).astype(np.float16)
        b = (rng.standard_normal((k, n)) * 0.05).astype(np.float16)
        ab = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        c1 = cl.Buffer(ctx, mf.READ_WRITE, M * n * 2)
        c2 = cl.Buffer(ctx, mf.READ_WRITE, M * n * 2)

        def blast():
            L.hgemm_nn(ab, bb, c1, M, n, k)

        def mine():
            kern(q, ((n + 63) // 64 * 16, (M + 127) // 128 * 16), (16, 16), ab, bb, c2,
                 np.int32(M), np.int32(n), np.int32(k), np.float32(1.0))

        def t(fn, nn=8):
            fn(); q.finish()
            t0 = time.time()
            for _ in range(nn):
                fn()
            q.finish()
            return (time.time() - t0) / nn

        try:
            tb, tm = t(blast), t(mine)
        except Exception as e:
            print('%-6s FAIL %s' % (nm, str(e)[:70])); continue
        o1 = np.empty((M, n), np.float16); cl.enqueue_copy(q, o1, c1)
        o2 = np.empty((M, n), np.float16); cl.enqueue_copy(q, o2, c2)
        q.finish()
        rel = float(np.abs(o2.astype(np.float32) - o1.astype(np.float32)).max()
                    / (np.abs(o1.astype(np.float32)).max() + 1e-12))
        gf = 2.0 * M * n * k / 1e9
        print('%-6s %8.0f GF %8.0f GF %8.2fx  rel=%.1e %s'
              % (nm, gf / tb, gf / tm, tb / tm, rel, 'OK' if rel < 5e-2 else 'BAD'))
        tot += tb / tm
    print('平均 %.2fx' % (tot / len(shapes)))


if __name__ == '__main__':
    main()
