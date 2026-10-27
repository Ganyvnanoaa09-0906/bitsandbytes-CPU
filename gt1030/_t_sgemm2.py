# -*- coding: utf-8 -*-
"""探针 45：自写 S = Q Kᵀ（k=64）tiled kernel 对比 CLBlast batched（T=1024 写死）。"""
from __future__ import annotations
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_multi as MM      # 设好 A.B=2 -> A.BH=16
import gt1030.attention_gpu as A

T, HD, BH, TS = 1024, 64, 16, 64
KSRC = """
__kernel void s_gemm(__global const half* Q, __global const half* Kt,
                     __global half* S, float scale) {
    int bh = get_global_id(2);
    int ti = get_global_id(0);
    int tj = get_global_id(1);
    __local half Qs[64][64];
    __local half Ks[64][64];
    int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int lx = get_local_id(0), ly = get_local_id(1);
    __global const half* Qp = Q + (long)bh * 1024 * 64;
    __global const half* Kp = Kt + (long)bh * 1024 * 64;
    __global half* Sp = S + (long)bh * 1024 * 1024;
    for (int i = lid; i < 64 * 64; i += 256) {
        int r = i / 64, c = i % 64;
        Qs[r][c] = Qp[(long)(ti * 64 + r) * 64 + c];
        Ks[r][c] = Kp[(long)(tj * 64 + r) * 64 + c];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int r0 = ly * 4, c0 = lx * 4;
    float acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    for (int kk = 0; kk < 64; ++kk) {
        float qv[4], kv[4];
        for (int i = 0; i < 4; ++i) qv[i] = (float)Qs[r0 + i][kk];
        for (int j = 0; j < 4; ++j) kv[j] = (float)Ks[c0 + j][kk];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) acc[i][j] += qv[i] * kv[j];
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            Sp[(long)(ti * 64 + r0 + i) * 1024 + (tj * 64 + c0 + j)] = (half)(acc[i][j] * scale);
}
"""
ctx, q_ = A.ctx, A.queue
prog = cl.Program(ctx, KSRC).build(options=['-cl-mad-enable'])
kern = cl.Kernel(prog, 's_gemm')
mf = cl.mem_flags
rng = np.random.default_rng(0)
Qh = (rng.standard_normal((BH, T, HD)) * 0.5).astype(np.float16)
Kh = (rng.standard_normal((BH, T, HD)) * 0.5).astype(np.float16)
Qb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Qh)
Kb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Kh)
S1 = cl.Buffer(ctx, mf.READ_WRITE, BH * T * T * 2)
S2 = cl.Buffer(ctx, mf.READ_WRITE, BH * T * T * 2)
SC = 1.0 / np.sqrt(HD)


def blast():
    A.hgemm_batched(Qb, Kb, S1, T, T, HD, A.NO, A.YES, HD, HD, T,
                    T * HD, T * HD, T * T)


def mine():
    kern(q_, (T // TS, T // TS, BH), (16, 16, 1), Qb, Kb, S2, np.float32(SC))


for fn in (blast, mine):
    fn(); q_.finish()
a1 = np.empty(BH * T * T, np.float16); cl.enqueue_copy(q_, a1, S1)
a2 = np.empty(BH * T * T, np.float16); cl.enqueue_copy(q_, a2, S2)
q_.finish()
d = float(np.abs(a1.astype(np.float32) - a2.astype(np.float32)).max())
ref = float(np.abs(a1.astype(np.float32)).max())
print('数值: max|diff|=%.3e (refmax=%.3f) rel=%.2e' % (d, ref, d / (ref + 1e-12)))


def bench(fn, n=10):
    fn(); q_.finish()
    t0 = time.time()
    for _ in range(n):
        fn()
    q_.finish()
    return (time.time() - t0) / n


gf = 2 * BH * T * T * HD
tb, tm = bench(blast), bench(mine)
print('CLBlast batched : %7.2f ms  %6.1f GFLOPS' % (tb * 1000, gf / tb / 1e9))
print('自写 tiled      : %7.2f ms  %6.1f GFLOPS  (%.2fx)' % (tm * 1000, gf / tm / 1e9, tb / tm))
