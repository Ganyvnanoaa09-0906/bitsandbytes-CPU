# -*- coding: utf-8 -*-
"""探针 50：s_gemm 改成 1D 工作组（256 线程，tile 由 group id 分解），先验证正确再测速。"""
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_multi as MM
import gt1030.attention_gpu as A

T, HD, BH = 1024, 64, 16
KSRC = r"""
__kernel void s_gemm(__global const half* Q, __global const half* Kt,
                     __global half* S, float scale) {
    int wg = get_group_id(0);
    int bh = wg / 256;
    int rem = wg % 256;
    int ti = rem / 16, tj = rem % 16;
    int lid = get_local_id(0);
    int lx = lid % 16, ly = lid / 16;
    __local half Qs[64][64];
    __local half Ks[64][64];
    __global const half* Qp = Q + (long)bh * 1024 * 64;
    __global const half* Kp = Kt + (long)bh * 1024 * 64;
    __global half* Sp = S + (long)bh * 1024 * 1024;
    for (int i = lid; i < 4096; i += 256) {
        int r = i / 64, c = i % 64;
        Qs[r][c] = Qp[(long)(ti * 64 + r) * 64 + c];
        Ks[r][c] = Kp[(long)(tj * 64 + r) * 64 + c];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int r0 = ly * 4, c0 = lx * 4;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float acc = 0.f;
            for (int kk = 0; kk < 64; ++kk)
                acc += (float)Qs[r0 + i][kk] * (float)Ks[c0 + j][kk];
            Sp[(long)(ti * 64 + r0 + i) * 1024 + (tj * 64 + c0 + j)] = (half)(acc * scale);
        }
    }
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
GRID = (BH * 256 * 256,)


def mine():
    kern(q_, GRID, (256,), Qb, Kb, S2, np.float32(SC))


def blast():
    A.hgemm_batched(Qb, Kb, S1, T, T, HD, A.NO, A.YES, HD, HD, T, T * HD, T * HD, T * T)


mine(); q_.finish()
out = np.empty(BH * T * T, np.float16); cl.enqueue_copy(q_, out, S2); q_.finish()
out = out.reshape(BH, T, T).astype(np.float32)
ref = (Qh.astype(np.float32) @ Kh.astype(np.float32).transpose(0, 2, 1)) * SC
rel = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-12))
print('vs numpy: rel=%.3e  %s  (nonzero %d)' % (rel, 'OK' if rel < 5e-3 else 'FAIL',
                                                int((out != 0).sum())))
blast(); q_.finish()
b1 = np.empty(BH * T * T, np.float16); cl.enqueue_copy(q_, b1, S1); q_.finish()
rel2 = float(np.abs(out.ravel() - b1.astype(np.float32)).max() / (np.abs(ref).max() + 1e-12))
print('vs CLBlast: rel=%.3e' % rel2)


def bench(fn, n=10):
    fn(); q_.finish()
    t0 = time.time()
    for _ in range(n):
        fn()
    q_.finish()
    return (time.time() - t0) / n


gf = 2 * BH * T * T * HD
tb = bench(blast)
tm = bench(mine)
print('CLBlast batched : %7.2f ms  %6.1f GFLOPS' % (tb * 1000, gf / tb / 1e9))
print('自写 1D tiled   : %7.2f ms  %6.1f GFLOPS  (%.2fx)' % (tm * 1000, gf / tm / 1e9, tb / tm))
