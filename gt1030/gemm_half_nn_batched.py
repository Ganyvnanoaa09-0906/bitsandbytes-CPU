# -*- coding: utf-8 -*-
"""batched GEMM：C[b] = A[b]·B[b]，A (T,T) row-major、B (T,64) row-major（k=T 可切块）。

用于 attention 的
    O  = P  · Vb      (前向)
    dQ = dS · Kn      (反向)
两者形状相同：(1024×1024) @ (1024×64)。

独占口径下 CLBlast 这两个各 11.7ms（184 GFLOPS）；本 kernel 走 MS=128/NS=64/BK=64、
A 转置进 local 以便计算循环用 vload4，实测见 gemm_half_nn_batched 的自测输出。

    python gt1030/gemm_half_nn_batched.py [--pid N] [--suspend 0|1]
"""
from __future__ import annotations
import argparse
import ctypes
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl

SRC = r"""
#define MS 128
#define NS 64
#define BK 64
#define TM 8
#define TN 4

__kernel void sgemm_nn_bt(__global const half* A, __global const half* B,
                          __global half* C, int T, float alpha) {
    __local half At[BK * MS];
    __local half Bt[BK * NS];
    int bh = get_group_id(2);
    int mb = get_group_id(0) * MS;
    int lx = get_local_id(0), ly = get_local_id(1);
    int tid = ly * 16 + lx;
    __global const half* Ap = A + (long)bh * T * T;      // (T,T)
    __global const half* Bp = B + (long)bh * T * 64;     // (T,64)
    __global half* Cp = C + (long)bh * T * 64;           // (T,64)
    int m0 = ly * TM, n0 = lx * TN;
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < T; k0 += BK) {
        // A tile: MS 行 x BK 列，读连续 half4，转置进 At[k][m]
        for (int t = tid; t < MS * (BK / 4); t += 256) {
            int r = t / (BK / 4), c4 = (t % (BK / 4)) * 4;
            half4 v = vload4(0, Ap + (long)(mb + r) * T + k0 + c4);
            At[(c4 + 0) * MS + r] = v.x; At[(c4 + 1) * MS + r] = v.y;
            At[(c4 + 2) * MS + r] = v.z; At[(c4 + 3) * MS + r] = v.w;
        }
        // B tile: BK 行 x 64 列，行内连续
        for (int t = tid; t < BK * 16; t += 256) {
            int r = t / 16, c4 = (t % 16) * 4;
            half4 v = vload4(0, Bp + (long)(k0 + r) * 64 + c4);
            vstore4(v, 0, (__local half*)Bt + r * NS + c4);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < BK; ++kk) {
            float av[TM], bv[TN];
#pragma unroll
            for (int t = 0; t < TM; t += 4) {
                half4 a = vload4(0, (__local half*)At + kk * MS + m0 + t);
                av[t] = (float)a.x; av[t+1] = (float)a.y;
                av[t+2] = (float)a.z; av[t+3] = (float)a.w;
            }
            half4 b = vload4(0, (__local half*)Bt + kk * NS + n0);
            bv[0] = (float)b.x; bv[1] = (float)b.y; bv[2] = (float)b.z; bv[3] = (float)b.w;
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += av[i] * bv[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = 0; i < TM; ++i) {
        int gm = mb + m0 + i;
        if (gm >= T) break;
        for (int j = 0; j < TN; ++j) {
            int gn = n0 + j;
            if (gn >= 64) break;
            Cp[(long)gm * 64 + gn] = (half)(acc[i][j] * alpha);
        }
    }
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, default=16288)
    ap.add_argument('--suspend', type=int, default=0)
    args = ap.parse_args()
    import gt1030.ar_v2_multi as MM
    import gt1030.attention_gpu as A

    ctx, q = A.ctx, A.queue
    mf = cl.mem_flags
    T, HD, BH = MM.T, MM.HD, MM.BH
    kern = cl.Kernel(cl.Program(ctx, SRC).build(options=['-cl-mad-enable']), 'sgemm_nn_bt')
    rng = np.random.default_rng(0)
    P = (rng.standard_normal((BH, T, T)) * 0.1).astype(np.float16)
    V = (rng.standard_normal((BH, T, HD)) * 0.5).astype(np.float16)
    pb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=P)
    vb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=V)
    c1 = cl.Buffer(ctx, mf.READ_WRITE, BH * T * HD * 2)
    c2 = cl.Buffer(ctx, mf.READ_WRITE, BH * T * HD * 2)

    def blast():
        A.hgemm_batched(pb, vb, c1, T, HD, T, A.NO, A.NO, T, HD, HD,
                        T * T, T * HD, T * HD)

    def mine():
        kern(q, ((T + 127) // 128 * 16, 16, BH), (16, 16, 1), pb, vb, c2,
             np.int32(T), np.float32(1.0))

    h = None
    if args.suspend:
        ntdll = ctypes.WinDLL('ntdll')
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        h = k32.OpenProcess(0x0800, False, args.pid)
        ntdll.NtSuspendProcess(ctypes.c_void_p(h))
        print('[nn_bt] CPU trainer 挂起', flush=True)
        time.sleep(3.0)
    try:
        for fn in (blast, mine):
            fn()
        q.finish()
        o1 = np.empty(BH * T * HD, np.float16); cl.enqueue_copy(q, o1, c1)
        o2 = np.empty(BH * T * HD, np.float16); cl.enqueue_copy(q, o2, c2)
        q.finish()
        rel = float(np.abs(o2.astype(np.float32) - o1.astype(np.float32)).max()
                    / (np.abs(o1.astype(np.float32)).max() + 1e-12))

        def t(fn, n=10):
            fn(); q.finish()
            t0 = time.time()
            for _ in range(n):
                fn()
            q.finish()
            return (time.time() - t0) / n

        t1, t2 = t(blast), t(mine)
        gf = 2.0 * BH * T * T * HD / 1e9
        print('O=P·Vb / dQ=dS·Kn  (16 x 1024x64x1024)')
        print('  CLBlast batched : %7.2f ms  %6.1f GFLOPS' % (t1 * 1000, gf / t1))
        print('  自写 nn batched : %7.2f ms  %6.1f GFLOPS  (%.2fx)  rel=%.2e %s'
              % (t2 * 1000, gf / t2, t1 / t2, rel, 'OK' if rel < 5e-2 else 'FAIL'))
    finally:
        if h:
            ntdll = ctypes.WinDLL('ntdll')
            k32 = ctypes.WinDLL('kernel32')
            print('resume rc=%d' % ntdll.NtResumeProcess(ctypes.c_void_p(h)), flush=True)
            k32.CloseHandle(ctypes.c_void_p(h))


if __name__ == '__main__':
    main()
