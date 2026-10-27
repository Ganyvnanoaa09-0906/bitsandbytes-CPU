# -*- coding: utf-8 -*-
"""k=64 batched GEMM 配置扫描（S=Qn·Knᵀ / dP=dO·Vbᵀ 共用）。

    python gt1030/gemm_half_attn_sweep.py [--pid N] [--suspend 0|1]
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
#define TM {TM}
#define TN {TN}
#define LX {LX}
#define LY {LY}
#define MS (LX*TM)
#define NS (LY*TN)
__kernel void sgemm_nt64(__global const half* A, __global const half* B,
                         __global half* C, int T, float alpha) {
    __local half At[64 * MS];
    __local half Bt[64 * NS];
    int bh = get_group_id(2);
    int ti = get_group_id(0), tj = get_group_id(1);
    int lx = get_local_id(0), ly = get_local_id(1);
    int tid = ly * LX + lx;
    __global const half* Ap = A + (long)bh * T * 64;
    __global const half* Bp = B + (long)bh * T * 64;
    __global half* Cp = C + (long)bh * T * T;
    int mb = ti * MS, nb = tj * NS;
    // A: 读 MS 行 x 64(16 个 half4)，转置进 At[kk][m]
    for (int t = tid; t < MS * 16; t += (LX*LY)) {
        int r = t / 16, c4 = (t % 16) * 4;
        half4 va = vload4(0, Ap + (long)(mb + r) * 64 + c4);
        At[(c4+0)*MS + r] = va.x; At[(c4+1)*MS + r] = va.y;
        At[(c4+2)*MS + r] = va.z; At[(c4+3)*MS + r] = va.w;
    }
    // B: 读 NS 行 x 64，转置进 Bt[kk][n]
    for (int t = tid; t < NS * 16; t += (LX*LY)) {
        int r = t / 16, c4 = (t % 16) * 4;
        half4 vb = vload4(0, Bp + (long)(nb + r) * 64 + c4);
        Bt[(c4+0)*NS + r] = vb.x; Bt[(c4+1)*NS + r] = vb.y;
        Bt[(c4+2)*NS + r] = vb.z; Bt[(c4+3)*NS + r] = vb.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int m0 = ly * TM, n0 = lx * TN;
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;
    for (int kk = 0; kk < 64; ++kk) {
        float av[TM], bv[TN];
#pragma unroll
        for (int t = 0; t < TM; t += 4) {
            half4 a = vload4(0, (__local half*)At + kk * MS + m0 + t);
            av[t] = (float)a.x; av[t+1] = (float)a.y;
            av[t+2] = (float)a.z; av[t+3] = (float)a.w;
        }
#pragma unroll
        for (int t = 0; t < TN; t += 4) {
            half4 b = vload4(0, (__local half*)Bt + kk * NS + n0 + t);
            bv[t] = (float)b.x; bv[t+1] = (float)b.y;
            bv[t+2] = (float)b.z; bv[t+3] = (float)b.w;
        }
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) acc[i][j] += av[i] * bv[j];
    }
    for (int i = 0; i < TM; ++i) {
        int gm = mb + m0 + i;
        if (gm >= T) break;
        for (int j = 0; j < TN; ++j) {
            int gn = nb + n0 + j;
            if (gn >= T) break;
            Cp[(long)gm * T + gn] = (half)(acc[i][j] * alpha);
        }
    }
}
"""

CFGS = [
    ('64x64  t4x4 16x16', dict(TM=4, TN=4, LX=16, LY=16)),
    ('64x64  t8x8 8x8', dict(TM=8, TN=8, LX=8, LY=8)),
    ('128x64 t8x4 16x16', dict(TM=8, TN=4, LX=16, LY=16)),
    ('64x128 t4x8 16x16', dict(TM=4, TN=8, LX=16, LY=16)),
    ('128x128 t8x8 16x16', dict(TM=8, TN=8, LX=16, LY=16)),
    ('128x64 t8x4 8x16', dict(TM=8, TN=4, LX=8, LY=16)),
    ('64x64  t8x8 16x4', dict(TM=8, TN=8, LX=4, LY=16)),
]


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
    rng = np.random.default_rng(0)
    Q = (rng.standard_normal((BH, T, HD)) * 0.5).astype(np.float16)
    Kb = (rng.standard_normal((BH, T, HD)) * 0.5).astype(np.float16)
    qb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Q)
    kb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Kb)
    c1 = cl.Buffer(ctx, mf.READ_WRITE, BH * T * T * 2)
    c2 = cl.Buffer(ctx, mf.READ_WRITE, BH * T * T * 2)

    def blast():
        A.hgemm_batched(qb, kb, c1, T, T, HD, A.NO, A.YES, HD, HD, T,
                        T * HD, T * HD, T * T)

    def t(fn, n=10):
        fn(); q.finish()
        t0 = time.time()
        for _ in range(n):
            fn()
        q.finish()
        return (time.time() - t0) / n

    h = None
    if args.suspend:
        ntdll = ctypes.WinDLL('ntdll')
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        h = k32.OpenProcess(0x0800, False, args.pid)
        ntdll.NtSuspendProcess(ctypes.c_void_p(h))
        time.sleep(3.0)
    try:
        tb = t(blast)
        gf = 2.0 * BH * T * T * HD / 1e9
        o1 = np.empty(BH * T * T, np.float16); cl.enqueue_copy(q, o1, c1); q.finish()
        print('CLBlast batched: %.2f ms  %.1f GFLOPS' % (tb * 1000, gf / tb))
        best = None
        for nm, cfg in CFGS:
            src = SRC
            for k, v in cfg.items():
                src = src.replace('{%s}' % k, str(v))
            try:
                kern = cl.Kernel(cl.Program(ctx, src).build(options=['-cl-mad-enable']),
                                 'sgemm_nt64')
            except Exception as e:
                print('%-20s BUILD FAIL %s' % (nm, str(e)[:50])); continue
            gx = (T + cfg['LX'] * cfg['TM'] - 1) // (cfg['LX'] * cfg['TM'])
            gy = (T + cfg['LY'] * cfg['TN'] - 1) // (cfg['LY'] * cfg['TN'])

            def run():
                kern(q, (gx * cfg['LX'], gy * cfg['LY'], BH), (cfg['LX'], cfg['LY'], 1),
                     qb, kb, c2, np.int32(T), np.float32(1.0))
            try:
                tm = t(run)
            except Exception as e:
                print('%-20s RUN FAIL %s' % (nm, str(e)[:60])); continue
            o2 = np.empty(BH * T * T, np.float16); cl.enqueue_copy(q, o2, c2); q.finish()
            rel = float(np.abs(o2.astype(np.float32) - o1.astype(np.float32)).max()
                        / (np.abs(o1.astype(np.float32)).max() + 1e-12))
            ok = rel < 5e-2
            print('%-20s %6.2f ms  %6.1f GFLOPS  %.2fx  rel=%.1e %s'
                  % (nm, tm * 1000, gf / tm, tb / tm, rel, 'OK' if ok else 'BAD'))
            if ok and (best is None or tm < best[0]):
                best = (tm, nm)
        print('BEST: %s' % (best,))
    finally:
        if h:
            ntdll = ctypes.WinDLL('ntdll')
            k32 = ctypes.WinDLL('kernel32')
            print('resume rc=%d' % ntdll.NtResumeProcess(ctypes.c_void_p(h)))
            k32.CloseHandle(ctypes.c_void_p(h))


if __name__ == '__main__':
    main()
