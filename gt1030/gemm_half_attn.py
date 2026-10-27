# -*- coding: utf-8 -*-
"""k=64 的 batched GEMM（attention 的 S = Qn·Knᵀ 与 dP = dO·Vbᵀ 都是这个形状）。

为什么专门写它：CLBlast 的 strided-batched 在这种"k 极薄"的形状上只有 **118 GFLOPS**（独占口径），
而它自己在大形状上能到 500~675。k=64 可以直接整个装进 local：
  · At[64][64]（把 A 按 k 转置，At[kk][m]）+ Bt[64][64]（同理 Bt[kk][n]），各 8KB
  · 每工作组算 64x64 的 C tile，16x16 线程、每线程 4x4 寄存器累加
  · 计算循环每 kk 只需 2 条 vload4（local）换 16 次 FMA
  · 全局读沿 k 连续（A[i][k]），用 half4 向量取；写 C 逐行连续

    python gt1030/gemm_half_attn.py [--pid 16288]      # 会自动挂起/恢复 CPU trainer
"""
from __future__ import annotations
import argparse
import ctypes
import sys
import time

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl

KSRC = r"""
#define TT 64
__kernel void sgemm_nt64(__global const half* A, __global const half* B,
                         __global half* C, int T, float alpha) {
    __local half At[64 * 64];
    __local half Bt[64 * 64];
    int bh = get_global_id(2);
    int ti = get_group_id(0), tj = get_group_id(1);
    int lx = get_local_id(0), ly = get_local_id(1);
    int tid = ly * 16 + lx;
    __global const half* Ap = A + (long)bh * T * 64;
    __global const half* Bp = B + (long)bh * T * 64;
    __global half* Cp = C + (long)bh * T * T;
    int mb = ti * 64, nb = tj * 64;
    // 装 A/B tile（行方向取 64 行，k 方向 64 全取，转置存 local）
    for (int t = tid; t < 64 * 16; t += 256) {          // 64 行 x 16 个 half4
        int r = t / 16, c4 = (t % 16) * 4;
        half4 va = vload4(0, Ap + (long)(mb + r) * 64 + c4);
        At[(c4 + 0) * 64 + r] = va.x; At[(c4 + 1) * 64 + r] = va.y;
        At[(c4 + 2) * 64 + r] = va.z; At[(c4 + 3) * 64 + r] = va.w;
        half4 vb = vload4(0, Bp + (long)(nb + r) * 64 + c4);
        Bt[(c4 + 0) * 64 + r] = vb.x; Bt[(c4 + 1) * 64 + r] = vb.y;
        Bt[(c4 + 2) * 64 + r] = vb.z; Bt[(c4 + 3) * 64 + r] = vb.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int m0 = ly * 4, n0 = lx * 4;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    for (int kk = 0; kk < 64; ++kk) {
        half4 a = vload4(0, (__local half*)At + kk * 64 + m0);
        half4 b = vload4(0, (__local half*)Bt + kk * 64 + n0);
        float av[4] = {(float)a.x, (float)a.y, (float)a.z, (float)a.w};
        float bv[4] = {(float)b.x, (float)b.y, (float)b.z, (float)b.w};
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] += av[i] * bv[j];
    }
    for (int i = 0; i < 4; ++i) {
        int gm = mb + m0 + i;
        for (int j = 0; j < 4; ++j) {
            int gn = nb + n0 + j;
            Cp[(long)gm * T + gn] = (half)(acc[i][j] * alpha);
        }
    }
}
"""


def build(ctx):
    prog = cl.Program(ctx, KSRC).build(options=['-cl-mad-enable'])
    return cl.Kernel(prog, 'sgemm_nt64')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, default=16288)
    ap.add_argument('--suspend', type=int, default=1)
    args = ap.parse_args()
    import gt1030.ar_v2_multi as MM
    import gt1030.attention_gpu as A

    ctx, q = A.ctx, A.queue
    mf = cl.mem_flags
    T, HD, BH = MM.T, MM.HD, MM.BH
    kern = build(ctx)
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

    def mine():
        kern(q, (T // 64 * 16, T // 64 * 16, BH), (16, 16, 1), qb, kb, c2,
             np.int32(T), np.float32(1.0))

    h = None
    if args.suspend:
        PROCESS_SUSPEND_RESUME = 0x0800
        ntdll = ctypes.WinDLL('ntdll')
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        h = k32.OpenProcess(PROCESS_SUSPEND_RESUME, False, args.pid)
        rc = ntdll.NtSuspendProcess(ctypes.c_void_p(h))
        print('[attn] CPU trainer 挂起 rc=%d' % rc, flush=True)
        time.sleep(3.0)
    try:
        for fn in (blast, mine):
            fn()
        q.finish()
        o1 = np.empty(BH * T * T, np.float16); cl.enqueue_copy(q, o1, c1)
        o2 = np.empty(BH * T * T, np.float16); cl.enqueue_copy(q, o2, c2)
        q.finish()
        ref = o1.astype(np.float32)
        rel = float(np.abs(o2.astype(np.float32) - ref).max() / (np.abs(ref).max() + 1e-12))

        def t(fn, n=10):
            fn(); q.finish()
            t0 = time.time()
            for _ in range(n):
                fn()
            q.finish()
            return (time.time() - t0) / n

        t1, t2 = t(blast), t(mine)
        gf = 2.0 * BH * T * T * HD / 1e9
        print('S=Qn·Knᵀ  (16 x 1024x1024x64)')
        print('  CLBlast batched : %7.2f ms  %6.1f GFLOPS' % (t1 * 1000, gf / t1))
        print('  自写 k64 kernel : %7.2f ms  %6.1f GFLOPS  (%.2fx)'
              % (t2 * 1000, gf / t2, t1 / t2))
        print('  数值 rel=%.2e %s' % (rel, 'OK' if rel < 5e-2 else 'FAIL'))
    finally:
        if h:
            PROCESS_SUSPEND_RESUME = 0x0800
            ntdll = ctypes.WinDLL('ntdll')
            k32 = ctypes.WinDLL('kernel32')
            print('[attn] resume rc=%d' % ntdll.NtResumeProcess(ctypes.c_void_p(h)), flush=True)
            k32.CloseHandle(ctypes.c_void_p(h))


if __name__ == '__main__':
    main()
