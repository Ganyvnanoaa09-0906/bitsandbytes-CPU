"""sweep_gemm_v5.py -- parameter sweep on a correct kernel, correctness checked each point.

v3 fixed the staging layout and is exactly correct but slow (92-126 GFLOPS); v4
added float4 access and 4x4 register tiles for 129-164. CLBlast reaches 493.8 on
this device, so a factor of three remains, and CLBlast's own tuning database names
the configuration it uses for gfx902:

    MDIMC=8  NDIMC=8   -> 64 threads per workgroup, not 256
    MWG=64   NWG=64    -> 64x64 output tile
    KWG=32             -> 32 elements along K per staging step, not 16
    VWM=4    VWN=4     -> 2x2 blocks of 4x4 outputs per thread = 8x8

Every one of those differs from v4. Rather than rewrite the kernel after CLBlast's
structure -- which is a large, error-prone edit at this hour -- parameterise v3/v4's
structure, which is already verified correct, and sweep it. Each configuration is
checked against numpy before its timing is believed, because a throughput number
from an unchecked kernel is what wasted several hours today.

Local memory stays within 32 KB: 2 * TS * TK * 4 bytes.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
import pyopencl as cl  # noqa: E402

TEMPLATE = r'''
#define TS  {TS}
#define TK  {TK}
#define TY  {TY}
#define TX  {TX}
#define TM  {TM}
#define TN  {TN}

__kernel void gemm( const int M, const int N, const int K,
                    const __global float* restrict A,
                    const __global float* restrict B,
                    __global float* restrict C) {
    __local float As[TS][TK + 1];
    __local float Bs[TK][TS + 1];

    const int tx = get_local_id(0);          // 0..TX-1
    const int ty = get_local_id(1);          // 0..TY-1
    const int bx = get_group_id(0) * TS;
    const int by = get_group_id(1) * TS;
    const int NT = TX * TY;
    const int lane = ty * TX + tx;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += TK) {
        for (int p = lane; p < TS * TK / 4; p += NT) {
            int f = p * 4;
            int r = f / TK, c = f % TK;
            int gr = by + r, gc = k0 + c;
            float4 v = (float4)0.f;
            if (gr < M && gc + 3 < K) v = *(const __global float4*)(A + gr * K + gc);
            As[r][c+0]=v.x; As[r][c+1]=v.y; As[r][c+2]=v.z; As[r][c+3]=v.w;
        }
        for (int p = lane; p < TK * TS / 4; p += NT) {
            int f = p * 4;
            int r = f / TS, c = f % TS;
            int gr = k0 + r, gc = bx + c;
            float4 v = (float4)0.f;
            if (gr < K && gc + 3 < N) v = *(const __global float4*)(B + gr * N + gc);
            Bs[r][c+0]=v.x; Bs[r][c+1]=v.y; Bs[r][c+2]=v.z; Bs[r][c+3]=v.w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] += a[i] * b[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gr = by + ty * TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gc = bx + tx * TN + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
    }
}
'''

CONFIGS = [
    (64,  8, 8, 8, 8, 8, 'TS=64 TK=8（4KB，上轮最佳）'),
    (64,  4, 8, 8, 8, 8, 'TS=64 TK=4（2KB）'),
    (64,  2, 8, 8, 8, 8, 'TS=64 TK=2（1KB）'),
    (48,  8, 8, 8, 6, 6, 'TS=48 TK=8（3KB）'),
    (48,  4, 8, 8, 6, 6, 'TS=48 TK=4（1.5KB）'),
    (32,  4, 8, 8, 4, 4, 'TS=32 TK=4（1KB）'),
    (32,  2, 8, 8, 4, 4, 'TS=32 TK=2（0.5KB）'),
    (16,  8, 8, 8, 2, 2, 'TS=16 TK=8（1KB，小 tile 多组）'),
    (16, 16, 8, 8, 2, 2, 'TS=16 TK=16（2KB）'),
    (128,  8, 8, 8, 16, 16, 'TS=128 TK=8（8KB）'),
]

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
print('设备 %s | %d CU | 本地 %d KB' % (dev.name, dev.max_compute_units, dev.local_mem_size // 1024))
print()
CPU = 198.5
M, N, K = 1024, 2736, 512        # 模型的真实形状
rng = np.random.default_rng(11)
A = rng.standard_normal((M, K)).astype(np.float32)
B = rng.standard_normal((K, N)).astype(np.float32)
ref = A @ B
print('形状 M=%d N=%d K=%d（模型真实形状）' % (M, N, K))
print()
print('%-34s %8s %10s %11s %10s %s' % ('配置', 'TS,TK', '线程', 'GFLOPS', '相对CPU', '正确性'))
print('-' * 88)
best = None
for (TS, TK, TX, TY, TM, TN, label) in CONFIGS:
    if TX * TM != TS or TY * TN != TS:
        print('%-34s %-8s %-10s %10s %10s %s'
              % (label[:34], '%d,%d' % (TS, TK), '%dx%d' % (TX, TY), 'SKIP', '-',
                 '线程×每线程 ≠ tile'))
        continue
    lds_kb = 2 * TS * TK * 4 / 1024.0
    if lds_kb > 32:
        print('%-34s %-8s %-10s %10s %10s %s'
              % (label[:34], '%d,%d' % (TS, TK), '%dx%d' % (TX, TY), 'SKIP', '-',
                 '本地内存 %.1f KB > 32' % lds_kb))
        continue
    try:
        # str.format cannot be used: the OpenCL source is full of braces that it
        # would try to interpret as fields. Substitute the six placeholders only.
        src = TEMPLATE
        for k, v in (('#define TS  {TS}', '#define TS  %d' % TS),
                     ('#define TK  {TK}', '#define TK  %d' % TK),
                     ('#define TY  {TY}', '#define TY  %d' % TY),
                     ('#define TX  {TX}', '#define TX  %d' % TX),
                     ('#define TM  {TM}', '#define TM  %d' % TM),
                     ('#define TN  {TN}', '#define TN  %d' % TN)):
            src = src.replace(k, v)
        prg = cl.Program(ctx, src).build()
        kern = cl.Kernel(prg, 'gemm')
        ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
        gsz = ((N + TS - 1) // TS * TX, (M + TS - 1) // TS * TY)

        def once():
            kern(queue, gsz, (TX, TY), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
            queue.finish()

        once()
        out = np.zeros((M, N), dtype=np.float32)
        cl.enqueue_copy(queue, out, bc).wait()
        bad = int((np.abs(out - ref) > 1e-2 * (np.abs(ref).max() or 1)).sum())
        t0 = time.perf_counter()
        for _ in range(3):
            once()
        dt = (time.perf_counter() - t0) / 3
        gf = 2.0 * M * N * K / dt / 1e9
        tag = '✓ 全对' if bad == 0 else '%d 错' % bad
        print('%-34s %-8s %-10s %10.1f %10.2fx %s'
              % (label[:34], '%d,%d' % (TS, TK), '%dx%d' % (TX, TY), gf, gf / CPU, tag))
        if bad == 0 and (best is None or gf > best[0]):
            best = (gf, label, (TS, TK, TX, TY, TM, TN))
        del ba, bb, bc
    except Exception as e:
        print('%-34s %-8s %-10s %10s %10s %s'
              % (label[:34], '%d,%d' % (TS, TK), '%dx%d' % (TX, TY), 'FAIL', '-', str(e)[:30]))
print()
if best:
    print('  ⇒ 最佳: %s  %.1f GFLOPS  %s' % (best[1], best[0], best[2]))
    print('     目标 500（CPU 的 2.5x），CLBlast 同设备实测 493.8')
