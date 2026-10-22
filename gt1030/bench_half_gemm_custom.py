"""Custom half GEMM for Qwen3-8B shapes (gfx902, pyopencl runtime compile).

Kernel: 64x64 tile, TK-deep shared tiles, 8x8 threads, 8x8 register accumulators.
Global A/B are fp16, loads use half4; shared and accumulation are fp32; C is fp16.
Tunable TS/TK (TM=TN=8 fixed).

Target: beat CLBlast's ~928-957 GFLOPS and reach >=1130 GFLOPS on Qwen3-8B mlp-up/down/qkv.
Correctness: fp16 inputs vs float64 reference, max|err| reported.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pyopencl as cl

SRC = r'''
#define TS %(TS)d
#define TK %(TK)d
#define TM %(TM)d
#define TN %(TN)d
#define NTH %(NTH)d
#define TX %(TX)d
#define TY %(TY)d
__kernel __attribute__((reqd_work_group_size(TX, TY, 1)))
void hgemm(const int M, const int N, const int K,
           const __global half* restrict A,
           const __global half* restrict B,
           __global half* restrict C) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * TX + tx;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    const int AROW4 = TK / 4, BCOL4 = TS / 4;
    const int ATOT4 = TS * AROW4, BTOT4 = TK * BCOL4;
    const int NLOAD = (ATOT4 + BTOT4 + NTH - 1) / NTH;
    for (int k0 = 0; k0 < K; k0 += TK) {
        #pragma unroll
        for (int p = 0; p < NLOAD; ++p) {
            int f = p * NTH + lane;
            if (f < ATOT4) {
                int row = f / AROW4, kq = f %% AROW4;
                int gr = by + row, gc = k0 + kq * 4;
                half4 hv = (half4)(0, 0, 0, 0);
                if (gr < M)
                    hv = vload4(0, A + (long)gr * K + gc);
                float4 v = convert_float4(hv);
                As[kq*4+0][row] = v.x; As[kq*4+1][row] = v.y;
                As[kq*4+2][row] = v.z; As[kq*4+3][row] = v.w;
            } else if (f < ATOT4 + BTOT4) {
                int g = f - ATOT4, kk = g / BCOL4, cq = g %% BCOL4;
                int gr = k0 + kk, gc = bx + cq * 4;
                half4 hv = (half4)(0, 0, 0, 0);
                if (gr < K)
                    hv = vload4(0, B + (long)gr * N + gc);
                float4 v = convert_float4(hv);
                Bs[kk][cq*4+0] = v.x; Bs[kk][cq*4+1] = v.y;
                Bs[kk][cq*4+2] = v.z; Bs[kk][cq*4+3] = v.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < TK; ++k) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = mad(a[i], b[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int rc = by + ty * TM + i;
        if (rc >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int cc = bx + tx * TN + j;
            if (cc >= N) continue;
            C[(long)rc * N + cc] = (half)acc[i][j];
        }
    }
}
'''

SHAPES = {
    'qkv/o 512x4096x4096': (512, 4096, 4096),
    'mlp up 512x12288x4096': (512, 12288, 4096),
    'mlp down 512x4096x12288': (512, 4096, 12288),
    '1.7B qkv 512x2048x2048': (512, 2048, 2048),
    '1.7B up 512x6144x2048': (512, 6144, 2048),
    '1.7B down 512x2048x6144': (512, 2048, 6144),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ts', type=int, default=64)
    ap.add_argument('--tk', type=int, default=16)
    ap.add_argument('--iters', type=int, default=5)
    ap.add_argument('--tx', type=int, default=8)
    ap.add_argument('--ty', type=int, default=8)
    ap.add_argument('--tm', type=int, default=8)
    ap.add_argument('--tn', type=int, default=8)
    args = ap.parse_args()
    if args.ts % 64 or args.tk % 4:
        raise SystemExit('ts%64/tk%4 required')

    ctx = cl.create_some_context(interactive=False)
    q = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    ts, tk = args.ts, args.tk
    tx, ty, tm, tn = args.tx, args.ty, args.tm, args.tn
    assert ts == tx * tm and tk % 4 == 0
    src = SRC % {'TS': ts, 'TK': tk, 'TM': tm, 'TN': tn, 'NTH': tx * ty, 'TX': tx, 'TY': ty}
    prog = cl.Program(ctx, src).build(options=['-cl-fast-relaxed-math', '-cl-mad-enable'])
    kern = cl.Kernel(prog, 'hgemm')
    dev = ctx.devices[0]
    print('device %s CU=%d tile TS=%d TK=%d TM=%d TN=%d threads=%dx%d' % (dev.name, dev.max_compute_units, ts, tk, tm, tn, tx, ty))

    for name, (M, N, K) in SHAPES.items():
        rng = np.random.default_rng(0)
        a = (rng.standard_normal((M, K)).astype(np.float32) * 0.02).astype(np.float16)
        b = (rng.standard_normal((K, N)).astype(np.float32) * 0.02).astype(np.float16)
        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 2)
        gws = (int(np.ceil(N / ts)) * tx, int(np.ceil(M / ts)) * ty)
        lws = (tx, ty)
        out = np.empty((M, N), dtype=np.float16)

        def call():
            kern(q, gws, lws, np.int32(M), np.int32(N), np.int32(K), a_buf, b_buf, c_buf)

        call(); q.finish()
        samples = []
        for _ in range(args.iters):
            t0 = time.perf_counter(); call(); q.finish(); samples.append(time.perf_counter() - t0)
        sec = np.median(samples)
        cl.enqueue_copy(q, out, c_buf).wait()
        ref = a.astype(np.float64) @ b.astype(np.float64)
        err = float(np.abs(out.astype(np.float64) - ref).max())
        gflops = 2.0 * M * N * K / sec / 1e9
        print('%-26s %8.1f ms  %8.1f GFLOPS  max|err| %.4e  %.2fx vs1127'
              % (name, sec * 1e3, gflops, err, gflops / 1127.0), flush=True)


if __name__ == '__main__':
    main()
