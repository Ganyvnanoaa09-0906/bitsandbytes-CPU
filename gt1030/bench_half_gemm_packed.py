"""Packed half2 custom GEMM for Qwen3-8B shapes.

half2 accumulators + packed half2 mad, half local tiles, half4 global loads, half2 output.
This targets the packed-fp16 path (the only way past the fp32-FMA ceiling of v1).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pyopencl as cl

SRC = r'''
#define TS %(TS)d
#define TK %(TK)d
#define TM 8
#define TN 8
#define NTN 4
#define NTH 64
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void hgemm2(const int M, const int N, const int K,
            const __global half* restrict A,
            const __global half* restrict B,
            __global half* restrict C) {
    __local half As[TK][TS + 8];
    __local half Bs[TK][TS + 8];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 8 + tx;
    half2 acc[TM][NTN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < NTN; ++j) acc[i][j] = (half2)(0, 0);

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
                if (gr < M && gc + 3 < K) hv = vload4(0, A + (long)gr * K + gc);
                As[kq*4+0][row] = hv.s0; As[kq*4+1][row] = hv.s1;
                As[kq*4+2][row] = hv.s2; As[kq*4+3][row] = hv.s3;
            } else if (f < ATOT4 + BTOT4) {
                int g = f - ATOT4, kk = g / BCOL4, cq = g %% BCOL4;
                int gr = k0 + kk, gc = bx + cq * 4;
                half4 hv = (half4)(0, 0, 0, 0);
                if (gr < K && gc + 3 < N) hv = vload4(0, B + (long)gr * N + gc);
                Bs[kk][cq*4+0] = hv.s0; Bs[kk][cq*4+1] = hv.s1;
                Bs[kk][cq*4+2] = hv.s2; Bs[kk][cq*4+3] = hv.s3;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < TK; ++k) {
            half a[TM];
            half2 b[NTN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < NTN; ++j)
                b[j] = vload2(0, &Bs[k][tx * TN + j * 2]);
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                half2 aa = (half2)(a[i], a[i]);
                #pragma unroll
                for (int j = 0; j < NTN; ++j)
                    acc[i][j] = mad(aa, b[j], acc[i][j]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int rc = by + ty * TM + i;
        if (rc >= M) continue;
        #pragma unroll
        for (int j = 0; j < NTN; ++j) {
            int cc = bx + tx * TN + j * 2;
            if (cc + 1 >= N) continue;
            vstore2(acc[i][j], 0, C + (long)rc * N + cc);
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
    ap.add_argument('--tk', type=int, default=8)
    ap.add_argument('--iters', type=int, default=5)
    args = ap.parse_args()
    ctx = cl.create_some_context(interactive=False)
    q = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    src = SRC % {'TS': args.ts, 'TK': args.tk}
    prog = cl.Program(ctx, src).build(options=['-cl-fast-relaxed-math', '-cl-mad-enable'])
    kern = cl.Kernel(prog, 'hgemm2')
    dev = ctx.devices[0]
    print('device %s CU=%d packed half2 TS=%d TK=%d' % (dev.name, dev.max_compute_units, args.ts, args.tk))

    for name, (M, N, K) in SHAPES.items():
        rng = np.random.default_rng(0)
        a = (rng.standard_normal((M, K)).astype(np.float32) * 0.02).astype(np.float16)
        b = (rng.standard_normal((K, N)).astype(np.float32) * 0.02).astype(np.float16)
        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 2)
        gws = (int(np.ceil(N / args.ts)) * 8, int(np.ceil(M / args.ts)) * 8)
        lws = (8, 8)
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
