"""igpu_gemm_fp16.py -- GEMM with fp16 inputs, packed arithmetic, split accumulation.

Established by measurement just now, on this device:

  * fp32 pure compute: 1089 GFLOPS, 95% of the 1150 peak.
  * fp16 packed (half2, fp16 accumulate): 4665 GFLOPS, 4.28x fp32.
  * fp16 storage with fp32 accumulate: 380 GFLOPS, 0.35x -- converting back and forth
    each operation is slower than staying in fp32.

While the project's real fp32 GEMM reaches only 300-494 GFLOPS. So compute is not the
constraint, data supply is -- and fp16 attacks both at once: half2 arithmetic for the
multiply-accumulate, and half the bytes through local memory, which directly relieves
the bank-conflict pressure that caps the fp32 kernel.

The obvious objection is precision, and it splits in two:

  * input precision: fp16 has a 10-bit mantissa, roughly 11 bits of significance. The
    3-bit weight quantisation already measured at +0.0195 of validation loss, so fp16
    inputs are far more precise than what is already accepted.
  * accumulation precision: summing hundreds of terms in fp16 would lose it. So
    accumulate in half2 for a short block, then promote to a float32 accumulator. The
    error grows as sqrt(K/S) rather than K, and S is tunable.

Correctness is checked against numpy on real model shapes before any timing is
believed, with the tolerance set by fp16's input precision rather than by fp32's.
"""
from __future__ import annotations

import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

import pyopencl as cl  # noqa: E402

SRC = r'''
// C[m,n] = sum_k A[m,k]*B[k,n], A and B in half, C in float32.
// Output tile TS x TS per workgroup, each thread TM x TN.
// Accumulation: SPLIT terms in half2, then promoted to float32.
#define TS 64
#define TK 16
#define TX 8
#define TY 8
#define TM 8
#define TN 8
#define SPLIT 8

__kernel void gemm_f16(const int M, const int N, const int K,
                       const __global half* restrict A,
                       const __global half* restrict B,
                       __global float* restrict C) {
    __local half As[TS][TK + 2];
    __local half Bs[TK][TS + 2];

    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int NT = TX * TY, lane = ty * TX + tx;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += TK) {
        #pragma unroll
        for (int p = 0; p < TS * TK / 64; p++) {
            int flat = p * 64 + lane;
            int r = flat / TK, c = flat % TK;
            int gr = by + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.f;
        }
        #pragma unroll
        for (int p = 0; p < TK * TS / 64; p++) {
            int flat = p * 64 + lane;
            int r = flat / TS, c = flat % TS;
            int gr = k0 + r, gc = bx + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0.f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // split accumulation: SPLIT terms in half2, then promote
        for (int k = 0; k < TK; k += SPLIT) {
            half2 part[TM][TN];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) part[i][j] = (half2)(0.f, 0.f);
            // both halves must carry a real term, or half the arithmetic is wasted:
            // pack adjacent K terms, which is what the packed units are for
            #pragma unroll
            for (int kk = 0; kk < SPLIT; kk += 2) {
                half2 a[TM], b[TN];
                #pragma unroll
                for (int i = 0; i < TM; i++)
                    a[i] = (half2)(As[ty * TM + i][k + kk], As[ty * TM + i][k + kk + 1]);
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    b[j] = (half2)(Bs[k + kk][tx * TN + j], Bs[k + kk + 1][tx * TN + j]);
                #pragma unroll
                for (int i = 0; i < TM; i++)
                    #pragma unroll
                    for (int j = 0; j < TN; j++) part[i][j] = fma(a[i], b[j], part[i][j]);
            }
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    // both lanes carry a term, so both must be promoted: taking only
                    // .s0 loses half the sum, which showed up as 0.7 relative error
                    acc[i][j] += (float)(part[i][j].s0) + (float)(part[i][j].s1);
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

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
prg = cl.Program(ctx, SRC).build()
kern = cl.Kernel(prg, 'gemm_f16')
print('设备 %s | %d CU | 本地 %d KB' % (dev.name, dev.max_compute_units, dev.local_mem_size // 1024))
print('fp16 输入 + half2 分块累加（SPLIT=8）→ float32 累加器，TS=64, TK=16, 8x8 线程')
print()

CPU = 198.5


def run(M, N, K, reps=3):
    rng = np.random.default_rng(3)
    Af = rng.standard_normal((M, K)).astype(np.float32)
    Bf = rng.standard_normal((K, N)).astype(np.float32)
    Ah = Af.astype(np.float16)
    Bh = Bf.astype(np.float16)
    ref = Af @ Bf                                   # fp32 reference
    ref16 = Ah.astype(np.float32) @ Bh.astype(np.float32)   # fp16-input reference
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Ah)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Bh)
    bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    gsz = ((N + TS - 1) // TS * 8, (M + TS - 1) // TS * 8)

    def once():
        kern(queue, gsz, (8, 8), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
        queue.finish()

    once()
    out = np.zeros((M, N), dtype=np.float32)
    cl.enqueue_copy(queue, out, bc).wait()
    # two tolerances matter: against the fp16-input product (this tests the kernel)
    # and against the fp32 product (this tests the format)
    d_kernel = float(np.abs(out - ref16).max()) / (float(np.abs(ref16).max()) or 1.0)
    d_format = float(np.abs(out - ref).max()) / (float(np.abs(ref).max()) or 1.0)
    t0 = time.perf_counter()
    for _ in range(reps):
        once()
    dt = (time.perf_counter() - t0) / reps
    gf = 2.0 * M * N * K / dt / 1e9
    del ba, bb, bc
    return gf, d_kernel, d_format


TS = 64
print('%-20s %10s %11s %11s  %s' % ('形状', 'GFLOPS', '相对CPU', '内核 vs fp16', '格式 vs fp32'))
print('-' * 78)
for (M, N, K) in [(1024, 512, 512), (1024, 2736, 512), (1024, 512, 2736),
                  (1024, 1024, 512), (2048, 2048, 2048), (256, 512, 128)]:
    try:
        gf, dk, df = run(M, N, K)
        print('%-20s %10.1f %10.2fx %11.2e %11.2e  %s'
              % ('%dx%dx%d' % (M, N, K), gf, gf / CPU, dk, df,
                 '✓' if dk < 2e-2 else '✗ 内核有误'))
    except Exception as e:
        print('%-20s %10s %11s %11s  %s' % ('%dx%dx%d' % (M, N, K), 'FAIL', '-', '-',
                                            str(e)[:30]))
print()
print('判据: 内核误差应 ≈ fp16 精度（1e-3 量级），显著更大说明累加有问题；')
print('      格式误差是 fp16 输入本身的代价，与内核无关。')
