"""igpu_gemm_v3.py -- a correct, occupancy-aware SGEMM for gfx902.

Established by measurement tonight:

  * CLBlast reaches 493.8 GFLOPS on this device for cubic shapes -- 43% of the
    1150 GFLOPS peak -- so the silicon can do it.
  * The project's sgemm_v2a stages A as __local As[TK][TS+4] = 16 rows x 64 cols,
    while a TS=64 tile needs 64 rows of A. Rows 0..15 come out exactly right and
    16..63 are garbage: 75% of the output wrong. Setting TS=16 to match TK drops the
    error count from 3072/4096 to 18/4096, confirming the layout is the cause.
  * With that mismatch the kernel measures 87-126 GFLOPS at the model's shapes,
    i.e. below the CPU's 198.5. But CLBlast's kernel structure explains the gap:
    its local buffers hold only KWG * MWG/VWM elements (2 KB for the tuned gfx902
    parameters), with most reuse happening in registers, whereas this kernel puts a
    full tile in local memory and reuses it less.
  * Every earlier iGPU figure in this project (245.3, 266.58, "+30-35%") came from
    benchmarks that never read their output back.

This kernel keeps the simple, understandable structure -- a 64x64 output tile per
workgroup, 8x8 threads, 8x8 outputs per thread in registers -- but fixes the
staging layout: As is [TS][TK+4], i.e. 64 rows x 16 columns of A, matching how the
tile is actually consumed. Both the staging and the compute index it the same way.

Correctness is checked on the first run, at cubic and non-cubic shapes, because a
throughput number without that check is what cost several hours today.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
import pyopencl as cl  # noqa: E402

KERNEL = r'''
// C = A * B, all row-major, A is (M,K), B is (K,N), C is (M,N)
// Workgroup computes a TS x TS output tile. 8x8 threads, each 8x8 outputs.
// As[m][k] holds TS rows x TK columns of A; Bs[k][n] holds TK rows x TS cols of B.
#define TS 64
#define TK 16
#define TM 8
#define TN 8

__kernel void sgemm_v3(const int M, const int N, const int K,
                       const __global float* restrict A,
                       const __global float* restrict B,
                       __global float* restrict C) {
    // As[TS][TK+1]: one row of A's tile, TK wide, padded to avoid bank conflicts
    __local float As[TS][TK + 1];
    // Bs[TK][TS+1]: one K-slice of B, TS wide along N
    __local float Bs[TK][TS + 1];

    const int tx = get_local_id(0);          // 0..7
    const int ty = get_local_id(1);          // 0..7
    const int bx = get_group_id(0) * TS;     // column (N) origin
    const int by = get_group_id(1) * TS;     // row (M) origin
    const int lane = ty * 8 + tx;            // 0..63

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += TK) {
        // ---- stage A: TS rows x TK cols. Row-major source: A[(by+r)*K + k0+c]
        #pragma unroll
        for (int p = 0; p < (TS * TK) / 64; p++) {
            int flat = p * 64 + lane;        // 0 .. TS*TK-1
            int r = flat / TK;               // row within tile  0..TS-1
            int c = flat % TK;               // col within tile  0..TK-1
            int gr = by + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        // ---- stage B: TK rows x TS cols. B[(k0+k)*N + bx+n]
        #pragma unroll
        for (int p = 0; p < (TK * TS) / 64; p++) {
            int flat = p * 64 + lane;
            int k = flat / TS;
            int n = flat % TS;
            int gk = k0 + k, gn = bx + n;
            Bs[k][n] = (gk < K && gn < N) ? B[gk * N + gn] : 0.f;
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

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
prg = cl.Program(ctx, KERNEL).build()
kern = cl.Kernel(prg, 'sgemm_v3')
print('设备 %s | %d CU | 本地内存 %d KB' % (dev.name, dev.max_compute_units, dev.local_mem_size // 1024))
print('v3: 64x64 tile, 8x8 线程, 每线程 8x8 输出, As[64][17] + Bs[16][65]')
print()


def run(M, N, K, reps=3, check=True):
    rng = np.random.default_rng(4)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    gsz = ((N + 63) // 64 * 8, (M + 63) // 64 * 8)

    def once():
        kern(queue, gsz, (8, 8), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
        queue.finish()

    once()
    bad = -1
    if check:
        out = np.zeros((M, N), dtype=np.float32)
        cl.enqueue_copy(queue, out, bc).wait()
        ref = A @ B
        bad = int((np.abs(out - ref) > 1e-2 * (np.abs(ref).max() or 1)).sum())
    t0 = time.perf_counter()
    for _ in range(reps):
        once()
    dt = (time.perf_counter() - t0) / reps
    gf = 2.0 * M * N * K / dt / 1e9
    del ba, bb, bc
    return gf, bad, M * N


CPU = 198.5
print('%-20s %10s %11s %10s %s' % ('形状 MxNxK', 'GFLOPS', '相对 CPU', '错误', '结论'))
print('-' * 70)
for (M, N, K) in [(64, 64, 64), (256, 256, 256), (1024, 512, 512),
                  (1024, 2736, 512), (1024, 512, 2736), (1024, 1024, 512),
                  (2048, 2048, 2048)]:
    try:
        gf, bad, tot = run(M, N, K)
        tag = '✓ 全对' if bad == 0 else ('%d/%d ✗' % (bad, tot))
        print('%-20s %10.1f %10.2fx %10s %s'
              % ('%dx%dx%d' % (M, N, K), gf, gf / CPU, tag,
                 '核显更快 ✓' if gf > CPU else 'CPU 更快 ✗'))
    except Exception as e:
        print('%-20s %10s %11s %10s %s' % ('%dx%dx%d' % (M, N, K), 'FAIL', '-', '-', str(e)[:40]))
print()
print('目标: 500 GFLOPS（CPU 的 2.5 倍）⇒ 与 CPU 并行可把 22 小时压到 9 小时')
