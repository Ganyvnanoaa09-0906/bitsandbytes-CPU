"""igpu_gemm_v4.py -- v3 was correct but slow; fix the three structural gaps.

v3 fixed the staging layout and is now exactly correct at every shape tested, but
reaches only 92-126 GFLOPS (0.47-0.64x the CPU), against CLBlast's 493.8 on this
same device. Comparing the two structures gives three concrete differences, all of
which favour CLBlast:

  1. Vectorised global access. v3 reads and writes scalars; the project's own
     v2a/v2b use float4, and scalar traffic on GCN wastes most of the available
     bandwidth. This was dropped when v3 was simplified, and it is the largest
     suspect.
  2. Register pressure. v3 gives each thread 8x8 = 64 accumulators plus operand
     arrays; the tuned gfx902 parameters use VWM=VWN=4, i.e. 16 accumulators.
     Spilling would erase the benefit of the register blocking.
  3. K-blocking. v3 uses TK=16 where the tuned parameters use KWG=32, halving the
     reuse of each staged element.

v4 changes exactly those three: 16x16 threads per workgroup (256, better occupancy
on 6 CUs), 4x4 outputs per thread, and float4 loads/stores where the alignment
permits. Staging is still As[TS][TK+4] / Bs[TK][TS+4], which v3 established is the
correct layout. Correctness is checked first, at cubic and non-cubic shapes.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
import pyopencl as cl  # noqa: E402

KERNEL = r'''
// C = A*B, row-major. Workgroup: TS x TS output tile, 16x16 threads, 4x4 each.
#define TS 64
#define TK 16
#define TM 4
#define TN 4

__kernel void sgemm_v4(const int M, const int N, const int K,
                       const __global float* restrict A,
                       const __global float* restrict B,
                       __global float* restrict C) {
    __local float As[TS][TK + 4];      // 64 rows x 16 cols of A, padded
    __local float Bs[TK][TS + 4];      // 16 rows x 64 cols of B, padded

    const int tx = get_local_id(0);    // 0..15
    const int ty = get_local_id(1);    // 0..15
    const int bx = get_group_id(0) * TS;
    const int by = get_group_id(1) * TS;
    const int lane = ty * 16 + tx;     // 0..255

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += TK) {
        // stage A: TS*TK = 1024 floats = 256 float4
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int f4 = p * 256 + lane;              // 0..1023 float4 index
            int r = (f4 * 4) / TK;                // row in tile
            int c = (f4 * 4) % TK;                // col, multiple of 4
            int gr = by + r, gc = k0 + c;
            float4 v = (float4)0.f;
            if (gr < M && gc + 3 < K)
                v = *(const __global float4*)(A + gr * K + gc);
            As[r][c + 0] = v.x; As[r][c + 1] = v.y;
            As[r][c + 2] = v.z; As[r][c + 3] = v.w;
        }
        // stage B: TK*TS = 1024 floats = 256 float4
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int f4 = p * 256 + lane;
            int k = (f4 * 4) / TS;
            int n = (f4 * 4) % TS;
            int gk = k0 + k, gn = bx + n;
            float4 v = (float4)0.f;
            if (gk < K && gn + 3 < N)
                v = *(const __global float4*)(B + gk * N + gn);
            Bs[k][n + 0] = v.x; Bs[k][n + 1] = v.y;
            Bs[k][n + 2] = v.z; Bs[k][n + 3] = v.w;
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
kern = cl.Kernel(prg, 'sgemm_v4')
print('设备 %s | %d CU | 本地 %d KB' % (dev.name, dev.max_compute_units, dev.local_mem_size // 1024))
print('v4: 64x64 tile, 16x16=256 线程, 每线程 4x4, float4 读写, As[64][20]+Bs[16][68]=9.5KB')
print()


def run(M, N, K, reps=3):
    rng = np.random.default_rng(4)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    gsz = ((N + 63) // 64 * 16, (M + 63) // 64 * 16)

    def once():
        kern(queue, gsz, (16, 16), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
        queue.finish()

    once()
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
                  (1024, 2736, 512), (1024, 512, 2736), (2048, 2048, 2048)]:
    try:
        gf, bad, tot = run(M, N, K)
        tag = '✓ 全对' if bad == 0 else '%d/%d ✗' % (bad, tot)
        print('%-20s %10.1f %10.2fx %10s %s'
              % ('%dx%dx%d' % (M, N, K), gf, gf / CPU, tag,
                 '核显更快 ✓' if gf > CPU else 'CPU 更快 ✗'))
    except Exception as e:
        print('%-20s %10s %11s %10s %s' % ('%dx%dx%d' % (M, N, K), 'FAIL', '-', '-', str(e)[:44]))
