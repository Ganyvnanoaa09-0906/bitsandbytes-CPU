"""igpu_gemm_v5.py -- transposed A in local memory, to remove the bank conflicts.

The evidence that local-memory access is the limit, from three independent directions:

  1. Occupancy sweep: shrinking the local tile from 16 KB to 4 KB raised throughput
     from 224 to 314 GFLOPS on the same shape.
  2. Pure compute reaches 1089 GFLOPS (95% of this device's 1150 peak) while the real
     fp32 GEMM reaches 300.
  3. An fp16 kernel with packed arithmetic reaches 165 GFLOPS against 4665 in the
     equivalent pure-compute test.

So there is 3.6x of compute headroom in fp32 and 28x in fp16, all of it lost to data
supply. The access pattern is the reason. A is read as As[row][k] with k fixed, which
walks down a column: consecutive threads differ in `row`, so their addresses differ by
(TK+1) = 9 floats and land in only about four of the 32 banks -- an eight-way conflict.
B is read as Bs[k][col] with col varying, which is contiguous and conflict-free.

Fix: store A transposed, As[k][row]. Then the compute-time read varies `row`
consecutively, addresses differ by one and the banks are used in sequence. The staging
write becomes the transposed side, which needs its own padding: writing As[gc][gr] from
a row-major global read makes consecutive threads' addresses differ by the row stride,
so a padding of 4 (68 floats, bank stride 4) still conflicts, while a padding of 1 (65,
bank stride 1) does not.

Everything else is held identical to the working v3/v4 kernel so the change is isolated,
and correctness is checked before the timing is believed.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

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
// C[m,n] = A[m,k]*B[k,n], row-major, A stored transposed in local memory.
#define TS 64
#define TK 8
#define TX 8
#define TY 8
#define TM 8
#define TN 8

__kernel void gemm_v5(const int M, const int N, const int K,
                      const __global float* restrict A,
                      const __global float* restrict B,
                      __global float* restrict C) {
    // As[k][row]: transposed, and +1 padding so the transposed *write* is
    // conflict-free too. +4 would give a bank stride of 4 and reintroduce conflicts.
    __local float As[TK][TS + 1];
    // Bs[k][col]: natural orientation and contiguity-preserving padding.
    __local float Bs[TK][TS + 4];

    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int NT = TX * TY, lane = ty * TX + tx;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += TK) {
        // read A row-major, write transposed: A[by+r][k0+c] -> As[c][r]
        #pragma unroll
        for (int p = 0; p < TS * TK / 64; p++) {
            int flat = p * 64 + lane;
            int r = flat / TK, c = flat % TK;
            int gr = by + r, gc = k0 + c;
            As[c][r] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        #pragma unroll
        for (int p = 0; p < TK * TS / 64; p++) {
            int flat = p * 64 + lane;
            int r = flat / TS, c = flat % TS;
            int gr = k0 + r, gc = bx + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            // now consecutive in `row`, so consecutive in memory and in bank
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[k][ty * TM + i];
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
prg = cl.Program(ctx, SRC).build()
kern = cl.Kernel(prg, 'gemm_v5')
print('设备 %s | %d CU | 本地 %d KB' % (dev.name, dev.max_compute_units, dev.local_mem_size // 1024))
print('v5: A 转置存 As[k][row]（+1 padding）+ Bs[k][col]（+4），TS=64 TK=8 8x8 线程')
print()
# TS must also exist on the Python side: the launch geometry is computed from it,
# and the #define inside the kernel source is not visible here.
TS = 64
CPU = 198.5
V4_BEST = 299.9      # the previous best on the model's real shape


def run(M, N, K, reps=3):
    rng = np.random.default_rng(5)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ref = A @ B
    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    gsz = ((N + TS - 1) // TS * 8, (M + TS - 1) // TS * 8)

    def once():
        kern(queue, gsz, (8, 8), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
        queue.finish()

    once()
    out = np.zeros((M, N), dtype=np.float32)
    cl.enqueue_copy(queue, out, bc).wait()
    bad = int((np.abs(out - ref) > 1e-2 * (np.abs(ref).max() or 1)).sum())
    t0 = time.perf_counter()
    for _ in range(reps):
        once()
    dt = (time.perf_counter() - t0) / reps
    gf = 2.0 * M * N * K / dt / 1e9
    del ba, bb, bc
    return gf, bad, M * N


print('%-20s %10s %11s %10s  %s' % ('形状', 'GFLOPS', '相对CPU', '正确性', '对 v4 最佳'))
print('-' * 76)
for (M, N, K) in [(1024, 2736, 512), (1024, 512, 512), (1024, 1024, 512),
                  (1024, 512, 2736), (2048, 2048, 2048), (256, 256, 256)]:
    try:
        gf, bad, tot = run(M, N, K)
        tag = '✓ 全对' if bad == 0 else '%d 错' % bad
        extra = '%.2fx' % (gf / V4_BEST) if (M, N, K) == (1024, 2736, 512) else ''
        print('%-20s %10.1f %10.2fx %10s  %s' % ('%dx%dx%d' % (M, N, K), gf, gf / CPU, tag, extra))
    except Exception as e:
        print('%-20s %10s %11s %10s  %s' % ('%dx%dx%d' % (M, N, K), 'FAIL', '-', '-', str(e)[:26]))
print()
print('对照: v4 最佳 299.9 GFLOPS（模型真实形状），CLBlast 493.8（仅立方），纯计算 1089')
