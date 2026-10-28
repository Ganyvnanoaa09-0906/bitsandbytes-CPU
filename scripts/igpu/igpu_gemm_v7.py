"""igpu_gemm_v7.py -- interleave B's columns in local memory to kill the 8-way conflict.

The decomposition says where the time goes, for the model's real shape:

    full       12.58 ms   staging + LDS reads + FMA
    no_lds      1.76 ms   staging + FMA            -> reads cost 86%
    no_stage    6.66 ms   LDS reads + FMA          -> staging costs 47%
    no_fma      7.34 ms   staging + LDS reads

Theoretical LDS traffic is about 1.7 ms; the measured read cost is near 10.8 ms, six
times more, so the access pattern really is conflicting.

Which one, precisely. With 8x8 threads, a warp spans four `ty` values and eight `tx`:

  a[i] = As[k][ty*8 + i]   address depends on ty, not tx, so the eight threads sharing
                           a ty read one address (broadcast) and four distinct ty give
                           four banks -> 4-way conflict.
  b[j] = Bs[k][tx*8 + j]   address = k*68 + tx*8 + j, so tx*8 mod 32 is one of
                           {0, 8, 16, 24} -> only four of 32 banks are touched, and
                           eight threads collide -> 8-way conflict.

An earlier attempt concluded A was broadcast and therefore conflict-free, which is true,
and then generalised it to the whole kernel, which was wrong: B is the worse of the two
and padding cannot help because the stride is 8 regardless of the row pitch.

Fix: store B's tile with its columns interleaved, so thread tx reads column
tx + j*TX. Consecutive tx then land on consecutive banks. The staging write uses the
same permutation, so the tile is self-consistent.

    slot(c) = (c % TX) * TN + (c / TX)     for c in 0..TS-1

A is left alone this time: it is only 4-way and fixing both at once would leave the
result ambiguous.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

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
#define TS 64
#define TK 8
#define TX 8
#define TY 8
#define TM 8
#define TN 8

// column c of the B tile lives at slot (c % TX) * TN + (c / TX), so that the eight
// threads of a warp read eight consecutive banks instead of four
__kernel void gemm_v7(const int M, const int N, const int K,
                      const __global float* restrict A,
                      const __global float* restrict B,
                      __global float* restrict C) {
    __local float As[TK][TS + 4];
    __local float Bs[TK][TS + 4];

    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * TX + tx;

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
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        // same permutation on the way in
        #pragma unroll
        for (int p = 0; p < TK * TS / 64; p++) {
            int flat = p * 64 + lane;
            int r = flat / TS, c = flat % TS;
            int gr = k0 + r, gc = bx + c;
            int slot = (c % TX) * TN + (c / TX);
            Bs[r][slot] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[k][ty * TM + i];
            // consecutive tx -> consecutive addresses -> consecutive banks
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][tx + j * TX];
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
            int gc = bx + tx + j * TX;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
    }
}
'''

TS, TK, TX, TY, TM, TN = 64, 8, 8, 8, 8, 8

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
prg = cl.Program(ctx, SRC).build()
kern = cl.Kernel(prg, 'gemm_v7')
print('设备 %s | %d CU' % (dev.name, dev.max_compute_units))
print('v7: B 的列交错存放（slot = (c%%8)*8 + c/8），使 warp 读连续 bank')
print()
CPU, V4 = 198.5, 299.9


def run(M, N, K, reps=3):
    rng = np.random.default_rng(5)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ref = A @ B
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
    for _ in range(reps):
        once()
    dt = (time.perf_counter() - t0) / reps
    gf = 2.0 * M * N * K / dt / 1e9
    del ba, bb, bc
    return gf, bad


print('%-20s %10s %11s %10s  %s' % ('形状', 'GFLOPS', '相对CPU', '正确性', '对 v4'))
print('-' * 74)
for (M, N, K) in [(1024, 2736, 512), (1024, 512, 512), (1024, 1024, 512),
                  (1024, 512, 2736), (2048, 2048, 2048), (256, 256, 256)]:
    try:
        gf, bad = run(M, N, K)
        tag = '✓ 全对' if bad == 0 else '%d 错' % bad
        rel = '%.2fx' % (gf / V4) if (M, N, K) == (1024, 2736, 512) else ''
        print('%-20s %10.1f %10.2fx %10s  %s' % ('%dx%dx%d' % (M, N, K), gf, gf / CPU, tag, rel))
    except Exception as e:
        print('%-20s %10s %11s %10s  %s' % ('%dx%dx%d' % (M, N, K), 'FAIL', '-', '-', str(e)[:26]))
