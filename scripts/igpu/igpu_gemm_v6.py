"""igpu_gemm_v6.py -- transposed A plus vectorised local reads, which the measurements say is the fix.

Isolated by measurement rather than inference, after five attempts that moved nothing:

    fma_only (registers)      3.159e11 FMA/s
    lds_only (local reads)    3.14e10 reads/s
    both (the real shape)     1.344e11 FMA/s

The real inner loop needs 0.25 reads per FMA, i.e. 3.36e10 reads/s to sustain the FMA
rate it reaches. Local memory supplies 3.14e10. Both lands at 1.344e11 FMA/s, which is
essentially the read rate times four -- the arithmetic is not the limit, the reads are,
and they are short by about 7%.

So the fix is to read less per FMA. Two ways: a larger per-thread tile, which raises
TM*TN/(TM+TN) but runs out of registers (16x16 needs 256 accumulators), or vectorised
reads, which divide the read count by four. Vectorised reads need four consecutive
elements per load, which for the A operand means four consecutive `row` values -- that
is, A stored transposed, which is exactly what the previous attempt did. It was slower
because transposing the *write* costs more than it saved, and it did not read
vectorised. Both halves are needed together.

Padding note: TS+1 keeps the transposed write conflict-free but leaves rows 65 floats
apart, so no float4 load is aligned. TS+4 restores alignment and costs a conflict on the
staging write, which happens once per K block rather than once per FMA. That trade is
worth taking here, and the measurement will say whether it was.
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
#define PAD 4

__kernel void gemm_v6(const int M, const int N, const int K,
                      const __global float* restrict A,
                      const __global float* restrict B,
                      __global float* restrict C) {
    // transposed so that four consecutive rows are contiguous and can be loaded as
    // one float4; PAD=4 keeps rows 16-byte aligned
    __local float As[TK][TS + PAD];
    __local float Bs[TK][TS + PAD];

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
            // vectorised: two float4 loads instead of eight scalar loads
            const __local float4* ap = (const __local float4*)&As[k][ty * TM];
            #pragma unroll
            for (int q = 0; q < TM / 4; q++) {
                float4 v = ap[q];
                a[q * 4 + 0] = v.x; a[q * 4 + 1] = v.y;
                a[q * 4 + 2] = v.z; a[q * 4 + 3] = v.w;
            }
            // B is naturally contiguous along `col`, so this is already vectorisable
            const __local float4* bp = (const __local float4*)&Bs[k][tx * TN];
            #pragma unroll
            for (int q = 0; q < TN / 4; q++) {
                float4 v = bp[q];
                b[q * 4 + 0] = v.x; b[q * 4 + 1] = v.y;
                b[q * 4 + 2] = v.z; b[q * 4 + 3] = v.w;
            }
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
kern = cl.Kernel(prg, 'gemm_v6')
print('设备 %s | %d CU' % (dev.name, dev.max_compute_units))
print('v6: A 转置 As[k][row]（PAD=4 对齐）+ float4 读 A 与 B，TS=64 TK=8 8x8')
print()
# TS must exist here too: the launch geometry uses it and the kernel #define is
# not visible from Python. Forgetting this cost a run on v5 as well.
TS = 64
CPU = 198.5
V4 = 299.9


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
