"""Register blocking: halve the LDS reads per FMA, now that the baseline is trustworthy.

The gap worth attacking. The kernel's inner loop reads

    a[i] = As[ty*TM + i][k]      k fixed, varies with ty
    b[j] = Bs[k][tx*TN + j]      varies with tx

so all threads sharing a ty read the same As address, and all sharing a tx read the same Bs
address. That is a broadcast pattern, and the standalone LDS microbenchmark measured broadcast at
271 GB/s -- the best of the three patterns it tested, against 66 for strided. So the shortfall
against the 544 GB/s needed to sustain the pure-compute rate is about 2x, not the 8x I reported
earlier when I assumed the strided pattern applied.

A 2x shortfall is exactly what register blocking addresses. At TM=TN=4 each k step reads TM+TN=8
floats to do 16 FMAs, i.e. 0.5 reads per FMA. At 8x8 it reads 16 floats to do 64 FMAs, i.e. 0.25.
Half the LDS traffic, same total compute.

The cost is registers: 64 accumulators rather than 16. GCN has 256 VGPRs per thread, so 64 plus
operands should fit, but occupancy will fall and the earlier occupancy sweep said occupancy is not
the lever -- which is consistent, because this changes the read *volume*, not the latency hiding.

Parameterised so TS = NTX*TM = NTY*TN holds automatically and the staging loop iterations are
derived rather than hard-coded -- the bug just fixed in 7.43 came from hard-coding those.

Decision rule from 7.39.4, fixed in advance: the single-kernel GEMM on our real shapes must reach
400 GFLOPS for LDS to be ruled not-a-hard-wall. Below that, 7.28's conclusion stands.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

import statistics
import sys
import time

import numpy as np

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


import pyopencl as cl  # noqa: E402
from scripts.lib.measure_guard import assert_idle, stamp

SRC = r'''
#define TK 16
#define NTX {ntx}
#define NTY {nty}
#define TM {tm}
#define TN {tn}
#define TS (NTX * TM)
#define LANES (NTX * NTY)
#define STAGE_A_ITERS ((TS * TK / 4) / LANES)
#define STAGE_B_ITERS ((TK * TS / 4) / LANES)

__kernel void gemm_blk(const int M, const int N, const int K,
                       const __global float* restrict A,
                       const __global float* restrict B,
                       __global float* restrict C) {
    __local float As[TS][TK + 4];
    __local float Bs[TK][TS + 4];

    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int bx = get_group_id(0) * TS;
    const int by = get_group_id(1) * TS;
    const int lane = ty * NTX + tx;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += TK) {
        #pragma unroll
        for (int p = 0; p < STAGE_A_ITERS; p++) {
            int f4 = p * LANES + lane;
            int r = (f4 * 4) / TK;
            int c = (f4 * 4) % TK;
            int gr = by + r, gc = k0 + c;
            float4 v = (float4)0.f;
            if (gr < M && gc + 3 < K) v = *(const __global float4*)(A + gr * K + gc);
            As[r][c + 0] = v.x; As[r][c + 1] = v.y;
            As[r][c + 2] = v.z; As[r][c + 3] = v.w;
        }
        #pragma unroll
        for (int p = 0; p < STAGE_B_ITERS; p++) {
            int f4 = p * LANES + lane;
            int k = (f4 * 4) / TS;
            int n = (f4 * 4) % TS;
            int gk = k0 + k, gn = bx + n;
            float4 v = (float4)0.f;
            if (gk < K && gn + 3 < N) v = *(const __global float4*)(B + gk * N + gn);
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

# create_some_context kept failing with OUT_OF_HOST_MEMORY on queue creation even though the
# machine had 5.9 GB free and the same call works in igpu_gemm_v4.py. Building the context from
# the platform explicitly avoids whatever that path was doing.
_plat = cl.get_platforms()[0]
_devs = _plat.get_devices()
ctx = cl.Context(devices=[_devs[0]])
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

assert_idle()
stamp('寄存器分块扫描（LDS 读取量减半）')
print()
print('设备 %s | %d CU | 本地 %d KB' % (dev.name, dev.max_compute_units,
                                        dev.local_mem_size // 1024))
print()

M, N, K = 1024, 2736, 512
rng = np.random.default_rng(3)
A = np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float32))
B = np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float32))
ref = A @ B
ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
out = np.empty((M, N), dtype=np.float32)

print('%-22s %10s %8s %12s %10s %8s' %
      ('线程×每线程', 'TS', 'LDS/组', '读/FMA', 'GFLOPS', '正确'))
print('-' * 76)
base = None
for ntx, nty, tm, tn in ((16, 16, 4, 4), (8, 8, 8, 8), (16, 8, 4, 8), (8, 16, 8, 4), (8, 8, 8, 4)):
    ts = ntx * tm
    if ts != nty * tn:
        continue
    if (ts * 16 // 4) % (ntx * nty) or (16 * ts // 4) % (ntx * nty):
        continue
    local_kb = (ts * 20 + 16 * (ts + 4)) * 4 / 1024
    src = SRC.replace('{ntx}', str(ntx)).replace('{nty}', str(nty))
    src = src.replace('{tm}', str(tm)).replace('{tn}', str(tn))
    try:
        kern = cl.Kernel(cl.Program(ctx, src).build(), 'gemm_blk')
    except Exception as e:
        print('%-22s 编译失败' % ('%dx%d / %dx%d' % (ntx, nty, tm, tn)))
        continue
    gsz = ((N + ts - 1) // ts * ntx, (M + ts - 1) // ts * nty)
    try:
        kern(queue, gsz, (ntx, nty), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
        queue.finish()
        cl.enqueue_copy(queue, out, bc).wait()
        bad = int((np.abs(out - ref) > 1e-2 * np.abs(ref).max()).sum())
        if bad:
            print('%-22s %8d %7.1fKB %12s %10s %8d 错' %
                  ('%dx%d / %dx%d' % (ntx, nty, tm, tn), ts, local_kb, '-', '-', bad))
            continue
        tss = []
        for _ in range(3):
            t0 = time.perf_counter()
            kern(queue, gsz, (ntx, nty), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
            queue.finish()
            tss.append(time.perf_counter() - t0)
        gf = 2.0 * M * N * K / sorted(tss)[1] / 1e9
        if base is None:
            base = gf
        per_fma = (tm + tn) / float(tm * tn)
        print('%-22s %8d %7.1fKB %12.3f %10.1f %8s   %.2fx' %
              ('%dx%d / %dx%d' % (ntx, nty, tm, tn), ts, local_kb, per_fma, gf, '✓全对', gf / base))
    except Exception as e:
        print('%-22s 运行失败 %s' % ('%dx%d / %dx%d' % (ntx, nty, tm, tn), str(e)[:30]))

print()
print('判据（7.39.4 事先固定）: 单内核 ≥ 400 GFLOPS ⇒ LDS 不是硬墙，7.28 的结论要改')
print('对照: 修复后的 v4 基线 170.5 GFLOPS（全对）· CPU 同形状 218~240')
