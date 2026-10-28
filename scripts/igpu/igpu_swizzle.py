"""Swizzle the LDS layout to kill two 2-way bank conflicts, borrowed from the GPU playbook.

The kernel's inner loop reads

    a[i] = As[ty*TM + i][k]        row stride is TK + PAD_A
    b[j] = Bs[k][tx*TN + j]        within-row, group stride TN

Working out the bank mapping for a 64-lane wavefront on GCN (32 banks, 4 bytes each):

  * As: the row stride is 20 floats with PAD_A=4, and gcd(20, 32) = 4, so the 16 distinct
    addresses touched collapse onto only 8 banks -- a 2-way conflict. PAD_A=1 gives stride 17,
    coprime with 32, so all 16 land on distinct banks.
  * Bs: the addresses are tx*TN + j for tx = 0..15, i.e. stride 4, so again only 8 banks are used
    and each is hit twice. Row padding cannot fix this because the conflict is inside the row.

The fix for Bs is a real swizzle rather than padding: store the row in TN-sized blocks, transposed
within each block, at position (c % TN) * (TS / TN) + (c / TN). Reading column tx*TN + j then
becomes reading j * (TS/TN) + tx -- 16 consecutive addresses for fixed j, which is 16 distinct
banks. The store side pays a slightly more complex index; the read side, which happens TK times
per K-block instead of once, gets the benefit.

Both are parameterised so each can be turned on independently. Baseline is PAD_A=4, PAD_B=4, no
swizzle -- which is what the kernel does today.

Measured on the real shape 1024x2736x512, correctness checked against numpy each time. A kernel
that computes the wrong answer faster is not an improvement, which is the lesson from 7.43.

Decision rule from 7.39.4, fixed in advance: reaching 400 GFLOPS would mean LDS is not a hard
wall and 7.28's conclusion needs revising. The current trustworthy baseline is 170.5 GFLOPS.
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

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


import pyopencl as cl
from scripts.lib.measure_guard import assert_idle, stamp

SRC = r'''
#define TS 64
#define TK 16
#define TM 4
#define TN 4
#define PAD_A {pad_a}
#define PAD_B {pad_b}
#define SWIZZLE_B {swz}

__kernel void gemm_sw(const int M, const int N, const int K,
                      const __global float* restrict A,
                      const __global float* restrict B,
                      __global float* restrict C) {
    __local float As[TS][TK + PAD_A];
    __local float Bs[TK][TS + PAD_B];

    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int bx = get_group_id(0) * TS;
    const int by = get_group_id(1) * TS;
    const int lane = ty * 16 + tx;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += TK) {
        // one iteration per thread: 1024 floats = 256 float4, 256 threads
        {
            int f4 = lane;
            int r = (f4 * 4) / TK;
            int c = (f4 * 4) % TK;
            int gr = by + r, gc = k0 + c;
            float4 v = (float4)0.f;
            if (gr < M && gc + 3 < K) v = *(const __global float4*)(A + gr * K + gc);
            As[r][c + 0] = v.x; As[r][c + 1] = v.y;
            As[r][c + 2] = v.z; As[r][c + 3] = v.w;
        }
        {
            int f4 = lane;
            int k = (f4 * 4) / TS;
            int n = (f4 * 4) % TS;
            int gk = k0 + k, gn = bx + n;
            float4 v = (float4)0.f;
            if (gk < K && gn + 3 < N) v = *(const __global float4*)(B + gk * N + gn);
#if SWIZZLE_B
            // store in TN-sized blocks, transposed inside each block
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                int cc = n + q;
                int dst = (cc % TN) * (TS / TN) + (cc / TN);
                Bs[k][dst] = (q == 0) ? v.x : (q == 1) ? v.y : (q == 2) ? v.z : v.w;
            }
#else
            Bs[k][n + 0] = v.x; Bs[k][n + 1] = v.y;
            Bs[k][n + 2] = v.z; Bs[k][n + 3] = v.w;
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[ty * TM + i][k];
#if SWIZZLE_B
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][j * (TS / TN) + tx];
#else
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][tx * TN + j];
#endif
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

_plat = cl.get_platforms()[0]
ctx = cl.Context(devices=[_plat.get_devices()[0]])
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

assert_idle()
stamp('Swizzle / padding 扫描（LDS bank conflict）')
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

print('%-30s %9s %9s' % ('配置', '本地/组', 'GFLOPS'))
print('-' * 52)
base = None
combined = None
for pad_a, pad_b, swz, note in ((4, 4, 0, '现状基线'),
                                (1, 4, 0, 'As 行距 17（互质）'),
                                (2, 4, 0, 'As 行距 18'),
                                (8, 4, 0, 'As 行距 24'),
                                (4, 4, 1, 'Bs 块内转置 swizzle'),
                                (1, 4, 1, 'As 17 + Bs swizzle'),
                                (1, 1, 1, 'As 17 + Bs swizzle + B 行距 65'),
                                (1, 1, 0, 'As 17 + B 行距 65（无 swizzle）')):
    src = SRC.replace('{pad_a}', str(pad_a)).replace('{pad_b}', str(pad_b))
    src = src.replace('{swz}', str(swz))
    local_kb = (64 * (16 + pad_a) + 16 * (64 + pad_b)) * 4 / 1024
    if local_kb > 32:
        print('%-30s %7.1fKB  超本地内存，跳过' % (note, local_kb))
        continue
    try:
        kern = cl.Kernel(cl.Program(ctx, src).build(), 'gemm_sw')
    except Exception as e:
        print('%-30s 编译失败 %s' % (note, str(e)[:40]))
        continue
    gsz = ((N + 63) // 64 * 16, (M + 63) // 64 * 16)
    try:
        kern(queue, gsz, (16, 16), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
        queue.finish()
        cl.enqueue_copy(queue, out, bc).wait()
        bad = int((np.abs(out - ref) > 1e-2 * np.abs(ref).max()).sum())
        if bad:
            print('%-30s %7.1fKB  %d 个元素错' % (note, local_kb, bad))
            continue
        tss = []
        for _ in range(5):
            t0 = time.perf_counter()
            kern(queue, gsz, (16, 16), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
            queue.finish()
            tss.append(time.perf_counter() - t0)
        gf = 2.0 * M * N * K / statistics.median(tss) / 1e9
        if base is None:
            base = gf
        print('%-30s %7.1fKB %9.1f  %.3fx  ✓全对' % (note, local_kb, gf, gf / base))
    except Exception as e:
        print('%-30s 运行失败 %s' % (note, str(e)[:40]))

print()
print('判据（7.39.4 事先固定）: ≥ 400 GFLOPS ⇒ LDS 不是硬墙，7.28 要改')
print('对照: 修复后的 v4 = 170.5 GFLOPS · CPU 同形状 218~240')
