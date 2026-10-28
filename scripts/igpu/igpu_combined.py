"""Combine register blocking with the swizzle. They attack different things and should stack.

Register blocking halves the LDS reads per FMA (0.5 to 0.25) by giving each thread an 8x8 tile
instead of 4x4. The swizzle removes two 2-way bank conflicts by changing the addressing, worth 2x
on its own. One reduces volume, the other removes interference between concurrent accesses --
there is no reason they would conflict.

Measured separately at 1024x2736x512:

    16x16 / 4x4, no swizzle      185.8 GFLOPS   (the fixed v4 baseline)
    16x16 / 4x4, As17 + swizzle  366.6          (1.97x)
     8x8 / 8x8,  no swizzle      283.9          (1.53x)

If they stack, the combination should be well above 400, which is the threshold 7.39.4 set for
concluding that LDS is not a hard wall. At 366.6 alone we are already past the CPU's 218-240, so
the remaining question is how far past.

A correction to the earlier sweep, recorded because it explains two failures: the layout is valid
only when TS = NTX*TN (columns) and TS = NTY*TM (rows), not merely when NTX*TM == NTY*TN. The
16x8/4x8 and 8x16/8x4 combinations passed the weaker check and produced wrong results because they
covered only half the tile's rows. The 8x8/8x8 case satisfies the correct constraint and computed
everything correctly, so its 1.53x stands.
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
#define TK 16
#define NTX {ntx}
#define NTY {nty}
#define TM {tm}
#define TN {tn}
#define TS (NTX * TN)
#define LANES (NTX * NTY)
#define PAD_A {pad_a}
#define PAD_B {pad_b}
#define SWZ {swz}
#define STAGE_A_ITERS ((TS * TK / 4) / LANES)
#define STAGE_B_ITERS ((TK * TS / 4) / LANES)

__kernel void gemm_both(const int M, const int N, const int K,
                        const __global float* restrict A,
                        const __global float* restrict B,
                        __global float* restrict C) {
    __local float As[TS][TK + PAD_A];
    __local float Bs[TK][TS + PAD_B];

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
#if SWZ
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                int cc = n + q;
                Bs[k][(cc % TN) * (TS / TN) + (cc / TN)] =
                    (q == 0) ? v.x : (q == 1) ? v.y : (q == 2) ? v.z : v.w;
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
#if SWZ
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
stamp('组合：寄存器分块 + Swizzle')
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

print('%-40s %10s %8s %9s' % ('配置', '读/FMA', '本地', 'GFLOPS'))
print('-' * 72)
base = None
CASES = [
    (16, 16, 4, 4, 4, 4, 0, '4x4 基线（无 swizzle）'),
    (16, 16, 4, 4, 1, 4, 1, '4x4 + As17 + swizzle'),
    (8, 8, 8, 8, 4, 4, 0, '8x8 仅寄存器分块'),
    (8, 8, 8, 8, 1, 4, 1, '★ 8x8 + As17 + swizzle'),
    (8, 8, 8, 8, 1, 1, 1, '★ 8x8 + As17 + swizzle + B65'),
    (16, 16, 4, 4, 1, 1, 1, '4x4 + As17 + swizzle + B65'),
    (8, 8, 8, 8, 2, 4, 1, '8x8 + As18 + swizzle'),
]
for ntx, nty, tm, tn, pad_a, pad_b, swz, note in CASES:
    ts = ntx * tn
    if ts != nty * tm:
        print('%-40s  布局不合法（TS 行列不一致），跳过' % note)
        continue
    if (ts * 16 // 4) % (ntx * nty) or (16 * ts // 4) % (ntx * nty):
        print('%-40s  装填迭代数不整除，跳过' % note)
        continue
    local_kb = (ts * (16 + pad_a) + 16 * (ts + pad_b)) * 4 / 1024
    if local_kb > 32:
        print('%-40s %8.1fKB 超本地内存' % (note, local_kb))
        continue
    src = SRC.replace('{ntx}', str(ntx)).replace('{nty}', str(nty))
    src = src.replace('{tm}', str(tm)).replace('{tn}', str(tn))
    src = src.replace('{pad_a}', str(pad_a)).replace('{pad_b}', str(pad_b))
    src = src.replace('{swz}', str(swz))
    try:
        kern = cl.Kernel(cl.Program(ctx, src).build(), 'gemm_both')
    except Exception as e:
        print('%-40s 编译失败 %s' % (note, str(e)[:30]))
        continue
    gsz = ((N + ts - 1) // ts * ntx, (M + ts - 1) // ts * nty)
    try:
        kern(queue, gsz, (ntx, nty), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
        queue.finish()
        cl.enqueue_copy(queue, out, bc).wait()
        bad = int((np.abs(out - ref) > 1e-2 * np.abs(ref).max()).sum())
        if bad:
            print('%-40s %8s %7.1fKB  %d 错' % (note, '-', local_kb, bad))
            continue
        tss = []
        for _ in range(5):
            t0 = time.perf_counter()
            kern(queue, gsz, (ntx, nty), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
            queue.finish()
            tss.append(time.perf_counter() - t0)
        gf = 2.0 * M * N * K / statistics.median(tss) / 1e9
        if base is None:
            base = gf
        print('%-40s %10.3f %7.1fKB %9.1f  %.2fx ✓' %
              (note, (tm + tn) / float(tm * tn), local_kb, gf, gf / base))
    except Exception as e:
        print('%-40s 运行失败 %s' % (note, str(e)[:30]))

print()
print('判据（7.39.4）: ≥ 400 GFLOPS ⇒ LDS 不是硬墙，7.28 要改')
print('对照: CPU 同形状 218~240 GFLOPS · 单 swizzle 366.6 · 单分块 283.9')
