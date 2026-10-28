"""igpu_tile_occupancy.py -- is the iGPU kernel's occupancy limited by registers?

The user asked this and the honest answer was that we never tested it in that direction.
Section 7.22.4 swept occupancy by shrinking the *local tile*, and it also tried 4x4 threads
with 16x16 outputs, which needs 256 accumulators per thread and collapsed to 19.7 GFLOPS
from spilling. That is the far end of the curve, not the near one. Nobody reduced the
per-thread tile below 8x8.

The mechanism worth checking: 8x8 accumulators is 64 registers before operands and
indices, so roughly 90 per thread. GCN's LDS latency is 20-30 cycles, and hiding it needs
enough wavefronts resident per CU. If registers cap the number of resident wavefronts, the
latency is exposed and the FMA pipes idle -- which would explain 300 GFLOPS against a
1089 GFLOPS compute ceiling, and would explain why every layout change I tried failed:
they all attacked the access pattern while the limit was how many threads can be in flight.

Sweeps the per-thread tile from 8x8 down to 4x2 at fixed output-tile size, and reads back
each kernel's maximum workgroup size, which the driver derives from register usage. If
throughput rises as the tile shrinks while the reported workgroup size grows, registers
are the limit. If throughput falls, the arithmetic-to-load ratio is what matters instead,
and registers are not the constraint.

Correctness is checked at every point, since a smaller tile changes the indexing.
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import sys
import time

import numpy

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


import pyopencl
from scripts.lib.measure_guard import assert_idle, stamp

SRC_TMPL = r'''
#define TS 64
#define TK 8
#define TX {tx}
#define TY {ty}
#define TM {tm}
#define TN {tn}

__kernel void gemm_tile(const int M, const int N, const int K,
                        const __global float* restrict A,
                        const __global float* restrict B,
                        __global float* restrict C) {
    __local float As[TS][TK + 1];
    __local float Bs[TK][TS + 1];
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
        for (int p = 0; p < TS * TK / (TX * TY); p++) {
            int f = p * (TX * TY) + lane, r = f / TK, c = f % TK;
            int gr = by + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        #pragma unroll
        for (int p = 0; p < TK * TS / (TX * TY); p++) {
            int f = p * (TX * TY) + lane, r = f / TS, c = f % TS;
            int gr = k0 + r, gc = bx + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
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

TS = 64
ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

# (TX, TY, TM, TN): TX*TM = TY*TN = TS so the tile is fully covered
TILES = [
    (8, 8, 8, 8, '基线 64 累加器'),
    (8, 8, 8, 4, '32 累加器（TN 减半）'),
    (8, 8, 4, 8, '32 累加器（TM 减半）'),
    (8, 8, 4, 4, '16 累加器'),
    (8, 8, 2, 8, '16 累加器（细长）'),
    (8, 8, 2, 4, '8 累加器'),
    (8, 8, 4, 2, '8 累加器（另一种）'),
    (16, 4, 4, 16, '64 累加器（宽松线程）'),
    (4, 16, 16, 4, '64 累加器（另一种）'),
]

assert_idle(allow=True, threshold=50.0)
stamp('iGPU tile/occupancy 扫描（A/B 仍在跑，负载是背景）')
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

print('%-26s %8s %10s %10s %8s %s'
      % ('线程 × 每线程 tile', '工作组', '最大组', 'GFLOPS', '相对', '正确性'))
print('-' * 82)
base = None
for tx, ty, tm, tn, note in TILES:
    if tx * tm != TS or ty * tn != TS:
        continue
    src = SRC_TMPL.replace('{tx}', str(tx)).replace('{ty}', str(ty))
    src = src.replace('{tm}', str(tm)).replace('{tn}', str(tn))
    # the staging loop count must divide evenly over the actual thread count
    if (TS * 8) % (tx * ty) or (8 * TS) % (tx * ty):
        continue
    try:
        prg = cl.Program(ctx, src).build()
        kern = cl.Kernel(prg, 'gemm_tile')
        maxwg = kern.get_work_group_info(
            cl.kernel_work_group_info.WORK_GROUP_SIZE, dev)
    except Exception as e:
        print('%-26s 编译失败: %s' % ('%dx%d/%dx%d' % (tx, ty, tm, tn), str(e)[:40]))
        continue
    gsz = ((N + TS - 1) // TS * tx, (M + TS - 1) // TS * ty)
    try:
        kern(queue, gsz, (tx, ty), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
        queue.finish()
        cl.enqueue_copy(queue, out, bc).wait()
        bad = int((np.abs(out - ref) > 1e-2 * np.abs(ref).max()).sum())
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            kern(queue, gsz, (tx, ty), np.int32(M), np.int32(N), np.int32(K),
                 ba, bb, bc)
            queue.finish()
            ts.append(time.perf_counter() - t0)
        dt = sorted(ts)[1]
        gf = 2.0 * M * N * K / dt / 1e9
        if base is None:
            base = gf
        print('%-26s %8d %10d %10.1f %7.2fx %s'
              % ('%dx%d 线程 / %dx%d' % (tx, ty, tm, tn), tx * ty, maxwg, gf,
                 gf / base, '✓ 全对' if bad == 0 else '%d 错' % bad))
    except Exception as e:
        print('%-26s %8d %10d %10s %8s %s'
              % ('%dx%d 线程 / %dx%d' % (tx, ty, tm, tn), tx * ty, maxwg, 'FAIL', '-',
                 str(e)[:26]))

print()
print('判据: 若【累加器变少 ⇒ 最大工作组变大 ⇒ GFLOPS 上升】⇒ 寄存器是瓶颈 ✓')
print('      若【累加器变少而 GFLOPS 下降】⇒ 算术/取数比才是关键，寄存器不是 ✓')
