"""Which engine for which operation. A probe over the training step's actual matmuls.

The question is not "is the iGPU faster" but "for this operation, on this shape, which one wins".
The duty-cycle sweep settled the interference cost at a roughly fixed 10-19%, so the assignment can
be decided per operation and the shared cost paid once.

The operations, taken from the model's real shapes at batch 2 and T=1024, so 2048 rows:

  qkv projection     2048 x 512  x 1536
  attention scores   1024 x  64  x 1024   per head, 8 heads
  attention output   1024 x 1024 x   64   per head
  mlp w12            2048 x 512  x 2736
  mlp w3             2048 x 2048 x  512
  optimizer (NS)      512 x 512  x  512   Newton-Schulz inner matmuls

The iGPU side uses the swizzled kernel, which was verified element-by-element against numpy in 7.45.
The CPU side is torch, which is what training actually uses.

Decision rule, fixed in advance: an operation belongs to whichever engine is faster on its own
shape, and operations the iGPU wins go there only if the shape is also large enough that kernel
launch and transfer overhead is amortised. Shapes the iGPU cannot handle at all -- the attention
scores, where the multiplication is batched over heads and the operands are small -- stay on the CPU
and are printed as such rather than being forced through a path that does not fit.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


SRC = r'''
#define TS 64
#define TK 16
#define TM 4
#define TN 4
__kernel void gemm_fast(const int M, const int N, const int K,
                        const __global float* restrict A,
                        const __global float* restrict B,
                        __global float* restrict C) {
    __local float As[TS][TK + 1];
    __local float Bs[TK][TS + 4];
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
        { int f4 = lane; int r = (f4*4)/TK, c = (f4*4)%TK;
          int gr = by + r, gc = k0 + c; float4 v = (float4)0.f;
          if (gr < M && gc + 3 < K) v = *(const __global float4*)(A + gr*K + gc);
          As[r][c+0]=v.x; As[r][c+1]=v.y; As[r][c+2]=v.z; As[r][c+3]=v.w; }
        { int f4 = lane; int k = (f4*4)/TS, n = (f4*4)%TS;
          int gk = k0 + k, gn = bx + n; float4 v = (float4)0.f;
          if (gk < K && gn + 3 < N) v = *(const __global float4*)(B + gk*N + gn);
          #pragma unroll
          for (int q = 0; q < 4; q++) { int cc = n + q;
              Bs[k][(cc % TN) * (TS / TN) + (cc / TN)] =
                  (q==0)?v.x:((q==1)?v.y:((q==2)?v.z:v.w)); } }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[ty*TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][j*(TS/TN) + tx];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] += a[i]*b[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i = 0; i < TM; i++) { int gr = by + ty*TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) { int gc = bx + tx*TN + j;
            if (gr < M && gc < N) C[gr*N + gc] = acc[i][j]; } }
}
'''

import pyopencl as cl  # noqa: E402
from measure_guard import assert_idle, stamp  # noqa: E402

_plat = cl.get_platforms()[0]
ctx = cl.Context(devices=[_plat.get_devices()[0]])
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

OPS = [
    ('qkv 投影',        2048, 1536,  512),
    ('attention 分数',  1024, 1024,   64),
    ('attention 输出',  1024,   64, 1024),
    ('mlp w12',         2048, 2736,  512),
    ('mlp w3',          2048,  512, 2048),
    ('优化器 NS(小)',     512,  512,  512),
    ('优化器 NS(大)',    2048,  512,  512),
]

assert_idle()
stamp('任务分派探针：哪个操作归哪个引擎')
torch.set_num_threads(6)
print()
print('  设备 %s | %d CU' % (ctx.devices[0].name, ctx.devices[0].max_compute_units))
print()
print('  %-16s %18s %12s %12s %10s %s' %
      ('操作', '形状 MxNxK', 'CPU ms', '核显 ms', '核显/CPU', '归属'))
print('  ' + '-' * 88)
rows = []
for name, M, N, K in OPS:
    A = np.ascontiguousarray(np.random.default_rng(5).standard_normal((M, K)).astype(np.float32))
    B = np.ascontiguousarray(np.random.default_rng(6).standard_normal((K, N)).astype(np.float32))
    fl = 2.0 * M * N * K

    ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
    kern = cl.Kernel(cl.Program(ctx, SRC).build(), 'gemm_fast')
    gsz = ((N + 63) // 64 * 16, (M + 63) // 64 * 16)

    def gpu():
        kern(queue, gsz, (16, 16), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
        queue.finish()

    ta = torch.from_numpy(A)
    tb = torch.from_numpy(B)

    def cpu():
        torch.mm(ta, tb)

    for _ in range(2):
        gpu(); cpu()
    tg = statistics.median([(lambda t0: (gpu(), time.perf_counter() - t0)[1])(time.perf_counter())
                            for _ in range(3)])
    tc = statistics.median([(lambda t0: (cpu(), time.perf_counter() - t0)[1])(time.perf_counter())
                            for _ in range(3)])
    gf_c, gf_g = fl / tc / 1e9, fl / tg / 1e9
    own = '核显' if tg < tc else 'CPU'
    rows.append((name, tc, tg, own))
    print('  %-16s %18s %12.3f %12.3f %9.2fx  %s' %
          (name, '%dx%dx%d' % (M, N, K), tc * 1e3, tg * 1e3, tc / tg, own))

print()
win_gpu = [r for r in rows if r[3] == '核显']
t_cpu_all = sum(r[1] for r in rows)
t_split = sum(min(r[1], r[2]) for r in rows)
print('  全部在 CPU:      %.1f ms（这些操作之和）' % (t_cpu_all * 1e3))
print('  按归属分派:      %.1f ms  ⇒ 【%.2fx】' % (t_split * 1e3, t_cpu_all / t_split))
print('  核显赢下 %d / %d 个操作' % (len(win_gpu), len(rows)))
print()
print('  ⇒ 而这些操作只占一步的一部分;其余是 elementwise 与 attention 的 SDPA')
