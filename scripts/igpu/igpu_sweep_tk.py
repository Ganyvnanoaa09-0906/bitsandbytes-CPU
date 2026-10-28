"""igpu_sweep_tk.py -- is the limit LDS bandwidth, or the staging volume?

The LDS microbenchmark gave 66-68 GB/s for the patterns the kernel uses, against the 544
GB/s that sustaining the pure-compute rate would need. That looks conclusive, but running
the arithmetic on the kernel's own inner loop says something else may be going on.

One K-block of the kernel, with TK=8:

    compute     8 k-steps x 64 FMA                =   512 FMA
    staging     As (64x8) + Bs (8x64) = 1024 floats written
                plus the same 1024 read from global

So each FMA is served by two staged floats. Staging traffic scales as (TS*TK + TK*TS) per
TK of compute, i.e. it is proportional to TK while compute per block is also proportional
to TK -- the *ratio* is fixed at 2 staged floats per FMA regardless of TK. What changes
with TK is the number of barriers and the number of times the pipeline drains, not the
volume.

Which means the two hypotheses make different predictions:

  LDS bandwidth     throughput should be roughly flat in TK, since reads per FMA are fixed
  staging overhead  throughput should improve with TK, as barriers and drains amortise
  register pressure should get worse with TK, eventually hurting

TK has never been swept. 7.22.4 swept the output-tile size and the local-memory footprint
by changing TS, and every later variant changed access patterns at fixed TK=8. Running the
sweep is cheap and it discriminates between the explanations rather than assuming one.

Measured at fixed TS=64, TX=TY=8, TM=TN=8, with correctness checked at each point.

Note on conditions: the SGDR A/B is running on the CPU. It does not use the GPU, so the
iGPU timings here are not competing for the device, but the CPU load means the host side of
each launch is slower. Any conclusion must therefore rest on the *shape* of the sweep, not
on absolute GFLOPS.
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

SRC_TMPL = r'''
#define TS 64
#define TK {tk}
#define TX 8
#define TY 8
#define TM 8
#define TN 8

__kernel void gemm_tk(const int M, const int N, const int K,
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
        for (int p = 0; p < TS * TK / 64; p++) {
            int f = p * 64 + lane, r = f / TK, c = f % TK;
            int gr = by + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        #pragma unroll
        for (int p = 0; p < TK * TS / 64; p++) {
            int f = p * 64 + lane, r = f / TS, c = f % TS;
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

assert_idle(allow=True, threshold=50.0)
stamp('TK 扫描（SGDR A/B 在 CPU 上跑；GPU 未被占用，但 launch 的 host 侧会被拖慢）')
print()
print('设备 %s | %d CU | 本地 %d KB' % (dev.name, dev.max_compute_units,
                                        dev.local_mem_size // 1024))

M, N, K = 1024, 2736, 512
rng = np.random.default_rng(3)
A = np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float32))
B = np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float32))
ref = A @ B
ba = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
bb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
bc = cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)
out = np.empty((M, N), dtype=np.float32)

# local memory per workgroup: As is TS*(TK+1) + Bs is TK*(TS+1), in floats
print()
print('%-8s %12s %10s %12s %8s %s'
      % ('TK', '本地内存/组', 'GFLOPS', '每 FMA 装填', '相对', '正确性'))
print('-' * 74)
base = None
for tk in (4, 8, 16, 32, 64):
    if (TS * tk) % 64 or (tk * TS) % 64:
        continue
    local_kb = (TS * (tk + 1) + tk * (TS + 1)) * 4 / 1024
    src = SRC_TMPL.replace('{tk}', str(tk))
    try:
        kern = cl.Kernel(cl.Program(ctx, src).build(), 'gemm_tk')
    except Exception as e:
        print('%-8d 编译失败: %s' % (tk, str(e)[:44]))
        continue
    gsz = ((N + TS - 1) // TS * 8, (M + TS - 1) // TS * 8)
    try:
        kern(queue, gsz, (8, 8), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
        queue.finish()
        cl.enqueue_copy(queue, out, bc).wait()
        bad = int((np.abs(out - ref) > 1e-2 * np.abs(ref).max()).sum())
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            kern(queue, gsz, (8, 8), np.int32(M), np.int32(N), np.int32(K),
                 ba, bb, bc)
            queue.finish()
            ts.append(time.perf_counter() - t0)
        dt = sorted(ts)[1]
        gf = 2.0 * M * N * K / dt / 1e9
        if base is None:
            base = gf
        # staged floats per FMA: (TS*TK + TK*TS) staged per (TK * TM*TN) FMAs per thread
        per_fma = (2.0 * TS * tk) / (tk * 8 * 8)
        print('%-8d %10.1f KB %10.1f %12.2f %7.2fx %s'
              % (tk, local_kb, gf, per_fma, gf / base,
                 '✓ 全对' if bad == 0 else '%d 错' % bad))
    except Exception as e:
        print('%-8d %10.1f KB %10s %12s %8s %s'
              % (tk, local_kb, 'FAIL', '-', '-', str(e)[:24]))
print()
print('判据:')
print('  LDS 带宽是限制 ⇒ 各 TK 的 GFLOPS 应大致持平（每 FMA 的 LDS 读固定）')
print('  装填/barrier 开销是限制 ⇒ 随 TK 增大应明显上升')
print('  寄存器压力是限制 ⇒ 随 TK 增大先升后降')
