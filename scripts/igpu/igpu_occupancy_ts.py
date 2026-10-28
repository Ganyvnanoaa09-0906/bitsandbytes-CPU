"""igpu_occupancy_ts.py -- does shrinking the workgroup's footprint raise throughput?

The chain of measurements so far:

    pure compute, register operands        1089 GFLOPS   (95% of peak -- the CUs are fine)
    inner loop, operands from local memory  ~530 GFLOPS   (half the compute rate)
    full GEMM, with staging from global      300 GFLOPS   (the number to explain)

So the arithmetic units are not the problem and, inside the inner-loop test, neither is
global memory -- that path touches none. What sits between is local memory, and the
standalone LDS microbenchmark reported only 66-68 GB/s for the patterns in use against the
544 GB/s that sustaining the compute rate would need. That points at a bandwidth limit, but
a synthetic microbenchmark can understate the real rate, and the TK sweep pointed somewhere
else entirely: throughput *fell* as TK grew, from 167.7 at TK=4 to 20.7 at TK=32. Larger TK
means a larger local footprint per workgroup, hence fewer workgroups resident per CU.

That is the signature of a latency problem rather than a bandwidth one. GCN's LDS latency is
tens of cycles; if too few wavefronts are resident to cover it, the FMA pipes stall no matter
how much bandwidth the memory has.

This tests it directly by shrinking the output tile, which shrinks both the local-memory
footprint and the registers per thread, so more workgroups fit per CU:

    TS=64   the current configuration, 4.3 KB local, 64 accumulators per thread
    TS=32   1.1 KB local, 16 accumulators
    TS=16   0.3 KB local, 4 accumulators

The kernel is copied verbatim from the version that measured 300 GFLOPS, with only TS and
the derived per-thread tile size parameterised -- an earlier sweep used a retyped template,
which is why its baseline came out at 158 rather than 300 and its numbers could not be
compared with anything.

Prediction to test, stated first: if latency hiding is the limit, GFLOPS should rise as TS
falls. If it falls too, then the arithmetic-to-load ratio is what matters and occupancy is
not the lever.
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

# verbatim from the kernel that measured 300 GFLOPS, with TS/TM/TN parameterised
SRC_TMPL = r'''
#define TS {ts}
#define TK 8
#define TX 8
#define TY 8
#define TM {tm}
#define TN {tn}

__kernel void gemm_ts(const int M, const int N, const int K,
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

ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

# allow=True is justified here and the reason is specific: this measures the iGPU, whose
# throughput depends on its own local memory and on shared-memory bandwidth, not on how
# busy the CPU cores are. The background load is a VS Installer downloader at ~24% and the
# agent harness itself at ~50%, neither of which touches the GPU. A CPU-side measurement
# would not get this exemption.
assert_idle(allow=True, threshold=50.0)
stamp('TS / occupancy 扫描（GPU 测量；CPU 有 VS 下载器与 harness 的背景负载）')
print()
print('设备 %s | %d CU | 本地 %d KB | 最大工作组 %d'
      % (dev.name, dev.max_compute_units, dev.local_mem_size // 1024,
         dev.max_work_group_size))
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

print('%-6s %8s %10s %12s %12s %10s %s'
      % ('TS', '每线程', '本地/组', '组/CU(理)', 'GFLOPS', '相对', '正确性'))
print('-' * 84)
base = None
for ts in (64, 32, 16):
    tm = ts // 8
    tn = ts // 8
    if (ts * 8) % 64 or (8 * ts) % 64:
        continue
    local_kb = (ts * 9 + 8 * (ts + 1)) * 4 / 1024
    src = SRC_TMPL.replace('{ts}', str(ts)).replace('{tm}', str(tm)).replace('{tn}', str(tn))
    try:
        kern = cl.Kernel(cl.Program(ctx, src).build(), 'gemm_ts')
    except Exception as e:
        print('%-6d 编译失败: %s' % (ts, str(e)[:50]))
        continue
    gsz = ((N + ts - 1) // ts * 8, (M + ts - 1) // ts * 8)
    # workgroups per CU from the local-memory limit alone
    groups_per_cu = int((dev.local_mem_size / 1024) // max(1, local_kb))
    try:
        kern(queue, gsz, (8, 8), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
        queue.finish()
        cl.enqueue_copy(queue, out, bc).wait()
        bad = int((np.abs(out - ref) > 1e-2 * np.abs(ref).max()).sum())
        tss = []
        for _ in range(3):
            t0 = time.perf_counter()
            kern(queue, gsz, (8, 8), np.int32(M), np.int32(N), np.int32(K),
                 ba, bb, bc)
            queue.finish()
            tss.append(time.perf_counter() - t0)
        dt = sorted(tss)[1]
        gf = 2.0 * M * N * K / dt / 1e9
        if base is None:
            base = gf
        print('%-6d %6dx%-3d %8.1f KB %12d %12.1f %9.2fx %s'
              % (ts, tm, tn, local_kb, groups_per_cu, gf, gf / base,
                 '✓ 全对' if bad == 0 else '%d 错' % bad))
    except Exception as e:
        print('%-6d %6dx%-3d %8.1f KB %12d %12s %9s %s'
              % (ts, tm, tn, local_kb, groups_per_cu, 'FAIL', '-', str(e)[:24]))

print()
print('判据: 若 TS 变小 ⇒ 组/CU 变多 ⇒ GFLOPS 上升 ⇒ 是【延迟没被掩盖】(occupancy 问题)')
print('      若 GFLOPS 反而下降 ⇒ 是【算术/取数比】(每次 LDS 读服务的 FMA 变少)')
print('基准对照: 纯计算 1089 · 内层循环约 530 · 此前 v4 全内核 300')
