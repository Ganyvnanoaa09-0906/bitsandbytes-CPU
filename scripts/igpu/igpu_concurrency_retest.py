"""Re-test concurrency with the fast kernel. The previous five all used the slow one.

Every earlier concurrency result -- NS offload at 0.23x, full offload at 0.39x, true async 5% worse
than serial, two independent tasks at 0.81x, splitting one matrix at 0.68-0.91x -- was measured
against a kernel that could not beat the CPU. That made the arithmetic trivial: you were handing
work to a slower device and paying memory contention on top.

Now the kernel does 361.5 GFLOPS at 1024x2736x512 against the CPU's 218-240. So the question is
different: hand work to a device that is genuinely faster, and see whether the shared DDR4 bus
costs more than the speed gains.

Method: measure each engine alone, then measure both running at once, and compare total
throughput. The metric that matters is FLOPs per second summed across both engines while they are
running, against the CPU's rate alone. If concurrency does not beat the CPU alone, offloading
cannot help no matter how fast the GPU kernel is, because the training would be waiting on both.

Shapes are the real ones: the model's matmuls at batch 2, T=1024, d=512, ffn=2048, plus the fused
w12 width of 2736.

Decision rule, fixed in advance per 7.39.4: concurrent throughput must reach 1.2x the CPU alone for
the iGPU path to reopen. Below that it stays closed and 7.28's conclusion stands on new evidence.

One caution learned from the earlier rounds: warm up both engines before timing, and re-measure the
"alone" numbers in the same session, since a comparison against a number taken at a different time
is exactly how the 10-13% error in 7.41 happened.
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
import threading
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


import pyopencl as cl  # noqa: E402
from scripts.lib.measure_guard import assert_idle, stamp

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
        {
            int f4 = lane;
            int r = (f4 * 4) / TK, c = (f4 * 4) % TK;
            int gr = by + r, gc = k0 + c;
            float4 v = (float4)0.f;
            if (gr < M && gc + 3 < K) v = *(const __global float4*)(A + gr * K + gc);
            As[r][c+0]=v.x; As[r][c+1]=v.y; As[r][c+2]=v.z; As[r][c+3]=v.w;
        }
        {
            int f4 = lane;
            int k = (f4 * 4) / TS, n = (f4 * 4) % TS;
            int gk = k0 + k, gn = bx + n;
            float4 v = (float4)0.f;
            if (gk < K && gn + 3 < N) v = *(const __global float4*)(B + gk * N + gn);
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                int cc = n + q;
                Bs[k][(cc % TN) * (TS / TN) + (cc / TN)] =
                    (q==0)?v.x:((q==1)?v.y:((q==2)?v.z:v.w));
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][j * (TS / TN) + tx];
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
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

assert_idle()
stamp('并发重测（快内核）')
torch.set_num_threads(6)
print()
print('设备 %s | %d CU' % (ctx.devices[0].name, ctx.devices[0].max_compute_units))
print()

SHAPES = [(2048, 512, 512), (2048, 2736, 512), (2048, 512, 2736), (2048, 2048, 512)]
rng = np.random.default_rng(7)

gpu_bufs = []
for (M, N, K) in SHAPES:
    A = np.ascontiguousarray(rng.standard_normal((M, K)).astype(np.float32))
    B = np.ascontiguousarray(rng.standard_normal((K, N)).astype(np.float32))
    gpu_bufs.append((M, N, K,
                     cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A),
                     cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B),
                     cl.Buffer(ctx, mf.WRITE_ONLY, M * N * 4)))

kern = cl.Kernel(cl.Program(ctx, SRC).build(), 'gemm_fast')

# torch tensors for the CPU side, same shapes
cpu_bufs = []
for (M, N, K) in SHAPES:
    cpu_bufs.append((torch.randn(M, K), torch.randn(K, N)))

FLOP = sum(2.0 * M * N * K for (M, N, K) in SHAPES)


def run_gpu(rounds=1):
    for _ in range(rounds):
        for (M, N, K, ba, bb, bc) in gpu_bufs:
            kern(queue, ((N + 63) // 64 * 16, (M + 63) // 64 * 16), (16, 16),
                 np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
    queue.finish()


def run_cpu(rounds=1):
    for _ in range(rounds):
        for (a, b) in cpu_bufs:
            torch.mm(a, b)


# correctness of the fast kernel, since a wrong kernel measuring fast is worthless.
# The first version of this check ran before the kernel had ever been launched, so it read an
# untouched buffer and reported a false failure. Run it first, then compare.
run_gpu(1)
(_, N0, K0, ba0, bb0, bc0) = gpu_bufs[0]
out = np.empty((SHAPES[0][0], N0), dtype=np.float32)
cl.enqueue_copy(queue, out, bc0).wait()
A0 = np.empty((SHAPES[0][0], K0), dtype=np.float32)
cl.enqueue_copy(queue, A0, ba0).wait()
B0 = np.empty((K0, N0), dtype=np.float32)
cl.enqueue_copy(queue, B0, bb0).wait()
ref0 = A0 @ B0
bad0 = int((np.abs(out - ref0) > 1e-2 * np.abs(ref0).max()).sum())
print('  内核正确性（对 numpy 校验）:', '✓ 全对' if bad0 == 0 else ('✗ %d 个元素错' % bad0))
print()

print('--- 预热 ---')
run_cpu(2); run_gpu(2)
torch_t = statistics.median([(lambda t0: (run_cpu(2), time.perf_counter() - t0)[1])(time.perf_counter())
                             for _ in range(3)])
cpu_gf = 2.0 * FLOP / torch_t / 1e9
print('  CPU  单独: %.4f s/组 · 【%.1f GFLOPS】' % (torch_t, cpu_gf))

gpu_t = statistics.median([(lambda t0: (run_gpu(2), time.perf_counter() - t0)[1])(time.perf_counter())
                           for _ in range(3)])
gpu_gf = 2.0 * FLOP / gpu_t / 1e9
print('  核显 单独: %.4f s/组 · 【%.1f GFLOPS】' % (gpu_t, gpu_gf))
print()

print('--- 并发（真并行：CPU 在主线程，核显在工作线程）---')
res = []
for _ in range(3):
    th = threading.Thread(target=run_gpu, args=(2,))
    t0 = time.perf_counter()
    th.start()
    run_cpu(2)
    th.join()
    res.append(time.perf_counter() - t0)
conc_t = statistics.median(res)
conc_gf = 2.0 * FLOP / conc_t / 1e9
print('  并发总吞吐: %.4f s/组 · 【%.1f GFLOPS】' % (conc_t, conc_gf))
print()

print('=' * 62)
print('  CPU 单独        %8.1f GFLOPS' % cpu_gf)
print('  核显 单独       %8.1f GFLOPS   (%.2fx of CPU)' % (gpu_gf, gpu_gf / cpu_gf))
print('  并发总和        %8.1f GFLOPS   【%.2fx of CPU 单独】' % (conc_gf, conc_gf / cpu_gf))
print('=' * 62)
print()
print('  理想叠加上限: %.1f GFLOPS (%.2fx) —— 若并发接近它，说明内存不是瓶颈'
      % (cpu_gf + gpu_gf, (cpu_gf + gpu_gf) / cpu_gf))
print()
print('  判据（7.39.4 事先固定）: 并发 ≥ CPU 单独 1.2 倍 ⇒ 核显重新打开；否则关闭')
print('  ⇒ %s' % ('【重新打开】✓' if conc_gf / cpu_gf >= 1.2 else '【仍然关闭】✗'))
