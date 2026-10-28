"""breakdown_igpu_overhead.py -- where the ~6 ms per call actually goes.

igpu_mm.py ended at 0.07-0.62x CPU, which is worse than not using the iGPU at all. Its
explanation was "about 6 ms of fixed overhead per call against 0.9 ms of compute for
1024x512x512", and that number was never decomposed -- it was inferred from the ratio of
measured times, not attributed to any particular step. Seven kernel attempts went after
throughput instead, which the decomposition in 7.24 shows was the wrong target.

pyopencl enqueues asynchronously, so wrapping a call in perf_counter measures the CPU
cost of submitting it, not the work. That distinction is what makes attribution possible:

  mode A -- time each step without waiting. Reveals CPU-side cost: numpy views,
            contiguous() copies, buffer lookups, the pyopencl call overhead itself.
  mode B -- finish() after each step. Attributes the device time to the step that
            issued it.

Anything large in A is Python-side and can be removed by restructuring. Anything large in
B is transfer or kernel time and needs a different fix.

Reported per step, for the model's real shape and for a shape where the iGPU should
plausibly win, with the total so the two modes can be compared against the 6 ms claim.
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

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）

import pyopencl as cl  # noqa: E402

_SRC = open(r'D:\work\bitsandbytes-CPU\igpu_mm_kernel.cl', 'w', encoding='utf-8')

KERNEL = r'''
#define TS 64
#define TK 8
#define TX 8
#define TY 8
#define TM 8
#define TN 8
__kernel void gemm(const int M, const int N, const int K,
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
prg = cl.Program(ctx, KERNEL).build()
kern = cl.Kernel(prg, 'gemm')
pool = {}

print('设备 %s | %d CU' % (dev.name, dev.max_compute_units))
print()


def buf(nbytes, tag):
    b = pool.get(tag)
    if b is None or b.size < nbytes:
        b = cl.Buffer(ctx, mf.READ_WRITE, nbytes)
        pool[tag] = b
    return b


def measure(M, N, K, reps=20):
    import torch
    torch.set_num_threads(6)
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    n_gpu = (N // 2) // TS * TS
    A = a.numpy()
    B = b.numpy()
    ba_ = buf(A.nbytes, 'a')
    bb_ = buf(B.nbytes, 'b')
    bc_ = buf(M * n_gpu * 4, 'c')
    gsz = ((n_gpu + TS - 1) // TS * 8, (M + TS - 1) // TS * 8)
    steps = {}

    def tick(name, t0, wait):
        if wait:
            queue.finish()
        steps.setdefault(name, []).append((time.perf_counter() - t0) * 1e3)

    # warm up
    for _ in range(2):
        cl.enqueue_copy(queue, ba_, A)
        cl.enqueue_copy(queue, bb_, B)
        kern(queue, gsz, (8, 8), np.int32(M), np.int32(n_gpu), np.int32(K), ba_, bb_, bc_)
        queue.finish()

    for mode in ('A: 只 enqueue（CPU 侧）', 'B: 每步 finish（归位 GPU）'):
        wait = mode.startswith('B')
        for name in list(steps):
            steps[name] = []
        for _ in range(reps):
            t0 = time.perf_counter()
            Aa = a.numpy(); Bb = b.numpy()
            tick('1 numpy 视图', t0, wait)

            t0 = time.perf_counter()
            Bg = Bb[:, :n_gpu]
            Bg = Bg if Bg.flags['C_CONTIGUOUS'] else np.ascontiguousarray(Bg)
            tick('2 B 切片+连续化', t0, wait)

            t0 = time.perf_counter()
            cl.enqueue_copy(queue, ba_, Aa)
            tick('3 上传 A', t0, wait)

            t0 = time.perf_counter()
            cl.enqueue_copy(queue, bb_, Bg)
            tick('4 上传 B(半)', t0, wait)

            t0 = time.perf_counter()
            kern(queue, gsz, (8, 8), np.int32(M), np.int32(n_gpu), np.int32(K),
                 ba_, bb_, bc_)
            tick('5 enqueue 内核', t0, wait)

            t0 = time.perf_counter()
            cpu_part = a @ b[:, n_gpu:].contiguous()
            tick('6 CPU 半边 GEMM', t0, wait)

            t0 = time.perf_counter()
            out_g = np.empty((M, n_gpu), dtype=np.float32)
            cl.enqueue_copy(queue, out_g, bc_)
            tick('7 下载 C', t0, wait)

            t0 = time.perf_counter()
            out = torch.empty(M, N)
            out[:, :n_gpu] = torch.from_numpy(out_g)
            out[:, n_gpu:] = cpu_part
            tick('8 torch 拼装', t0, wait)

        tot = sum(sum(v) / len(v) for v in steps.values() if v)
        print('  %s  （%dx%dx%d，iGPU 半边 %d 列）' % (mode, M, N, K, n_gpu))
        print('    %-22s %9s' % ('步骤', '毫秒'))
        for name, v in steps.items():
            if v:
                print('    %-22s %9.3f' % (name, statistics.median(v) * 1000 / 1000))
        print('    %-22s %9.3f' % ('合计', tot))
        print()

    # what the same work costs on the CPU alone
    t0 = time.perf_counter()
    for _ in range(5):
        a @ b
    t_cpu = (time.perf_counter() - t0) / 5 * 1e3
    print('    对照: 纯 CPU 同一 GEMM %.3f ms' % t_cpu)
    print()


for shape in [(1024, 512, 512), (1024, 2736, 512)]:
    measure(*shape)
print('读法: A 里大的项是 Python 侧开销，可以直接重构成掉；')
print('      B 里大的项是传输或内核时间，需要换别的手段。')
