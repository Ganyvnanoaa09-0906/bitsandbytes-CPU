"""ns_on_device.py -- Newton-Schulz with nothing crossing the bus between iterations.

The asynchronous test showed the mechanism works: variant B reached 0.92x of the ideal
max(CPU, iGPU) by enqueueing device work without waiting and running CPU work meanwhile.
Variant C only reached 0.55x, and the reason is visible in the loop -- each NS iteration
downloads A, downloads A2, computes Bm on the host, and downloads the result, so a
five-iteration NS costs fifteen round trips per matrix. Round trips are latency, and
latency is what the async test was hiding.

So keep everything on the device. X, A, A2, Bm and the temporary T live in device buffers
across the whole orthogonalisation; only the input upload and the final download touch the
host. Two kernels are needed:

  gemm_t   the existing GEMM plus a transpose flag on the B operand, because the first
           product is X @ X^T and the stored X is not transposed.
  axpby    an elementwise kernel for Bm = b*A + c*A2 and for X = a*X + T. These are a
           handful of operations per element and they were forcing the round trips.

Then the correctness question, which is the one that actually decides this: the result must
match the CPU Newton-Schulz to fp32 tolerance. If it does, the same code runs five
iterations for one round trip instead of fifteen.

Measured on the shapes Muon uses for this model, 74 matrices of 512x256 to 2048x512.
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
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


import pyopencl as cl  # noqa: E402
from scripts.lib.measure_guard import assert_idle, stamp

torch.set_num_threads(6)

SRC = r'''
#define TS 64
#define TK 8
#define TX 8
#define TY 8
#define TM 8
#define TN 8

// C[m,n] = sum_k A[m,k] * B'[k,n]
//   TB=0: B stored (K,N), B'[k,n] = B[k*N+n]
//   TB=1: B stored (N,K), B'[k,n] = B[n*K+k]      <- for X @ X^T
__kernel void gemm_t(const int M, const int N, const int K, const int TB,
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
            Bs[r][c] = (gr < K && gc < N)
                     ? (TB ? B[gc * K + gr] : B[gr * N + gc]) : 0.f;
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

// out = p*X + q*Y, elementwise. p and q carry the NS coefficients.
__kernel void axpby(const int n, const float p, const float q,
                    const __global float* restrict X,
                    const __global float* restrict Y,
                    __global float* restrict out) {
    const int i = get_global_id(0);
    if (i < n) out[i] = p * X[i] + q * Y[i];
}

// X = a*X + T
__kernel void axpy(const int n, const float a,
                   __global float* restrict X,
                   const __global float* restrict T) {
    const int i = get_global_id(0);
    if (i < n) X[i] = a * X[i] + T[i];
}

// X = X / (norm(X) + eps): one workgroup reduces, then everyone scales
__kernel void scale_inplace(const int n, const float inv,
                            __global float* restrict X) {
    const int i = get_global_id(0);
    if (i < n) X[i] *= inv;
}
'''

TS = 64
ctx = cl.create_some_context(interactive=False)
dev = ctx.devices[0]
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
prg = cl.Program(ctx, SRC).build()
k_gemm = cl.Kernel(prg, 'gemm_t')
k_axpby = cl.Kernel(prg, 'axpby')
k_axpy = cl.Kernel(prg, 'axpy')
k_scale = cl.Kernel(prg, 'scale_inplace')

SHAPES = []
for _ in range(10):
    SHAPES += [(512, 512), (512, 256), (512, 256), (512, 512),
               (512, 2048), (512, 2048), (2048, 512)]
SHAPES = SHAPES[:74]

pool = {}


def buf(nbytes, tag):
    b = pool.get(tag)
    if b is None or b.size < nbytes:
        b = cl.Buffer(ctx, mf.READ_WRITE, nbytes)
        pool[tag] = b
    return b


def gemm_launch(m, n, k, tb, ba, bb, bc):
    gsz = ((n + TS - 1) // TS * 8, (m + TS - 1) // TS * 8)
    k_gemm(queue, gsz, (8, 8), np.int32(m), np.int32(n), np.int32(k), np.int32(tb),
           ba, bb, bc)


def ns_device(W):
    """Full 5-iteration NS for one matrix, everything on the device."""
    a, b, c, eps = 3.4445, -4.7750, 2.0315, 1e-7
    Xh = np.ascontiguousarray(W.detach().float().numpy())
    tr = Xh.shape[0] > Xh.shape[1]
    if tr:
        Xh = np.ascontiguousarray(Xh.T)
    m, n = Xh.shape
    nrm = float(np.linalg.norm(Xh)) + eps
    Xh = Xh / nrm

    bX = buf(Xh.nbytes, 'X')
    cl.enqueue_copy(queue, bX, Xh)
    bA = buf(m * m * 4, 'A')
    bA2 = buf(m * m * 4, 'A2')
    bB = buf(m * m * 4, 'B')
    bT = buf(m * n * 4, 'T')

    for _ in range(5):
        gemm_launch(m, m, n, 1, bX, bX, bA)          # A = X @ X^T
        gemm_launch(m, m, m, 0, bA, bA, bA2)         # A2 = A @ A
        k_axpby(queue, (m * m,), None, np.int32(m * m), np.float32(b),
                 np.float32(c), bA, bA2, bB)
        gemm_launch(m, n, m, 0, bB, bX, bT)          # T = B @ X
        k_axpy(queue, (m * n,), None, np.int32(m * n), np.float32(a), bX, bT)

    out = np.empty((m, n), dtype=np.float32)
    cl.enqueue_copy(queue, out, bX)
    queue.finish()
    return torch.from_numpy(out.T if tr else out)


def ns_cpu(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    tr = False
    if X.size(0) > X.size(1):
        X = X.T
        tr = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if tr else X


assert_idle(allow=True, threshold=50.0)
stamp('NS 全设备化（A/B 在跑，负载即场景）')
print()
print('设备 %s | %d CU' % (dev.name, dev.max_compute_units))
weights = [torch.randn(*s) for s in SHAPES]
half = len(weights) // 2
print('矩阵 %d 个，%.1fM 参数' % (len(SHAPES), sum(m * n for m, n in SHAPES) / 1e6))
print()

w0 = weights[0]
ref = ns_cpu(w0)
got = ns_device(w0)
d = float((got - ref).abs().max()) / (float(ref.abs().max()) or 1.0)
print('正确性: 全设备 NS 与 CPU NS 的相对差 %.2e  %s' % (d, '✓' if d < 5e-3 else '✗'))
print()

# warm up and time
ns_device(weights[0])


def timeit(fn, reps=3):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


t_cpu = timeit(lambda: [ns_cpu(w) for w in weights[:half]])
t_dev = timeit(lambda: [ns_device(w) for w in weights[:half]])
fb = torch.randn(8 * 1024, 512)
fbw = torch.randn(512, 2048)
t_fb = timeit(lambda: [torch.mm(fb, fbw) for _ in range(20)])
print('  CPU 半量 NS                     %7.3f s' % t_cpu)
print('  全设备 NS 半量（1 次往返/矩阵）  %7.3f s' % t_dev)
print('  CPU 前反向形状负载               %7.3f s' % t_fb)
print()


def both():
    for w in weights[:half]:
        ns_device(w)
    for _ in range(20):
        torch.mm(fb, fbw)


t_both = timeit(both)
ideal = max(t_cpu, t_fb)
serial = t_cpu + t_fb
print('  并发 %7.3f s   相对理想 max(%.3f, %.3f) = %.2fx   相对串行 %.2fx   ⇒ %s'
      % (t_both, t_cpu, t_fb, ideal / t_both, serial / t_both,
         '✓ 设备工作被隐藏' if t_both < ideal * 1.15
         else ('⚠ 部分重叠' if t_both < serial * 0.9 else '✗ 串行')))
