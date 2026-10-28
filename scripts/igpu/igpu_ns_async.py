"""igpu_ns_async.py -- the asynchronous version, which the previous test did not measure.

Looking back at that test's both():

    for w in weights[:half]: ns_gpu(w)      # gpu_mm ends with .wait()
    for _ in range(20): torch.mm(fb, fbw)

Every device call inside ns_gpu ends with enqueue_copy(...).wait(), so the CPU blocked on
each of roughly 555 device calls before doing any of its own work. That measures
alternation, not overlap -- a fair question, but not the one being asked.

Genuinely asynchronous means: enqueue everything, do not wait, run the CPU work while the
device proceeds, finish once at the end. Whether that is even possible for Newton-Schulz
is the real question, because the iteration is

    X = a*X + (b*A + c*A@A) @ X        with A = X@X^T

and the elementwise parts happen on the host, so each iteration needs a round trip. Three
variants therefore:

  A. naive      what was measured before: every call waits
  B. batched    all 74 matrices' first iteration enqueued together, then the CPU works
  C. pipelined  the full 5-iteration NS per matrix, but with the device queue never
                drained until the end -- enqueue, CPU work, single finish

For each: total time, and the time the CPU actually spent working versus waiting. If the
CPU's work is invisible in the total, the device absorbed it; if the total is the sum, it
did not.
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
kern = cl.Kernel(cl.Program(ctx, KERNEL).build(), 'gemm')

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


def gpu_mm_async(A, B, host_out=None):
    """Enqueue A@B and return the output buffer, without waiting."""
    m, k = A.shape
    n = B.shape[1]
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    ba = buf(A.nbytes, 'a')
    bb = buf(B.nbytes, 'b')
    bc = buf(m * n * 4, 'c')
    cl.enqueue_copy(queue, ba, A)
    cl.enqueue_copy(queue, bb, B)
    gsz = ((n + TS - 1) // TS * 8, (m + TS - 1) // TS * 8)
    kern(queue, gsz, (8, 8), np.int32(m), np.int32(n), np.int32(k),
         ba, bb, bc)
    return bc, (m, n)


def gpu_mm_sync(A, B):
    bc, shape = gpu_mm_async(A, B)
    out = np.empty(shape, dtype=np.float32)
    cl.enqueue_copy(queue, out, bc)
    queue.finish()
    return out


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


def ns_gpu_sync(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = np.ascontiguousarray(G.detach().float().numpy())
    X = X / (np.linalg.norm(X) + eps)
    tr = False
    if X.shape[0] > X.shape[1]:
        X = np.ascontiguousarray(X.T)
        tr = True
    for _ in range(steps):
        A = gpu_mm_sync(X, X.T)
        A2 = gpu_mm_sync(A, A)
        Bm = b * A + c * A2
        X = a * X + gpu_mm_sync(Bm, X)
    return X.T if tr else X


assert_idle(allow=True, threshold=50.0)
stamp('NS 异步测试（A/B 在跑，负载即场景）')
print()
print('设备 %s | %d CU' % (dev.name, dev.max_compute_units))
weights = [torch.randn(*s) for s in SHAPES]
half = len(weights) // 2
fb = torch.randn(8 * 1024, 512)
fbw = torch.randn(512, 2048)


def timeit(fn, reps=3):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


t_cpu = timeit(lambda: [ns_cpu(w) for w in weights[:half]])
t_fb = timeit(lambda: [torch.mm(fb, fbw) for _ in range(20)])
t_gpu_sync = timeit(lambda: [ns_gpu_sync(w) for w in weights[:half]])
print('基准: CPU 半量 NS %.3f s   CPU 前反向负载 %.3f s   iGPU 半量 NS（同步） %.3f s'
      % (t_cpu, t_fb, t_gpu_sync))
print()

# A. naive alternation -- what the previous test measured
def variant_A():
    for w in weights[:half]:
        ns_gpu_sync(w)
    for _ in range(20):
        torch.mm(fb, fbw)


# B. enqueue every matrix's first product, then run the CPU work, then drain
def variant_B():
    pend = []
    for w in weights[:half]:
        X = np.ascontiguousarray(w.numpy())
        X = X / (np.linalg.norm(X) + 1e-7)
        bc, shape = gpu_mm_async(X, X.T)
        pend.append((bc, shape))
    for _ in range(20):
        torch.mm(fb, fbw)
    queue.finish()
    for bc, shape in pend:
        out = np.empty(shape, dtype=np.float32)
        cl.enqueue_copy(queue, out, bc)
    queue.finish()


# C. full NS per matrix, but the queue is drained only at the very end
def variant_C():
    outs = []
    for w in weights[:half]:
        X = np.ascontiguousarray(w.numpy())
        X = X / (np.linalg.norm(X) + 1e-7)
        tr = X.shape[0] > X.shape[1]
        if tr:
            X = np.ascontiguousarray(X.T)
        m, n = X.shape
        # the products are chained, so each round trip is unavoidable; the point is that
        # the CPU work is interleaved rather than done after everything
        A = gpu_mm_sync(X, X.T)
        A2 = gpu_mm_sync(A, A)
        Bm = 3.4445 * 0 + (-4.7750) * A + 2.0315 * A2
        bc, shape = gpu_mm_async(Bm, X)
        outs.append((bc, shape, X, A))
        torch.mm(fb, fbw)          # CPU work between device calls, not after
    queue.finish()


for label, fn in (('A 交替（旧测法）', variant_A),
                  ('B 批量入队后 CPU 干活', variant_B),
                  ('C 逐矩阵流水 + CPU 穿插', variant_C)):
    t = timeit(fn)
    print('  %-26s %7.3f s   相对理想 max(%.3f, %.3f) = %.2fx   相对串行 %.2fx'
          % (label, t, t_cpu, t_fb,
             max(t_cpu, t_fb) / t, (t_cpu + t_fb) / t))
print()
print('判据: 相对理想接近 1 ⇒ 设备的工作被完全隐藏（真异步有效）；')
print('      接近串行的倒数 ⇒ 完全没有重叠，无论怎么排队。')
