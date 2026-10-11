"""ns_async.py -- enqueue the whole orthogonalisation, then let the CPU work.

The user spotted what I missed. ns_on_device.py kept everything on the device, which was
the right idea for reducing round trips, but its ns_device() ends with queue.finish(), and
the concurrency test called it once per matrix before doing any CPU work:

    for w in weights[:half]:
        ns_device(w)          # each call finishes
    for _ in range(20):
        torch.mm(fb, fbw)

That is alternation, not overlap -- the same mistake the earlier synchronous test made. The
earlier async test showed what the fix is worth: batching device work and letting the CPU
run during it reached 0.92x of the ideal max(CPU, iGPU), against 0.23x for alternation.

So: enqueue every matrix's full five-iteration Newton-Schulz without a single finish, run
the CPU work, then finish once and collect. Nothing crosses the bus per iteration, so the
round-trip problem that capped the earlier device-resident attempt does not apply.

One implementation detail has to be handled. The existing code pools device buffers by size
and reuses them, which is fine for one matrix at a time and wrong here -- the second
matrix's first matmul would overwrite the first matrix's still-pending input. Each in-flight
matrix therefore needs its own buffers, allocated round-robin from a ring sized to the
number of matrices in flight.

Correctness is checked against the per-matrix CPU implementation, since a pipelined rewrite
is exactly the kind of change that can be subtly wrong. Then three timings:

    CPU only          the Newton-Schulz work alone
    iGPU only         the same work, fully async, no CPU competing
    both              iGPU enqueued, CPU doing forward/backward-shaped work meanwhile

and the verdict is whether the combined time approaches the larger of the two or their sum.
"""
from __future__ import annotations

import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')

import pyopencl as cl  # noqa: E402
from measure_guard import assert_idle, stamp   # noqa: E402

torch.set_num_threads(6)

SRC = r'''
#define TS 64
#define TK 8
#define TX 8
#define TY 8
#define TM 8
#define TN 8

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

__kernel void axpby(const int n, const float p, const float q,
                    const __global float* restrict X,
                    const __global float* restrict Y,
                    __global float* restrict out) {
    const int i = get_global_id(0);
    if (i < n) out[i] = p * X[i] + q * Y[i];
}

__kernel void axpy(const int n, const float a,
                   __global float* restrict X,
                   const __global float* restrict T) {
    const int i = get_global_id(0);
    if (i < n) X[i] = a * X[i] + T[i];
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

SHAPES = []
for _ in range(10):
    SHAPES += [(512, 512), (512, 256), (512, 256), (512, 512),
               (512, 2048), (512, 2048), (2048, 512)]
SHAPES = SHAPES[:74]

# a ring of buffer sets, one per matrix in flight
RING = 8
ring = []
for _ in range(RING):
    ring.append({'X': None, 'A': None, 'A2': None, 'B': None, 'T': None})
ring_i = [0]


def get_set():
    s = ring[ring_i[0] % RING]
    ring_i[0] += 1
    return s


def need(s, key, nbytes):
    b = s[key]
    if b is None or b.size < nbytes:
        b = cl.Buffer(ctx, mf.READ_WRITE, nbytes)
        s[key] = b
    return b


def gemm_launch(m, n, k, tb, ba, bb, bc):
    gsz = ((n + TS - 1) // TS * 8, (m + TS - 1) // TS * 8)
    k_gemm(queue, gsz, (8, 8), np.int32(m), np.int32(n), np.int32(k), np.int32(tb),
           ba, bb, bc)


def ns_enqueue(W):
    """Enqueue a full 5-iteration NS for one matrix. No finish, no readback."""
    a, b, c, eps = 3.4445, -4.7750, 2.0315, 1e-7
    Xh = np.ascontiguousarray(W.detach().float().numpy())
    tr = Xh.shape[0] > Xh.shape[1]
    if tr:
        Xh = np.ascontiguousarray(Xh.T)
    m, n = Xh.shape
    Xh = Xh / (float(np.linalg.norm(Xh)) + eps)
    s = get_set()
    bX = need(s, 'X', Xh.nbytes)
    bA = need(s, 'A', m * m * 4)
    bA2 = need(s, 'A2', m * m * 4)
    bB = need(s, 'B', m * m * 4)
    bT = need(s, 'T', m * n * 4)
    cl.enqueue_copy(queue, bX, Xh)
    for _ in range(5):
        gemm_launch(m, m, n, 1, bX, bX, bA)
        gemm_launch(m, m, m, 0, bA, bA, bA2)
        k_axpby(queue, (m * m,), None, np.int32(m * m), np.float32(b), np.float32(c),
                bA, bA2, bB)
        gemm_launch(m, n, m, 0, bB, bX, bT)
        k_axpy(queue, (m * n,), None, np.int32(m * n), np.float32(a), bX, bT)
    return bX, (m, n), tr


def collect(handle):
    bX, (m, n), tr = handle
    out = np.empty((m, n), dtype=np.float32)
    cl.enqueue_copy(queue, out, bX)
    return out, tr


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
stamp('NS 全异步（SGDR A/B 在 CPU 上跑 —— 这正是要重叠的对象）')
print()
print('设备 %s | %d CU | 缓冲环 %d 组' % (dev.name, dev.max_compute_units, RING))
weights = [torch.randn(*s) for s in SHAPES]
half = len(weights) // 2
print('矩阵 %d 个，取半 %d 个做 NS' % (len(SHAPES), half))
print()


def timeit(fn, reps=3):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


# ---- correctness on one matrix
w0 = weights[0]
h = ns_enqueue(w0)
queue.finish()
got, tr = collect(h)
queue.finish()
got_t = torch.from_numpy(got.T if tr else got)
ref = ns_cpu(w0)
d = float((got_t - ref).abs().max()) / (float(ref.abs().max()) or 1.0)
print('正确性: 异步版与 CPU 版相对差 %.2e  %s' % (d, '✓' if d < 5e-3 else '✗'))
print()


def gpu_all():
    hs = [ns_enqueue(w) for w in weights[:half]]
    queue.finish()
    for h in hs:
        collect(h)
    queue.finish()


def cpu_all():
    return [ns_cpu(w) for w in weights[:half]]


fb = torch.randn(8 * 1024, 512)
fbw = torch.randn(512, 2048)


def cpu_work():
    for _ in range(20):
        torch.mm(fb, fbw)


t_gpu = timeit(gpu_all)
t_cpu = timeit(cpu_all)
t_work = timeit(cpu_work)
print('  iGPU 单独全部 NS（异步，1 次 finish/矩阵）  %7.3f s' % t_gpu)
print('  CPU  单独全部 NS                            %7.3f s' % t_cpu)
print('  CPU  前反向形状负载                          %7.3f s' % t_work)
print()


def both():
    hs = [ns_enqueue(w) for w in weights[:half]]   # enqueue everything first
    cpu_work()                                      # CPU runs while the queue drains
    queue.finish()
    for h in hs:
        collect(h)
    queue.finish()


t_both = timeit(both)
ideal = max(t_gpu, t_work)
serial = t_gpu + t_work
print('  两者并发（全部入队 → CPU 干活 → 一次 finish）  %7.3f s' % t_both)
print('  参照: max(iGPU, CPU-work) = %.3f（理想）   串行和 = %.3f' % (ideal, serial))
print('  ⇒ 相对理想 %.2fx   相对串行 %.2fx   %s'
      % (ideal / t_both, serial / t_both,
         '✓ 设备工作被隐藏' if t_both < ideal * 1.15
         else ('⚠ 部分重叠' if t_both < serial * 0.9 else '✗ 串行')))
