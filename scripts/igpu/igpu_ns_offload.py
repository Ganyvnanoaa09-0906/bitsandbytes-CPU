"""igpu_ns_offload.py -- put Muon's Newton-Schulz on the iGPU, run it beside the CPU job.

The version I started to write did not use the GPU at all: ns_gpu did its matmuls with
numpy on the CPU, so it would have measured the CPU twice and concluded nothing. This one
routes every product through the OpenCL kernel.

The shape of the idea. Muon's orthogonalisation is 74 small matmuls on 2-D weights, 1.9 s
per step measured, and it does not scale with batch. It runs during the optimiser phase.
So if the iGPU can absorb it while the CPU does forward and backward, that 1.9 s comes off
the step for free, and the CPU never contends for the same matrix -- unlike the earlier
split tests, which divided one matmul between the two devices and measured 0.68-0.91x.

The counter-evidence is real: two independent workloads gave 0.81x, the iGPU slowed 1.84x
under CPU load, and even a compute-only test interfered by 1.7x. So this is genuinely open.

Deliberately measured **while the 2000-step Muon A/B is running**. That is not pollution
here, it is the experiment: the question is precisely whether the iGPU can work alongside a
saturated CPU. The guard is called with allow=True for that reason, and the A/B's own
timings are unaffected because it never touches the GPU.

Reported: CPU-only NS, iGPU-only NS with its round trips, a correctness check against the
CPU result, and both together. The verdict is whether the combined time approaches the
larger of the two or their sum.
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
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


import pyopencl as cl
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

# Muon's shapes here: per layer Q(512,512) K(512,256) V(512,256) O(512,512)
# and FFN gate/up(512,2048) down(2048,512), ten layers, 74 matrices total
SHAPES = []
for _ in range(10):
    SHAPES += [(512, 512), (512, 256), (512, 256), (512, 512),
               (512, 2048), (512, 2048), (2048, 512)]
SHAPES = SHAPES[:74]

_pool = {}


def _buf(nbytes, tag):
    b = _pool.get(tag)
    if b is None or b.size < nbytes:
        b = cl.Buffer(ctx, mf.READ_WRITE, nbytes)
        _pool[tag] = b
    return b


def gpu_mm(A, B):
    """A @ B with the OpenCL kernel. Both operands fp32, row major."""
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    ba = _buf(A.nbytes, 'a')
    bb = _buf(B.nbytes, 'b')
    bc = _buf(m * n * 4, 'c')
    cl.enqueue_copy(queue, ba, A)
    cl.enqueue_copy(queue, bb, B)
    gsz = ((n + TS - 1) // TS * 8, (m + TS - 1) // TS * 8)
    kern(queue, gsz, (8, 8), np.int32(m), np.int32(n), np.int32(k), ba, bb, bc)
    out = np.empty((m, n), dtype=np.float32)
    cl.enqueue_copy(queue, out, bc).wait()
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


def ns_gpu(G, steps=5, eps=1e-7):
    """Same maths, every product through the device."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = np.ascontiguousarray(G.detach().float().numpy())
    X = X / (np.linalg.norm(X) + eps)
    tr = False
    if X.shape[0] > X.shape[1]:
        X = np.ascontiguousarray(X.T)
        tr = True
    for _ in range(steps):
        A = gpu_mm(X, X.T)                 # m x m
        A2 = gpu_mm(A, A)                  # m x m
        Bm = b * A + c * A2
        X = a * X + gpu_mm(Bm, X)          # m x n
    return torch.from_numpy(np.ascontiguousarray(X.T if tr else X))


# the A/B is the intended CPU load here, so the guard is told to allow it
assert_idle(allow=True, threshold=50.0)
stamp('NS offload（与 A/B 并发，负载即被测对象）')
print()
print('设备 %s | %d CU' % (dev.name, dev.max_compute_units))

weights = [torch.randn(*s) for s in SHAPES]
print('NS 权重 %d 个矩阵，%.1fM 参数（%.0f MB fp32）'
      % (len(SHAPES), sum(m * n for m, n in SHAPES) / 1e6,
         sum(m * n for m, n in SHAPES) * 4 / 2**20))
print()


def timeit(fn, reps=3):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


# correctness first
w0 = weights[0]
ref = ns_cpu(w0)
got = ns_gpu(w0)
d = float((got - ref).abs().max()) / (float(ref.abs().max()) or 1.0)
print('正确性: iGPU NS 与 CPU NS 的相对差 %.2e  %s' % (d, '✓' if d < 5e-3 else '✗'))
print()

t_cpu = timeit(lambda: [ns_cpu(w) for w in weights])
print('  CPU 单独全部 NS                %7.3f s' % t_cpu)
t_gpu = timeit(lambda: [ns_gpu(w) for w in weights])
print('  iGPU 单独全部 NS（含往返）     %7.3f s' % t_gpu)

# the CPU's forward/backward-shaped load: 20 GEMMs the size of one MLP layer at batch 8
fb = torch.randn(8 * 1024, 512)
fbw = torch.randn(512, 2048)
t_fb = timeit(lambda: [torch.mm(fb, fbw) for _ in range(20)])
print('  CPU 前反向形状负载（20×GEMM）  %7.3f s' % t_fb)
print()

half = len(weights) // 2
half_cpu = timeit(lambda: [ns_cpu(w) for w in weights[:half]])


def both():
    for w in weights[:half]:        # iGPU does half the orthogonalisation
        ns_gpu(w)
    for _ in range(20):             # CPU does its own work meanwhile
        torch.mm(fb, fbw)


t_both = timeit(both)
ideal = max(half_cpu, t_fb)
serial = half_cpu + t_fb
print('  并发：iGPU 半量 NS + CPU 前反向形状负载   %7.3f s' % t_both)
print('  参照 max = %.3f（理想）   串行和 = %.3f' % (ideal, serial))
print('  ⇒ 相对理想 %.2fx   相对串行 %.2fx   %s'
      % (ideal / t_both, serial / t_both,
         '✓ iGPU 的活完全被隐藏' if t_both < ideal * 1.15
         else ('⚠ 部分重叠' if t_both < serial * 0.9 else '✗ 完全串行')))
