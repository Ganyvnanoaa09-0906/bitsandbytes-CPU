"""igpu_mm.py -- run part of a matmul on the iGPU, the rest on the CPU, with autograd.

Measured in test_cpu_gpu_parallel.py: splitting B's columns between the two devices,
with the OpenCL kernel enqueued without a blocking finish, takes 0.019 s against
0.031 s serial -- 2.10x. Report table 69's 0.83x for concurrent *training* does not
reproduce, and its own explanation (per-operation synchronisation, not the memory
bus) is consistent with that: this split synchronises once per matmul rather than
once per operator.

So this module makes it usable from PyTorch. A training step is roughly one forward
matmul per layer plus two backward ones, and the backward is the expensive half, so
all three are routed the same way:

    forward   y      = a @ b        a:(M,K) b:(K,N) -> (M,N)     split along N
    grad_a           = g @ b^T      g:(M,N) b:(K,N) -> (M,K)     split along K
    grad_b           = a^T @ g      a:(M,K) g:(M,N) -> (K,N)     split along N

The first and third need B transposed as stored, the second needs A transposed, so
the kernel takes transA/transB flags rather than requiring a materialised transpose,
which would cost a full copy and defeat the purpose.

Everything is checked against torch before any timing is believed: the day's lesson
is that a throughput number from an unverified path is worth nothing.
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

import os
import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


_KERNEL = r'''
#define TS 64
#define TK 8
#define TX 8
#define TY 8
#define TM 8
#define TN 8

// C[m,n] = sum_k A'[m,k] * B'[k,n]
//   transA=0: A stored (M,K) row-major, A'[m,k] = A[m*K+k]
//   transA=1: A stored (K,M) row-major, A'[m,k] = A[k*M+m]
//   transB=0: B stored (K,N) row-major, B'[k,n] = B[k*N+n]
//   transB=1: B stored (N,K) row-major, B'[k,n] = B[n*K+k]
__kernel void gemm(const int M, const int N, const int K,
                   const int transA, const int transB,
                   const __global float* restrict A,
                   const __global float* restrict B,
                   __global float* restrict C) {
    __local float As[TS][TK + 1];
    __local float Bs[TK][TS + 1];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int NT = TX * TY, lane = ty * TX + tx;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    for (int k0 = 0; k0 < K; k0 += TK) {
        #pragma unroll
        for (int p = 0; p < TS * TK / 64; p++) {
            int flat = p * 64 + lane;
            int r = flat / TK, c = flat % TK;
            int gr = by + r, gc = k0 + c;
            As[r][c] = (gr < M && gc < K)
                ? (transA ? A[gc * M + gr] : A[gr * K + gc]) : 0.f;
        }
        #pragma unroll
        for (int p = 0; p < TK * TS / 64; p++) {
            int flat = p * 64 + lane;
            int r = flat / TS, c = flat % TS;
            int gr = k0 + r, gc = bx + c;
            Bs[r][c] = (gr < K && gc < N)
                ? (transB ? B[gc * K + gr] : B[gr * N + gc]) : 0.f;
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
_ctx = None
_queue = None
_kern = None
_frac = float(os.environ.get('IGPU_FRAC', '0.5'))
_min_elems = int(os.environ.get('IGPU_MIN_ELEMS', '200000'))


def _init():
    global _ctx, _queue, _kern
    if _kern is not None:
        return True
    try:
        import pyopencl as cl
        _ctx = cl.create_some_context(interactive=False)
        _queue = cl.CommandQueue(_ctx)
        prg = cl.Program(_ctx, _KERNEL).build()
        _kern = cl.Kernel(prg, 'gemm')
        return True
    except Exception as e:
        print('[igpu_mm] 初始化失败，全部回退 CPU: %s' % str(e)[:80], flush=True)
        _kern = None
        return False


def available():
    return _init()


# Buffer pool. Allocating three buffers per call was the reason the first version
# lost to plain torch by up to 14x: the compute is milliseconds, the setup was not
# amortised. Keyed by byte size, never shrunk -- these are a few MB each.
_pool = {}


def _buf(nbytes, read_only=True):
    import pyopencl as cl
    key = (nbytes, read_only)
    b = _pool.get(key)
    if b is None:
        mf = cl.mem_flags
        b = cl.Buffer(_ctx, mf.READ_ONLY if read_only else cl.mem_flags.READ_WRITE, nbytes)
        _pool[key] = b
    return b


def _gpu_gemm(A, B, M, N, K, ta, tb):
    """Enqueue on the iGPU and return the output buffer, without waiting."""
    import pyopencl as cl
    A = A if A.flags['C_CONTIGUOUS'] else np.ascontiguousarray(A)
    B = B if B.flags['C_CONTIGUOUS'] else np.ascontiguousarray(B)
    ba = _buf(A.nbytes)
    bb = _buf(B.nbytes)
    bc = _buf(M * N * 4, read_only=False)
    cl.enqueue_copy(_queue, ba, A)
    cl.enqueue_copy(_queue, bb, B)
    gsz = ((N + TS - 1) // TS * 8, (M + TS - 1) // TS * 8)
    _kern(_queue, gsz, (8, 8), np.int32(M), np.int32(N), np.int32(K),
          np.int32(ta), np.int32(tb), ba, bb, bc)
    return bc


def _collect(bc, shape, out=None):
    import pyopencl as cl
    if out is None:
        out = np.empty(shape, dtype=np.float32)
    cl.enqueue_copy(_queue, out, bc)
    _queue.finish()
    return out


def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b with the work split between iGPU and CPU. Falls back to torch alone."""
    if (not a.is_contiguous()) or (not b.is_contiguous()):
        a = a.contiguous(); b = b.contiguous()
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError('shape mismatch %s @ %s' % (tuple(a.shape), tuple(b.shape)))
    if not _init() or a.numel() * b.shape[1] < _min_elems or _frac <= 0:
        return a @ b
    n_gpu = int(N * _frac // TS * TS)
    if n_gpu < TS or N - n_gpu < TS:
        return a @ b
    try:
        A = a.numpy(); B = b.numpy()
        # iGPU: first n_gpu columns of the output, enqueued without waiting
        bc = _gpu_gemm(A, B[:, :n_gpu], M, n_gpu, K, 0, 0)
        # CPU: remaining columns, executing while the kernel runs.
        # b[:, n_gpu:] is non-contiguous, so torch copies it -- unavoidable without
        # a strided kernel, and it showed up in the first measurement.
        cpu_part = a @ b[:, n_gpu:].contiguous()
        gpu_part = _collect(bc, (M, n_gpu))
        out = torch.empty(M, N)
        out[:, :n_gpu] = torch.from_numpy(gpu_part)
        out[:, n_gpu:] = cpu_part
        return out
    except Exception as e:
        print('[igpu_mm] 回退 CPU: %s' % str(e)[:80], flush=True)
        return a @ b


class IgpuParallelMm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return mm(a, b)

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        g = g.contiguous()
        # grad_a = g @ b^T  ->  B operand is b stored (K,N), so B' is transposed
        ga = mm(g, b.t().contiguous())
        # grad_b = a^T @ g  ->  A operand is a stored (M,K), so A' is transposed
        gb = mm(a.t().contiguous(), g)
        return ga, gb


def linear(x, w, bias=None):
    """y = x @ w.T (+ bias), matching nn.Linear's weight layout (out,in)."""
    y = IgpuParallelMm.apply(x, w.t().contiguous())
    return y if bias is None else y + bias


if __name__ == '__main__':
    import time
    torch.set_num_threads(6)
    print('设备可用:', available())
    print('分给核显的比例: %.2f' % _frac)
    print()
    ok = True
    for (M, K, N) in [(64, 32, 64), (256, 512, 256), (1024, 512, 512),
                      (1024, 512, 2736), (1024, 2736, 512), (2048, 2048, 2048)]:
        a = torch.randn(M, K)
        b = torch.randn(K, N)
        ref = a @ b
        got = mm(a, b)
        d = float((got - ref).abs().max()) / (float(ref.abs().max()) or 1.0)
        ok &= d < 1e-4
        t0 = time.perf_counter()
        for _ in range(3):
            a @ b
        t_cpu = (time.perf_counter() - t0) / 3
        t0 = time.perf_counter()
        for _ in range(3):
            mm(a, b)
        t_both = (time.perf_counter() - t0) / 3
        print('  %-16s 相对差 %.2e %s   CPU %.4fs  并行 %.4fs  ⇒ %.2fx'
              % ('%dx%dx%d' % (M, K, N), d, '✓' if d < 1e-4 else '✗',
                 t_cpu, t_both, t_cpu / t_both))
    print()
    print('全部正确 ✓' if ok else '有错误 ✗')
