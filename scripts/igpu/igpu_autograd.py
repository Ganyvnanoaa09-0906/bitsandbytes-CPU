"""An OpenCL GEMM as a PyTorch autograd Function, verified against torch's own gradients.

This is the piece iGPU training needs and does not have. Everything measured so far concerns throughput
of a kernel in isolation; a training loop needs the operation inside the autograd graph, with a backward
that is correct, and correctness has to be established before any speed number means anything.

The verification is gradient parity, not agreement of the forward pass. A forward-only check passes on a
function whose backward is wrong, and the resulting model trains without raising and learns the wrong
thing -- the same class of failure as the silently unquantised model and the checkpoint whose
copy_ into a meta tensor did nothing. So gradients are compared against torch's own autograd on the same
inputs, and the tolerance is stated rather than chosen after seeing the numbers.

The backward is done with torch.mm on the CPU here. That is deliberate: the first question is whether
the graph works, and putting the backward on the GPU in the same step would confound "the wiring is
wrong" with "the second kernel is wrong". Once forward parity holds, moving the backward is a separate
change with its own test.

Transfers are part of the cost and are included. On an APU this is a shared-memory copy rather than a
PCIe round trip, which is what makes the approach plausible here at all, but it is not free and a
function that ignores it would report a speedup that no training loop would see.
"""
from __future__ import annotations
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
    pass

import pyopencl as cl  # noqa: E402

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
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
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

_plat = cl.get_platforms()[0]
_ctx = cl.Context(devices=[_plat.get_devices()[0]])
_q = cl.CommandQueue(_ctx)
_mf = cl.mem_flags
_kern = cl.Kernel(cl.Program(_ctx, SRC).build(), 'gemm_fast')
_cache = {}


def gpu_mm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """C = A @ B on the iGPU, no autograd. A [M,K] row-major, B [K,N] row-major."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, 'K 不一致'
    An = np.ascontiguousarray(A.detach().to(torch.float32).numpy())
    Bn = np.ascontiguousarray(B.detach().to(torch.float32).numpy())
    Cn = np.empty((M, N), dtype=np.float32)
    key = (M, N, K)
    if key not in _cache:
        _cache.clear()                       # 只留一份,避免显存被旧形状占住
        ba = cl.Buffer(_ctx, _mf.READ_ONLY, M * K * 4)
        bb = cl.Buffer(_ctx, _mf.READ_ONLY, K * N * 4)
        bc = cl.Buffer(_ctx, _mf.WRITE_ONLY, M * N * 4)
        _cache[key] = (ba, bb, bc)
    ba, bb, bc = _cache[key]
    cl.enqueue_copy(_q, ba, An).wait()
    cl.enqueue_copy(_q, bb, Bn).wait()
    gsz = ((N + 63) // 64 * 16, (M + 63) // 64 * 16)
    _kern(_q, gsz, (16, 16), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
    cl.enqueue_copy(_q, Cn, bc).wait()
    return torch.from_numpy(Cn)


class IgpuLinear(torch.autograd.Function):
    """y = x @ w.T, with the forward GEMM on the iGPU.

    ★ 反传【故意】留在 CPU（torch.mm）。
      第一步要回答的是"图接得对不对",而如果同一步里也把反传搬上核显,
      就会把"接线错了"和"第二个内核错了"混在一起 —— 那是本报告反复记过的那类混杂。
      前向等价性成立之后,再单独搬反传,而它有自己的测试。
    """
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return gpu_mm(x, w.t().contiguous())

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        # y = x @ w.T, x [M,K], w [N,K], y [M,N]
        # dx = gy @ w        [M,N] @ [N,K] -> [M,K]
        # dw = gy.T @ x      [N,M] @ [M,K] -> [N,K]
        return gy @ w, gy.t() @ x


print()
print('  === 1. 前向正确性 ===')
M, N, K = 128, 512, 512
torch.manual_seed(0)
x = torch.randn(M, K)
w = torch.randn(N, K)
ref = x @ w.t()
got = IgpuLinear.apply(x, w)
err = (got - ref).abs().max().item()
rel = err / ref.abs().max().item()
print('    %dx%dx%d · 最大绝对误差 %.3e · 相对 %.3e' % (M, N, K, err, rel))
print('    ⇒ %s（判据:相对 < 1e-5,float32 的累积舍入）' % ('✓ 通过' if rel < 1e-5 else '✗ 不通过'))

print()
print('  === 2. ★ 梯度等价性（与 torch 自己的 autograd 比）===')
xg = torch.randn(M, K, requires_grad=True)
wg = torch.randn(N, K, requires_grad=True)
loss_ref = (xg @ wg.t()).pow(2).sum()
loss_ref.backward()
dx_ref, dw_ref = xg.grad.clone(), wg.grad.clone()
xg.grad = None; wg.grad = None

xg2 = xg.detach().clone().requires_grad_(True)
wg2 = wg.detach().clone().requires_grad_(True)
loss_ig = IgpuLinear.apply(xg2, wg2).pow(2).sum()
loss_ig.backward()

dl = abs(loss_ref.item() - loss_ig.item()) / abs(loss_ref.item())
dxe = (xg2.grad - dx_ref).abs().max().item() / dx_ref.abs().max().item()
dwe = (wg2.grad - dw_ref).abs().max().item() / dw_ref.abs().max().item()
print('    loss    相对误差 %.3e' % dl)
print('    dX      相对误差 %.3e' % dxe)
print('    dW      相对误差 %.3e' % dwe)
ok = dl < 1e-5 and dxe < 1e-4 and dwe < 1e-4
print('    ⇒ %s' % ('★【梯度等价 ✓ —— 这个 Function 可以进训练循环】' if ok
                    else '✗ 梯度不等价 —— 而前向是过的,所以问题在 backward 的接线'))

print()
print('  === 3. 它在训练步里的成本 ===')
for (m, n, k) in [(128, 512, 512), (128, 2048, 2048), (128, 6144, 2048)]:
    xx = torch.randn(m, k)
    ww = torch.randn(n, k)
    fl = 2.0 * m * n * k
    for _ in range(3):
        gpu_mm(xx, ww.t().contiguous())
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        gpu_mm(xx, ww.t().contiguous())
        ts.append(time.perf_counter() - t0)
    tg = statistics.median(ts)
    for _ in range(2):
        xx @ ww.t()
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        xx @ ww.t()
        ts.append(time.perf_counter() - t0)
    tc = statistics.median(ts)
    print('    %-16s 核显 %.2f ms (%6.1f GFLOPS) · CPU %.2f ms (%6.1f GFLOPS) ⇒ %s'
          % ('%dx%dx%d' % (m, n, k), tg * 1e3, fl / tg / 1e9, tc * 1e3, fl / tc / 1e9,
             '核显快 %.2f×' % (tc / tg) if tg < tc else 'CPU 快 %.2f×' % (tg / tc)))
print()
print('  ★ 注意:上表【含传输】,而形状很小 ⇒ 传输占比高。')
print('    ⇒ 而这正是"找平衡点"要解决的问题:什么形状、什么分工,合计最大。')
print()
