"""iGPU training end to end: forward and backward both on the device, weights resident.

Two corrections from the measurements go into this version.

First, 7.96's backward ran on the CPU and its cost profile was dominated by re-uploading the 50 MB weight
matrix on every call. 7.97 showed that is where the iGPU's apparent slowness came from, so weights are
cached on the device by tensor identity here and uploaded once. That is what a training loop actually
does and it is the difference between 64 and 374 GFLOPS.

Second, 7.97 found the balance is a handover rather than a division: M=1 belongs to the CPU, M>=64 to
the iGPU, and running both together adds 7% because the shared L3 is zero-sum. So this does not try to
split work. It runs the whole loop on one engine and measures which.

What decides whether it worked is not speed. It is that the loss falls, that the gradients stay within
parity of torch's own, and that the step time is reported honestly with transfers included. A training
loop that runs faster and learns nothing is the failure mode this whole project keeps finding in other
people's code, and there is no reason to add another instance.
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
_wcache = {}   # ★ 权重常驻:key -> buffer
# ★★ 而"常驻"的正确语义不是"永不重传",是【权重没变就不重传】。
#    第一版用 data_ptr() 当键,而 opt.step() 是原地更新 ⇒ ptr 不变 ⇒ 缓存永远返回第一步的权重
#    ⇒ 核显那条臂对着【冻结的权重】训练,而【梯度等价性检查抓不到它】(它只测一次前向+反向)。
#    修法:键里带一个【权重版本号】,训练循环在 opt.step() 之后调 igpu_step() 把它 +1。
_wver = [0]


def igpu_step():
    """★ 在 optimizer.step() 之后调用 —— 声明"权重变了,缓存作废"。"""
    _wver[0] += 1
_abuf = {}     # A/C 缓冲按形状复用


def _buf(store, key, size):
    if key not in store:
        store[key] = cl.Buffer(_ctx, _mf.READ_WRITE, size)
    return store[key]


def gpu_mm(A, B, B_is_weight=False):
    """C = A @ B. B_is_weight=True ⇒ B 按 data_ptr 常驻,只上传一次。"""
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    _, N = B.shape
    An = np.ascontiguousarray(A.detach().to(torch.float32).numpy())
    key_b = (A.shape[0], B.shape[0], B.shape[1])
    ba = _buf(_abuf, ('a',) + key_b, M * K * 4)
    bc = _buf(_abuf, ('c',) + key_b, M * N * 4)
    cl.enqueue_copy(_q, ba, An).wait()
    if B_is_weight:
        wk = (int(B.data_ptr()), tuple(B.shape), _wver[0])   # ★ 带版本号
        if wk not in _wcache:
            Bn = np.ascontiguousarray(B.detach().to(torch.float32).numpy())
            bb = cl.Buffer(_ctx, _mf.READ_ONLY, K * N * 4)
            cl.enqueue_copy(_q, bb, Bn).wait()
            _wcache[wk] = bb
        bb = _wcache[wk]
    else:
        Bn = np.ascontiguousarray(B.detach().to(torch.float32).numpy())
        bb = _buf(_abuf, ('b',) + key_b, K * N * 4)
        cl.enqueue_copy(_q, bb, Bn).wait()
    gsz = ((N + 63) // 64 * 16, (M + 63) // 64 * 16)
    _kern(_q, gsz, (16, 16), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
    cl.enqueue_copy(_q, np.empty(0)) if False else None
    Cn = np.empty((M, N), dtype=np.float32)
    cl.enqueue_copy(_q, Cn, bc).wait()
    return torch.from_numpy(Cn)


class IgpuLinear(torch.autograd.Function):
    """y = x @ w.T, 前向与反向都在核显,权重常驻。"""
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return gpu_mm(x, w.t().contiguous() if False else w, B_is_weight=True) \
            if False else gpu_mm_t(x, w)

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        # y = x @ w.T   x[M,K] w[N,K] gy[M,N]
        # dX = gy @ w          [M,N]@[N,K] -> [M,K]
        # dW = gy.T @ x        [N,M]@[M,K] -> [N,K]
        dX = gpu_mm(gy, w, B_is_weight=False)
        dW = gpu_mm(gy.t().contiguous(), x, B_is_weight=False)
        return dX, dW


def gpu_mm_t(x, w):
    """w 是 [N,K]，要算 x @ w.T ⇒ 传 w.T 的 [K,N]。缓存在 w 的 data_ptr 上,
    而 transpose 后的张量每次是新对象 ⇒ 用 w 自己的 ptr + 标记当键。"""
    M, K = x.shape
    N = w.shape[0]
    An = np.ascontiguousarray(x.detach().to(torch.float32).numpy())
    wk = (int(w.data_ptr()), 'T', M, _wver[0])   # ★ 带版本号
    ba = _buf(_abuf, ('a', M, K, N), M * K * 4)
    bc = _buf(_abuf, ('c', M, K, N), M * N * 4)
    cl.enqueue_copy(_q, ba, An).wait()
    if wk not in _wcache:
        Bn = np.ascontiguousarray(w.detach().t().contiguous().to(torch.float32).numpy())
        bb = cl.Buffer(_ctx, _mf.READ_ONLY, K * N * 4)
        cl.enqueue_copy(_q, bb, Bn).wait()
        _wcache[wk] = bb
    bb = _wcache[wk]
    gsz = ((N + 63) // 64 * 16, (M + 63) // 64 * 16)
    _kern(_q, gsz, (16, 16), np.int32(M), np.int32(N), np.int32(K), ba, bb, bc)
    Cn = np.empty((M, N), dtype=np.float32)
    cl.enqueue_copy(_q, Cn, bc).wait()
    return torch.from_numpy(Cn)


print()
print('  === 1. 梯度等价（前向与反传都在核显,权重常驻）===')
torch.manual_seed(0)
M, N, K = 512, 1024, 768
x0 = torch.randn(M, K)
w0 = torch.randn(N, K)

xr = x0.clone().requires_grad_(True)
wr = w0.clone().requires_grad_(True)
(xr @ wr.t()).pow(2).sum().backward()

xi = x0.clone().requires_grad_(True)
wi = w0.clone().requires_grad_(True)
IgpuLinear.apply(xi, wi).pow(2).sum().backward()

for nm, a, b in (('loss', None, None), ('dX', xr.grad, xi.grad), ('dW', wr.grad, wi.grad)):
    if a is None:
        continue
    e = (a - b).abs().max().item() / a.abs().max().item()
    print('    %-4s 相对误差 %.3e  %s' % (nm, e, '✓' if e < 1e-4 else '✗'))
ok = all((a - b).abs().max().item() / a.abs().max().item() < 1e-4
         for a, b in ((xr.grad, xi.grad), (wr.grad, wi.grad)))
print('    ⇒ %s' % ('★【前向+反向都在核显,而梯度等价 ✓】' if ok else '✗ 不等价'))

print()
print('  === 2. 真训练循环：两层 MLP，看 loss 掉不掉 ===')
torch.manual_seed(1)
H, B = 512, 1024
tgt_W1 = torch.randn(H, 512) * 0.1
Xt = torch.randn(4096, 512)
Yt = torch.tanh(Xt @ tgt_W1.t()) @ torch.randn(4, H).t() * 0.1


def make_layers(gpu):
    torch.manual_seed(2)
    l1w = torch.randn(H, 512, requires_grad=True)
    l2w = torch.randn(4, H, requires_grad=True)
    return l1w, l2w


def step(l1w, l2w, opt, gpu):
    idx = torch.randint(0, Xt.shape[0], (B,))
    xb, yb = Xt[idx], Yt[idx]
    h = IgpuLinear.apply(xb, l1w) if gpu else xb @ l1w.t()
    h = torch.tanh(h)
    y = IgpuLinear.apply(h, l2w) if gpu else h @ l2w.t()
    loss = torch.nn.functional.mse_loss(y, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if gpu:
        igpu_step()          # ★ 声明权重已变 ⇒ 缓存作废
    return float(loss.detach())


for gpu in (False, True):
    torch.set_num_threads(5)
    l1w, l2w = make_layers(gpu)
    opt = torch.optim.Adam([l1w, l2w], lr=1e-3)
    ls = []
    ts = []
    for i in range(20):
        t0 = time.perf_counter()
        ls.append(step(l1w, l2w, opt, gpu))
        ts.append(time.perf_counter() - t0)
    print('    %-6s loss %.4f → %.4f · 中位步时 %.1f ms · 前5步均 %.4f · 后5步均 %.4f'
          % ('核显' if gpu else 'CPU', ls[0], ls[-1], statistics.median(ts) * 1e3,
             statistics.mean(ls[:5]), statistics.mean(ls[-5:])))
print()
print('  ★ 判据: 核显那条的【后5步均 loss】必须明显低于【前5步均】⇒ 真的在学,而不只是跑得快')
print()
print('  === 3. 第 4 层检验：权重真的变了吗 ===')
print('  （这一层是第一版【缺】的,而它会在那一刻就抓到冻结的权重）')
for gpu in (False, True):
    torch.set_num_threads(5)
    l1w, l2w = make_layers(gpu)
    w0 = l1w.detach().clone()
    opt = torch.optim.Adam([l1w, l2w], lr=1e-3)
    for _ in range(20):
        step(l1w, l2w, opt, gpu)
    d = (l1w.detach() - w0).abs().mean().item()
    print('    %-6s 权重的 mean|Δl| = %.3e  %s'
          % ('核显' if gpu else 'CPU', d, '✓ 变了' if d > 1e-5 else '✗ 【冻结】'))
print()
