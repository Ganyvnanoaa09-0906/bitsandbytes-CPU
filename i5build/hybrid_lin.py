"""hybrid_lin.py — 最小可用的「核显主力 GEMM」实验原型（v2，2026-10-06）。

v2 变化：
  · 约束改为 M,N,K 全 64 倍数（wgrad 的 M 维 = 输入 K，所以 K 也必须 64 倍数）
  · forward 内部自行判断能否上核显，不能就整体回退 torch；autograd 路径统一
  · backward 加 try/except 兜底
  · 新增 min_flops：小于该 FLOPs 的 GEMM 直接走 CPU（同步开销会吃掉小 GEMM 的收益）
"""
import os
import sys
import numpy as np
import pyopencl as cl
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import igpu_repro as R


def _ok(M, N, K):
    return (M % 64 == 0) and (N % 64 == 0) and (K % 64 == 0)


class OCLGEMM:
    def __init__(self):
        self.dev = cl.get_platforms()[0].get_devices()[0]
        self.ctx = cl.Context([self.dev])
        self.q = cl.CommandQueue(self.ctx)
        self._progs = {}
        self.stats = {'calls': 0, 'copy_back': 0}

    def _prog(self, N, K):
        key = (int(N), int(K))
        if key not in self._progs:
            self._progs[key] = cl.Program(self.ctx, R.GEMM_SRC).build(
                options=['-cl-fast-relaxed-math', '-cl-mad-enable',
                         '-DKC=%d' % K, '-DNC=%d' % N])
        return self._progs[key]

    def gemm_nn(self, A, B, C):
        M, K = A.shape
        K2, N = B.shape
        assert K == K2 and tuple(C.shape) == (M, N), 'shape mismatch'
        a = A.detach().contiguous().numpy()
        b = B.detach().contiguous().numpy()
        c = C.numpy()
        mf = cl.mem_flags
        try:
            ab = (cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=a),
                  cl.Buffer(self.ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=b))
            cb = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=c)
            copy_back = False
        except cl.Error:
            ab = (cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a),
                  cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b))
            cb = cl.Buffer(self.ctx, mf.WRITE_ONLY, size=c.nbytes)
            copy_back = True
        kern = cl.Kernel(self._prog(N, K), 'sgemm')
        gws = (int(np.ceil(N / 64)) * 8, int(np.ceil(M / 64)) * 8)
        kern(self.q, gws, (8, 8), np.int32(M), np.int32(N), np.int32(K), ab[0], ab[1], cb)
        self.q.finish()
        if copy_back:
            cl.enqueue_copy(self.q, c, cb).wait()
            self.stats['copy_back'] += 1
        self.stats['calls'] += 1
        return C


class HybridLinear(torch.autograd.Function):
    """y = x @ W.T；满足条件时三次 GEMM 走核显，否则整体回退 CPU。"""
    ocl = None
    enabled = True
    min_flops = 0.0
    n_gpu = 0
    n_cpu = 0

    @staticmethod
    def forward(ctx, x, w):
        x = x.contiguous()
        w = w.contiguous()
        M, K = x.shape
        N = w.shape[0]
        use_gpu = (HybridLinear.enabled and HybridLinear.ocl is not None
                   and x.dtype == torch.float32 and _ok(M, N, K)
                   and 2.0 * M * N * K >= HybridLinear.min_flops)
        ctx.cpu_fallback = not use_gpu
        ctx.save_for_backward(x, w)
        if not use_gpu:
            HybridLinear.n_cpu += 1
            return torch.nn.functional.linear(x, w)
        HybridLinear.n_gpu += 1
        wt = w.t().contiguous()
        y = torch.empty((M, N), dtype=torch.float32)
        HybridLinear.ocl.gemm_nn(x, wt, y)
        return y

    @staticmethod
    def backward(ctx, g):
        need_x, need_w = ctx.needs_input_grad[:2]
        x, w = ctx.saved_tensors
        g = g.contiguous()
        if ctx.cpu_fallback:
            dx = g @ w if need_x else None
            dw = g.t().contiguous() @ x if need_w else None
            return dx, dw
        try:
            dx = torch.empty((x.shape[0], x.shape[1]), dtype=torch.float32)
            HybridLinear.ocl.gemm_nn(g, w, dx)
            xt = x.t().contiguous()
            dwt = torch.empty((x.shape[1], w.shape[0]), dtype=torch.float32)
            HybridLinear.ocl.gemm_nn(xt, g, dwt)
            dw = dwt.t().contiguous()
        except Exception:
            dx = g @ w
            dw = g.t().contiguous() @ x
        return (dx if need_x else None), (dw if need_w else None)
