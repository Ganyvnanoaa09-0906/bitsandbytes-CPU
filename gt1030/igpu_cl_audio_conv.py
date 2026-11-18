# -*- coding: utf-8 -*-
"""igpu_cl_audio_conv.py — OpenCL/CLBlast 做音频卷积（前向 + 反向）的实测。

思路：卷积 = im2col + GEMM，CLBlast 提供 GEMM：
  forward :  y   = cols @ W^T        (rows × Cout)
  dgrad   :  dcols = dy @ W          (rows × cols) → col2im 散回 dx
  wgrad   :  dW^T  = cols^T @ dy     (cols × Cout)

对比纯 CPU torch fp32 的前向/反向耗时，形状取自关键词识别网络（KWS）与 VITS 级 conv1d。
"""
from __future__ import annotations

import ctypes
import json
import sys
import time

import numpy as np
import pyopencl as cl
import torch
import torch.nn.functional as F

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
OUT = r"D:\work\research\bench_results\audio_lab"
torch.set_num_threads(6)

IM2COL = r"""
__kernel void im2col_f32(__global const float* src, __global float* dst,
                         int N, int C, int H, int W, int K, int S, int P, int Ho, int Wo) {
    int idx = get_global_id(0);
    int rows = N * Ho * Wo;
    int cols = C * K * K;
    if (idx >= rows * cols) return;
    int r = idx / cols;
    int c = idx % cols;
    int n = r / (Ho * Wo);
    int ohw = r % (Ho * Wo);
    int oh = ohw / Wo, ow = ohw % Wo;
    int ch = c / (K * K);
    int kk = c % (K * K);
    int kh = kk / K, kw = kk % K;
    int ih = oh * S - P + kh, iw = ow * S - P + kw;
    float v = 0.f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
        v = src[((n * C + ch) * H + ih) * W + iw];
    dst[idx] = v;
}
"""


class CLConv:
    """最小 conv2d（stride=1，square padding），fwd/bwd 都走 im2col+GEMM。"""

    def __init__(self, ctx, q, bl, kern_im2col, kern_col2im, qp):
        self.ctx, self.q, self.bl = ctx, q, bl
        self.ki, self.kc, self.qp = kern_im2col, kern_col2im, qp
        self.mf = cl.mem_flags

    def _buf(self, arr, flags=None):
        if flags is None:
            flags = self.mf.READ_ONLY | self.mf.COPY_HOST_PTR
        return cl.Buffer(self.ctx, flags, hostbuf=arr) if flags & self.mf.COPY_HOST_PTR else cl.Buffer(self.ctx, flags, arr.nbytes)

    def forward(self, x, wt, pad=1):
        """x (N,C,H,W), wt (Cout,C,K,K) -> y (N,Cout,Ho,Wo); 返回 (y, cols_buf, meta) 供反传。"""
        N, C, H, W = x.shape
        Cout, _, K, _ = wt.shape
        S, P = 1, pad
        Ho, Wo = (H + 2 * P - K) // S + 1, (W + 2 * P - K) // S + 1
        rows, cols = N * Ho * Wo, C * K * K
        wmat = np.ascontiguousarray(wt.reshape(Cout, cols).T)      # (cols, Cout)
        xb = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(x, np.float32))
        wb = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=wmat)
        cb = cl.Buffer(self.ctx, self.mf.READ_WRITE, rows * cols * 4)
        ob = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, rows * Cout * 4)
        self.ki(self.q, (rows * cols,), (256,), xb, cb,
                np.int32(N), np.int32(C), np.int32(H), np.int32(W),
                np.int32(K), np.int32(S), np.int32(P), np.int32(Ho), np.int32(Wo))
        rc = self.bl.CLBlastSgemm(101, 111, 111, rows, Cout, cols, ctypes.c_float(1.0),
                                  ctypes.c_void_p(int(cb.int_ptr)), 0, cols,
                                  ctypes.c_void_p(int(wb.int_ptr)), 0, Cout, ctypes.c_float(0.0),
                                  ctypes.c_void_p(int(ob.int_ptr)), 0, Cout, self.qp, None)
        if rc:
            raise RuntimeError(f"Sgemm forward rc={rc}")
        out = np.empty((rows, Cout), np.float32)
        cl.enqueue_copy(self.q, out, ob).wait()
        y = out.reshape(N, Ho, Wo, Cout).transpose(0, 3, 1, 2).copy()
        return y, dict(cols_buf=cb, wmat=wmat, meta=(N, C, H, W, K, S, P, Ho, Wo, rows, cols), Cout=Cout)

    def backward(self, dy, cache):
        """dy (N,Cout,Ho,Wo) -> (dx (N,C,H,W), dW (Cout,C,K,K))。

        dgrad 用「翻转卷积」实现（避开 float atomic）：
            dx = conv2d(dy, W', pad=K-1-P)，W' 是 W 的核翻转 + 通道转置
        wgrad 直接 GEMM：dW^T(cols,Cout) = cols^T @ dy
        """
        N, C, H, W, K, S, P, Ho, Wo, rows, cols = cache["meta"]
        Cout = cache["Cout"]
        cols_buf = cache["cols_buf"]
        wmat = cache["wmat"]                       # (cols, Cout)
        dy = np.ascontiguousarray(dy, np.float32)
        # ---- dgrad ----
        w4 = wmat.T.reshape(Cout, C, K, K)
        wflip = np.ascontiguousarray(w4[:, :, ::-1, ::-1].transpose(1, 0, 2, 3))  # (C, Cout, K, K)
        dx, _ = self.forward(dy, wflip, pad=K - 1 - P)
        # ---- wgrad ----
        dym = np.ascontiguousarray(dy.transpose(0, 2, 3, 1).reshape(rows, Cout), np.float32)
        dyb = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=dym)
        dwb = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, cols * Cout * 4)
        rc = self.bl.CLBlastSgemm(101, 112, 111, cols, Cout, rows, ctypes.c_float(1.0),
                                  ctypes.c_void_p(int(cols_buf.int_ptr)), 0, cols,
                                  ctypes.c_void_p(int(dyb.int_ptr)), 0, Cout, ctypes.c_float(0.0),
                                  ctypes.c_void_p(int(dwb.int_ptr)), 0, Cout, self.qp, None)
        if rc:
            raise RuntimeError(f"Sgemm wgrad rc={rc}")
        dwn = np.empty((cols, Cout), np.float32)
        cl.enqueue_copy(self.q, dwn, dwb).wait()
        dW = dwn.T.reshape(Cout, C, K, K)
        return dx, dW


def bench(clc, shapes, tag):
    rows = {}
    for (N, C, H, W, Cout, K) in shapes:
        x = np.random.randn(N, C, H, W).astype(np.float32)
        wt = (np.random.randn(Cout, C, K, K) * 0.1).astype(np.float32)
        xt = torch.from_numpy(x).requires_grad_(True)
        wtt = torch.from_numpy(wt).requires_grad_(True)
        g = torch.randn(N, Cout, H, W)
        # CPU 参考：前向 + 反向（梯度已给好，测总耗时）
        def cpu_step():
            y0 = F.conv2d(xt, wtt, padding=K // 2)
            torch.autograd.grad(y0, (xt, wtt), grad_outputs=g)
        cpu_step()
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            cpu_step()
            ts.append(time.perf_counter() - t)
        cpu = min(ts) * 1e3
        y_ref = F.conv2d(torch.from_numpy(x), torch.from_numpy(wt), padding=K // 2).detach().numpy()

        clc.forward(x, wt, pad=K // 2)
        clc.q.finish()
        ts = []
        for _ in range(3):
            t = time.perf_counter()
            y, cache = clc.forward(x, wt, pad=K // 2)
            dy = np.random.randn(*y.shape).astype(np.float32)
            clc.backward(dy, cache)
            clc.q.finish()
            ts.append(time.perf_counter() - t)
        gpu = min(ts) * 1e3
        # 正确性（前向）
        y_cpu = y_ref
        err = float(np.max(np.abs(y.astype(np.float32) - y_cpu)))
        rows[tag.format(N=N, C=C, H=H, W=W, Cout=Cout, K=K)] = dict(
            cpu_ms=round(cpu, 2), igpu_ms=round(gpu, 2), speedup=round(cpu / gpu, 2), fwd_err=err)
        print(f"  {tag.format(N=N,C=C,H=H,W=W,Cout=Cout,K=K):<26} CPU {cpu:7.2f} ms | iGPU {gpu:7.2f} ms | "
              f"{cpu/gpu:5.2f}× | err {err:.2e}", flush=True)
    return rows


def main():
    import gt1030.hybrid_gpu as G
    G.init()
    ctx, q = G._ctx, G._q
    bl = G._bl
    prog = cl.Program(ctx, IM2COL).build(options=["-cl-mad-enable"])
    ki = cl.Kernel(prog, "im2col_f32")
    qp = ctypes.byref(ctypes.c_void_p(int(q.int_ptr)))
    clc = CLConv(ctx, q, bl, ki, None, qp)

    res = {}
    # KWS 的三层卷积（fwd+bwd），batch=64
    kws = [(64, 1, 64, 64, 16, 3), (64, 16, 32, 32, 32, 3), (64, 32, 16, 16, 64, 3)]
    res.update(bench(clc, kws, "kws N{N} C{C} {H}x{W} k{K}"))
    # VITS 级 conv1d（用 H=1 表示），C=512
    vits = [(8, 512, 1, 100, 512, 3), (32, 512, 1, 100, 513, 3)]
    res.update(bench(clc, vits, "vits N{N} C{C} L{W} k{K}"))

    with open(rf"{OUT}\cl_audio_conv_bench.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
