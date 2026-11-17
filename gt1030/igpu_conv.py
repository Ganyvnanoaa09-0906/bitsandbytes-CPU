# -*- coding: utf-8 -*-
"""igpu_conv.py — 核显 Conv2d 的**训练反向**（im2col + GEMM + col2im），补上阶段 2 的缺口。

现状：`opencl_fp16_ops.conv2d_h` 只有前向（im2col + Hgemm）。本模块补齐：
    fwd : X_col = im2col(x) ; Y = X_col @ Wᵀ (hgemm_nt) ; reshape -> (N,O,Ho,Wo)
    dW  : dW = dYᵀ @ X_col   (hgemm_tn)
    dX  : dX_col = dY @ W    (hgemm_nn) ; dX = col2im(dX_col)

col2im 用 **gather 版**（每个输出元素反向汇总），避免 atomic 竞争：
    dX[n,c,h,w] = Σ_{kh,kw,ho,wo 映射到 (h,w)} dX_col[n,ho,wo,c,kh,kw]

用法/验收：
    python gt1030/igpu_conv.py --check     # 与 torch conv2d 的 fwd/dW/dX 对拍
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import ctypes  # noqa: E402

import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.ar_v2_multi as MM  # noqa: E402
import gt1030.igpu_ctx as CTX  # noqa: E402

A = MM.A
L = MM.L

SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// X_col[m, k]，m = ((n*Ho + ho)*Wo + wo)，k = ((c*KH + kh)*KW + kw)
__kernel void im2col_h(__global const half* x, __global half* col,
                       int N, int C, int H, int W, int KH, int KW,
                       int Ho, int Wo, int stride, int pad, int dil) {
    int m = get_global_id(0), k = get_global_id(1);
    int K = C * KH * KW;
    if (k >= K) return;
    int kw = k % KW, kh = (k / KW) % KH, c = k / (KW * KH);
    int wo = m % Wo, ho = (m / Wo) % Ho, n = m / (Wo * Ho);
    int hi = ho * stride - pad + kh * dil;
    int wi = wo * stride - pad + kw * dil;
    half v = (half)0;
    if (hi >= 0 && hi < H && wi >= 0 && wi < W)
        v = x[(((long)n * C + c) * H + hi) * W + wi];
    col[(long)m * K + k] = v;
}

// dX[n,c,h,w] = Σ dX_col[m, k]  (gather)
__kernel void col2im_h(__global const half* dcol, __global half* dx,
                       int N, int C, int H, int W, int KH, int KW,
                       int Ho, int Wo, int stride, int pad, int dil) {
    int idx = get_global_id(0);
    long total = (long)N * C * H * W;
    if (idx >= total) return;
    int w = idx % W, h = (idx / W) % H, c = (h / 1) % H, n;
    h = (idx / W) % H;
    c = (idx / (W * H)) % C;
    n = idx / (W * H * C);
    int K = C * KH * KW;
    float acc = 0.f;
    for (int kh = 0; kh < KH; ++kh) {
        int t = h + pad - kh * dil;
        if (t % stride != 0) continue;
        int ho = t / stride;
        if (ho < 0 || ho >= Ho) continue;
        for (int kw = 0; kw < KW; ++kw) {
            int u = w + pad - kw * dil;
            if (u % stride != 0) continue;
            int wo = u / stride;
            if (wo < 0 || wo >= Wo) continue;
            long m = ((long)n * Ho + ho) * Wo + wo;
            int k = ((c * KH + kh) * KW + kw);
            acc += (float)dcol[m * K + k];
        }
    }
    dx[idx] = (half)acc;
}
"""


class Conv2dCtx(object):
    """groups=1 的 Conv2d（fp16，设备常驻）。所有缓冲来自一个 DevCtx 的 arena。"""

    def __init__(self, n, c, h, w, o, kh, kw, stride=1, pad=1, dil=1, dev=None):
        self.N, self.C, self.H, self.W = n, c, h, w
        self.O, self.KH, self.KW = o, kh, kw
        self.stride, self.pad, self.dil = stride, pad, dil
        self.Ho = (h + 2 * pad - dil * (kh - 1) - 1) // stride + 1
        self.Wo = (w + 2 * pad - dil * (kw - 1) - 1) // stride + 1
        self.M = n * self.Ho * self.Wo
        self.K = c * kh * kw
        self.dev = dev or CTX.DevCtx(arena_mb=400, name="conv")
        d = self.dev
        d.activate()
        self.K_ = d.program(SRC, ["im2col_h", "col2im_h"])
        self.x = d.buf(n * c * h * w * 2)
        self.wt = d.buf(o * self.K * 2)          # torch 布局 (O, C*KH*KW)
        self.col = d.buf(self.M * self.K * 2)
        self.y = d.buf(self.M * o * 2)
        self.dy = d.buf(self.M * o * 2)
        self.dx = d.buf(n * c * h * w * 2)
        self.dcol = d.buf(self.M * self.K * 2)
        self.dw = d.buf(o * self.K * 2)

    # ---- 数据进出 ----
    def set_x(self, x):
        d = self.dev
        cl.enqueue_copy(d.q, self.x, np.ascontiguousarray(x, dtype=np.float16))
        d.q.finish()

    def set_w(self, w):
        d = self.dev
        cl.enqueue_copy(d.q, self.wt, np.ascontiguousarray(w, dtype=np.float16))
        d.q.finish()

    def get(self, buf, shape):
        return self.dev.down(buf, shape)

    # ---- 计算 ----
    def forward(self):
        d = self.dev
        d.activate()
        self.K_["im2col_h"](d.q, (self.M, self.K), None, self.x, self.col,
                            np.int32(self.N), np.int32(self.C), np.int32(self.H),
                            np.int32(self.W), np.int32(self.KH), np.int32(self.KW),
                            np.int32(self.Ho), np.int32(self.Wo), np.int32(self.stride),
                            np.int32(self.pad), np.int32(self.dil))
        L.hgemm_nt(self.col, self.wt, self.y, self.M, self.O, self.K)
        d.q.finish()
        return self.y

    def backward(self, dy):
        d = self.dev
        d.activate()
        cl.enqueue_copy(d.q, self.dy, np.ascontiguousarray(dy, dtype=np.float16))
        d.q.finish()
        L.hgemm_tn(self.dy, self.col, self.dw, self.O, self.K, self.M)      # dW
        L.hgemm_nn(self.dy, self.wt, self.dcol, self.M, self.K, self.O)     # dX_col
        self.K_["col2im_h"](d.q, (self.N * self.C * self.H * self.W,), None,
                            self.dcol, self.dx, np.int32(self.N), np.int32(self.C),
                            np.int32(self.H), np.int32(self.W), np.int32(self.KH),
                            np.int32(self.KW), np.int32(self.Ho), np.int32(self.Wo),
                            np.int32(self.stride), np.int32(self.pad), np.int32(self.dil))
        d.q.finish()
        return self.dx, self.dw


def _check(n=2, c=3, h=8, w=8, o=4, kh=3, kw=3, stride=1, pad=1):
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((n, c, h, w)) * 0.3).astype(np.float16)
    wt = (rng.standard_normal((o, c, kh, kw)) * 0.2).astype(np.float16)
    cv = Conv2dCtx(n, c, h, w, o, kh, kw, stride, pad)
    cv.set_x(x)
    cv.set_w(wt)
    cv.forward()
    y = cv.get(cv.y, (cv.M, o)).reshape(n, cv.Ho, cv.Wo, o).transpose(0, 3, 1, 2)
    dy = (rng.standard_normal((n, o, cv.Ho, cv.Wo)) * 0.1).astype(np.float16)
    cv.backward(dy.transpose(0, 2, 3, 1).reshape(cv.M, o))
    dx = cv.get(cv.dx, (n, c, h, w))
    dw = cv.get(cv.dw, (o, c * kh * kw)).reshape(o, c, kh, kw)

    tx = torch.from_numpy(x.astype(np.float32)).requires_grad_(True)
    tw = torch.from_numpy(wt.astype(np.float32)).requires_grad_(True)
    ty = F.conv2d(tx, tw, stride=stride, padding=pad)
    ty.backward(torch.from_numpy(dy.astype(np.float32)))

    def rel(a, b):
        a = a.astype(np.float32); b = b.detach().numpy().astype(np.float32)
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

    ey, ex, ew = rel(y, ty), rel(dx, tx.grad), rel(dw, tw.grad)
    print("  conv2d N%d C%d %dx%d O%d k%dx%d s%d p%d | relL2 y=%.2e dx=%.2e dw=%.2e"
          % (n, c, h, w, o, kh, kw, stride, pad, ey, ex, ew))
    ok = max(ey, ex, ew) < 5e-2
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return _check()
    print("  用 --check 运行对拍")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
