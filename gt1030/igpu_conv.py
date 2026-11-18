# -*- coding: utf-8 -*-
"""igpu_conv.py — Conv2d on the iGPU via im2col + CLBlast GEMM（fp16，groups=1，dilation=1）。

布局约定：
    输入 x   (N, C, H, W)  fp16
    权重 w   (OC, C, KH, KW) → 展平成 (OC, K), K = C*KH*KW（**正好是 hgemm_nt 要的 B 布局**）
    输出 y   (N, OC, OH, OW)

    forward : cols (M, K) = im2col(x)，M = N*OH*OW；y = cols @ wᵀ → hgemm_nt
    dW      : hgemm_tn(dy, cols) → (OC, K)
    dcols   : hgemm_nn(dy, w)    → (M, K)，再 col2im 累加回 dx

    python gt1030/igpu_conv.py --check     # 与 torch 对拍 fwd/dx/dw
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.igpu_ctx as CTX  # noqa: E402
from gt1030.ar_v2_multi import A  # noqa: E402
from gt1030.ar_v2_multi import L as GEMM  # noqa: E402  (hgemm_nt/nn/tn)

SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// im2col：(N,C,H,W) → (M, K)，M = N*OH*OW，K = C*KH*KW
__kernel void im2col(__global const half* x, __global half* cols,
                     int N, int C, int H, int W, int KH, int KW,
                     int OH, int OW, int SH, int SW, int PH, int PW) {
    int m = get_global_id(0);          // n*OH*OW + oh*OW + ow
    int k = get_global_id(1);          // c*KH*KW + kh*KW + kw
    int K = C * KH * KW;
    int M = N * OH * OW;
    if (m >= M || k >= K) return;
    int ow = m % OW, t = m / OW;
    int oh = t % OH, n = t / OH;
    int kw = k % KW, t2 = k / KW;
    int kh = t2 % KH, c = t2 / KH;
    int ih = oh * SH - PH + kh;
    int iw = ow * SW - PW + kw;
    float v = 0.f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
        v = (float)x[((long)(n * C + c) * H + ih) * W + iw];
    cols[(long)m * K + k] = (half)v;
}

// col2im（gather 版）：每个输入元素汇总所有覆盖它的 (oh,ow,kh,kw)。
// 原实现用 read-modify-write 累加，重叠窗口会丢更新（stride1 时错得一塌糊涂）。
__kernel void col2im(__global const half* dcols, __global half* dx,
                     int N, int C, int H, int W, int KH, int KW,
                     int OH, int OW, int SH, int SW, int PH, int PW) {
    long idx = get_global_id(0);
    long total = (long)N * C * H * W;
    if (idx >= total) return;
    int iw = idx % W, t = idx / W;
    int ih = t % H; t /= H;
    int c = t % C, n = t / C;
    int K = C * KH * KW;
    float acc = 0.f;
    for (int kh = 0; kh < KH; ++kh) {
        int a = ih + PH - kh;
        if (a < 0 || a % SH != 0) continue;
        int oh = a / SH;
        if (oh >= OH) continue;
        for (int kw = 0; kw < KW; ++kw) {
            int b = iw + PW - kw;
            if (b < 0 || b % SW != 0) continue;
            int ow = b / SW;
            if (ow >= OW) continue;
            long m = (long)(n * OH + oh) * OW + ow;
            acc += (float)dcols[m * K + (c * KH + kh) * KW + kw];
        }
    }
    dx[idx] = (half)acc;
}


// (N,OH,OW,OC) → (N,OC,OH,OW)
__kernel void nhwc2nchw(__global const half* src, __global half* dst,
                        int N, int OC, int OH, int OW) {
    long i = get_global_id(0);
    long total = (long)N * OC * OH * OW;
    if (i >= total) return;
    int ow = i % OW, t = i / OW;
    int oh = t % OH; t /= OH;
    int oc = t % OC, n = t / OC;
    dst[i] = src[(((long)n * OH + oh) * OW + ow) * OC + oc];
}

// (N,OC,OH,OW) → (N,OH,OW,OC)
__kernel void nchw2nhwc(__global const half* src, __global half* dst,
                        int N, int OC, int OH, int OW) {
    long i = get_global_id(0);
    long total = (long)N * OC * OH * OW;
    if (i >= total) return;
    int ow = i % OW, t = i / OW;
    int oh = t % OH; t /= OH;
    int oc = t % OC, n = t / OC;
    dst[(((long)n * OH + oh) * OW + ow) * OC + oc] = src[i];
}


// (N,C,H,W) → (N,C,H+1,W+1)，右/下补零（VAE 的 Downsample2D 需要）
__kernel void pad01(__global const half* src, __global half* dst,
                    int N, int C, int H, int W) {
    long i = get_global_id(0);
    int Hp = H + 1, Wp = W + 1;
    long total = (long)N * C * Hp * Wp;
    if (i >= total) return;
    int wp = i % Wp, t = i / Wp;
    int hp = t % Hp;
    long nc = t / Hp;
    float v = 0.f;
    if (hp < H && wp < W) v = (float)src[nc * (long)H * W + (long)hp * W + wp];
    dst[i] = (half)v;
}

// 反向：(N,C,H+1,W+1) → (N,C,H,W)，裁掉右/下
__kernel void crop01(__global const half* src, __global half* dst,
                     int N, int C, int H, int W) {
    long i = get_global_id(0);
    long total = (long)N * C * H * W;
    if (i >= total) return;
    int w = i % W, t = i / W;
    int h = t % H;
    long nc = t / H;
    int Wp = W + 1, Hp = H + 1;
    dst[i] = src[nc * (long)Hp * Wp + (long)h * Wp + w];
}

// y += bias[o]
__kernel void bias_add_oc(__global half* y, __global const half* b, int M, int OC) {
    int m = get_global_id(0), o = get_global_id(1);
    if (m >= M || o >= OC) return;
    long i = (long)m * OC + o;
    y[i] = (half)((float)y[i] + (float)b[o]);
}

// dy 的 bias 梯度：sum over m
__kernel void bias_grad_oc(__global const half* dy, __global half* db, int M, int OC) {
    int o = get_global_id(0);
    if (o >= OC) return;
    float acc = 0.f;
    for (int m = 0; m < M; ++m) acc += (float)dy[(long)m * OC + o];
    db[o] = (half)((float)db[o] + acc);
}
"""

_C = {}


def K(dev):
    key = int(dev.ctx.int_ptr)
    if key not in _C:
        prog = cl.Program(dev.ctx, SRC).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"]
        )
        _C[key] = {
            n: cl.Kernel(prog, n)
            for n in (
                "im2col",
                "col2im",
                "bias_add_oc",
                "bias_grad_oc",
                "nhwc2nchw",
                "nchw2nhwc",
                "pad01",
                "crop01",
            )
        }
    return _C[key]


class Conv2d(object):
    """单层 Conv2d（训练用）：forward / backward 都只用 im2col + CLBlast GEMM。"""

    def __init__(self, dev, n, c, h, w, oc, kh, kw, sh=1, sw=1, ph=0, pw=0, bias=True):
        self.dev = dev
        self.N, self.C, self.H, self.W = n, c, h, w
        self.OC, self.KH, self.KW = oc, kh, kw
        self.SH, self.SW, self.PH, self.PW = sh, sw, ph, pw
        self.OH = (h + 2 * ph - kh) // sh + 1
        self.OW = (w + 2 * pw - kw) // sw + 1
        self.Kdim = c * kh * kw
        self.M = n * self.OH * self.OW
        self.has_bias = bias
        d = dev
        self.cols = d.buf(max(4096, self.M * self.Kdim * 2))
        self.dcols = d.buf(max(4096, self.M * self.Kdim * 2))
        self.y = d.buf(max(4096, self.M * self.OC * 2))
        self.dy = d.buf(max(4096, self.M * self.OC * 2))
        self.dx = d.buf(max(4096, self.N * self.C * self.H * self.W * 2))
        self.dw = d.buf(max(4096, self.OC * self.Kdim * 2))
        if bias:
            self.b = d.buf(max(4096, self.OC * 2))
            self.db = d.buf(max(4096, self.OC * 2))
            # bias_grad_oc 是 +=；首次 backward 前必须清零，否则会把 arena 旧数据算进梯度。
            cl.enqueue_fill_buffer(d.q, self.db, np.uint16(0), 0, self.OC * 2)

    # ---- 前向 ----
    # 布局：x (N,C,H,W)；w (OC, C*KH*KW)；输出 y (N, OH, OW, OC)——**通道在最后**
    def forward(self, x, w, b=None):
        d = self.dev
        K(d)["im2col"](
            d.q,
            (self.M, self.Kdim),
            None,
            x,
            self.cols,
            np.int32(self.N),
            np.int32(self.C),
            np.int32(self.H),
            np.int32(self.W),
            np.int32(self.KH),
            np.int32(self.KW),
            np.int32(self.OH),
            np.int32(self.OW),
            np.int32(self.SH),
            np.int32(self.SW),
            np.int32(self.PH),
            np.int32(self.PW),
        )
        GEMM.hgemm_nt(self.cols, w, self.y, self.M, self.OC, self.Kdim)
        if self.has_bias and b is not None:
            K(d)["bias_add_oc"](
                d.q,
                (self.M, self.OC),
                None,
                self.y,
                b,
                np.int32(self.M),
                np.int32(self.OC),
            )
        d.q.finish()
        return self.y

    # ---- 反向：返回 (dx, dw, db) ----
    def backward(self, dy, x, w):
        d = self.dev
        cl.enqueue_copy(d.q, self.dy, dy)
        GEMM.hgemm_tn(self.dy, self.cols, self.dw, self.OC, self.Kdim, self.M)
        GEMM.hgemm_nn(self.dy, w, self.dcols, self.M, self.Kdim, self.OC)
        K(d)["col2im"](
            d.q,
            (self.N * self.C * self.H * self.W,),
            None,
            self.dcols,
            self.dx,
            np.int32(self.N),
            np.int32(self.C),
            np.int32(self.H),
            np.int32(self.W),
            np.int32(self.KH),
            np.int32(self.KW),
            np.int32(self.OH),
            np.int32(self.OW),
            np.int32(self.SH),
            np.int32(self.SW),
            np.int32(self.PH),
            np.int32(self.PW),
        )
        if self.has_bias:
            K(d)["bias_grad_oc"](
                d.q,
                (self.OC,),
                None,
                self.dy,
                self.db,
                np.int32(self.M),
                np.int32(self.OC),
            )
        d.q.finish()
        return self.dx, self.dw, (self.db if self.has_bias else None)


def check(n=2, c=3, h=9, w=9, oc=4, kh=3, kw=3, sh=2, sw=2, ph=1, pw=1, seed=0):
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((n, c, h, w)) * 0.5).astype(np.float32)
    wt = (rng.standard_normal((oc, c, kh, kw)) * 0.2).astype(np.float32)
    bt = (rng.standard_normal(oc) * 0.1).astype(np.float32)

    dev = CTX.DevCtx(arena_mb=600, name="conv2d")
    dev.activate()
    layer = Conv2d(dev, n, c, h, w, oc, kh, kw, sh, sw, ph, pw)
    xd = dev.up(x.astype(np.float16))
    wd = dev.up(wt.reshape(oc, -1).astype(np.float16))
    bd = dev.up(bt.astype(np.float16))

    y_dev = layer.forward(xd, wd, bd)
    # 设备侧输出布局是 (N,OH,OW,OC)；torch 是 (N,OC,OH,OW)
    yv = (
        dev.down(y_dev, (n, layer.OH, layer.OW, oc))
        .transpose(0, 3, 1, 2)
        .astype(np.float32)
    )

    dy = (rng.standard_normal((n, oc, layer.OH, layer.OW)) * 0.3).astype(np.float32)
    dy_dev = dev.up(np.ascontiguousarray(dy.transpose(0, 2, 3, 1)).astype(np.float16))
    dx_dev, dw_dev, db_dev = layer.backward(dy_dev, xd, wd)
    dxv = dev.down(dx_dev, (n, c, h, w)).astype(np.float32)
    dwv = dev.down(dw_dev, (oc, c * kh * kw)).astype(np.float32)
    dbv = dev.down(db_dev, (oc,)).astype(np.float32) if db_dev is not None else None

    # torch 参考
    tx = torch.from_numpy(x).requires_grad_(True)
    tw = torch.from_numpy(wt).requires_grad_(True)
    tb = torch.from_numpy(bt).requires_grad_(True)
    ty = F.conv2d(tx, tw, tb, stride=(sh, sw), padding=(ph, pw))
    ty.backward(torch.from_numpy(dy))

    def rel(a, b):
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

    r_f = rel(yv, ty.detach().numpy())
    r_x = rel(dxv, tx.grad.numpy())
    r_w = rel(dwv, tw.grad.reshape(oc, -1).numpy())
    r_b = rel(dbv, tb.grad.numpy()) if dbv is not None else 0.0
    print(
        "  conv2d N%d C%d HxW%dx%d → OC%d %dx%d s%d p%d"
        % (n, c, h, w, oc, kh, kw, sh, ph)
    )
    print("    fwd %.2e   dx %.2e   dw %.2e   db %.2e   (fp16)" % (r_f, r_x, r_w, r_b))
    return max(r_f, r_x, r_w, r_b) < 3e-2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        print(__doc__)
        return 0
    ok = True
    ok &= check(n=2, c=3, h=9, w=9, oc=4, kh=3, kw=3, sh=1, sw=1, ph=1, pw=1)
    ok &= check(n=2, c=4, h=8, w=8, oc=6, kh=3, kw=3, sh=2, sw=2, ph=1, pw=1)
    ok &= check(n=1, c=4, h=7, w=7, oc=3, kh=1, kw=1, sh=1, sw=1, ph=0, pw=0)
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
