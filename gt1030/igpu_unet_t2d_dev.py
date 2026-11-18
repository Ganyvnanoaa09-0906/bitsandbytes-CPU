# -*- coding: utf-8 -*-
"""igpu_unet_t2d_dev.py — 设备侧 UNet `Transformer2DModel`（GN→proj_in→self→cross→FF→proj_out）。

与 VAE decoder attention 的差异（别复用 decoder 那套"GN→attn→residual"）：
  1. 外层 `norm` 是 GroupNorm(32)，但**块内 norm1/2/3 是 LayerNorm（带 affine, eps=1e-5）**；
  2. 还有 `proj_in` / `proj_out` 两个 **1×1 卷积**；
  3. attention 是 **self + cross 两个**，中间夹一个 **GEGLU FF**；
  4. 残差是**最后统一加**（proj_out 之后 + 输入），不是每个子块外面各一层。

复用已验证部件：`igpu_gn.GroupNorm`、`igpu_conv.Conv2d`(1×1)、`igpu_attn_cross.CrossAttn`。

对拍抓 **5 个中间张量**：proj_in 后、self-attn 后、cross-attn 后、FF 后、最终 out。

    python gt1030/igpu_unet_t2d_dev.py --check
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.igpu_ctx as CTX  # noqa: E402
from gt1030.ar_v2_multi import L as GEMM  # noqa: E402
from gt1030.igpu_attn_chunk import attn_chunked_heads  # noqa: E402
from gt1030.igpu_conv import Conv2d, K as CVK  # noqa: E402
from gt1030.igpu_gn import GroupNorm  # noqa: E402
from gt1030.igpu_graph import _kernels as GK  # noqa: E402

SRC = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// LayerNorm：一行一个 work-item（token 布局，rows×C）
__kernel void ln_fwd(__global const half* x, __global const half* g,
                     __global const half* b, __global half* y,
                     int rows, int C, float eps) {
    int r = get_global_id(0);
    if (r >= rows) return;
    long base = (long)r * C;
    float s = 0.f;
    for (int c = 0; c < C; ++c) s += (float)x[base + c];
    float mean = s / C;
    float v = 0.f;
    for (int c = 0; c < C; ++c) { float d = (float)x[base + c] - mean; v += d * d; }
    float rstd = rsqrt(v / C + eps);
    for (int c = 0; c < C; ++c) {
        float n = ((float)x[base + c] - mean) * rstd;
        y[base + c] = (half)(n * (float)g[c] + (float)b[c]);
    }
}

// SiLU（UNet 的 FF 内部没有，但 resnet 用；这里一并提供）
__kernel void silu_fwd(__global const half* x, __global half* y, long n) {
    long i = get_global_id(0);
    if (i >= n) return;
    float v = (float)x[i];
    y[i] = (half)(v / (1.f + exp(-v)));
}

// GEGLU 门控：out = a * gelu(b)，a = 前 C 列、b = 后 C 列
// GEGLU：half 是内层维度（UNet 里 = 4*C），前 half 列做 hidden、后 half 列做 gate
__kernel void geglu(__global const half* ab, __global half* y, int rows, int hd) {
    long i = get_global_id(0);
    if (i >= (long)rows * hd) return;
    int r = (int)(i / hd), c = (int)(i % hd);
    float a = (float)ab[(long)r * 2 * hd + c];
    float b = (float)ab[(long)r * 2 * hd + hd + c];
    y[i] = (half)(a * 0.5f * b * (1.f + erf(b * 0.70710678f)));
}
"""

_K = {}


def K(dev):
    key = int(dev.ctx.int_ptr)
    if key not in _K:
        prog = cl.Program(dev.ctx, SRC).build(
            options=["-cl-mad-enable", "-cl-fast-relaxed-math"])
        _K[key] = {n: cl.Kernel(prog, n) for n in ("ln_fwd", "silu_fwd", "geglu")}
    return _K[key]


def _bias(m):
    return None if m is None else m.detach().numpy()


def layernorm(dev, x, g, b, y, rows, C, eps=1e-5):
    K(dev)["ln_fwd"](dev.q, (rows,), None, x, g, b, y,
                     np.int32(rows), np.int32(C), np.float32(eps))
    return y


def token_linear(dev, tok, w, bias, out, rows, cout, cin):
    """tok (rows,cin) × wᵀ → out (rows,cout)（w 形状 (cout,cin)，diffusers Linear 权重）。"""
    GEMM.hgemm_nt(tok, w, out, rows, cout, cin)
    if bias is not None:
        # bias_add_oc 的索引是 m*OC+o，正好就是 token 布局 (rows, cout)
        CVK(dev)["bias_add_oc"](dev.q, (rows, cout), None, out, bias,
                                np.int32(rows), np.int32(cout))
    return out


def _nhwc2nchw(dev, src, dst, N, C, H, W):
    CVK(dev)["nhwc2nchw"](dev.q, (N * C * H * W,), None, src, dst,
                          np.int32(N), np.int32(C), np.int32(H), np.int32(W))
    return dst


def _nchw2nhwc(dev, src, dst, N, C, H, W):
    CVK(dev)["nchw2nhwc"](dev.q, (N * C * H * W,), None, src, dst,
                          np.int32(N), np.int32(C), np.int32(H), np.int32(W))
    return dst



def _proj(dev, tok, lin, rows, cout, cin, buf):
    """Linear 作用在 token 布局上（diffusers 权重 (cout,cin)）。"""
    w = dev.up(np.ascontiguousarray(lin.weight.detach().numpy().astype(np.float16)))
    b = None if lin.bias is None else dev.up(
        np.ascontiguousarray(lin.bias.detach().numpy().astype(np.float16)))
    out = buf(rows * cout)
    token_linear(dev, tok, w, b, out, rows, cout, cin)
    return out


def dev_transformer2d(dev, t2d, x_np, enc_np, groups=32):
    """返回 (out_np, 中间张量 dict)；enc_np (B,Tk,D) fp16。"""
    N, C, H, W = x_np.shape
    T = H * W
    x = dev.up(np.asarray(x_np, np.float16))

    # 1) 外层 GroupNorm
    gn = GroupNorm(dev, N, C, H, W, groups, eps=float(t2d.norm.eps))
    cl.enqueue_copy(dev.q, gn.gamma, np.ascontiguousarray(t2d.norm.weight.detach().numpy().astype(np.float16)))
    cl.enqueue_copy(dev.q, gn.beta, np.ascontiguousarray(t2d.norm.bias.detach().numpy().astype(np.float16)))
    y1 = dev.buf(max(4096, N * C * H * W * 2))
    gn.forward(x, y1)

    # 2) proj_in 1×1 卷积 → NHWC = token 布局
    c_in = Conv2d(dev, N, C, H, W, C, 1, 1, 1, 1, 0, 0, bias=True)
    w_in = dev.up(np.ascontiguousarray(t2d.proj_in.weight.detach().numpy()[:, :, 0, 0].astype(np.float16)))
    b_in = dev.up(np.ascontiguousarray(t2d.proj_in.bias.detach().numpy().astype(np.float16)))
    tok = c_in.forward(y1, w_in, b_in)                     # (N*T, C) 已经是 token 布局
    rq = N * T
    enc = dev.up(np.asarray(enc_np, np.float16))
    rk = enc_np.shape[0] * enc_np.shape[1]

    buf = lambda k: dev.buf(max(4096, int(k) * 2))
    tmp = buf(rq * C)
    ln_out = buf(rq * C)
    att_out = buf(rq * C)
    nchw = buf(N * C * H * W)
    nchw2 = buf(N * C * H * W)
    out5 = {}

    for blk in t2d.transformer_blocks:
        # --- self-attn（k/v 用同一份 token）：token 布局直接进 attn_chunked ---
        g1 = dev.up(np.ascontiguousarray(blk.norm1.weight.detach().numpy().astype(np.float16)))
        b1 = dev.up(np.ascontiguousarray(blk.norm1.bias.detach().numpy().astype(np.float16)))
        layernorm(dev, tok, g1, b1, ln_out, rq, C, float(blk.norm1.eps))
        q_ = _proj(dev, ln_out, blk.attn1.to_q, rq, C, C, buf)
        k_ = _proj(dev, ln_out, blk.attn1.to_k, rq, C, C, buf)
        v_ = _proj(dev, ln_out, blk.attn1.to_v, rq, C, C, buf)
        ao = buf(rq * C)
        attn_chunked_heads(dev, q_, k_, v_, ao, N, T, T, C, blk.attn1.heads, bq=256)
        o = _proj(dev, ao, blk.attn1.to_out[0], rq, C, C, buf)
        GK(dev)["add_h"](dev.q, (rq * C,), None, tok, o, att_out, np.int64(rq * C))
        tok, att_out = att_out, tok
        out5["self"] = dev.down(tok, (N, T, C)).astype(np.float32)

        # --- cross-attn（k/v 来自 encoder_hidden_states）---
        g2 = dev.up(np.ascontiguousarray(blk.norm2.weight.detach().numpy().astype(np.float16)))
        b2 = dev.up(np.ascontiguousarray(blk.norm2.bias.detach().numpy().astype(np.float16)))
        layernorm(dev, tok, g2, b2, ln_out, rq, C, float(blk.norm2.eps))
        q2 = _proj(dev, ln_out, blk.attn2.to_q, rq, C, C, buf)
        k2 = _proj(dev, enc, blk.attn2.to_k, rk, C, enc_np.shape[2], buf)
        v2 = _proj(dev, enc, blk.attn2.to_v, rk, C, enc_np.shape[2], buf)
        ao2 = buf(rq * C)
        attn_chunked_heads(dev, q2, k2, v2, ao2, N, T, enc_np.shape[1], C,
                           blk.attn2.heads, bq=256)
        o2 = _proj(dev, ao2, blk.attn2.to_out[0], rq, C, C, buf)
        GK(dev)["add_h"](dev.q, (rq * C,), None, tok, o2, att_out, np.int64(rq * C))
        tok, att_out = att_out, tok
        out5["cross"] = dev.down(tok, (N, T, C)).astype(np.float32)

        # --- GEGLU FF ---
        g3 = dev.up(np.ascontiguousarray(blk.norm3.weight.detach().numpy().astype(np.float16)))
        b3 = dev.up(np.ascontiguousarray(blk.norm3.bias.detach().numpy().astype(np.float16)))
        layernorm(dev, tok, g3, b3, ln_out, rq, C, float(blk.norm3.eps))
        w0 = blk.ff.net[0].proj.weight          # (2*half, C)，UNet 的 half = 4C = 1280
        half = w0.shape[0] // 2
        hidden = buf(rq * 2 * half)
        hw = dev.up(np.ascontiguousarray(w0.detach().numpy().astype(np.float16)))
        hb = dev.up(np.ascontiguousarray(blk.ff.net[0].proj.bias.detach().numpy().astype(np.float16)))
        token_linear(dev, ln_out, hw, hb, hidden, rq, 2 * half, C)
        gated = buf(rq * half)
        K(dev)["geglu"](dev.q, (rq * half,), None, hidden, gated, np.int32(rq), np.int32(half))
        w2 = dev.up(np.ascontiguousarray(blk.ff.net[2].weight.detach().numpy().astype(np.float16)))
        b2f = dev.up(np.ascontiguousarray(blk.ff.net[2].bias.detach().numpy().astype(np.float16)))
        ffout = buf(rq * C)
        token_linear(dev, gated, w2, b2f, ffout, rq, C, half)
        GK(dev)["add_h"](dev.q, (rq * C,), None, tok, ffout, att_out, np.int64(rq * C))
        tok, att_out = att_out, tok
        out5["ff"] = dev.down(tok, (N, T, C)).astype(np.float32)

    # 3) proj_out 1×1 卷积 + 残差
    _nhwc2nchw(dev, tok, nchw, N, C, H, W)
    c_out = Conv2d(dev, N, C, H, W, C, 1, 1, 1, 1, 0, 0, bias=True)
    w_out = dev.up(np.ascontiguousarray(t2d.proj_out.weight.detach().numpy()[:, :, 0, 0].astype(np.float16)))
    b_out = dev.up(np.ascontiguousarray(t2d.proj_out.bias.detach().numpy().astype(np.float16)))
    po = c_out.forward(nchw, w_out, b_out)
    pon = buf(N * C * H * W)
    _nhwc2nchw(dev, po, pon, N, C, H, W)
    res = buf(N * C * H * W)
    GK(dev)["add_h"](dev.q, (N * C * H * W,), None, pon, x, res, np.int64(N * C * H * W))
    dev.q.finish()
    out = dev.down(res, (N, C, H, W)).astype(np.float32)
    return out, out5


def check(seed=0, N=1, C=320, H=8, W=8, idx=0):
    import torch
    from diffusers import UNet2DConditionModel

    torch.manual_seed(seed)
    u = UNet2DConditionModel.from_pretrained(r"D:\work\textmodel\_sd15_combined\unet")
    u.eval()
    t2d = u.down_blocks[0].attentions[idx]
    D = u.config.cross_attention_dim
    x = torch.randn(N, C, H, W)
    enc = torch.randn(N, 77, D)

    # torch 侧逐中间量
    with torch.no_grad():
        h = t2d.norm(x)
        h = t2d.proj_in(h)
        B, Cc, Hh, Ww = h.shape
        tok = h.flatten(2).transpose(1, 2)
        blk = t2d.transformer_blocks[0]
        t1 = blk.norm1(tok)
        tok = tok + blk.attn1(t1).reshape(B, -1, Cc) if False else tok
        # 用最朴素的方式重算：直接调 diffusers 的 Attention
        a1 = blk.attn1(blk.norm1(tok))[0]
        ref_self = tok + a1
        a2 = blk.attn2(blk.norm2(ref_self), encoder_hidden_states=enc)[0]
        ref_cross = ref_self + a2
        ff = blk.ff(blk.norm3(ref_cross))
        ref_ff = ref_cross + ff
        ref_out = t2d.proj_out(ref_ff.transpose(1, 2).reshape(B, Cc, Hh, Ww)) + x

    dev = CTX.DevCtx(arena_mb=2000, name="t2d")
    dev.activate()
    out, mid = dev_transformer2d(dev, t2d, x.numpy(), enc.numpy())
    rel = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    rs = rel(mid["self"], ref_self.numpy())
    rc = rel(mid["cross"], ref_cross.numpy())
    rf = rel(mid["ff"], ref_ff.numpy())
    ro = rel(out, ref_out.numpy())
    print("  T2D 设备侧: self %.2e · cross %.2e · ff %.2e · out %.2e" % (rs, rc, rf, ro))
    return max(rs, rc, rf, ro) < 3e-2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        print(__doc__)
        return 0
    ok = check()
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
