# -*- coding: utf-8 -*-
"""igpu_unet_full_dev.py — **整链走设备**（中间张量不落 host）+ 逐级 arena 水位。

块级测试每个 `dev.down/up` 都会把中间张量搬回 host 再量化一次 fp16，链越长误差越大
（up/mid 已到 2.2e-2）。整链必须保证：
  · hidden / skip **全程留 device**；
  · skip 拼接用 device `cat2`（不 host cat）；
  · 只在最后 `dev.down(out)` 一次。

    python gt1030/igpu_unet_full_dev.py --check     # 对拍 + 逐级水位
    python gt1030/igpu_unet_full_dev.py --peak      # 只测水位
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.igpu_ctx as CTX  # noqa: E402
from gt1030.igpu_attn_chunk import attn_chunked_heads  # noqa: E402
from gt1030.igpu_cat import cat2  # noqa: E402
from gt1030.igpu_conv import Conv2d, K as CVK  # noqa: E402
from gt1030.igpu_gn import GroupNorm  # noqa: E402
from gt1030.igpu_graph import _kernels as GK  # noqa: E402
from gt1030.igpu_unet_t2d_dev import K as T2DK  # noqa: E402
from gt1030.igpu_unet_t2d_dev import layernorm, token_linear  # noqa: E402

STAGES = []


def _mb(dev):
    """历史峰值（不是活跃值）：量水位要用 peak。"""
    return max(dev._peak, dev._off) / 2 ** 20


def _probe(dev, name, buf, shape):
    import os

    if not os.environ.get("DSH_PROBE"):
        return
    arr = dev.down(buf, shape).astype(np.float32)
    print("    [probe] %-8s mean %+.5f std %.5f max %.5f"
          % (name, arr.mean(), arr.std(), np.abs(arr).max()))


def _free(dev, *bufs):
    """释放；同一对象只释放一次（up 的第一个 skip 可能与 mid 输出是同一缓冲）。"""
    seen = set()
    for b in bufs:
        if b is not None and id(b) not in seen:
            seen.add(id(b))
            dev.release(b)


def _note(dev, name):
    STAGES.append((name, _mb(dev)))


def _buf(dev, n):
    return dev.buf(max(4096, int(n) * 2))


def _up_h(dev, arr):
    a = np.ascontiguousarray(arr, np.float16)
    b = dev.buf(a.nbytes)
    cl.enqueue_copy(dev.q, b, a)
    return b


class W(object):
    """权重/常量上传缓存（同一模块只传一次）。"""

    def __init__(self, dev):
        self.dev, self.c = dev, {}

    def lin(self, m):
        k = id(m)
        if k not in self.c:
            self.c[k] = (_up_h(self.dev, m.weight.detach().numpy()),
                         None if m.bias is None else _up_h(self.dev, m.bias.detach().numpy()))
        return self.c[k]

    def arr(self, t):
        k = id(t)
        if k not in self.c:
            self.c[k] = _up_h(self.dev, t.detach().numpy())
        return self.c[k]


def _gn(dev, x, g, b, y, N, C, H, Wd, groups, eps):
    m = GroupNorm(dev, N, C, H, Wd, groups, eps=eps)
    cl.enqueue_copy(dev.q, m.gamma, g)
    cl.enqueue_copy(dev.q, m.beta, b)
    m.forward(x, y)
    return y


def _conv(dev, x, w, b, N, C, H, Wd, OC, kh=3, kw=3, sh=1, sw=1, ph=1, pw=1):
    cv = Conv2d(dev, N, C, H, Wd, OC, kh, kw, sh, sw, ph, pw, bias=b is not None)
    y = cv.forward(x, w, b)
    OH, OW = cv.OH, cv.OW
    out = _buf(dev, N * OC * OH * OW)
    CVK(dev)["nhwc2nchw"](dev.q, (N * OC * OH * OW,), None, y, out,
                          np.int32(N), np.int32(OC), np.int32(OH), np.int32(OW))
    # ★ 释放 Conv2d 内部缓冲（cols 是 im2col 的 M×Kdim，整链 ~30 个卷积时是水位大头）
    _free(dev, getattr(cv, "cols", None), getattr(cv, "y", None),
          getattr(cv, "dy", None), getattr(cv, "dcols", None),
          getattr(cv, "dw", None), getattr(cv, "db", None))
    return out, OH, OW


def _n2h(dev, x, N, C, H, Wd):
    out = _buf(dev, N * C * H * Wd)
    CVK(dev)["nchw2nhwc"](dev.q, (N * C * H * Wd,), None, x, out,
                          np.int32(N), np.int32(C), np.int32(H), np.int32(Wd))
    return out


def _h2n(dev, x, N, C, H, Wd):
    out = _buf(dev, N * C * H * Wd)
    CVK(dev)["nhwc2nchw"](dev.q, (N * C * H * Wd,), None, x, out,
                          np.int32(N), np.int32(C), np.int32(H), np.int32(Wd))
    return out


def _silu(dev, x, n):
    out = _buf(dev, n)
    T2DK(dev)["silu_fwd"](dev.q, (n,), None, x, out, np.int64(n))
    return out


def _add(dev, a, b, n):
    out = _buf(dev, n)
    GK(dev)["add_h"](dev.q, (n,), None, a, b, out, np.int64(n))
    return out


def _ln(dev, x, m, rows, C, wc):
    g, b = wc.lin(m)
    out = _buf(dev, rows * C)
    layernorm(dev, x, g, b, out, rows, C, float(m.eps))
    return out


def _attn(dev, h, ctx, a, N, Tq, Tk, C, wc):
    wq, bq = wc.lin(a.to_q)
    wk, bk = wc.lin(a.to_k)
    wv, bv = wc.lin(a.to_v)
    wo, bo = wc.lin(a.to_out[0])
    q = _buf(dev, N * Tq * C)
    token_linear(dev, h, wq, bq, q, N * Tq, C, C)
    src = h if ctx is None else ctx
    k = _buf(dev, N * Tk * C)
    v = _buf(dev, N * Tk * C)
    ck = a.to_k.in_features
    token_linear(dev, src, wk, bk, k, N * Tk, C, ck)
    token_linear(dev, src, wv, bv, v, N * Tk, C, ck)
    ao = _buf(dev, N * Tq * C)
    attn_chunked_heads(dev, q, k, v, ao, N, Tq, Tk, C, a.heads, bq=min(256, Tq))
    _free(dev, q, k, v)
    out = _buf(dev, N * Tq * C)
    token_linear(dev, ao, wo, bo, out, N * Tq, C, C)
    _free(dev, ao)
    return out


def _resnet(dev, r, x, temb, tembD, N, C, H, Wd, wc):
    y = _buf(dev, N * C * H * Wd)
    _gn(dev, x, wc.arr(r.norm1.weight), wc.arr(r.norm1.bias), y,
        N, C, H, Wd, r.norm1.num_groups, float(r.norm1.eps))
    s = _silu(dev, y, N * C * H * Wd)
    _free(dev, y)
    w1, b1 = wc.lin(r.conv1)
    C2 = r.conv1.out_channels
    c1, _, _ = _conv(dev, s, w1, b1, N, C, H, Wd, C2)
    _free(dev, s)
    if temb is not None and r.time_emb_proj is not None:
        ts = _silu(dev, temb, N * tembD)
        tw, tb = wc.lin(r.time_emb_proj)
        tp = _buf(dev, N * C2)
        token_linear(dev, ts, tw, tb, tp, N, C2, r.time_emb_proj.in_features)
        c1h = _n2h(dev, c1, N, C2, H, Wd)
        CVK(dev)["bias_add_oc"](dev.q, (N * H * Wd, C2), None, c1h, tp,
                                np.int32(N * H * Wd), np.int32(C2))
        c1n = _h2n(dev, c1h, N, C2, H, Wd)
        _free(dev, c1, c1h, tp, ts)
        c1 = c1n
    y2 = _buf(dev, N * C2 * H * Wd)
    _gn(dev, c1, wc.arr(r.norm2.weight), wc.arr(r.norm2.bias), y2,
        N, C2, H, Wd, r.norm2.num_groups, float(r.norm2.eps))
    s2 = _silu(dev, y2, N * C2 * H * Wd)
    _free(dev, y2, c1)
    w2, b2 = wc.lin(r.conv2)
    c2, _, _ = _conv(dev, s2, w2, b2, N, C2, H, Wd, C2)
    _free(dev, s2)
    if r.conv_shortcut is not None:
        ws, bs = wc.lin(r.conv_shortcut)
        sc, _, _ = _conv(dev, x, ws, bs, N, C, H, Wd, C2, 1, 1, 1, 1, 0, 0)
        out = _add(dev, sc, c2, N * C2 * H * Wd)
        _free(dev, sc, c2)
        return out, C2
    out = _add(dev, x, c2, N * C2 * H * Wd)
    _free(dev, c2)
    return out, C2


def _t2d(dev, t, x, enc, N, C, H, Wd, rk, wc):
    y = _buf(dev, N * C * H * Wd)
    _gn(dev, x, wc.arr(t.norm.weight), wc.arr(t.norm.bias), y,
        N, C, H, Wd, t.norm.num_groups, float(t.norm.eps))
    wi, bi = wc.lin(t.proj_in)
    c1, _, _ = _conv(dev, y, wi, bi, N, C, H, Wd, C, 1, 1, 1, 1, 0, 0)
    _free(dev, y)
    tok = _n2h(dev, c1, N, C, H, Wd)
    _free(dev, c1)
    T = H * Wd
    for blk in t.transformer_blocks:
        s = _ln(dev, tok, blk.norm1, N * T, C, wc)
        a1 = _attn(dev, s, None, blk.attn1, N, T, T, C, wc)
        _free(dev, s)
        nxt = _add(dev, tok, a1, N * T * C)
        _free(dev, tok, a1)
        tok = nxt
        s = _ln(dev, tok, blk.norm2, N * T, C, wc)
        a2 = _attn(dev, s, enc, blk.attn2, N, T, rk, C, wc)
        _free(dev, s)
        nxt = _add(dev, tok, a2, N * T * C)
        _free(dev, tok, a2)
        tok = nxt
        s = _ln(dev, tok, blk.norm3, N * T, C, wc)
        w0, b0 = wc.lin(blk.ff.net[0].proj)
        hd = blk.ff.net[0].proj.out_features // 2
        hid = _buf(dev, N * T * 2 * hd)
        token_linear(dev, s, w0, b0, hid, N * T, 2 * hd, C)
        gated = _buf(dev, N * T * hd)
        T2DK(dev)["geglu"](dev.q, (N * T * hd,), None, hid, gated, np.int32(N * T), np.int32(hd))
        w2, b2 = wc.lin(blk.ff.net[2])
        ff = _buf(dev, N * T * C)
        token_linear(dev, gated, w2, b2, ff, N * T, C, hd)
        _free(dev, s, hid, gated)
        nxt = _add(dev, tok, ff, N * T * C)
        _free(dev, tok, ff)
        tok = nxt
    back = _h2n(dev, tok, N, C, H, Wd)
    _free(dev, tok)
    wo_, bo_ = wc.lin(t.proj_out)
    po, _, _ = _conv(dev, back, wo_, bo_, N, C, H, Wd, C, 1, 1, 1, 1, 0, 0)
    _free(dev, back)
    out = _add(dev, po, x, N * C * H * Wd)
    _free(dev, po)
    return out


def _cat(dev, h, skip, N, C1, C2, H, Wd):
    dst = _buf(dev, N * (C1 + C2) * H * Wd)
    cat2(dev, dst, h, skip, N, C1, C2, H, Wd)
    return dst


def _upsample(dev, us, x, N, C, H, Wd, wc):
    up = _buf(dev, N * C * H * Wd * 4)
    CVK(dev)["upsample2x"](dev.q, (N * C * H * Wd * 4,), None, x, up,
                           np.int32(N), np.int32(C), np.int32(H), np.int32(Wd))
    w, b = wc.lin(us.conv)
    out, _, _ = _conv(dev, up, w, b, N, C, H, Wd, us.conv.out_channels)
    return out


def full_forward(dev, u, sample_np, timestep_np, enc_np, wc=None, verbose=True):
    wc = wc or W(dev)
    N = sample_np.shape[0]
    D = u.config.block_out_channels[-1]
    STAGES.clear()

    x = _up_h(dev, sample_np)
    wci, bci = wc.lin(u.conv_in)
    OCi = u.conv_in.out_channels   # ★ 320，不是 block_out_channels[-1]
    h, H, Wd = _conv(dev, x, wci, bci, N, 4, sample_np.shape[2], sample_np.shape[3], OCi, 3, 3, 1, 1, 1, 1)
    _note(dev, "conv_in")

    t = _up_h(dev, timestep_np.astype(np.float32))
    # sinusoidal（沿用已验证实现）
    from gt1030.igpu_time_emb import sinusoidal

    tdim = u.config.block_out_channels[0]   # sinusoidal 维度 = 首层通道（SD1.5: 320）
    tb = _buf(dev, N * tdim)
    sinusoidal(dev, t, tb, N, tdim)
    tw1, tb1 = wc.lin(u.time_embedding.linear_1)
    te1 = _buf(dev, N * u.time_embedding.linear_2.out_features)
    token_linear(dev, tb, tw1, tb1, te1, N, u.time_embedding.linear_1.out_features, tdim)
    ts = _silu(dev, te1, N * u.time_embedding.linear_1.out_features)
    tw2, tb2 = wc.lin(u.time_embedding.linear_2)
    temb = _buf(dev, N * D)
    token_linear(dev, ts, tw2, tb2, temb, N, D, u.time_embedding.linear_2.in_features)
    tembD = D
    _note(dev, "time_emb")

    rk = enc_np.shape[1]
    enc = _up_h(dev, enc_np)
    skips = [h]
    c = OCi
    for i, db in enumerate(u.down_blocks):
        atts = getattr(db, "attentions", None) or ()
        for j, res in enumerate(db.resnets):
            # ★ down 每层的输出都会进 skips（up 要用），所以这里**不释放**上一层张量；
            #   只有 resnet/t2d 内部的临时缓冲会被它们各自释放。
            h, c = _resnet(dev, res, h, temb, tembD, N, c, H, Wd, wc)
            if j < len(atts):
                h = _t2d(dev, atts[j], h, enc, N, c, H, Wd, rk, wc)
            skips.append(h)
        ups = getattr(db, "downsamplers", None)
        if ups is not None and len(ups) > 0:
            w, b = wc.lin(ups[0].conv)
            nxt, H, Wd = _conv(dev, h, w, b, N, c, H, Wd, ups[0].conv.out_channels, 3, 3, 2, 2, 1, 1)
            # ★ 不能释放 h：它（pre-downsample）和 nxt（post-downsample）**都是 skip**，up 还要用
            h = nxt
            c = ups[0].conv.out_channels
            skips.append(h)
        _probe(dev, "down%d" % i, h, (N, c, H, Wd))
        _note(dev, "down%d" % i)

    mid = u.mid_block
    # ★ 这里的输入 h 同时是 skips 的最后一个元素（up 还要 pop 它），不能释放
    h, c = _resnet(dev, mid.resnets[0], h, temb, tembD, N, c, H, Wd, wc)
    prev = h
    h = _t2d(dev, mid.attentions[0], prev, enc, N, c, H, Wd, rk, wc)
    _free(dev, prev)
    prev = h
    h, c = _resnet(dev, mid.resnets[1], prev, temb, tembD, N, c, H, Wd, wc)
    _free(dev, prev)
    _probe(dev, "mid", h, (N, c, H, Wd))
    _note(dev, "mid")

    for i, ub in enumerate(u.up_blocks):
        atts = getattr(ub, "attentions", None) or ()
        for j, res in enumerate(ub.resnets):
            skip = skips.pop()
            # ★ 通道数用缓冲大小反推：up 里 hidden 的通道数不一定等于 res.conv2.out_channels，
            #   用 in-out 之差会在深层算错，导致 conv 的 Kdim 与实际权重不符（CLBlast -1010）
            Cskip = skip.size // 2 // (N * H * Wd)
            cat = _cat(dev, h, skip, N, c, Cskip, H, Wd)
            _free(dev, skip, h)          # skip 在 cat2 之后才释放
            c = c + Cskip
            h, c = _resnet(dev, res, cat, temb, tembD, N, c, H, Wd, wc)
            _free(dev, cat)
            if j < len(atts):
                prev = h
                h = _t2d(dev, atts[j], prev, enc, N, c, H, Wd, rk, wc)
                _free(dev, prev)
        ups = getattr(ub, "upsamplers", None)
        if ups is not None and len(ups) > 0:
            prev = h
            h = _upsample(dev, ups[0], prev, N, c, H, Wd, wc)
            _free(dev, prev)
            H, Wd = H * 2, Wd * 2
            c = ups[0].conv.out_channels
        _probe(dev, "up%d" % i, h, (N, c, H, Wd))
        _note(dev, "up%d" % i)

    import os as _os

    def _stat(name, buf, shape):
        if not _os.environ.get("DSH_DEBUG_TAIL"):
            return
        arr = dev.down(buf, shape).astype(np.float32)
        print("    [tail] %-4s mean %.5f std %.5f max %.5f" % (name, arr.mean(), arr.std(), np.abs(arr).max()))

    _stat("h", h, (N, c, H, Wd))
    y = _buf(dev, N * c * H * Wd)
    _gn(dev, h, wc.arr(u.conv_norm_out.weight), wc.arr(u.conv_norm_out.bias), y,
        N, c, H, Wd, u.conv_norm_out.num_groups, float(u.conv_norm_out.eps))
    _stat("y_gn", y, (N, c, H, Wd))
    s = _silu(dev, y, N * c * H * Wd)
    _stat("s", s, (N, c, H, Wd))
    wco, bco = wc.lin(u.conv_out)
    out, _, _ = _conv(dev, s, wco, bco, N, c, H, Wd, u.conv_out.out_channels)
    _stat("out", out, (N, u.conv_out.out_channels, H, Wd))
    dev.q.finish()
    _note(dev, "conv_out")
    if verbose:
        for name, mb in STAGES:
            print("    %-10s %7.0f MB" % (name, mb))
    return out, (N, u.conv_out.out_channels, H, Wd)


def check(seed=0, size=64):
    import torch
    from diffusers import UNet2DConditionModel

    torch.manual_seed(seed)
    u = UNet2DConditionModel.from_pretrained(r"D:\work\textmodel\_sd15_combined\unet")
    u.eval()
    sample = torch.randn(1, 4, size, size)
    t = torch.tensor([500])
    enc = torch.randn(1, 77, u.config.cross_attention_dim)
    with torch.no_grad():
        ref = u(sample, t, encoder_hidden_states=enc).sample
    dev = CTX.DevCtx(arena_mb=6000, name="unet")
    dev.activate()
    out, shape = full_forward(dev, u, sample.numpy(), torch.tensor([500.0]).numpy(), enc.numpy())
    ov = dev.down(out, shape).astype(np.float32)
    rel = float(np.linalg.norm(ov - ref.numpy()) / np.linalg.norm(ref.numpy()))
    print("  整链 out %s rel %.2e · arena 峰值 %.0f MB" % (tuple(shape), rel, _mb(dev)))
    return rel < 1e-1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--peak", action="store_true")
    a = ap.parse_args()
    if not (a.check or a.peak):
        print(__doc__)
        return 0
    if a.peak:
        import torch
        from diffusers import UNet2DConditionModel

        u = UNet2DConditionModel.from_pretrained(r"D:\work\textmodel\_sd15_combined\unet")
        u.eval()
        dev = CTX.DevCtx(arena_mb=6000, name="peak")
        dev.activate()
        full_forward(dev, u, np.zeros((1, 4, 64, 64), np.float32),
                     np.array([500.0], np.float32), np.zeros((1, 77, 768), np.float32))
        print("  RESULT peak %.0f MB" % _mb(dev))
        return 0
    ok = check()
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
