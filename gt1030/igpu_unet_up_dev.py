# -*- coding: utf-8 -*-
"""igpu_unet_up_dev.py — 设备侧 up_block / mid_block（含 arena 峰值记录）。

up_block 要点（SD1.5）：
  · 每个 resnet 前：`skip = skips.pop()`（**从末尾取，深→浅**），`cat([hidden, skip])`（hidden 在前）；
  · resnet 也带 temb；
  · 有 attention 的层再走 Transformer2D；
  · 末端 upsample = **nearest ×2 + 3×3 conv**（不是 transpose）。

    python gt1030/igpu_unet_up_dev.py --check
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.igpu_ctx as CTX  # noqa: E402
from gt1030.igpu_cat import cat2  # noqa: E402
from gt1030.igpu_conv import Conv2d, K as CVK  # noqa: E402
from gt1030.igpu_unet_dev import dev_resnet  # noqa: E402
from gt1030.igpu_unet_t2d_dev import dev_transformer2d  # noqa: E402


def _mb(dev):
    return dev._off / 2 ** 20


def dev_upsample(dev, us, x_np):
    """nearest ×2 → 3×3 conv。"""
    N, C, H, W = x_np.shape
    N2, C2, H2, W2 = N, C, H * 2, W * 2
    xb = dev.up(np.asarray(x_np, np.float16))
    up = dev.buf(max(4096, N2 * C2 * H2 * W2 * 2))
    CVK(dev)["upsample2x"](dev.q, (N2 * C2 * H2 * W2,), None, xb, up,
                           np.int32(N), np.int32(C), np.int32(H), np.int32(W))
    conv = Conv2d(dev, N2, C2, H2, W2, us.conv.out_channels, 3, 3, 1, 1, 1, 1, bias=True)
    w = dev.up(np.ascontiguousarray(us.conv.weight.detach().numpy().astype(np.float16)))
    b = dev.up(np.ascontiguousarray(us.conv.bias.detach().numpy().astype(np.float16)))
    y = conv.forward(up, w, b)
    out = dev.buf(max(4096, N2 * us.conv.out_channels * H2 * W2 * 2))
    CVK(dev)["nhwc2nchw"](dev.q, (N2 * us.conv.out_channels * H2 * W2,), None, y, out,
                          np.int32(N2), np.int32(us.conv.out_channels), np.int32(H2), np.int32(W2))
    dev.q.finish()
    return dev.down(out, (N2, us.conv.out_channels, H2, W2)).astype(np.float32)


def dev_cat_skip(dev, h_np, skip_np):
    """通道拼接：h 在前、skip 在后（diffusers torch.cat([hidden, res])）。"""
    N, C1, H, W = h_np.shape
    C2 = skip_np.shape[1]
    a = dev.up(np.asarray(h_np, np.float16))
    b = dev.up(np.asarray(skip_np, np.float16))
    dst = dev.buf(max(4096, N * (C1 + C2) * H * W * 2))
    cat2(dev, dst, a, b, N, C1, C2, H, W)
    dev.q.finish()
    return dev.down(dst, (N, C1 + C2, H, W)).astype(np.float32)


def dev_up_block(dev, ub, x_np, temb_np, enc_np, skips):
    h = x_np
    atts = getattr(ub, "attentions", None) or ()
    peaks = [_mb(dev)]
    for i, res in enumerate(ub.resnets):
        skip = skips.pop()
        h = dev_cat_skip(dev, h, skip)
        h, _ = dev_resnet(dev, res, h, temb_np)
        if i < len(atts):
            h, _ = dev_transformer2d(dev, atts[i], h, enc_np)
        peaks.append(_mb(dev))
    ups = getattr(ub, "upsamplers", None)
    if ups is not None and len(ups) > 0:
        h = dev_upsample(dev, ups[0], h)
        peaks.append(_mb(dev))
    return h, peaks


def dev_mid(dev, mid, x_np, temb_np, enc_np):
    h, _ = dev_resnet(dev, mid.resnets[0], x_np, temb_np)
    h, _ = dev_transformer2d(dev, mid.attentions[0], h, enc_np)
    h, _ = dev_resnet(dev, mid.resnets[1], h, temb_np)
    return h, _mb(dev)


def check_up(seed=0, N=1, H=8, W=8):
    """up_blocks[3]（最深层，3 个 resnet，无 attention、无 upsample）。"""
    import torch
    from diffusers import UNet2DConditionModel

    torch.manual_seed(seed)
    u = UNet2DConditionModel.from_pretrained(r"D:\work\textmodel\_sd15_combined\unet")
    u.eval()
    ub = u.up_blocks[3]
    # up[3] 的输入通道 = 上一层输出（640），skip 是 320，cat 后 960（与 resnets[0].norm1 对齐）
    Cin = ub.resnets[0].norm1.num_channels - u.config.block_out_channels[0]
    Cskip = u.config.block_out_channels[0]
    D = u.config.cross_attention_dim
    x = torch.randn(N, Cin, H, W)
    temb = torch.randn(N, u.config.block_out_channels[-1])
    enc = torch.randn(N, 77, D)
    skips = [torch.randn(N, Cskip, H, W) for _ in range(len(ub.resnets))]
    with torch.no_grad():
        ref = ub(x, tuple(skips), temb, encoder_hidden_states=enc)
    dev = CTX.DevCtx(arena_mb=3000, name="up3")
    dev.activate()
    mine, peaks = dev_up_block(dev, ub, x.numpy(), temb.numpy(), enc.numpy(),
                               [s.numpy().copy() for s in skips])
    rel = float(np.linalg.norm(mine - ref.numpy()) / np.linalg.norm(ref.numpy()))
    print("  up[3]: out %.2e · arena 峰值 %.0f MB" % (rel, max(peaks)))
    return rel < 3e-2


def check_mid(seed=0, N=1, H=8, W=8):
    import torch
    from diffusers import UNet2DConditionModel

    torch.manual_seed(seed)
    u = UNet2DConditionModel.from_pretrained(r"D:\work\textmodel\_sd15_combined\unet")
    u.eval()
    mid = u.mid_block
    print("  mid 结构: resnets=%d attentions=%d · attn block 子模块 %s"
          % (len(mid.resnets), len(mid.attentions),
             [n for n, _ in mid.attentions[0].transformer_blocks[0].named_children()]))
    C = u.config.block_out_channels[-1]
    D = u.config.cross_attention_dim
    x = torch.randn(N, C, H, W)
    temb = torch.randn(N, u.config.block_out_channels[-1])
    enc = torch.randn(N, 77, D)
    with torch.no_grad():
        ref = mid(x, temb, encoder_hidden_states=enc)
    dev = CTX.DevCtx(arena_mb=3000, name="mid")
    dev.activate()
    mine, peak = dev_mid(dev, mid, x.numpy(), temb.numpy(), enc.numpy())
    rel = float(np.linalg.norm(mine - ref.numpy()) / np.linalg.norm(ref.numpy()))
    print("  mid: out %.2e · arena 峰值 %.0f MB" % (rel, peak))
    return rel < 3e-2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        print(__doc__)
        return 0
    ok = check_up()
    ok &= check_mid()
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
