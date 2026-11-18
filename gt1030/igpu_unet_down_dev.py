# -*- coding: utf-8 -*-
"""igpu_unet_down_dev.py — 设备侧 down_block 组装（resnet + Transformer2D + downsample）。

按 peer 要求：**抓 3 个 skip（2 层输出 + 下采样后）+ conv_in skip 的 shape/值**，
不只看最终 out；downsample 用 conv 自带 padding=1（SD1.5），不再手动 pad。

    python gt1030/igpu_unet_down_dev.py --check
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402

import gt1030.igpu_ctx as CTX  # noqa: E402
from gt1030.igpu_conv import Conv2d  # noqa: E402
from gt1030.igpu_unet_dev import dev_resnet  # noqa: E402
from gt1030.igpu_unet_t2d_dev import dev_transformer2d  # noqa: E402


def dev_downsample(dev, ds, x_np):
    """Downsample2D（use_conv, padding=1）：直接 3x3 stride2 pad1 卷积。"""
    N, C, H, W = x_np.shape
    C2 = ds.conv.out_channels
    conv = Conv2d(dev, N, C, H, W, C2, 3, 3, 2, 2,
                  int(getattr(ds, "padding", 1)), int(getattr(ds, "padding", 1)), bias=True)
    w = dev.up(np.ascontiguousarray(ds.conv.weight.detach().numpy().astype(np.float16)))
    b = dev.up(np.ascontiguousarray(ds.conv.bias.detach().numpy().astype(np.float16)))
    xb = dev.up(np.asarray(x_np, np.float16))
    y = conv.forward(xb, w, b)
    from gt1030.igpu_conv import K as CVK

    OH, OW = conv.OH, conv.OW
    out = dev.buf(max(4096, N * C2 * OH * OW * 2))
    CVK(dev)["nhwc2nchw"](dev.q, (N * C2 * OH * OW,), None, y, out,
                          np.int32(N), np.int32(C2), np.int32(OH), np.int32(OW))
    dev.q.finish()
    return dev.down(out, (N, C2, OH, OW)).astype(np.float32)


def dev_down_block(dev, db, x_np, temb_np, enc_np):
    """返回 (out_np, skips)。skips 顺序 = 每层输出 + 下采样后（与 diffusers 一致）。"""
    h = x_np
    skips = []
    atts = getattr(db, "attentions", None) or ()
    for i, res in enumerate(db.resnets):
        h, _ = dev_resnet(dev, res, h, temb_np)
        if i < len(atts):
            h, _ = dev_transformer2d(dev, atts[i], h, enc_np)
        skips.append(h.copy())
    ups = getattr(db, "downsamplers", None)
    if ups is not None and len(ups) > 0:
        h = dev_downsample(dev, ups[0], h)
        skips.append(h.copy())
    return h, skips


def check(idx=0, H=16, W=16, seed=0):
    import torch
    from diffusers import UNet2DConditionModel

    torch.manual_seed(seed)
    u = UNet2DConditionModel.from_pretrained(r"D:\work\textmodel\_sd15_combined\unet")
    u.eval()
    db = u.down_blocks[idx]
    C = u.config.block_out_channels[idx]
    D = u.config.cross_attention_dim
    x = torch.randn(1, C, H, W)
    temb = torch.randn(1, u.config.block_out_channels[-1])
    enc = torch.randn(1, 77, D)
    with torch.no_grad():
        out = db(x, temb, encoder_hidden_states=enc)
        ref_h, ref_skips = out[0], out[1]

    dev = CTX.DevCtx(arena_mb=2500, name="down%d" % idx)
    dev.activate()
    mine_h, mine_skips = dev_down_block(dev, db, x.numpy(), temb.numpy(), enc.numpy())
    rel = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    rel_h = rel(mine_h, ref_h.numpy())
    rs = [rel(a, b.numpy()) for a, b in zip(mine_skips, ref_skips)]
    shapes_ok = [tuple(a.shape) == tuple(b.shape) for a, b in zip(mine_skips, ref_skips)]
    print("  down[%d] out %.2e · skips %s · shape 对齐 %s"
          % (idx, rel_h, ["%.1e" % v for v in rs], all(shapes_ok)))
    return rel_h < 3e-2 and max(rs) < 3e-2 and all(shapes_ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        print(__doc__)
        return 0
    ok = check(0)
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
