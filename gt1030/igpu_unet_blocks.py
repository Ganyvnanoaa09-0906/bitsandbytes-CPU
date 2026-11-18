# -*- coding: utf-8 -*-
"""igpu_unet_blocks.py — UNet 各块的**手写前向**（先做 torch 侧等价，再上设备）。

为什么要手写：diffusers 的 `ResnetBlock2D` / `Transformer2DModel` 内部有大量控制流
（`time_embedding_norm` 分支、`chunk`、`upsample` 判断），`symbolic_trace` 过不了。

**与 VAE resnet 的关键差异（别直接复用）**：
  · UNet resnet 有 `time_emb_proj(temb)`，且它加在 **conv1 之后**（不是加在 norm1 之后）；
  · `temb` 要先过一遍 `nonlinearity`（SiLU）再进 `time_emb_proj`；
  · 末端是 `(shortcut + h) / output_scale_factor`。

    python gt1030/igpu_unet_blocks.py --check-resnet
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

UNET = r"D:\work\textmodel\_sd15_combined\unet"


def resnet_forward(r, x, temb):
    """等价于 diffusers `ResnetBlock2D.forward(x, temb)`（SD1.5 的配置，非 up/down 采样）。"""
    h = r.norm1(x)
    h = r.nonlinearity(h)
    if getattr(r, "conv_shortcut", None) is not None:
        inp = r.conv_shortcut(x)
    else:
        inp = x
    h = r.conv1(h)

    if temb is not None and getattr(r, "time_emb_proj", None) is not None:
        t = r.nonlinearity(temb)
        t = r.time_emb_proj(t)[:, :, None, None]
        h = h + t                      # ★ 加在 conv1 之后
    h = r.norm2(h)
    h = r.nonlinearity(h)
    if getattr(r, "dropout", None) is not None:
        h = r.dropout(h)
    h = r.conv2(h)
    return (inp + h) / r.output_scale_factor


def check_resnet(seed=0):
    from diffusers import UNet2DConditionModel

    torch.manual_seed(seed)
    unet = UNet2DConditionModel.from_pretrained(UNET)
    unet.eval()
    r = unet.down_blocks[0].resnets[0]
    C = unet.config.block_out_channels[0]
    x = torch.randn(2, C, 16, 16)
    temb = torch.randn(2, unet.config.block_out_channels[-1])
    with torch.no_grad():
        ref = r(x, temb)
        mine = resnet_forward(r, x, temb)
    rel = float((mine - ref).norm() / ref.norm())
    # 反例：把 temb 去掉，看误差是否明显变大（证明 temb 确实参与）
    with torch.no_grad():
        no_t = resnet_forward(r, x, None)
    rel_no = float((no_t - ref).norm() / ref.norm())
    print("  resnet(UNet, 含 temb): rel %.2e   （去掉 temb 后 %.2e，应显著变大）" % (rel, rel_no))
    return rel < 1e-6 and rel_no > 1e-3




def _sdpa_attn(a, h, enc=None):
    """token 布局 (B,T,C) 上的多头注意力（等价 diffusers AttnProcessor2_0）。"""
    B, T, C = h.shape
    q = a.to_q(h)
    k = a.to_k(h if enc is None else enc)
    v = a.to_v(h if enc is None else enc)
    heads = a.heads
    dh = C // heads

    def split(x):
        return x.reshape(x.shape[0], x.shape[1], heads, dh).permute(0, 2, 1, 3)

    o = torch.nn.functional.scaled_dot_product_attention(split(q), split(k), split(v))
    o = o.permute(0, 2, 1, 3).reshape(B, q.shape[1], C)
    o = a.to_out[0](o)
    if len(a.to_out) > 1 and a.to_out[1] is not None:
        o = a.to_out[1](o)
    return o


def _geglu_ff(ff, h):
    """FeedForward: net[0]=GEGLU（Linear→2C 再门控）, net[1]=Dropout, net[2]=Linear。"""
    return ff.net[2](ff.net[1](ff.net[0](h)))


def transformer2d_forward(t, x, enc):
    """diffusers Transformer2DModel.forward（SD1.5）：norm→proj_in→token→blocks→proj_out+残差。"""
    residual = x
    h = t.norm(x)
    h = t.proj_in(h)
    B, C, H, W = h.shape
    h = h.flatten(2).transpose(1, 2)
    for blk in t.transformer_blocks:
        h = h + _sdpa_attn(blk.attn1, blk.norm1(h))
        h = h + _sdpa_attn(blk.attn2, blk.norm2(h), enc)
        h = h + _geglu_ff(blk.ff, blk.norm3(h))
    h = h.transpose(1, 2).reshape(B, C, H, W)
    h = t.proj_out(h)
    return h + residual


def check_transformer2d(seed=0):
    from diffusers import UNet2DConditionModel

    torch.manual_seed(seed)
    unet = UNet2DConditionModel.from_pretrained(UNET)
    unet.eval()
    t = unet.down_blocks[0].attentions[0]
    C = unet.config.block_out_channels[0]
    L, D = 77, unet.config.cross_attention_dim
    x = torch.randn(2, C, 16, 16)
    enc = torch.randn(2, L, D)
    with torch.no_grad():
        ref = t(x, encoder_hidden_states=enc).sample
        mine = transformer2d_forward(t, x, enc)
    rel = float((mine - ref).norm() / ref.norm())
    with torch.no_grad():
        bad = transformer2d_forward(t, x, torch.zeros_like(enc))
    rel_bad = float((bad - ref).norm() / ref.norm())
    print("  Transformer2D(self+cross+FF): rel %.2e   （enc 置零后 %.2e，应显著变大）"
          % (rel, rel_bad))
    return rel < 1e-6 and rel_bad > 1e-3



def downsample_forward(ds, x):
    """Downsample2D（use_conv）：`padding == 0` 时才右下手动 pad 1 像素，否则靠 conv 自带 padding。"""
    if getattr(ds, "padding", 0) == 0:
        x = torch.nn.functional.pad(x, (0, 1, 0, 1))
    return ds.conv(x)


def down_block_forward(db, x, temb, enc):
    """CrossAttnDownBlock2D：逐层 resnet→attention，第 0 层输出一个 skip，末端 downsample。"""
    skips = []
    h = x
    atts = getattr(db, "attentions", None) or ()
    for i, res in enumerate(db.resnets):
        h = resnet_forward(res, h, temb)
        if i < len(atts):
            h = transformer2d_forward(atts[i], h, enc)
        skips.append(h)          # 每层输出都进 skip（diffusers 实测返回 3 个：层0、层1、下采样后）
    if getattr(db, "downsamplers", None) and len(db.downsamplers) > 0:
        h = downsample_forward(db.downsamplers[0], h)
        skips.append(h)
    return h, skips


def check_down_block0(seed=0):
    from diffusers import UNet2DConditionModel

    torch.manual_seed(seed)
    unet = UNet2DConditionModel.from_pretrained(UNET)
    unet.eval()
    db = unet.down_blocks[0]
    C = unet.config.block_out_channels[0]
    D = unet.config.cross_attention_dim
    x = torch.randn(2, C, 16, 16)
    temb = torch.randn(2, unet.config.block_out_channels[-1])
    enc = torch.randn(2, 77, D)
    with torch.no_grad():
        out = db(x, temb, encoder_hidden_states=enc)
        ref_h, ref_skips = out[0], out[1]
        mine_h, mine_skips = down_block_forward(db, x, temb, enc)
    rel_h = float((mine_h - ref_h).norm() / ref_h.norm())
    rel_s = [float((a - b).norm() / b.norm()) for a, b in zip(mine_skips, ref_skips)]
    print("  down_blocks[0]: out %.2e · skip0 %.2e · skip_last %.2e · shape %s -> %s"
          % (rel_h, rel_s[0], rel_s[-1], tuple(x.shape), tuple(mine_h.shape)))
    return rel_h < 1e-6 and max(rel_s) < 1e-6



def upsample_forward(us, x):
    """Upsample2D（SD1.5：use_conv=True, use_conv_transpose=False）：nearest ×2 再 3x3 卷积。"""
    x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
    return us.conv(x)


def mid_forward(mid, h, temb, enc):
    """UNetMidBlock2DCrossAttn：resnet → (self+cross) attention → resnet。"""
    h = resnet_forward(mid.resnets[0], h, temb)
    h = transformer2d_forward(mid.attentions[0], h, enc)
    h = resnet_forward(mid.resnets[1], h, temb)
    return h


def up_block_forward(db, h, temb, enc, skips):
    """CrossAttnUpBlock2D：**从 skips 末尾 pop**，先 cat([h, skip]) 再 resnet，最后 upsample。"""
    for i, res in enumerate(db.resnets):
        skip = skips.pop()                      # ★ 从深到浅
        h = torch.cat([h, skip], dim=1)         # ★ hidden 在前、skip 在后
        h = resnet_forward(res, h, temb)
        atts = getattr(db, "attentions", None)
        if atts is not None and len(atts) > 0:
            h = transformer2d_forward(atts[i], h, enc)
    ups = getattr(db, "upsamplers", None)
    if ups is not None and len(ups) > 0:
        h = upsample_forward(ups[0], h)
    return h


def unet_forward(u, sample, timestep, enc):
    """整条 UNet2DConditionModel 前向（SD1.5 路径）。"""
    h = u.conv_in(sample)
    temb = u.time_proj(timestep)
    temb = u.time_embedding(temb)
    # ★ diffusers 的 skip 栈以 conv_in 输出开头：down_block_res_samples = (sample,)
    #   down 路径共 11 个 + 这 1 个 = 12，正好等于 up 路径要 pop 的 12 个。
    skips = [h]
    for db in u.down_blocks:
        h, sk = down_block_forward(db, h, temb, enc)
        skips.extend(sk)
    h = mid_forward(u.mid_block, h, temb, enc)
    for ub in u.up_blocks:
        h = up_block_forward(ub, h, temb, enc, skips)
    if len(skips) != 0:
        raise RuntimeError("skip 没用完，剩 %d 个" % len(skips))
    h = u.conv_norm_out(h)
    h = u.conv_act(h)
    h = u.conv_out(h)
    return h


def check_mid_and_up(seed=0):
    from diffusers import UNet2DConditionModel

    torch.manual_seed(seed)
    u = UNet2DConditionModel.from_pretrained(UNET)
    u.eval()
    C, D = u.config.block_out_channels[-1], u.config.cross_attention_dim
    temb = torch.randn(2, u.config.block_out_channels[-1])
    enc = torch.randn(2, 77, D)
    x = torch.randn(2, C, 8, 8)
    with torch.no_grad():
        ref = u.mid_block(x, temb, encoder_hidden_states=enc)
        mine = mid_forward(u.mid_block, x, temb, enc)
    rel_mid = float((mine - ref).norm() / ref.norm())

    print("  mid_block %.2e" % rel_mid)
    return rel_mid < 1e-6


def check_full_unet(seed=0):
    from diffusers import UNet2DConditionModel

    torch.manual_seed(seed)
    u = UNet2DConditionModel.from_pretrained(UNET)
    u.eval()
    sample = torch.randn(1, 4, 64, 64)
    t = torch.tensor([500])
    enc = torch.randn(1, 77, u.config.cross_attention_dim)
    with torch.no_grad():
        ref = u(sample, t, encoder_hidden_states=enc).sample
        mine = unet_forward(u, sample, t, enc)
        bad = unet_forward(u, sample, t, torch.zeros_like(enc))
    rel = float((mine - ref).norm() / ref.norm())
    rel_bad = float((bad - ref).norm() / ref.norm())
    print("  full UNet: out %s rel %.2e   （enc 置零后 %.2e）"
          % (tuple(mine.shape), rel, rel_bad))
    return rel < 1e-6 and rel_bad > 1e-3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-resnet", action="store_true")
    ap.add_argument("--check-t2d", action="store_true")
    ap.add_argument("--check-down0", action="store_true")
    ap.add_argument("--check-mid", action="store_true")
    ap.add_argument("--check-full", action="store_true")
    a = ap.parse_args()
    if not (a.check_resnet or a.check_t2d or a.check_down0
            or a.check_mid or a.check_full):
        print(__doc__)
        return 0
    ok = True
    if a.check_resnet:
        ok &= check_resnet()
    if a.check_t2d:
        ok &= check_transformer2d()
    if a.check_down0:
        ok &= check_down_block0()
    if a.check_mid:
        ok &= check_mid_and_up()
    if a.check_full:
        ok &= check_full_unet()
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
