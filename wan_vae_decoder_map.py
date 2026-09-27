# -*- coding: utf-8 -*-
"""wan_vae_decoder_map.py — 只映射 Wan VAE 的 **decoder**（够用来把 latent 解成像素）

范围收敛（务实）:
    采样循环只需要 **decode**（Wan 生成 latent → 解成像素）。
    encode 只在"从图像反推 latent"时才用（img2video / 训练），本轮不需要。
    所以先只映射 decoder，把 encoder 标为**已知缺口**，
    而不是为了完整性在上下文耗尽时硬凑一个可能错位的映射。

decoder 结构（实测自原始权重清单）:
    upsamples.N 共 15 组（N=0..14），每 3 个 resnet 一组，通道按 384 -> 192 -> 96 递减：
        upsamples.{0,1,2}   -> up_blocks.0.resnets.{0,1,2}   (384)
        upsamples.3         -> up_blocks.0.resample         (含 resample + time_conv)
        upsamples.{4,5,6}   -> up_blocks.1.resnets.{0,1,2}   (4 带 shortcut: 384<-192)
        upsamples.7         -> up_blocks.1.resample
        upsamples.{8,9,10}  -> up_blocks.2.resnets.{0,1,2}   (192)
        upsamples.11        -> up_blocks.2.resample
        upsamples.{12,13,14}-> up_blocks.3.resnets.{0,1,2}   (96)
    另有：
        decoder.conv1        -> decoder.conv_in
        decoder.middle.{0,2,4} -> mid_block.resnets.{0,1,2}
        decoder.middle.{1,3}   -> mid_block.attentions.{0,1}
        decoder.head.0       -> norm_out
        decoder.head.2       -> conv_out

判据:
    逐键核对形状，任何不符立即报告并**拒绝加载**。
    只装 decoder（strict=False）并跑一次 decode 自检。
"""
from __future__ import annotations

import re
import sys
import os

import torch

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
sys.path.insert(0, r"D:\work\bitsandbytes-CPU\bitsandbytes")
torch.set_num_threads(int(os.environ.get("THREADS", "6")))

VAE_PTH = os.environ.get(
    "WAN_VAE", r"D:\work\textmodel\Wan2.1-T2V-1.3B\Wan2.1_VAE.pth")

# upsamples.N -> (up_block, resnet_idx) 或 'resample'
UP_MAP = {}
for base, ch in ((0, 0), (3, 1), (7, 2), (11, 3)):
    if base + 1 <= 14:
        pass
_GROUPS = [(0, 0, [0, 1, 2]), (3, 0, None), (4, 1, [0, 1, 2]), (7, 1, None),
           (8, 2, [0, 1, 2]), (11, 2, None), (12, 3, [0, 1, 2])]
for start, blk, idxs in _GROUPS:
    if idxs is None:
        UP_MAP[start] = ("resample", blk)
    else:
        for i, idx in enumerate(idxs):
            UP_MAP[start + i] = ("resnet", blk, idx)

RES_TAIL = {
    r"^residual\.0\.gamma$": "norm1.gamma",
    r"^residual\.2\.(weight|bias)$": r"conv1.\1",
    r"^residual\.3\.gamma$": "norm2.gamma",
    r"^residual\.6\.(weight|bias)$": r"conv2.\1",
    r"^shortcut\.(weight|bias)$": r"conv_shortcut.\1",
}
ATTN_TAIL = {
    # ⚠️ middle 块里的注意力键**不带 `residual.` 前缀**（与 upsamples 里的不同）：
    #    原始键就是 decoder.middle.1.norm.gamma / .to_qkv.* / .proj.*
    #    第一版照抄了 upsamples 的 `residual.1.` 前缀，导致 5 个键漏映射。
    r"^norm\.gamma$": "norm.gamma",
    r"^to_qkv\.(weight|bias)$": r"to_qkv.\1",
    r"^proj\.(weight|bias)$": r"proj.\1",
}


def map_tail(t):
    for pat, dst in RES_TAIL.items():
        if re.match(pat, t):
            return re.sub(pat, dst, t)
    return None


def map_key(k):
    m = re.match(r"^decoder\.(.*)$", k)
    if not m:
        return None                     # encoder / conv1 / conv2 本轮不映射
    t = m.group(1)
    if t.startswith("conv1."):
        return "decoder.conv_in." + t.split(".", 1)[1]
    if re.match(r"^head\.0\.gamma$", t):
        return "decoder.norm_out.gamma"
    if re.match(r"^head\.2\.(weight|bias)$", t):
        # diffusers 侧就是 decoder.conv_out.{weight,bias}，**没有**中间下标。
        # （我在此处连续猜了三次：去掉 .2 → 加 .2 → 又加一层，全部错。
        #  正确做法是直接读 ref 键名，diffusers 的 state_dict 每次都把它列在
        #  missing 里，答案就在报错信息中。）
        return "decoder.conv_out." + t.rsplit(".", 1)[-1]
    m = re.match(r"^middle\.(\d+)\.(.*)$", t)
    if m:
        n, rest = int(m.group(1)), m.group(2)
        if n % 2 == 0:
            nt = map_tail(rest)
            if nt:
                return "decoder.mid_block.resnets.%d.%s" % (n // 2, nt)
        else:
            idx = n // 2
            for pat, dst in ATTN_TAIL.items():
                if re.match(pat, rest):
                    return "decoder.mid_block.attentions.%d.%s" % (idx, re.sub(pat, dst, rest))
    m = re.match(r"^upsamples\.(\d+)\.(.*)$", t)
    if m:
        n, rest = int(m.group(1)), m.group(2)
        if n not in UP_MAP:
            return None
        spec = UP_MAP[n]
        if spec[0] == "resample":
            # ⚠️ diffusers 里 resample 包在 `upsamplers.0.` 下面：
            #    up_blocks.B.upsamplers.0.resample.{0,1}.*   和
            #    up_blocks.B.upsamplers.0.time_conv.*         （time_conv 是兄弟不是子级）
            #    第一版直接写成 up_blocks.B.resample.* 导致 12 个 unexpected。
            base = "decoder.up_blocks.%d.upsamplers.0" % spec[1]
            if rest.startswith("time_conv."):
                return "%s.time_conv.%s" % (base, rest.split(".", 1)[1])
            # ⚠️ 原始键是 `upsamples.N.resample.1.weight`（父级就叫 resample），
            #    而 diffusers 是 `...upsamplers.0.resample.1.weight`。
            #    第一版写成 base + "." + rest 会得到 `resample.resample.1`，
            #    多了一层（8 个键对不上）。
            if rest.startswith("resample."):
                return "%s.%s" % (base, rest)
            return base + "." + rest
        _, blk, idx = spec
        nt = map_tail(rest)
        if nt is None:
            return None
        return "decoder.up_blocks.%d.resnets.%d.%s" % (blk, idx, nt)
    return None


def main():
    print("=" * 84)
    print("Wan2.1 VAE decoder -> diffusers AutoencoderKLWan.decoder（只映射解码器）")
    print("=" * 84)
    from diffusers import AutoencoderKLWan
    sd = torch.load(VAE_PTH, map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    print("[1] 原始 VAE: %d 键（含 encoder/conv1/conv2，本轮只取 decoder 部分）" % len(sd))

    new, unmapped = {}, []
    for k, v in sd.items():
        nk = map_key(k)
        if nk is None:
            if k.startswith("decoder."):
                unmapped.append(k)
            continue
        new[nk] = v
    print("[2] decoder 映射出 %d 键；decoder 内未映射 %d" % (len(new), len(unmapped)))
    for k in unmapped[:10]:
        print("     未映射: %s" % k)

    m = AutoencoderKLWan()
    ref = m.state_dict()
    dec_ref = {k: v for k, v in ref.items() if k.startswith("decoder.")}
    missing = [k for k in dec_ref if k not in new]
    extra = [k for k in new if k not in dec_ref]
    bad = [(k, tuple(new[k].shape), tuple(dec_ref[k].shape))
           for k in new if k in dec_ref and tuple(new[k].shape) != tuple(dec_ref[k].shape)]
    print("[3] decoder: 期望 %d 键；missing=%d unexpected=%d shape不符=%d"
          % (len(dec_ref), len(missing), len(extra), len(bad)))
    for k in missing[:8]:
        print("     missing   : %s %s" % (k, tuple(dec_ref[k].shape)))
    for k in extra[:6]:
        print("     unexpected: %s" % k)
    for k, a, b in bad[:8]:
        print("     shape     : %s 我有%s 期望%s" % (k, a, b))

    if missing or extra or bad or unmapped:
        print("\n判据: decoder 映射不完整 —— 拒绝加载")
        return 1

    m.load_state_dict(new, strict=False)
    m.eval()
    print("[4] decoder 权重已装入（strict=False，encoder 保持随机初始化）")

    # 自检：直接 decode 一个随机 latent（不再依赖 encoder）
    z = torch.randn(1, 16, 5, 8, 8)
    try:
        with torch.no_grad():
            y = m.decode(z)
            y = y[0] if isinstance(y, tuple) else (y.sample if hasattr(y, "sample") else y)
        print("[5] decode: latent %s -> 输出 %s  finite=%s"
              % (tuple(z.shape), tuple(y.shape), bool(torch.isfinite(y).all())))
        ok = bool(torch.isfinite(y).all()) and y.shape[1] == 3
    except Exception as e:
        print("[5] decode 失败: %s: %s" % (type(e).__name__, e))
        ok = False
    print("\n判据: %s" % ("PASS —— decoder 可用（encoder 为已知缺口）" if ok else "FAIL"))
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
