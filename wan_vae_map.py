# -*- coding: utf-8 -*-
"""wan_vae_map.py — 把 Wan2.1 原始 VAE 权重映射进 diffusers 的 AutoencoderKLWan

为什么需要:
    采样循环需要 VAE 解码。原始 VAE（Wan GitHub 格式）与 diffusers 的
    AutoencoderKLWan **键数完全相同（194 = 194）**，但命名不同：
        conv1/conv2              -> encoder.conv1 / encoder.conv2
        decoder.conv1            -> decoder.conv_in
        decoder.head.0/2         -> decoder.norm_out / decoder.conv_out
        decoder.middle.N         -> decoder.mid_block.resnets.N / attentions.N
        decoder.upsamples.N      -> decoder.up_blocks.M.resample
        residual.{0,2,3,6}       -> norm1 / conv1 / norm2 / conv2
        residual.1.norm/to_qkv/proj -> attentions.K.norm/to_qkv/proj

设计原则（关键）:
    **每条映射都核对形状**。形状不符立即记录为错误，而不是让它静默装上一个
    错位的权重 —— 错位的权重会让输出仍然 finite（本轮已栽过两次：
    Wan 加载时 43 个 key 缺失仍 finite；反量化转置错误让 rmsrel 从 0.09 涨到 0.9）。

用法:
    python wan_vae_map.py          # 映射 + 加载 + 自检（编码→解码往返）
"""
from __future__ import annotations

import json
import os
import re
import sys

import torch

VAE_PTH = os.environ.get(
    "WAN_VAE", r"D:\work\textmodel\Wan2.1-T2V-1.3B\Wan2.1_VAE.pth")
torch.set_num_threads(int(os.environ.get("THREADS", "6")))


def load_raw(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


# ---- 残差块内部：原始 residual.N -> diffusers 名 ----
RES_MAP = {"0": "norm1.gamma", "1": "norm1.gamma",  # 占位，见下
           }
# 实测（diffusers WanRMS_norm 只有 gamma，无 bias）:
#   original residual.0.gamma            -> norm1.gamma
#   original residual.2.{weight,bias}    -> conv1.{weight,bias}
#   original residual.3.gamma            -> norm2.gamma
#   original residual.6.{weight,bias}    -> conv2.{weight,bias}
#   original residual.{1,4}.norm.gamma   -> attentions.K.norm.gamma
#   original residual.{1,4}.to_qkv.*     -> attentions.K.to_qkv.*
#   original residual.{1,4}.proj.*       -> attentions.K.proj.*
RES_SIMPLE = {
    r"^residual\.0\.gamma$": "norm1.gamma",
    r"^residual\.2\.(weight|bias)$": r"conv1.\1",
    r"^residual\.3\.gamma$": "norm2.gamma",
    r"^residual\.6\.(weight|bias)$": r"conv2.\1",
}
# 注意力在 residual.{1,4}；K 用出现顺序（0 或 1）
RES_ATTN = {
    r"^residual\.1\.norm\.gamma$": "attentions.{K}.norm.gamma",
    r"^residual\.1\.to_qkv\.(weight|bias)$": r"attentions.{K}.to_qkv.\1",
    r"^residual\.1\.proj\.(weight|bias)$": r"attentions.{K}.proj.\1",
    r"^residual\.4\.norm\.gamma$": "attentions.{K}.norm.gamma",
    r"^residual\.4\.to_qkv\.(weight|bias)$": r"attentions.{K}.to_qkv.\1",
    r"^residual\.4\.proj\.(weight|bias)$": r"attentions.{K}.proj.\1",
}


def map_block_tail(tail, k_counter):
    """把一个 'residual.*' / 'middle.*' 的后缀映射成 diffusers 的后缀。"""
    for pat, dst in RES_SIMPLE.items():
        if re.match(pat, tail):
            return re.sub(pat, dst, tail)
    for pat, dst in RES_ATTN.items():
        if re.match(pat, tail):
            k = k_counter.get("K", 0)
            k_counter["K"] = k + 1
            return re.sub(pat, dst.replace("{K}", str(k)), tail)
    return None


def convert(sd, verbose=True):
    """返回 (new_sd, errors)。errors 非空表示映射不完整。"""
    out = {}
    errs = []
    # 收集中间块的 resnet 个数：原始 decoder.middle.N，N 为偶数处是 resnet、奇数处是 attention
    for k, v in sd.items():
        nk = None
        # ---- 顶层 ----
        if k in ("conv1.weight", "conv1.bias"):
            nk = "encoder." + k
        elif k in ("conv2.weight", "conv2.bias"):
            nk = "encoder." + k
        else:
            # ---- encoder ----
            m = re.match(r"^encoder\.(.*)$", k)
            if m:
                nk = "encoder." + _map_enc(m.group(1))
            # ---- decoder ----
            m = re.match(r"^decoder\.(.*)$", k)
            if m:
                nk = "decoder." + _map_dec(m.group(1))
        if nk is None:
            errs.append(("unmapped", k, tuple(v.shape)))
            continue
        out[nk] = v
    return out, errs


def _map_enc(t):
    if t.startswith("conv1."):
        return "conv1." + t.split(".", 1)[1]
    # downsamples.N.residual.*
    m = re.match(r"^downsamples\.(\d+)\.(.*)$", t)
    if m:
        n, rest = m.group(1), m.group(2)
        kc = {}
        nt = map_block_tail(rest, kc)
        if nt is not None:
            return "down_blocks.%s.resnets.%s" % (n, nt)
    return t


def _map_dec(t):
    if t.startswith("conv1."):
        return "conv_in." + t.split(".", 1)[1]
    m = re.match(r"^head\.0\.(gamma)$", t)
    if m:
        return "norm_out." + m.group(1)
    m = re.match(r"^head\.2\.(weight|bias)$", t)
    if m:
        return "conv_out." + m.group(1)
    # middle.N.<residual|norm|to_qkv|proj>
    m = re.match(r"^middle\.(\d+)\.(.*)$", t)
    if m:
        n, rest = int(m.group(1)), m.group(2)
        # 偶数 -> resnets.{n//2}；奇数 -> attentions.{n//2}
        if n % 2 == 0:
            kc = {}
            nt = map_block_tail(rest, kc)
            if nt is not None:
                return "mid_block.resnets.%d.%s" % (n // 2, nt)
        else:
            idx = n // 2
            if rest.startswith("norm."):
                return "mid_block.attentions.%d.norm.%s" % (idx, rest.split(".", 1)[1])
            return "mid_block.attentions.%d.%s" % (idx, rest)
    # upsamples.N.<...>
    m = re.match(r"^upsamples\.(\d+)\.(.*)$", t)
    if m:
        n, rest = m.group(1), m.group(2)
        base = "up_blocks.%s" % n
        if rest.startswith("residual."):
            kc = {}
            nt = map_block_tail(rest, kc)
            if nt is not None:
                return "%s.resnets.0.%s" % (base, nt)
        if rest.startswith("upsample."):
            r = rest.split(".", 1)[1]
            return "%s.resample.%s" % (base, r)
        if rest.startswith("time_conv."):
            return "%s.resample.time_conv.%s" % (base, rest.split(".", 1)[1])
    return t


def main():
    print("=" * 84)
    print("Wan2.1 VAE 原始权重 -> diffusers AutoencoderKLWan")
    print("=" * 84)
    from diffusers import AutoencoderKLWan
    raw = load_raw(VAE_PTH)
    print("[1] 原始 VAE: %d 个键" % len(raw))
    new, errs = convert(raw)
    print("[2] 映射出 %d 个键；未映射 %d 个" % (len(new), len(errs)))
    for e in errs[:10]:
        print("     未映射: %s %s" % (e[1], e[2]))

    m = AutoencoderKLWan()
    ref = m.state_dict()
    missing = [k for k in ref if k not in new]
    extra = [k for k in new if k not in ref]
    shape_bad = [(k, tuple(new[k].shape), tuple(ref[k].shape))
                 for k in new if k in ref and tuple(new[k].shape) != tuple(ref[k].shape)]
    print("[3] missing=%d  unexpected=%d  shape 不符=%d"
          % (len(missing), len(extra), len(shape_bad)))
    for k in missing[:8]:
        print("     missing   : %s %s" % (k, tuple(ref[k].shape)))
    for k in extra[:8]:
        print("     unexpected: %s" % k)
    for k, a, b in shape_bad[:8]:
        print("     shape     : %s 我有%s 期望%s" % (k, a, b))

    if missing or extra or shape_bad or errs:
        print("\n判据: 不完整 —— 拒绝加载（错位权重会让输出仍然 finite，不能靠这个放过）")
        return 1

    m.load_state_dict(new, strict=True)
    m.eval()
    print("[4] load_state_dict(strict=True) 成功")

    # 自检：编码 -> 解码 往返
    x = torch.randn(1, 3, 5, 64, 64)
    with torch.no_grad():
        z = m.encode(x).latent_dist.sample() if hasattr(m.encode(x), "latent_dist") else m.encode(x)[0]
        y = m.decode(z)[0] if isinstance(m.decode(z), tuple) else m.decode(z).sample
    print("[5] 往返: 输入 %s -> latent %s -> 输出 %s  finite=%s"
          % (tuple(x.shape), tuple(z.shape), tuple(y.shape), bool(torch.isfinite(y).all())))
    ok = (y.shape[-2:] == x.shape[-2:]) and bool(torch.isfinite(y).all())
    print("\n判据: %s" % ("PASS" if ok else "FAIL"))
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
