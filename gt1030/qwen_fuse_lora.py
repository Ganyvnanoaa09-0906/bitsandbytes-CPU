# -*- coding: utf-8 -*-
"""
⚠️ 重要限制（2026-11-13 实测）：
  融合后的权重只适用于**纯 CPU** 路径。一旦走 iGPU 混合（qwen4bit_igpu_linear_fp32.to_igpu_f32），
  会出编织状/条纹花屏——重新 quantize_4bit 生成的 nested quant_state 与 iGPU 反量化 kernel
  所期望的 absmax 布局不一致（1080p fp16 35.3min 花、1080p fp32 41.7min 也花、768² 也花；
  同模型同精度用 hook 版则干净）。**iGPU 场景请用 LoRAInjector（hook）**。
qwen_fuse_lora.py — 把 Lightning LoRA 直接融进 NF4 权重（一次性），去掉每步的 CPU 侧 hook。

为什么：hook 版在每个 Linear 输出上多做两次小 matmul + dtype 转换，
实测在 512² 上让 iGPU 路径从 266.5 s 变慢到 320.7 s（+20%）。
融合做法：dequantize_4bit → W + (up@down)*scale → quantize_4bit 重新打包（nf4/64/compress）。
"""
from __future__ import annotations

import argparse
import re
import time

import torch

from qwen4bit_loader import load_transformer

LORA = r"D:\work\textmodel\qwen-image-2.1-lightning\pytorch_lora_weights.safetensors"


def fuse(model, lora_path=LORA, scale=1.0, verbose=True):
    import bitsandbytes as bnb
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit
    from bitsandbytes.nn import Params4bit
    from safetensors.torch import load_file

    sd = load_file(lora_path)
    pairs = {}
    for k, v in sd.items():
        if k.endswith(".lora.down.weight"):
            pairs.setdefault(k.replace(".lora.down.weight", ""), {})["down"] = v
        elif k.endswith(".lora.up.weight"):
            pairs.setdefault(k.replace(".lora.up.weight", ""), {})["up"] = v
    # 去掉 "transformer." 前缀，与模型命名对齐
    pairs = {k.replace("transformer.", "", 1): v for k, v in pairs.items() if "down" in v and "up" in v}

    named = dict(model.named_modules())
    ok = skip = 0
    t0 = time.perf_counter()
    for name, d in pairs.items():
        mod = named.get(name)
        if mod is None or not isinstance(mod, bnb.nn.Linear4bit):
            skip += 1
            continue
        w = mod.weight
        W = dequantize_4bit(w.data, w.quant_state).float()
        delta = (d["up"].float() @ d["down"].float()) * scale
        if tuple(delta.shape) != tuple(W.shape):
            print(f"  [skip] {name} 形状不符 {tuple(delta.shape)} vs {tuple(W.shape)}")
            skip += 1
            continue
        packed, qs = quantize_4bit((W + delta).to(torch.bfloat16), blocksize=64,
                                   compress_statistics=True, quant_type="nf4")
        new = Params4bit.from_prequantized(packed, qs.as_dict(packed=True), requires_grad=False, device="cpu")
        mod.weight = new
        ok += 1
    if verbose:
        print(f"  LoRA→NF4 融合: {ok} 层，跳过 {skip}，用时 {time.perf_counter()-t0:.1f}s", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--check", action="store_true", help="融合前后做一次前向对拍（慢）")
    a = ap.parse_args()
    tr = load_transformer()
    fuse(tr, scale=a.scale)
    if a.check:
        torch.set_num_threads(4)
        pe = torch.randn(1, 16, 4096).float()
        lat = torch.randn(1, 1024, 64)
        ts = torch.tensor([500.0])
        shapes = [[(1, 32, 32)]]
        with torch.no_grad():
            out = tr(hidden_states=lat, timestep=ts, encoder_hidden_states=pe,
                     encoder_hidden_states_mask=None, img_shapes=shapes,
                     img_mask=torch.ones(1, 16 + 256, dtype=torch.bool), return_dict=False)[0]
        print("  fused forward:", tuple(out.shape), "finite=", bool(torch.isfinite(out).all()))


if __name__ == "__main__":
    main()
