# -*- coding: utf-8 -*-
"""qwen_steps_lab.py — 步数 × LoRA 的实验室：Lightning 4/3/2 步 vs Viggle turbo 4/6 步。

同一 prompt、同一 seed、512²，逐个跑：计时 + 出图 + 锐度指标，最后拼一张对照图。
用法：python qwen_steps_lab.py [--size 512] [--threads 4]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from qwen4bit_loader import load_transformer
from qwen_lora_any import AnyLoRA
from qwen_lightning_bench import calculate_shift

FULL = r"D:\work\qwen_env\Qwen-Image-2.1"
EMB = r"D:\work\research\bench_results\qwen_prompt_embeds.pt"
L_LIGHTNING = r"D:\work\textmodel\qwen-image-2.1-lightning\pytorch_lora_weights.safetensors"
L_TURBO4 = r"D:\work\textmodel\qwen-viggle-turbo\Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors"
L_TURBO6 = r"D:\work\textmodel\qwen-viggle-turbo\Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors"
OUT = r"D:\work\research\bench_results\qwen_steps_lab"

CONFIGS = [
    ("lightning4", L_LIGHTNING, [1.0, 0.75, 0.5, 0.25]),
    ("lightning3", L_LIGHTNING, [1.0, 0.6667, 0.3333]),
    ("lightning2", L_LIGHTNING, [1.0, 0.5]),
    ("turbo4", L_TURBO4, [1.0, 0.75, 0.5, 0.25]),
    ("turbo6", L_TURBO6, [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]),
]


def sharpness(a):
    g = a.astype(np.float32).mean(2)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps-filter", default="", help="逗号分隔，只跑部分配置")
    a = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    only = set(x.strip() for x in a.steps_filter.split(",") if x.strip())

    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.models import AutoencoderKLQwenImage21
    from PIL import Image

    tr = load_transformer()
    vae = AutoencoderKLQwenImage21.from_pretrained(FULL, subfolder="vae",
                                                    torch_dtype=torch.float32, local_files_only=True).eval()
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(FULL, subfolder="scheduler",
                                                          local_files_only=True)
    d = torch.load(EMB, map_location="cpu")
    pe = d["prompt_embeds"].float()
    pm = d.get("prompt_embeds_mask")
    base_mask = d.get("image_pad_mask", torch.zeros(1, pe.shape[1], dtype=torch.bool)).bool()

    h = w = a.size // 16
    seq = h * w
    img_mask = torch.cat([base_mask, torch.ones(1, seq // 4, dtype=torch.bool)], dim=1)
    shapes = [[(1, h, w)]]
    mu = calculate_shift(seq)

    rows = []
    images = []
    for name, lora, sig in CONFIGS:
        if only and name not in only:
            continue
        out_png = os.path.join(OUT, f"{name}.png")
        if os.path.exists(out_png):
            print(f"  [skip] {name}", flush=True)
            images.append((name, Image.open(out_png).convert("RGB")))
            continue
        inj = AnyLoRA(tr, lora, verbose=True) if lora else None
        g = torch.Generator().manual_seed(0)
        lat = torch.randn(1, 1, 64, h, w, generator=g).float()
        lat = lat.view(1, 64, seq).transpose(1, 2)
        sched.set_timesteps(sigmas=sig, mu=mu, device="cpu")
        ts = sched.timesteps
        t0 = time.perf_counter()
        for i, t in enumerate(ts):
            with torch.no_grad():
                pred = tr(hidden_states=lat, timestep=(t.expand(1).to(lat.dtype) / 1000.0),
                          encoder_hidden_states=pe, encoder_hidden_states_mask=pm,
                          img_shapes=shapes, img_mask=img_mask, return_dict=False)[0][:, -seq:]
            lat = sched.step(pred, t, lat, return_dict=False)[0]
        den = time.perf_counter() - t0
        z = lat.transpose(1, 2).reshape(1, 64, 1, h, w)
        mean = torch.tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, 64, 1, 1, 1)
        t1 = time.perf_counter()
        with torch.no_grad():
            img = vae.decode(z * std + mean, return_dict=False)[0][:, :, 0][:, :3]
        vac = time.perf_counter() - t1
        arr = ((img / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(out_png)
        if inj:
            inj.remove()
        sh = sharpness(arr)
        rows.append(dict(name=name, steps=len(ts), denoise_s=round(den, 1), vae_s=round(vac, 1),
                         sharpness=round(sh, 1), out=out_png))
        images.append((name, Image.fromarray(arr)))
        print(f"  {name}: {len(ts)} 步 | 降噪 {den:.1f}s | VAE {vac:.1f}s | 锐度 {sh:.0f}", flush=True)

    # 对照图
    if images:
        tw = 512
        sheet = Image.new("RGB", (tw * len(images), tw + 8), "white")
        for i, (n, im) in enumerate(images):
            sheet.paste(im.resize((tw, tw)), (i * tw, 0))
        sheet.save(os.path.join(OUT, "sheet.png"))
    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("RESULT " + json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
