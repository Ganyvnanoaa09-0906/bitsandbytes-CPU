# -*- coding: utf-8 -*-
"""qwen_img.py — Qwen-Image-2.1 一条命令出图（CPU / 4-bit / 4-step Lightning / 可选超分）。

本机实测的几档（详见 QWEN_IMAGE21_ACCEL_2026-11-12.md）：
  preset 1024   512² 4步 + ESRGAN×2              ≈7.6 min   （原生 1024² 要 25 min）
  preset 1080p  896×512 4步 + ESRGAN×4 → 1920×1080 ≈12.6 min （原生 1080p 要 53–56 min）
  preset 4k     896×512 4步 + ESRGAN×4 → 3840×2160 ≈13–14 min（细节由超分先验决定，偏软）
  preset native1080p 1920x1088 原生 4步           约 37.3 min（--igpu fp16）／约 56 min（纯 CPU）

用法：
  python qwen_img.py --prompt "..." --preset 1080p --out fox.png
  python qwen_img.py --prompt "..." --preset native1080p --igpu fp16 --out real1080.png
  python qwen_img.py --prompt "..." --preset native768 --out big.png
  python qwen_img.py --prompt "..." --size 512 --steps 4 --no-upscale --out raw.png

注意：--igpu 只支持 hook 版 LoRA（融合版在核显上会花屏，见技术报告 A.16.24.8）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import psutil
import torch

from qwen4bit_loader import load_transformer
from qwen_lightning_bench import LORA, LoRAInjector, calculate_shift
from sr_models import DEFAULT as DEFAULT_SR, WEIGHTS as SR_WEIGHTS

FULL = r"D:\work\qwen_env\Qwen-Image-2.1"
CACHE = r"D:\work\research\bench_results\qwen_embeds_cache"
VAE_SCALE = 16
SR_X4PLUS = r"D:\work\textmodel\RealESRGAN_x4plus.pth"

PRESETS = {
    # name: (gen_w, gen_h, upscale, out_w, out_h)
    "1024": (512, 512, "x2", 1024, 1024),
    "1080p": (896, 512, "x4", 1920, 1080),
    "4k": (896, 512, "x4", 3840, 2160),
    "native768": (768, 768, "none", 768, 768),
    # 原生 1080p：iGPU fp16 + hook 实测 37.3 min；纯 CPU ≈56 min（a.igpu 选）
    "native1080p": (1920, 1088, "none", 1920, 1088),
}


def rss():
    return psutil.Process().memory_info().rss / 1e9


def prompt_embeds(prompt: str, cache_dir: str = CACHE):
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16] + ".pt")
    if os.path.exists(p):
        print(f"  [embeds] 缓存命中 {os.path.basename(p)}", flush=True)
        return torch.load(p, map_location="cpu")
    print("  [embeds] 编码 prompt（Qwen3-VL 8B，首次约 60 s）...", flush=True)
    from qwen_te_4bit_loader import encode_prompt, load_text_encoder
    te = load_text_encoder()
    pe, mask, img_pad = encode_prompt(te, prompt)
    d = dict(prompt_embeds=pe, prompt_embeds_mask=mask, image_pad_mask=img_pad, prompt=prompt)
    torch.save(d, p)
    del te
    return d


@torch.no_grad()
def generate(w_px, h_px, steps, sigmas, embeds, seed=0, threads=4, vae_tiling=None,
             igpu="none"):
    torch.set_num_threads(threads)
    tr = load_transformer()
    if igpu != "none":
        from qwen4bit_igpu_linear_fp32 import to_igpu_f32
        to_igpu_f32(tr, mode_fn=(lambda n: "fp16") if igpu == "fp16" else None)
        print(f"  [iGPU] 线性层包装完成（{igpu}）", flush=True)
    # LoRA 必须在 iGPU 包装【之后】注入；且只能用 hook（融合版在 iGPU 上会花屏，见 A.16.24.8）
    inj = LoRAInjector(tr, LORA, scale=1.0)
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.models import AutoencoderKLQwenImage21
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(FULL, subfolder="scheduler",
                                                          local_files_only=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(FULL, subfolder="vae",
                                                    torch_dtype=torch.float32, local_files_only=True)
    vae.eval()

    h_lat, w_lat = h_px // VAE_SCALE, w_px // VAE_SCALE
    ch = vae.config.z_dim
    g = torch.Generator().manual_seed(seed)
    lat = torch.randn(1, 1, ch, h_lat, w_lat, generator=g).float()
    lat = lat.view(1, ch, h_lat * w_lat).transpose(1, 2)
    seq = lat.shape[1]

    pe = embeds["prompt_embeds"].float()
    pm = embeds.get("prompt_embeds_mask")
    base = embeds.get("image_pad_mask", torch.zeros(1, pe.shape[1], dtype=torch.bool)).bool()
    img_mask = torch.cat([base, torch.ones(1, seq // 4, dtype=torch.bool)], dim=1)
    shapes = [[(1, h_lat, w_lat)]]

    mu = calculate_shift(seq)
    sched.set_timesteps(sigmas=sigmas, mu=mu, device="cpu")
    ts = sched.timesteps
    print(f"  [gen] {w_px}x{h_px} | latent {h_lat}x{w_lat} seq={seq} | {len(ts)} 步", flush=True)

    t0 = time.perf_counter()
    for i, t in enumerate(ts):
        t1 = time.perf_counter()
        pred = tr(hidden_states=lat, timestep=(t.expand(1).to(lat.dtype) / 1000.0),
                  encoder_hidden_states=pe, encoder_hidden_states_mask=pm,
                  img_shapes=shapes, img_mask=img_mask, return_dict=False)[0][:, -seq:]
        lat = sched.step(pred, t, lat, return_dict=False)[0]
        print(f"    step {i+1}/{len(ts)} {time.perf_counter()-t1:.1f}s RSS {rss():.2f} GB", flush=True)
    denoise = time.perf_counter() - t0
    inj.remove()

    if igpu != "none":
        import gc as _gc
        del tr
        _gc.collect()
        print(f"  [iGPU] 已释放包装缓冲 RSS {rss():.2f} GB", flush=True)
    z = lat.transpose(1, 2).reshape(1, ch, 1, h_lat, w_lat)
    mean = torch.tensor(vae.config.latents_mean).view(1, ch, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, ch, 1, 1, 1)
    if vae_tiling if vae_tiling is not None else max(w_px, h_px) >= 1024:
        vae.enable_tiling()          # 1080p+ 直解会 OOM
        print("   VAE 分块解码", flush=True)
    t2 = time.perf_counter()
    img = vae.decode(z * std + mean, return_dict=False)[0][:, :, 0][:, :3]
    vae_s = time.perf_counter() - t2
    img = (img / 2 + 0.5).clamp(0, 1)
    arr = (img[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    # 超分前必须把 transformer/VAE 释放掉：ESRGAN 在 7.3 MP 输出时需要数 GB，
    # 两者叠加会 OOM（2026-11-13 实测：1080p preset 在超分阶段被系统杀掉）
    import gc
    tr = None
    del vae
    gc.collect()
    print(f'  [mem] 已释放生成模型，RSS {rss():.2f} GB', flush=True)
    return arr, dict(denoise_s=round(denoise, 1), vae_s=round(vae_s, 1))


def upscale(arr, factor, out_w, out_h, sr=DEFAULT_SR):
    """统一走 sr_models（支持 x4plus / ultrasharp / remacri / nmkd / anime6b）。"""
    from sr_models import upscale_array
    return upscale_array(arr, name=sr, out_w=out_w, out_h=out_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--preset", choices=list(PRESETS), default="1024")
    ap.add_argument("--size", default="", help="覆盖 preset 的生成尺寸，如 512 或 896x512")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--sigmas", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-upscale", action="store_true")
    ap.add_argument("--sr", choices=sorted(SR_WEIGHTS), default=DEFAULT_SR,
                    help="超分模型（默认 %s）" % DEFAULT_SR)
    ap.add_argument("--igpu", choices=["none", "fp16", "fp32"], default="none",
                    help="把 232 个 Linear 放到核显（seq>=1024 才有收益；只支持 hook 版 LoRA）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gw, gh, up, ow, oh = PRESETS[a.preset]
    if a.size:
        if "x" in a.size.lower():
            gw, gh = (int(x) for x in a.size.lower().split("x"))
        else:
            gw = gh = int(a.size)
    if a.no_upscale:
        up, ow, oh = "none", gw, gh
    sig = ([float(x) for x in a.sigmas.split(",")] if a.sigmas
           else ([1.0, 0.75, 0.5, 0.25] if a.steps == 4
                 else list(np.linspace(1.0, 1.0 / a.steps, a.steps))))

    t_all = time.time()
    emb = prompt_embeds(a.prompt)
    arr, st = generate(gw, gh, a.steps, sig, emb, seed=a.seed, threads=a.threads,
                       igpu=a.igpu)
    from PIL import Image
    raw = a.out.replace(".png", "_raw.png")
    Image.fromarray(arr).save(raw)
    print(f"  [save] {raw}  ({arr.shape[1]}x{arr.shape[0]})", flush=True)

    up_info = None
    if up != "none":
        t_sr = time.time()
        arr2, _scale2 = upscale(arr, 4 if up == "x4" else 2, ow, oh, sr=a.sr)
        sr_s = round(time.time() - t_sr, 1)
        Image.fromarray(arr2).save(a.out)
        up_info = dict(sr=a.sr, sr_s=sr_s, size=[arr2.shape[1], arr2.shape[0]])
        print(f"  [save] {a.out}  ({arr2.shape[1]}x{arr2.shape[0]}, 超分 {sr_s}s)", flush=True)
    else:
        Image.fromarray(arr).save(a.out)

    print("RESULT " + json.dumps(dict(
        prompt=a.prompt, preset=a.preset, gen=[gw, gh], steps=a.steps, igpu=a.igpu,
        denoise_s=st["denoise_s"], vae_s=st["vae_s"], upscale=up_info,
        total_s=round(time.time() - t_all, 1), out=a.out), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
