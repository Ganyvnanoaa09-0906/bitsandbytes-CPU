# -*- coding: utf-8 -*-
"""qwen_lightning_bench.py — Qwen-Image-2.1 4-bit + LightX2V 4-step Lightning LoRA 加速实测。

LoRA 不融合进 nf4 权重（避免重量化误差），而是用 forward hook 在输出上加低秩增量：
    y = W_nf4(x) + (alpha/r) * up(down(x))
r=16, alpha=16 ⇒ 系数 1.0（来自 safetensors metadata）。

用法：
  python qwen_lightning_bench.py --size 256 --steps 4 --lightning 1
  python qwen_lightning_bench.py --size 256 --steps 8 --lightning 0     # 基线
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from qwen4bit_loader import load_transformer

FULL = r"D:\work\qwen_env\Qwen-Image-2.1"
LORA = r"D:\work\textmodel\qwen-image-2.1-lightning\pytorch_lora_weights.safetensors"
VAE_SCALE = 16


def rss():
    return psutil.Process().memory_info().rss / 1e9


def calculate_shift(seq, base_seq=256, max_seq=8192, base_shift=0.5, max_shift=0.9):
    m = (max_shift - base_shift) / (max_seq - base_seq)
    return seq * m + (base_shift - m * base_seq)


class LoRAInjector:
    """把 LoRA 低秩增量挂到指定模块的 forward 输出上。"""

    def __init__(self, model, lora_path, scale=1.0):
        from safetensors.torch import load_file
        sd = load_file(lora_path)
        pairs = {}
        for k, v in sd.items():
            if ".lora.down.weight" in k:
                mod = k.replace(".lora.down.weight", "")
                pairs.setdefault(mod, {})["down"] = v
            elif ".lora.up.weight" in k:
                mod = k.replace(".lora.up.weight", "")
                pairs.setdefault(mod, {})["up"] = v
        self.pairs = {}
        for mod, d in pairs.items():
            if "down" in d and "up" in d:
                self.pairs[mod.replace("transformer.", "", 1)] = (d["down"].float(), d["up"].float())
        self.scale = scale
        self.n = 0
        self.handles = []
        named = dict(model.named_modules())
        for name, (A, B) in self.pairs.items():
            m = named.get(name)
            if m is None:
                continue
            self.handles.append(m.register_forward_hook(self._make_hook(A, B)))
            self.n += 1
        print(f"  LoRA 注入 {self.n}/{len(self.pairs)} 个模块（scale={scale}）", flush=True)

    def _make_hook(self, A, B):
        scale = self.scale

        def hook(module, inputs, output):
            x = inputs[0]
            dt = output.dtype
            xf = x.float()
            delta = F.linear(F.linear(xf, A), B) * scale
            return (output.float() + delta).to(dt)

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="256")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lightning", type=int, default=1)
    ap.add_argument("--sigmas", default="", help="逗号分隔；Lightning 用 1.0,0.75,0.5,0.25")
    ap.add_argument("--prompt-embeds", default=os.path.join(
        r"D:\work\research\bench_results", "qwen_prompt_embeds.pt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--igpu", choices=["none", "fp32", "fp16", "mixed"], default="none")
    ap.add_argument("--fuse-lora", action="store_true",
                    help="把 LoRA 融进 nf4 权重（去掉 hook，iGPU 路径可省 ~20%%")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)

    print("1) 载入 4-bit transformer ...", flush=True)
    t0 = time.perf_counter()
    tr = load_transformer()
    print(f"   transformer 就绪 {time.perf_counter()-t0:.1f}s RSS {rss():.2f} GB", flush=True)
    inj = None
    if a.lightning and a.fuse_lora:   # 必须在 iGPU 包装之前（此时还是 Linear4bit 权重）
        from qwen_fuse_lora import fuse
        fuse(tr, LORA, scale=1.0)
    if a.igpu != 'none':
        t0 = time.perf_counter()
        from qwen4bit_igpu_linear_fp32 import to_igpu_f32
        if a.igpu == 'fp32':
            to_igpu_f32(tr, mode_fn=None)
        elif a.igpu == 'fp16':
            to_igpu_f32(tr, mode_fn=lambda n: 'fp16')
        else:   # mixed: MLP 走 fp16，attention 走 fp32
            to_igpu_f32(tr, mode_fn=lambda n: 'fp16' if '.img_mlp' in n or '.txt_mlp' in n else 'fp32')
        print(f'   iGPU 包装完成 {time.perf_counter()-t0:.1f}s', flush=True)
    if a.lightning and not a.fuse_lora:   # 必须在 iGPU 包装【之后】注入，否则 hook 挂在被替换掉的 Linear4bit 上会失效
        inj = LoRAInjector(tr, LORA, scale=1.0)

    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.models import AutoencoderKLQwenImage21
    print("2) 载入 scheduler + VAE ...", flush=True)
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(FULL, subfolder="scheduler",
                                                           local_files_only=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(FULL, subfolder="vae",
                                                   torch_dtype=torch.float32, local_files_only=True)
    vae.eval()

    if 'x' in str(a.size):      # 非方形："896x512"，VAE 16× 下采样
        w_px, h_px = (int(x) for x in str(a.size).lower().split('x'))
        w_lat, h_lat = w_px // VAE_SCALE, h_px // VAE_SCALE
    else:
        h_lat = w_lat = 2 * (int(a.size) // (VAE_SCALE * 2))
    channels = vae.config.z_dim
    g = torch.Generator().manual_seed(a.seed)
    lat = torch.randn(1, 1, channels, h_lat, w_lat, generator=g).float()
    lat = lat.view(1, channels, h_lat * w_lat).transpose(1, 2)
    seq = lat.shape[1]
    print(f"   latent {h_lat}x{w_lat} seq={seq} | RSS {rss():.2f} GB", flush=True)

    d = torch.load(a.prompt_embeds, map_location="cpu")
    pe = d["prompt_embeds"].float()
    pm = d.get("prompt_embeds_mask")
    base_mask = d.get("image_pad_mask", torch.zeros(1, pe.shape[1], dtype=torch.bool)).bool()
    img_mask = torch.cat([base_mask, torch.ones(1, seq // 4, dtype=torch.bool)], dim=1)
    img_shapes = [[(1, h_lat, w_lat)]]
    print("   prompt:", str(d.get("prompt", "?"))[:70], flush=True)

    # 时间表
    if a.sigmas:
        sig = [float(x) for x in a.sigmas.split(",")]
    elif a.lightning:
        sig = [1.0, 0.75, 0.5, 0.25][: a.steps] if a.steps <= 4 else list(np.linspace(1.0, 0.0, a.steps + 1)[:-1])
    else:
        sig = list(np.linspace(1.0, 1.0 / a.steps, a.steps))
    mu = calculate_shift(seq)
    sched.set_timesteps(sigmas=sig, mu=mu, device="cpu")
    ts_list = sched.timesteps
    print(f"3) denoise {len(ts_list)} 步 | mu={mu:.3f} | sigmas={np.round(sig,3).tolist()}", flush=True)

    t_all = time.perf_counter()
    per_step = []
    for i, t in enumerate(ts_list):
        t1 = time.perf_counter()
        ts = t.expand(1).to(lat.dtype) / 1000.0
        with torch.no_grad():
            pred = tr(hidden_states=lat, timestep=ts, encoder_hidden_states=pe,
                      encoder_hidden_states_mask=pm, img_shapes=img_shapes, img_mask=img_mask,
                      return_dict=False)[0]
        pred = pred[:, -seq:]
        lat = sched.step(pred, t, lat, return_dict=False)[0]
        dt = time.perf_counter() - t1
        per_step.append(dt)
        print(f"   step {i+1}/{len(ts_list)} | {dt:.1f} s | RSS {rss():.2f} GB", flush=True)
    denoise_s = time.perf_counter() - t_all

    if a.igpu != 'none':
        import gc
        del tr
        gc.collect()
        print(f'   已释放 transformer（iGPU 包装缓冲）| RSS {rss():.2f} GB', flush=True)
    torch.save(lat, os.path.join(os.path.dirname(a.out) or '.', 'qwen_latent_last.pt'))
    z = lat.transpose(1, 2).reshape(1, channels, 1, h_lat, w_lat)
    mean = torch.tensor(vae.config.latents_mean).view(1, channels, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, channels, 1, 1, 1)
    z = z * std + mean
    if max(h_lat, w_lat) * VAE_SCALE >= 1024:
        vae.enable_tiling()      # 1080p+ 直接解码会 OOM（单张 2.4 GB 分配失败）
        print('   VAE 分块解码已启用', flush=True)
    t2 = time.perf_counter()
    with torch.no_grad():
        img = vae.decode(z, return_dict=False)[0][:, :, 0][:, :3]
    img = (img / 2 + 0.5).clamp(0, 1)
    vae_s = time.perf_counter() - t2
    arr = (img[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    from PIL import Image
    out = a.out or os.path.join(
        r"D:\work\research\bench_results",
        f"qwen21_light{'on' if a.lightning else 'off'}_{a.size}_{a.steps}steps.png")
    Image.fromarray(arr).save(out)
    res = dict(size=a.size, steps=len(ts_list), lightning=bool(a.lightning), threads=a.threads,
               denoise_s=round(denoise_s, 1), step_s=[round(x, 1) for x in per_step],
               vae_s=round(vae_s, 1), rss_gb=round(rss(), 2), out=out,
               sigmas=[round(float(x), 3) for x in sig])
    print("RESULT " + json.dumps(res, ensure_ascii=False), flush=True)
    if inj:
        inj.remove()


if __name__ == "__main__":
    main()
