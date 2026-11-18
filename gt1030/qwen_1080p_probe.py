# -*- coding: utf-8 -*-
"""qwen_1080p_probe.py — 原生 1080p（seq≈8040）单次 forward 的可行性与耗时探测。

目的：回答"能不能直接跑 1080p、甚至更高"——量一次前向的时间与峰值内存，再乘以 4 步外推。
"""
from __future__ import annotations

import sys
import time

import torch
import psutil

from qwen4bit_loader import load_transformer

FULL = r"D:\work\qwen_env\Qwen-Image-2.1"
EMB = r"D:\work\research\bench_results\qwen_prompt_embeds.pt"


def rss():
    return psutil.Process().memory_info().rss / 1e9


def main(w=1920, h=1088, igpu='none'):
    torch.set_num_threads(4)
    torch.manual_seed(0)
    seq = (w // 16) * (h // 16)
    print(f"  目标 {w}x{h} -> latent {w//16}x{h//16} seq={seq}  ({seq/1024:.1f}x of 512²)", flush=True)
    tr = load_transformer()
    d = torch.load(EMB, map_location="cpu")
    pe = d["prompt_embeds"].float()
    pm = d.get("prompt_embeds_mask")
    base = d.get("image_pad_mask", torch.zeros(1, pe.shape[1], dtype=torch.bool)).bool()
    img_mask = torch.cat([base, torch.ones(1, seq // 4, dtype=torch.bool)], dim=1)
    if igpu != 'none':
        from qwen4bit_igpu_linear_fp32 import to_igpu_f32
        to_igpu_f32(tr, mode_fn=(lambda n: 'fp16') if igpu == 'fp16' else None)
        print(f'   iGPU 包装完成 | RSS {rss():.2f} GB', flush=True)
    lat = torch.randn(1, seq, 64)
    shapes = [[(1, h // 16, w // 16)]]
    ts = torch.tensor([500.0])
    base_rss = rss()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = tr(hidden_states=lat, timestep=ts, encoder_hidden_states=pe,
                 encoder_hidden_states_mask=pm, img_shapes=shapes, img_mask=img_mask,
                 return_dict=False)[0]
    dt = time.perf_counter() - t0
    print(f"  forward: {dt:.1f} s | 峰值 RSS {rss():.2f} GB (基线 {base_rss:.2f}) | out {tuple(out.shape)}", flush=True)
    print(f"  外推 4 步降噪 ≈ {dt*4:.0f} s = {dt*4/60:.1f} min（+ VAE 约 {seq/1024*20:.0f} s）", flush=True)
    print("RESULT " + str(dict(size=[w, h], seq=seq, forward_s=round(dt, 1),
                               rss_gb=round(rss(), 2), est_4step_min=round(dt * 4 / 60, 1))))


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 1920
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 1088
    g = sys.argv[3] if len(sys.argv) > 3 else 'none'
    main(w, h, g)
