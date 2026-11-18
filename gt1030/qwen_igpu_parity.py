# -*- coding: utf-8 -*-
"""qwen_igpu_parity.py — 单次 forward 对拍：CPU(4bit) vs iGPU 混合，seq=1024（≈512²）。

目的：验证 §7.179 的"核显混合正确性"是否只在 seq=256 成立。
  - 同一份随机 latent + 同一 prompt embeds，各跑一次 forward；
  - 比较 relL2 / max|Δ|。
"""
from __future__ import annotations

import os
import time

import torch

from qwen4bit_loader import load_transformer

FULL = r"D:\work\qwen_env\Qwen-Image-2.1"
EMB = r"D:\work\research\bench_results\qwen_prompt_embeds.pt"


def main(seq=1024, seed=0):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    tr = load_transformer()
    d = torch.load(EMB, map_location="cpu")
    pe = d["prompt_embeds"].float()
    pm = d.get("prompt_embeds_mask")
    base = d.get("image_pad_mask", torch.zeros(1, pe.shape[1], dtype=torch.bool)).bool()
    img_mask = torch.cat([base, torch.ones(1, seq // 4, dtype=torch.bool)], dim=1)
    lat = torch.randn(1, seq, 64)
    ts = torch.tensor([500.0])
    h_lat = int(round((seq ** 0.5)))
    shapes = [[(1, h_lat, h_lat)]]

    def fwd():
        with torch.no_grad():
            out = tr(hidden_states=lat, timestep=ts, encoder_hidden_states=pe,
                     encoder_hidden_states_mask=pm, img_shapes=shapes, img_mask=img_mask,
                     return_dict=False)[0]
        return out.float()

    t0 = time.perf_counter()
    cpu_out = fwd()
    cpu_t = time.perf_counter() - t0
    print(f"  CPU forward seq={seq}: {cpu_t:.1f} s  out {tuple(cpu_out.shape)}", flush=True)

    from qwen4bit_igpu_linear_fp32 import to_igpu_f32
    t0 = time.perf_counter()
    to_igpu_f32(tr, mode_fn=None)          # fp32 模式（正确性模式）
    print(f"  iGPU 包装 {time.perf_counter()-t0:.1f} s", flush=True)
    t0 = time.perf_counter()
    gpu_out = fwd()
    gpu_t = time.perf_counter() - t0
    d_ = (cpu_out - gpu_out)
    rel = float(d_.norm() / cpu_out.norm())
    print(f"  iGPU fp32 forward: {gpu_t:.1f} s | relL2 {rel:.3e} | max|Δ| {float(d_.abs().max()):.3e}", flush=True)
    print("  RESULT " + str(dict(seq=seq, cpu_s=round(cpu_t, 1), gpu_s=round(gpu_t, 1),
                                rel=rel, maxabs=float(d_.abs().max()))))


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1024)
