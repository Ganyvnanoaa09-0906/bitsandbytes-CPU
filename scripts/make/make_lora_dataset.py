# -*- coding: utf-8 -*-
"""Build a small character-LoRA dataset: varied angles/expressions of ONE character.

Layout produced (this is what train_diffusion.py's SDDataset expects):
    <out>/images/*.png
    <out>/prompts.json      { "fname.png": {"prompt": "..."} }

Each variant is sampled with the FROZEN production settings (512, DPM-Solver++ 2M, 5 steps,
SD1.5 VAE, full schedule from t=999) so the training set matches what the pipeline actually
generates. `--per-prompt` frames are drawn per variant and all are kept; the motion module
needs >=2 frames to have a temporal axis, hence the default of 2.

Usage:
  python make_lora_dataset.py --out D:\work\lora_char2 --per-prompt 2
"""
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse
import json
import os
import sys
import time

import numpy
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scripts.lib.anime_adiff import build, ANIME_NEG, OUT_DIR

GUIDANCE = 7.5

BASE = ("1girl, anime style, silver hair, red eyes, detailed face, beautiful detailed face, "
        "masterpiece, best quality, solo")

VARIANTS = [
    ("front_smile", "looking at viewer, happy smile, upper body"),
    ("profile_neutral", "from side, profile view, neutral expression, upper body"),
    ("above_surprised", "from above, looking up, surprised expression, upper body"),
    ("closeup_blush", "close-up face, blushing, embarrassed, looking away"),
    ("below_smirk", "from below, confident smirk, upper body"),
    ("fullbody", "full body, standing, arms crossed, neutral expression"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--per-prompt", type=int, default=2)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--scheduler", default="dpm")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True
    idir = os.path.join(a.out, "images")
    os.makedirs(idir, exist_ok=True)

    print(f"[cfg] res={a.res} steps={a.steps} per_prompt={a.per_prompt} "
          f"variants={len(VARIANTS)}")
    pipe = build(a.scheduler)
    with torch.no_grad():
        ne = pipe.encode_prompt("", "cpu", 1, True, negative_prompt=ANIME_NEG)[1]

    lat_h = a.res // pipe.vae_scale_factor
    sched = pipe.scheduler
    manifest = {}
    t_all = time.time()

    for vi, (tag, extra) in enumerate(VARIANTS):
        prompt = f"{BASE}, {extra}"
        with torch.no_grad():
            pe = pipe.encode_prompt(prompt, "cpu", 1, True,
                                    negative_prompt=ANIME_NEG)[0]
        embeds = torch.cat([ne, pe])

        sched.set_timesteps(a.steps)
        sched._step_index = 0
        sched.model_outputs = [None] * sched.config.solver_order
        sched._lower_order_nums = 0
        ts = list(sched.timesteps)

        gen = torch.Generator().manual_seed(a.seed + vi * 100)
        lat = torch.randn((1, pipe.unet.config.in_channels, a.per_prompt, lat_h, lat_h),
                          generator=gen, dtype=torch.float32) * sched.init_noise_sigma

        t0 = time.time()
        with torch.no_grad():
            for t in ts:
                mi = torch.cat([lat] * 2)
                mi = sched.scale_model_input(mi, t)
                npr = pipe.unet(mi, t, encoder_hidden_states=embeds).sample
                nu, nt = npr.chunk(2)
                npr = nu + GUIDANCE * (nt - nu)
                lat = sched.step(npr, t, lat,
                                 **pipe.prepare_extra_step_kwargs(gen, 0.0)).prev_sample
            b, c, T, h, w = lat.shape
            flat = (lat / pipe.vae.config.scaling_factor).permute(0, 2, 1, 3, 4
                                                                 ).reshape(b * T, c, h, w)
            dec = pipe.vae.decode(flat, return_dict=False)[0]
            arr = (((dec / 2 + 0.5).clamp(0, 1)).reshape(b, T, 3, dec.shape[-2], dec.shape[-1])
                   .permute(0, 1, 3, 4, 2)[0].cpu().numpy() * 255).round().astype(np.uint8)

        for k in range(arr.shape[0]):
            fn = f"{vi:02d}_{tag}_{k}.png"
            Image.fromarray(arr[k]).save(os.path.join(idir, fn))
            manifest[fn] = {"prompt": prompt}
        print(f"  [{vi+1}/{len(VARIANTS)}] {tag:<18} {arr.shape[0]} frames  "
              f"{time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(a.out, "prompts.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    n = len(manifest)
    print(f"\n[done] {n} images in {time.time()-t_all:.0f}s -> {idir}")
    print(f"[done] prompts.json -> {os.path.join(a.out, 'prompts.json')}")


if __name__ == "__main__":
    main()
