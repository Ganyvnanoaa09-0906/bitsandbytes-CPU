# -*- coding: utf-8 -*-
"""Isolate the LoRA: merge it into a PLAIN 2D StableDiffusion pipeline (no AnimateDiff).

The AnimateDiff merge produced garbage even at 0.11% perturbation, which is numerically
implausible, so the suspicion is that something about the *combination* is wrong (motion
model wrapping, temporal modules, scheduler mismatch). A plain txt2img probe removes every
AnimateDiff factor:

  * plain UNet2DConditionModel (the exact model training wrapped)
  * same VAE / text encoder / tokenizer as production
  * same sampler family and step count (DPM-Solver++ 2M, 5 steps)
  * single image, so no temporal axis at all

Then: scale 0.0 must reproduce the no-LoRA image; scale 0.8 shows what the LoRA alone does.

Usage:
  python lora_probe.py --lora <dir> --scales 0.0,0.3,0.8 --out <dir>
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

BASE = r"D:\work\textmodel\counterfeit-v3-diffusers"
VAE_DIR = os.environ.get("ADIFF_VAE", os.path.join(BASE, "vae"))
PROMPT = ("1girl, anime style, cute face, detailed eyes, beautiful detailed face, masterpiece, "
          "best quality, solo, upper body, gentle smile, silver hair, red eyes")
NEG = ("lowres, bad anatomy, bad hands, extra digits, fewer digits, cropped, worst quality, "
       "low quality, bad face, deformed, blurry")


def lapvar(a):
    a = np.asarray(a).astype(np.float32)
    g = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    return float((4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
                  - g[1:-1, :-2] - g[1:-1, 2:]).var())


def merge_lora(unet, lora_dir, scale):
    """W += scale * (B @ A), no peft required."""
    import json as _json
    from safetensors.torch import load_file as lf
    p = lora_dir
    if os.path.isdir(p):
        sd = lf(os.path.join(p, "adapter_model.safetensors"))
        alpha = r = None
        cfgp = os.path.join(p, "adapter_config.json")
        if os.path.exists(cfgp):
            c = _json.load(open(cfgp, encoding="utf-8"))
            alpha, r = c.get("lora_alpha"), c.get("r")
    else:
        sd = lf(p)
        alpha = r = None
    eff = scale * ((alpha / r) if (alpha and r) else 1.0)
    mods = dict(unet.named_modules())
    pairs, applied, missing = {}, 0, []
    for k, v in sd.items():
        if not k.startswith("base_model.model."):
            continue
        rest = k[len("base_model.model."):]
        if rest.endswith(".lora_A.weight"):
            pairs.setdefault(rest[:-len(".lora_A.weight")], {})["A"] = v
        elif rest.endswith(".lora_B.weight"):
            pairs.setdefault(rest[:-len(".lora_B.weight")], {})["B"] = v
    for path, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        if path not in mods or not hasattr(mods[path], "weight"):
            missing.append(path)
            continue
        mod = mods[path]
        delta = (ab["B"].float() @ ab["A"].float()) * eff
        with torch.no_grad():
            mod.weight.add_(delta.to(mod.weight.dtype))
        applied += 1
    print(f"  [merge] scale={scale} effective={eff:.3f} applied={applied} "
          f"missing={len(missing)}")
    return applied


def build():
    from diffusers import (StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel,
                           DPMSolverMultistepScheduler)
    from transformers import CLIPTextModel, CLIPTokenizer
    unet = UNet2DConditionModel.from_pretrained(os.path.join(BASE, "unet"),
                                                torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(VAE_DIR, torch_dtype=torch.float32)
    te = CLIPTextModel.from_pretrained(os.path.join(BASE, "text_encoder"),
                                       torch_dtype=torch.float32)
    tok = CLIPTokenizer.from_pretrained(os.path.join(BASE, "tokenizer"))
    sch = DPMSolverMultistepScheduler.from_pretrained(os.path.join(BASE, "scheduler"))
    return StableDiffusionPipeline(unet=unet, vae=vae, text_encoder=te, tokenizer=tok,
                                   scheduler=sch, safety_checker=None,
                                   feature_extractor=None, requires_safety_checker=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=r"D:\work\lora_char2_run500\final")
    ap.add_argument("--scales", default="0.0,0.3,0.8")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True
    os.makedirs(a.out, exist_ok=True)

    scales = [float(x) for x in a.scales.split(",")]
    print(f"[probe] PLAIN 2D pipeline (no AnimateDiff), scales={scales}, steps={a.steps}",
          flush=True)

    pipe = build()
    pipe.set_progress_bar_config(disable=True)

    # reference: no merge at all
    g = torch.Generator().manual_seed(1234)
    t0 = time.time()
    with torch.no_grad():
        img0 = pipe(prompt=PROMPT, negative_prompt=NEG, num_inference_steps=a.steps,
                    guidance_scale=7.5, height=512, width=512, generator=g).images[0]
    img0.save(os.path.join(a.out, "s_none.png"))
    print(f"  no-lora      lap={lapvar(img0):>8.1f} white%={100*(np.asarray(img0)>245).mean():.2f}"
          f"  ({time.time()-t0:.0f}s)", flush=True)

    for s in scales:
        pipe2 = build()                      # fresh pipeline each time (no accumulation)
        pipe2.set_progress_bar_config(disable=True)
        merge_lora(pipe2.unet, a.lora, s)
        g = torch.Generator().manual_seed(1234)
        t0 = time.time()
        with torch.no_grad():
            img = pipe2(prompt=PROMPT, negative_prompt=NEG, num_inference_steps=a.steps,
                        guidance_scale=7.5, height=512, width=512, generator=g).images[0]
        img.save(os.path.join(a.out, f"s_{s:.1f}.png"))
        w = 100 * (np.asarray(img) > 245).mean()
        print(f"  merge s={s:<4} lap={lapvar(img):>8.1f} white%={w:.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n[save] {a.out}")


if __name__ == "__main__":
    main()
