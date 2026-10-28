# -*- coding: utf-8 -*-
"""In-Context Regeneration (H3 memo) vs FSRCNN for detail restoration.

Idea from the H3/ByteDance memo: instead of a pure super-resolution pass, let the base model
REGENERATE the frame at target resolution under its original context (same prompt, image as
init) -- claimed to recover fine texture better than SR.

Test (single-frame img2img with the frozen prompt and the same base/V AE as production):
  * the 2D UNet is used directly (a single frame has no temporal axis, so the motion adapter
    is not involved);
  * `strength` controls how much of the schedule to re-run -- low values stay close to the
    original composition while re-synthesising detail;
  * output is compared against the FSRCNN result on the same frames (Laplacian + face metric).

Cost note: each frame costs roughly strength x 5 steps of the frozen pipeline, i.e. far more
than FSRCNN's 0.27 s/frame -- this test exists to decide whether that cost buys real quality.

Usage:
  python icr_test.py --src <frames dir> --picks 26,28,30,32 --strength 0.4 --out <dir>
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
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

import numpy
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

BASE_DIR = r"D:\work\textmodel\counterfeit-v3-diffusers"
SD15_FULL = r"D:\work\textmodel\sd15_full"
VAE_DIR = os.environ.get("ADIFF_VAE", os.path.join(BASE_DIR, "vae"))
OUT_DIR = r"D:\work\output\adiff_out"

ANIME_PROMPT = ("1girl, anime style, cute face, detailed eyes, beautiful detailed face, "
                "masterpiece, best quality, solo, upper body, gentle smile")
ANIME_NEG = ("lowres, bad anatomy, bad hands, extra digits, fewer digits, cropped, "
             "worst quality, low quality, bad face, deformed, blurry")


def lapvar(a):
    a = np.asarray(a).astype(np.float32)
    g = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    return float((4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
                  - g[1:-1, :-2] - g[1:-1, 2:]).var())


def build():
    from diffusers import (StableDiffusionImg2ImgPipeline, AutoencoderKL,
                           DPMSolverMultistepScheduler, UNet2DConditionModel)
    from transformers import CLIPTextModel, CLIPTokenizer
    unet = UNet2DConditionModel.from_pretrained(os.path.join(BASE_DIR, "unet"),
                                                torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(VAE_DIR, torch_dtype=torch.float32)
    te = CLIPTextModel.from_pretrained(os.path.join(BASE_DIR, "text_encoder"),
                                       torch_dtype=torch.float32)
    tok = CLIPTokenizer.from_pretrained(os.path.join(BASE_DIR, "tokenizer"))
    sch = DPMSolverMultistepScheduler.from_pretrained(os.path.join(BASE_DIR, "scheduler"))
    pipe = StableDiffusionImg2ImgPipeline(unet=unet, vae=vae, text_encoder=te, tokenizer=tok,
                                          scheduler=sch, safety_checker=None,
                                          feature_extractor=None,
                                          requires_safety_checker=False)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--picks", default="26,28,30,32")
    ap.add_argument("--strength", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--with-fsrcnn", action="store_true", default=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True
    os.makedirs(a.out, exist_ok=True)

    print(f"[cfg] strength={a.strength} steps={a.steps} vae={VAE_DIR}", flush=True)
    pipe = build()
    pipe.set_progress_bar_config(disable=True)
    sr = None
    if a.with_fsrcnn:
        from scripts.lib.detail_restore import load_sr
        import torch.nn.functional
        sr = load_sr()

    picks = [int(x) for x in a.picks.split(",")]
    print(f"\n{'frame':>6} {'variant':>12} {'Laplacian':>10} {'x src':>7} {'sec':>7}")
    print("-" * 48)
    for pi in picks:
        p = os.path.join(a.src, f"f{pi:03d}.png")
        if not os.path.exists(p):
            print(f"{pi:>6}  (missing)")
            continue
        pil = Image.open(p).convert("RGB")
        sl = lapvar(pil)
        print(f"{pi:>6} {'source':>12} {sl:>10.1f} {1.0:>7.2f} {'-':>7}")

        if sr is not None:
            t = torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1
                                                                ).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                up = sr(t)
                dn = F.interpolate(up, size=pil.size[::-1], mode="area").clamp(0, 1)
            arr = (dn[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(a.out, f"f{pi:03d}_fsrcnn.png"))
            print(f"{pi:>6} {'fsrcnn':>12} {lapvar(arr):>10.1f} {lapvar(arr)/sl:>7.2f} "
                  f"{'-':>7}")

        g = torch.Generator().manual_seed(1234 + pi)
        t0 = time.time()
        with torch.no_grad():
            out = pipe(prompt=ANIME_PROMPT, negative_prompt=ANIME_NEG, image=pil,
                       strength=a.strength, num_inference_steps=a.steps,
                       guidance_scale=7.5, generator=g)
        img = out.images[0]
        img.save(os.path.join(a.out, f"f{pi:03d}_icr_s{a.strength}.png"))
        print(f"{pi:>6} {'ICR s=%.2f' % a.strength:>12} {lapvar(img):>10.1f} "
              f"{lapvar(img)/sl:>7.2f} {time.time()-t0:>7.1f}", flush=True)

    print(f"\n[save] {a.out}")


if __name__ == "__main__":
    main()
