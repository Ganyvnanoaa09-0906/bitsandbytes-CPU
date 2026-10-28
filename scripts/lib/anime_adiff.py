# -*- coding: utf-8 -*-
"""Load the converted Counterfeit-V3 diffusers dir into AnimateDiff and verify it generates anime frames.

Runs a short low-res pass first (fast sanity check that weights are valid and the pipeline
accepts them), then optionally a 512 pass.

Usage:
  python anime_adiff.py --res 256 --frames 8 --steps 8 --scheduler dpm
  python anime_adiff.py --res 512 --frames 16 --steps 8
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

BASE_DIR = r"D:\work\textmodel\counterfeit-v3-diffusers"
SD15_FULL = r"D:\work\textmodel\sd15_full"
ADAPTER_V15 = r"D:\work\textmodel\animatediff-motion-adapter-v1-5"
OUT_DIR = r"D:\work\output\adiff_out"

# Which VAE to decode with. Counterfeit-V3's bundled VAE is measurably "hot": on the same
# latent it amplifies green artefacts 3.15x (vs 1.05x for the plain SD1.5 VAE), over-sharpens
# (Laplacian 333 vs source 163) and blows highlights (white% 0.54 vs 0.00). Override with
# the ADIFF_VAE env var to decode with a different VAE without touching this file:
#   $env:ADIFF_VAE = "D:\work\textmodel\sd15_full\vae"
VAE_DIR = os.environ.get("ADIFF_VAE", os.path.join(BASE_DIR, "vae"))

ANIME_PROMPT = ("1girl, anime style, cute face, detailed eyes, beautiful detailed face, "
                "masterpiece, best quality, solo, upper body, gentle smile")
ANIME_NEG = ("lowres, bad anatomy, bad hands, extra digits, fewer digits, cropped, "
             "worst quality, low quality, bad face, deformed, blurry")


def build(scheduler="dpm", adapter=ADAPTER_V15):
    import torch
    from diffusers import (
        AnimateDiffPipeline, AutoencoderKL, DDIMScheduler,
        DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler,
        MotionAdapter, UNet2DConditionModel,
    )
    from diffusers.models import UNetMotionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    t0 = time.time()
    unet = UNet2DConditionModel.from_pretrained(os.path.join(BASE_DIR, "unet"),
                                                torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(VAE_DIR, torch_dtype=torch.float32)
    te = CLIPTextModel.from_pretrained(os.path.join(BASE_DIR, "text_encoder"),
                                       torch_dtype=torch.float32)
    tok = CLIPTokenizer.from_pretrained(os.path.join(BASE_DIR, "tokenizer"))
    ma = MotionAdapter.from_pretrained(adapter, torch_dtype=torch.float32)
    um = UNetMotionModel.from_unet2d(unet, ma)

    if scheduler == "ddim":
        sch = DDIMScheduler.from_pretrained(os.path.join(BASE_DIR, "scheduler"))
    elif scheduler == "dpm":
        sch = DPMSolverMultistepScheduler.from_pretrained(os.path.join(BASE_DIR, "scheduler"))
    else:
        sch = EulerAncestralDiscreteScheduler.from_pretrained(os.path.join(BASE_DIR, "scheduler"))

    pipe = AnimateDiffPipeline(unet=um, vae=vae, text_encoder=te, tokenizer=tok,
                               scheduler=sch, motion_adapter=ma)
    print(f"[load] pipeline built in {time.time()-t0:.1f}s "
          f"(unet {sum(p.numel() for p in um.parameters())/1e6:.0f}M params)", flush=True)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--scheduler", default="dpm")
    ap.add_argument("--adapter", default=ADAPTER_V15)
    ap.add_argument("--prompt", default=ANIME_PROMPT)
    ap.add_argument("--neg", default=ANIME_NEG)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", default="anime")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True

    print(f"[cfg] res={a.res} frames={a.frames} steps={a.steps} sched={a.scheduler} "
          f"base=counterfeit-v3 adapter={os.path.basename(a.adapter)}", flush=True)

    pipe = build(a.scheduler, a.adapter)
    g = torch.Generator().manual_seed(a.seed)

    t0 = time.time()
    with torch.no_grad():
        out = pipe(prompt=a.prompt, negative_prompt=a.neg,
                   num_frames=a.frames, num_inference_steps=a.steps,
                   guidance_scale=7.5, height=a.res, width=a.res, generator=g)
    dt = time.time() - t0
    frames = out.frames[0]
    print(f"[gen] {a.res}x{a.res} x{a.frames}f x{a.steps}step = {dt:.1f}s "
          f"({dt/a.steps:.1f}s/step)", flush=True)

    import numpy
    from PIL import Image
    os.makedirs(OUT_DIR, exist_ok=True)
    imgs = [Image.fromarray(np.asarray(f)).convert("RGB") for f in frames]
    tag = f"{a.tag}_{a.scheduler}_{a.res}_{a.frames}f_{a.steps}s"
    gif = os.path.join(OUT_DIR, tag + ".gif")
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=200, loop=0)
    # every frame into its own subfolder, so the frames can be browsed as a sequence
    frames_dir = os.path.join(OUT_DIR, tag + "_frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i, im in enumerate(imgs):
        im.save(os.path.join(frames_dir, f"f{i:03d}.png"))
    print(f"[save] {gif}\n[save] {frames_dir}\\f000..f{len(imgs)-1:03d}.png", flush=True)


if __name__ == "__main__":
    main()
