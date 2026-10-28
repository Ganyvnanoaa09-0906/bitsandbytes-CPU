# -*- coding: utf-8 -*-
"""Run ONE sliding-window config at a given carry-noise, saving a GIF + all frames
into adiff_out/scan_cn_<value>/ per the requested layout.

Prints the boundary/intra diagnostics so the next decision (continue vs stop) can be made
from a single number.

Usage:
  python scan_cn_one.py --res 512 --chunk 10 --overlap 2 --chunks 3 --steps 6 --carry-noise 500
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
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import argparse
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scripts.lib.anime_adiff import build, ANIME_PROMPT, ANIME_NEG
from scripts.lib import temporal_metrics
from scripts.lib.sweep_carry_noise import generate, measure

OUT_ROOT = r"D:\work\output\adiff_out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--carry-noise", type=float, required=True)
    ap.add_argument("--prompt", default=ANIME_PROMPT)
    ap.add_argument("--neg", default=ANIME_NEG)
    ap.add_argument("--outdir", default=None,
                    help="output directory (defaults to adiff_out/scan_cn<value>)")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True

    cn_tag = f"cn{int(a.carry_noise)}"
    outdir = a.outdir or os.path.join(OUT_ROOT, f"scan_{cn_tag}")
    os.makedirs(outdir, exist_ok=True)

    print(f"[cfg] res={a.res} chunk={a.chunk} overlap={a.overlap} chunks={a.chunks} "
          f"steps={a.steps} carry-noise={a.carry_noise} solver=DPM-Solver++(2M) "
          f"base=counterfeit-v3", flush=True)
    print(f"[out] {outdir}", flush=True)

    pipe = build("dpm")
    with torch.no_grad():
        pe, ne = pipe.encode_prompt(a.prompt, "cpu", 1, True, negative_prompt=a.neg)
    embeds = torch.cat([ne, pe])

    # deterministic noise stream (same warm state as the sweep script uses)
    gprobe = torch.Generator().manual_seed(a.seed)
    torch.randn((1, 4, a.chunk, a.res // pipe.vae_scale_factor,
                 a.res // pipe.vae_scale_factor), generator=gprobe)
    gen_state = {"seed": a.seed, "warm": gprobe.get_state().clone()}

    frames, dt, log = generate(pipe, embeds, gen_state, a.res, a.chunk, a.overlap,
                               a.chunks, a.steps, a.carry_noise, a.prompt, a.neg)
    m = measure(frames, a.chunk, a.overlap)

    tag = f"scan_{cn_tag}_{a.res}_{a.chunk}c{a.overlap}o{a.chunks}n_{a.steps}s"
    imgs = [Image.fromarray(f).convert("RGB") for f in frames]
    gif = os.path.join(outdir, tag + ".gif")
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=200, loop=0)
    # Frame sequence goes into its own subfolder so the output directory stays browsable
    # (GIF + metrics at the top level, the individual frames in frames/).
    frames_dir = os.path.join(outdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i, im in enumerate(imgs):
        im.save(os.path.join(frames_dir, f"f{i:03d}.png"))

    with open(os.path.join(outdir, f"{tag}.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"res={a.res} chunk={a.chunk} overlap={a.overlap} chunks={a.chunks} "
                 f"steps={a.steps} carry_noise={a.carry_noise}\n")
        fh.write(f"frames={len(frames)} seconds={dt:.1f}\n")
        for k, v in m.items():
            fh.write(f"{k}={v}\n")
        fh.write("log:\n")
        for ln in log:
            fh.write("  " + ln + "\n")

    print(f"\n[RESULT] carry_noise={a.carry_noise:.0f} res={a.res} frames={len(frames)}")
    print(f"  intra MAD   = {m['intra']:.2f}")
    print(f"  boundary MAD= {m['boundary']:.2f}")
    print(f"  boundary/intra RATIO = {m['ratio']:.3f}")
    print(f"  hist_drift  = {m['hist']:.4f}")
    print(f"  full-mean MAD = {m['mad']:.2f}")
    print(f"  time        = {dt:.1f}s")
    print(f"  chunks: {'; '.join(log)}")
    print(f"[save] {gif}  (+ {len(imgs)} PNGs)", flush=True)


if __name__ == "__main__":
    main()
