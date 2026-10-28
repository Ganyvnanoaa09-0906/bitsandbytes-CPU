# -*- coding: utf-8 -*-
"""Locate the green artefacts and test whether they map onto latent outliers.

Two zero-cost questions (one VAE encode, no diffusion sampling):

  A. Are the green blobs STATIC across frames? If they sit in the same image region in
     every frame they are a persistent content/decoder feature; if they wander they are
     stochastic corruption.
     -> coarse 8x8 block histogram of green pixels per sampled frame.

  B. Do the green blobs coincide with LATENT OUTLIERS? The VAE downsamples by 8, so one
     latent cell maps to an 8x8 pixel block. If the green regions line up with latent cells
     whose |z| exceeds k sigma, the artefact is a latent-overflow/decode effect and can be
     attacked with clamping; if not, it originates elsewhere.
     -> encode the frame, find outlier cells, map them back to pixels, compare masks.

Always cast to float before subtracting: uint8 underflow silently corrupts green counts.

Usage:
  python green_diag.py --frames <dir> --picks 0,17,33
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

VAE_DIR = r"D:\work\textmodel\counterfeit-v3-diffusers\vae"
GRID = 8  # blocks per side for localisation


def green_mask(a, thr=25):
    a = a.astype(np.float32)  # uint8 subtraction would underflow
    return (a[:, :, 1] - np.maximum(a[:, :, 0], a[:, :, 2])) > thr


def block_hist(mask, grid=GRID):
    h, w = mask.shape
    bh, bw = h // grid, w // grid
    out = np.zeros((grid, grid), dtype=int)
    for i in range(grid):
        for j in range(grid):
            out[i, j] = int(mask[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw].sum())
    return out


def top_blocks(h, n=4):
    idx = np.dstack(np.unravel_index(np.argsort(h.ravel())[::-1], h.shape))[0]
    return [(int(i), int(j), int(h[i, j])) for i, j in idx[:n] if h[i, j] > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--picks", default="0,17,33")
    a = ap.parse_args()

    picks = [int(x) for x in a.picks.split(",") if x.strip()]
    ps = sorted(glob.glob(os.path.join(a.frames, "*_f*.png")),
                key=lambda p: int(re.search(r"_f(\d+)\.png$", p).group(1)))

    print("=== A. are the green blobs static? (8x8 block histogram, top blocks) ===")
    for pi in picks:
        img = np.asarray(Image.open(ps[pi]).convert("RGB"))
        gm = green_mask(img)
        h = block_hist(gm)
        H, W = img.shape[:2]
        bh, bw = H // GRID, W // GRID
        locs = [(i, j, c, f"px(x={j*bw}-{(j+1)*bw}, y={i*bh}-{(i+1)*bh})")
                for i, j, c in top_blocks(h)]
        print(f"  frame {pi:>3}: green_px={int(gm.sum()):>5}  top blocks:")
        for i, j, c, desc in locs:
            print(f"      block({i},{j}) count={c:>4}  {desc}")

    print("\n=== B. do green blobs sit on latent outliers? ===")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_DIR, torch_dtype=torch.float32)
    vae.eval()
    sf = vae.config.scaling_factor

    for pi in picks[:2]:
        img = np.asarray(Image.open(ps[pi]).convert("RGB")).astype(np.float32)
        t = torch.from_numpy(img / 255.0).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        with torch.no_grad():
            z = vae.encode(t).latent_dist.mode() * sf     # [1,4,64,64] for 512
        zz = z[0]                                          # [4,64,64]
        sd = float(zz.std())
        gm = green_mask(img)
        H, W = img.shape[:2]
        ch, cw = H // zz.shape[1], W // zz.shape[2]        # 8 for 512

        print(f"\n  frame {pi}: latent {tuple(zz.shape)}  z.std={sd:.3f}  "
              f"cell->pixel factor={ch}x{cw}")
        for k in (2.0, 3.0, 4.0):
            out = (zz.abs() > k * sd).any(0).cpu().numpy()   # [64,64] bool
            # upscale latent mask to pixel space
            big = np.kron(out, np.ones((ch, cw), dtype=bool))
            big = big[:H, :W]
            inter = int((big & gm).sum())
            nz = int(big.sum())
            print(f"    |z|>{k:.1f}sigma: {int(out.sum()):>5} latent cells -> {nz:>7} px; "
                  f"overlap with green = {inter:>5} "
                  f"({100*inter/max(int(gm.sum()),1):5.1f}% of green, "
                  f"{100*inter/max(nz,1):5.1f}% of outliers)")
        # where are the biggest outliers
        flat = zz.abs().reshape(zz.shape[0], -1).max(0).values
        top = torch.topk(flat, 6).indices.cpu().numpy()
        cells = [(int(i // zz.shape[2]), int(i % zz.shape[2])) for i in top]
        print(f"    top outlier cells (row,col): {cells}  "
              f"-> pixel y={(cells[0][0]*ch)}, x={(cells[0][1]*cw)}")


if __name__ == "__main__":
    main()
