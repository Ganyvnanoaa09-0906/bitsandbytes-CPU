# -*- coding: utf-8 -*-
"""Is the green blob the Counterfeit-V3 VAE's fault, or is it in the content?

Two independent VAEs are available:
  A. counterfeit-v3-diffusers/vae  (converted from the Counterfeit-V3 checkpoint)
  B. sd15_full/vae                 (the plain SD1.5 VAE)

Cross round-trip on a frame that visibly carries the green blob:
  encode(frame) with VAE_X -> decode with VAE_X -> compare green counts.
If VAE A amplifies green (input 385 -> 898 as measured) while VAE B does not, the
converted VAE is the culprit and swapping it is a one-line fix.

Also tests a surgical repair: clamp the latent cells that sit under the green blobs and
re-decode, to see whether green can be removed without wrecking the image.

Usage:
  python vae_compare_diag.py --frame <png> [--frames-dir <dir> --pick N]
"""
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
import glob
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

VAES = {
    "counterfeit": r"D:\work\textmodel\counterfeit-v3-diffusers\vae",
    "sd15": r"D:\work\textmodel\sd15_full\vae",
}


def green_mask(a, thr=25):
    a = a.astype(np.float32)          # uint8 subtraction underflows -> always cast
    return (a[:, :, 1] - np.maximum(a[:, :, 0], a[:, :, 2])) > thr


def stats(a):
    a = a.astype(np.float32)
    gm = green_mask(a)
    g = a[:, :, 1] - np.maximum(a[:, :, 0], a[:, :, 2])
    r, gg, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = 0.299 * r + 0.587 * gg + 0.114 * b
    mx, mn = a.max(2), a.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    lap = (4 * lum[1:-1, 1:-1] - lum[:-2, 1:-1] - lum[2:, 1:-1]
           - lum[1:-1, :-2] - lum[1:-1, 2:]).var()
    return dict(green=int(gm.sum()), gmax=float(g.max()), mean=float(a.mean()),
                white=float(100 * (a > 245).mean()), lap=float(lap))


def show(tag, s):
    print(f"  {tag:<34} green={s['green']:>5}  Gmax={s['gmax']:>5.0f}  "
          f"mean={s['mean']:>6.1f}  white%={s['white']:>6.2f}  lap={s['lap']:>8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--pick", type=int, default=0)
    a = ap.parse_args()

    ps = sorted(glob.glob(os.path.join(a.frames_dir, "*_f*.png")),
                key=lambda p: int(re.search(r"_f(\d+)\.png$", p).group(1)))
    p = ps[a.pick]
    img = np.asarray(Image.open(p).convert("RGB"))
    print(f"[src] {os.path.basename(p)}  {img.shape}")
    show("original PNG", stats(img))

    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0) * 2 - 1

    print("\n=== round-trip with each VAE (same input frame) ===")
    for name, path in VAES.items():
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(path, torch_dtype=torch.float32)
        vae.eval()
        sf = vae.config.scaling_factor
        with torch.no_grad():
            z = vae.encode(t).latent_dist.mode() * sf
            d = vae.decode(z / sf, return_dict=False)[0]
        out = (((d / 2 + 0.5).clamp(0, 1))[0].permute(1, 2, 0).cpu().numpy() * 255
               ).round().astype(np.uint8)
        s = stats(out)
        show(f"{name} VAE round-trip", s)
        print(f"      -> green amplification vs input: {s['green']/max(stats(img)['green'],1):.2f}x")
        Image.fromarray(out).save(
            os.path.join(a.frames_dir, f"vae_rt_{name}_f{a.pick:03d}.png"))

        # surgical test: clamp latent cells under the green blobs
        gm = green_mask(img)
        H, W = img.shape[:2]
        ch, cw = H // z.shape[2], W // z.shape[3]
        cell_mask = np.zeros(z.shape[2:], dtype=bool)
        for i in range(z.shape[2]):
            for j in range(z.shape[3]):
                if gm[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw].sum() > 4:
                    cell_mask[i, j] = True
        if cell_mask.sum():
            zc = z.clone()
            cm = torch.from_numpy(cell_mask).to(z.device)
            # pull those cells toward the global mean magnitude
            for c in range(zc.shape[1]):
                sel = zc[0, c][cm]
                if sel.numel():
                    med = sel.median()
                    zc[0, c][cm] = torch.clamp(sel, -abs(med) * 1.0, abs(med) * 1.0)
            with torch.no_grad():
                d2 = vae.decode(zc / sf, return_dict=False)[0]
            out2 = (((d2 / 2 + 0.5).clamp(0, 1))[0].permute(1, 2, 0).cpu().numpy() * 255
                    ).round().astype(np.uint8)
            show(f"{name} + clamp green cells ({int(cell_mask.sum())})", stats(out2))
            Image.fromarray(out2).save(
                os.path.join(a.frames_dir, f"vae_clamp_{name}_f{a.pick:03d}.png"))


if __name__ == "__main__":
    main()
