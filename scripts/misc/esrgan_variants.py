# -*- coding: utf-8 -*-
"""ESRGAN downscale/blend variants, to find why anime-6B looked worse than FSRCNN.

The first attempt did  x4 -> bicubic back to 512. Bicubic by an exact factor of 4 with
align_corners=False is a known source of ringing/aliasing, so the loss of quality may be in
the RESAMPLE step rather than in the ESRGAN output itself.

Variants produced per frame:
  src             original
  fsrcnn          the version judged better  (x4 -> bicubic down)
  e_bicubic       ESRGAN x4 -> bicubic down            (the first attempt)
  e_area          ESRGAN x4 -> area (box) down         (correct filter for integer decimation)
  e_lanczos       ESRGAN x4 -> lanczos down
  e_blend50       50/50 mix of e_area with the source  (milder, keeps original texture)
  e_blend35       35% e_area + 65% source              (mildest)
  e_native_crop   centre crop of the native 2048 output (no resample at all)

Usage:
  python esrgan_variants.py --src <frames dir> --picks 9,30 --out <dir>
"""
import os, sys
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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scripts.lib.detail_restore import load_sr, lapvar
from scripts.lib.realesrgan_anime_test import load_realesrgan


def to_t(pil):
    return torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def to_img(t):
    return Image.fromarray((t.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--picks", default="9,30")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    esr = load_realesrgan()
    fsc = load_sr()

    for pi in [int(x) for x in a.picks.split(",")]:
        p = os.path.join(a.src, f"f{pi:03d}.png")
        pil = Image.open(p).convert("RGB")
        src_lap = lapvar(pil)
        t = to_t(pil)
        W, H = pil.size

        with torch.no_grad():
            eup = esr(t)                                   # 2048
            fup = fsc(t)                                   # 2048

        outs = {"src": pil}
        outs["fsrcnn"] = to_img(F.interpolate(fup, size=(H, W), mode="bicubic",
                                              align_corners=False))
        outs["e_bicubic"] = to_img(F.interpolate(eup, size=(H, W), mode="bicubic",
                                                 align_corners=False))
        outs["e_area"] = to_img(F.interpolate(eup, size=(H, W), mode="area"))
        # torch has no lanczos; PIL does
        outs["e_lanczos"] = to_img(eup).resize((W, H), Image.LANCZOS)
        for tag, w in (("e_blend50", 0.5), ("e_blend35", 0.35)):
            base = F.interpolate(eup, size=(H, W), mode="area")
            outs[tag] = to_img(base * w + t * (1 - w))
        # native centre crop of the 2048 output, shown at 512 for a like-for-like view
        c = eup.shape[-1] // 2
        half = W // 2
        outs["e_native_crop"] = to_img(eup[..., c - half:c + half, c - half:c + half])

        print(f"\n=== f{pi:03d}  source lap={src_lap:.1f} ===")
        print(f"{'variant':>16} {'Laplacian':>10} {'x src':>7} {'note':>34}")
        print("-" * 72)
        notes = {"src": "(reference)", "fsrcnn": "the version judged better",
                 "e_bicubic": "first attempt: x4 then bicubic down (ringing risk)",
                 "e_area": "box filter, correct for 4x decimation",
                 "e_lanczos": "lanczos down",
                 "e_blend50": "50% area + 50% source",
                 "e_blend35": "35% area + 65% source",
                 "e_native_crop": "native 2048 crop, no resample"}
        for name, im in outs.items():
            im.save(os.path.join(a.out, f"f{pi:03d}_{name}.png"))
            l = lapvar(im)
            print(f"{name:>16} {l:>10.1f} {l/src_lap:>7.2f} {notes[name]:>34}")
    print(f"\n[save] {a.out}")


if __name__ == "__main__":
    main()
