# -*- coding: utf-8 -*-
"""Detail-restoration comparison for the soft relay frames.

The relay chunks plateau ~25-50% below full-schedule frames in high-frequency energy, and
the user judges even f009 (a full-schedule frame) as slightly soft. Three cheap restorers are
compared on the same frames:

  A  fsrcnn4_down   : FSRCNN x4 upscale then bicubic back to source size. Injecting learned
                      detail and resampling back is a classic way to add plausible texture
                      without changing the output resolution.
  B  fsrcnn2        : FSRCNN x4 then downscale to 2x the source, i.e. an actual 2x upscale
                      deliverable (1024x1024 from 512x512).
  C  rcas           : FSR 1.0 RCAS sharpening (contrast-adaptive, designed to avoid halos).
  D  unsharp        : plain unsharp mask, included as the cheap baseline.

Reports Laplacian variance for each so the amount of injected high frequency is comparable.

Usage:
  python detail_restore.py --src <frames dir> --picks 9,30 --out <dir>
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
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

FSRCNN_PATH = r"D:\work\textmodel\fsrcnn_x4_v2.pth"


class FSRCNN(nn.Module):
    def __init__(self, in_ch=3, d=64, s=16, m=4, upscale=4):
        super().__init__()
        self.feats = nn.Sequential()
        self.feats.add_module('0', nn.Conv2d(in_ch, d, 5, padding=2))
        self.feats.add_module('1', nn.PReLU(d))
        self.feats.add_module('2', nn.Conv2d(d, s, 1))
        self.feats.add_module('3', nn.PReLU(s))
        for i in range(m):
            self.feats.add_module(f'{4+i}',
                                  nn.Sequential(nn.Conv2d(s, s, 3, padding=1), nn.PReLU(s)))
        self.feats.add_module('8', nn.Conv2d(s, d, 3, padding=1))
        self.feats.add_module('9', nn.PReLU(d))
        self.feats.add_module('10', nn.Conv2d(d, in_ch * upscale * upscale, 3, padding=1))
        self.upscale = upscale

    def forward(self, x):
        return F.pixel_shuffle(self.feats(x), self.upscale)


def load_sr(path=FSRCNN_PATH):
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    m = FSRCNN(d=sd['feats.0.weight'].shape[0], s=sd['feats.2.weight'].shape[0])
    m.load_state_dict(sd)
    m.eval()
    return m


def rcas(img, sharpness=0.8, limit=0.25 - (1.0 / 16.0)):
    """FSR 1.0 RCAS - contrast-adaptive sharpening (implementation from make_manju_video.py)."""
    pad = F.pad(img, (1, 1, 1, 1), mode='replicate')
    b_ = pad[:, :, :-2, 1:-1]
    d_ = pad[:, :, 2:, 1:-1]
    f_ = pad[:, :, 1:-1, :-2]
    h_ = pad[:, :, 1:-1, 2:]
    e_ = img

    def lum(x):
        return 0.5 * x[:, 2:3] + (0.5 * x[:, 0:1] + x[:, 1:2])

    bL, dL, eL, fL, hL = lum(b_), lum(d_), lum(e_), lum(f_), lum(h_)
    nz = 0.25 * (bL + dL + fL + hL) - eL
    mx = torch.maximum(torch.maximum(torch.maximum(bL, dL), eL), torch.maximum(fL, hL))
    mn = torch.minimum(torch.minimum(torch.minimum(bL, dL), eL), torch.minimum(fL, hL))
    nz = torch.clamp(torch.abs(nz) / (mx - mn + 1e-6), 0, 1)
    nz = -0.5 * nz + 1.0
    ring_min = torch.minimum(torch.minimum(torch.minimum(b_, d_), f_), h_)
    ring_max = torch.maximum(torch.maximum(torch.maximum(b_, d_), f_), h_)
    hitMin = torch.minimum(ring_min, e_) / (4.0 * ring_max + 1e-8)
    hitMax = (1.0 - torch.maximum(ring_max, e_)) / (4.0 * ring_min - 4.0 + 1e-8)
    lobe_rgb = torch.maximum(-hitMin, hitMax)
    lobe = torch.maximum(torch.tensor(-limit),
                         torch.minimum(lobe_rgb.max(1, keepdim=True).values,
                                       torch.tensor(0.0))) * sharpness * nz
    rcpL = 1.0 / (4.0 * lobe + 1.0)
    return torch.clamp((lobe * (b_ + d_ + h_ + f_) + e_) * rcpL, 0, 1)


def unsharp(pil, radius=1.2, amount=0.7, thresh=3):
    from PIL import ImageFilter
    blur = pil.filter(ImageFilter.GaussianBlur(radius))
    a = np.asarray(pil).astype(np.float32)
    b = np.asarray(blur).astype(np.float32)
    d = a - b
    d = np.where(np.abs(d) > thresh, d, 0)
    return Image.fromarray(np.clip(a + amount * d, 0, 255).astype(np.uint8))


def lapvar(a):
    a = np.asarray(a).astype(np.float32)
    g = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    return float((4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
                  - g[1:-1, :-2] - g[1:-1, 2:]).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--picks", default="9,30")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    sr = load_sr()
    picks = [int(x) for x in a.picks.split(",")]

    print(f"{'frame':>6} {'variant':>16} {'Laplacian':>10} {'vs src':>8} {'sec':>6}")
    print("-" * 52)
    for pi in picks:
        p = os.path.join(a.src, f"f{pi:03d}.png")
        pil = Image.open(p).convert("RGB")
        src_lap = lapvar(pil)
        print(f"{pi:>6} {'source':>16} {src_lap:>10.1f} {1.0:>8.2f} {'-':>6}")
        t = torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        t0 = time.time()
        with torch.no_grad():
            up = sr(t)                                  # 4x -> 2048
        down = F.interpolate(up, size=pil.size[::-1], mode='bicubic', align_corners=False)
        arr = (down.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(a.out, f"f{pi:03d}_A_fsrcnn4down.png"))
        print(f"{pi:>6} {'A fsrcnn4->down':>16} {lapvar(arr):>10.1f} "
              f"{lapvar(arr)/src_lap:>8.2f} {time.time()-t0:>6.1f}")

        t0 = time.time()
        arr2 = (up.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr2).save(os.path.join(a.out, f"f{pi:03d}_B_fsrcnn2x1024.png"))
        print(f"{pi:>6} {'B fsrcnn 2x(1024)':>16} {lapvar(arr2):>10.1f} "
              f"{lapvar(arr2)/src_lap:>8.2f} {time.time()-t0:>6.1f}")

        t0 = time.time()
        with torch.no_grad():
            rc = rcas(t, sharpness=0.9)
        arr3 = (rc.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr3).save(os.path.join(a.out, f"f{pi:03d}_C_rcas.png"))
        print(f"{pi:>6} {'C rcas':>16} {lapvar(arr3):>10.1f} "
              f"{lapvar(arr3)/src_lap:>8.2f} {time.time()-t0:>6.1f}")

        u = unsharp(pil)
        u.save(os.path.join(a.out, f"f{pi:03d}_D_unsharp.png"))
        print(f"{pi:>6} {'D unsharp':>16} {lapvar(u):>10.1f} "
              f"{lapvar(u)/src_lap:>8.2f} {'-':>6}")
        print()

    print(f"[save] variants -> {a.out}")


if __name__ == "__main__":
    main()
