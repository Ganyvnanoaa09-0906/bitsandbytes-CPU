# -*- coding: utf-8 -*-
"""Noise-injection test: is the "frosted glass" detail melt caused by residual noise in
the relay latent, or by VAE non-linearity at high latent energy?

Method (no diffusion sampling, seconds per point):
  z = VAE.encode(frame0).mode() * sf        # clean, fully-denoised 5-step latent
  z_noisy = z + ratio * z.std() * N(0,1)    # inject residual noise
  decode(z_noisy / sf) -> measure detail / blowout / green

Controls:
  * baseline is the ROUND-TRIP (ratio=0), not the original PNG: re-encoding a frame and
    decoding it already changes Laplacian variance (f0: PNG 423.8 -> roundtrip 749.9),
    so comparing injected noise against the original PNG would confound VAE sharpening
    with the noise effect.
  * 3 noise seeds per ratio, averaged.
  * the actual late frame (f33) is encoded and decoded as a reference target, since its
    stats are what we are trying to reproduce.

Target (observed f33): lap ~26.8, blown% ~7.69, mean ~158.7, green_px ~684.

Usage:
  python noise_inject_test.py --frames <dir> --src 0 --target 33
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
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

VAE_DIR = r"D:\work\textmodel\counterfeit-v3-diffusers\vae"
RATIOS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
SEEDS = [0, 1, 2]


def lapvar(a):
    g = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    return float(lap.var())


def stats(a):
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    mx = a.max(2)
    mn = a.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    green = g - np.maximum(r, b)
    return dict(lap=lapvar(a), mean=float(a.mean()),
                blown=float(((lum > 235) & (sat < 0.08)).mean() * 100),
                green_px=int((green > 25).sum()),
                gmax=float(green.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--src", type=int, default=0)
    ap.add_argument("--target", type=int, default=33)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    outdir = a.out_dir or os.path.join(a.frames, "noise_test")
    os.makedirs(outdir, exist_ok=True)
    ps = sorted(glob.glob(os.path.join(a.frames, "*_f*.png")),
                key=lambda p: int(re.search(r"_f(\d+)\.png$", p).group(1)))

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_DIR, torch_dtype=torch.float32)
    vae.eval()
    sf = vae.config.scaling_factor
    print(f"[cfg] vae scaling_factor={sf}")

    def enc(p):
        img = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
        t = torch.from_numpy(img / 255.0).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        with torch.no_grad():
            return vae.encode(t).latent_dist.mode() * sf, img

    def dec(z_unet):
        with torch.no_grad():
            d = vae.decode(z_unet / sf, return_dict=False)[0]
        d = (d / 2 + 0.5).clamp(0, 1)
        return (d[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)

    z0, img0 = enc(ps[a.src])
    zt, imgt = enc(ps[a.target])
    print(f"[src ] frame {a.src}: z.std={float(z0.std()):.4f}  PNG " +
          f"lap={stats(img0)['lap']:.1f} blown={stats(img0)['blown']:.2f} "
          f"green={stats(img0)['green_px']}")
    print(f"[tgt ] frame {a.target}: z.std={float(zt.std()):.4f}  PNG " +
          f"lap={stats(imgt)['lap']:.1f} blown={stats(imgt)['blown']:.2f} "
          f"green={stats(imgt)['green_px']}")

    tgt_rt = stats(dec(zt).astype(np.float32))
    print(f"[tgt ] frame {a.target} round-trip: lap={tgt_rt['lap']:.1f} "
          f"blown={tgt_rt['blown']:.2f} mean={tgt_rt['mean']:.1f} "
          f"green={tgt_rt['green_px']}   <-- the stats we want to reproduce")
    print()

    print(f"{'noise ratio':>12} {'lap':>9} {'blown%':>8} {'mean':>7} {'green_px':>9} "
          f"{'Gmax':>6}   (avg of {len(SEEDS)} seeds)")
    print("-" * 70)
    rows = []
    for ratio in RATIOS:
        acc = []
        for sd in SEEDS:
            g = torch.Generator().manual_seed(sd)
            z_n = z0 + ratio * float(z0.std()) * torch.randn(z0.shape, generator=g)
            acc.append(stats(dec(z_n).astype(np.float32)))
        m = {k: float(np.mean([x[k] for x in acc])) for k in acc[0]}
        rows.append((ratio, m))
        print(f"{ratio:>12.2f} {m['lap']:>9.1f} {m['blown']:>8.2f} {m['mean']:>7.1f} "
              f"{m['green_px']:>9.0f} {m['gmax']:>6.0f}")
        if ratio in (0.05, 0.1, 0.2):
            Image.fromarray(dec(z0 + ratio * float(z0.std()) *
                                torch.randn(z0.shape, generator=torch.Generator().manual_seed(0)))
                            ).save(os.path.join(outdir, f"noise_{ratio:.2f}.png"))

    base_lap = rows[0][1]["lap"]
    tgt_lap = tgt_rt["lap"]
    print(f"\n[analysis] baseline round-trip lap={base_lap:.1f}; target f{a.target} lap={tgt_lap:.1f}")
    hit = None
    for ratio, m in rows:
        if m["lap"] <= tgt_lap * 1.3:
            hit = (ratio, m)
            break
    if hit:
        print(f"  -> noise ratio {hit[0]:.2f} reproduces the detail loss "
              f"(lap {hit[1]['lap']:.1f} vs target {tgt_lap:.1f})")
        if hit[0] <= 0.1:
            print("  -> VERDICT: residual noise explains the frosted glass. Fix = more relay steps.")
        elif hit[0] <= 0.3:
            print("  -> VERDICT: residual noise explains it but only at high levels; "
                  "check whether relay chunks plausibly leave that much noise.")
        else:
            print("  -> VERDICT: implausible noise level needed; VAE non-linearity likely dominant.")
    else:
        print("  -> VERDICT: NO noise level reproduces the detail loss -> the blur is baked in "
              "during generation, not a decode-side effect.")


if __name__ == "__main__":
    main()
