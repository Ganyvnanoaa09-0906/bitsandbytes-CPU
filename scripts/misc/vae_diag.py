# -*- coding: utf-8 -*-
"""Zero-cost VAE diagnostics on EXISTING PNG frames (no diffusion sampling).

There is no latent cache on disk, so we synthesise latents by VAE-encoding the saved
frames. That is enough to answer:

  1. Is the "frosted glass / detail melt" a decode-side response to latent magnitude?
     -> decode the SAME latent at several magnitude scales and watch Laplacian variance
        (detail) and blown-highlight fraction.
        If amplifying reproduces the frosted+blown look, the fault is latent magnitude
        (and normalisation can fix it). If it merely brightens uniformly, the detail was
        destroyed during generation and is not recoverable.

  2. Is the green hard-edged blob a latent overflow (fixable by clipping) or a decoder
     artefact?
     -> round-trip f000 (which carries a strong green blob, Gmax~128) and also round-trip
        a frame, then look for green blobs in the output.

  3. Round-trip fidelity baseline: encode->decode of a clean frame should reproduce it.

Usage:
  python vae_diag.py --frames <dir with _fNNN.png> --pick 0,6,18,33
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


def lapvar(a):
    """Laplacian variance on luma: proxy for high-frequency detail energy."""
    g = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    return float(lap.var())


def stats(a):
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    mx = a.max(2)
    mn = a.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    blown = float(((lum > 235) & (sat < 0.08)).mean() * 100)
    green = g - np.maximum(r, b)
    return dict(lap=lapvar(a), mean=float(a.mean()), blown=blown,
                green_px=int((green > 25).sum()), gmax=float(green.max()),
                sat=float(sat.mean()))


def show(tag, s):
    print(f"  {tag:<26} lap={s['lap']:>8.1f}  mean={s['mean']:>6.1f}  "
          f"blown%={s['blown']:>6.2f}  green_px={s['green_px']:>5}  "
          f"Gmax={s['gmax']:>5.0f}  sat={s['sat']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--pick", default="0,6,18,33")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    picks = [int(x) for x in a.pick.split(",") if x.strip()]
    outdir = a.out_dir or os.path.join(a.frames, "vae_diag")
    os.makedirs(outdir, exist_ok=True)

    ps = sorted(glob.glob(os.path.join(a.frames, "*_f*.png")),
                key=lambda p: int(re.search(r"_f(\d+)\.png$", p).group(1)))

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_DIR, torch_dtype=torch.float32)
    vae.eval()
    sf = vae.config.scaling_factor
    print(f"[cfg] vae scaling_factor={sf}  shift_factor={getattr(vae.config,'shift_factor',None)}")

    def dec(z):
        with torch.no_grad():
            d = vae.decode(z, return_dict=False)[0]
        d = (d / 2 + 0.5).clamp(0, 1)
        return (d[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)

    for pi in picks:
        p = ps[pi]
        img = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
        t = torch.from_numpy(img / 255.0).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        with torch.no_grad():
            z = vae.encode(t).latent_dist.mode() * sf   # store in "latent * sf" convention

        print(f"\n=== frame {pi} ({os.path.basename(p)}) ===")
        show("original PNG", stats(img))

        # round-trip baseline
        rt = dec(z / sf)
        show("roundtrip x1.00", stats(rt.astype(np.float32)))

        # magnitude sweep: does amplifying the latent reproduce frosted+blown?
        for m in (0.7, 0.85, 1.15, 1.3, 1.5, 2.0):
            out = dec(z / sf * m)
            show(f"decode x{m:.2f}", stats(out.astype(np.float32)))
            if pi in (0, 33) and m in (1.3, 2.0):
                Image.fromarray(out).save(os.path.join(outdir, f"f{pi:03d}_x{m:.2f}.png"))

        # clipping test for the green blob
        zc = z.clone()
        zc = zc.clamp(-4 * sf, 4 * sf)
        out_c = dec(zc / sf)
        show("clip(latent +-4)", stats(out_c.astype(np.float32)))
        Image.fromarray(out_c).save(os.path.join(outdir, f"f{pi:03d}_clip4.png"))

        zc2 = z.clamp(-2 * sf, 2 * sf)
        out_c2 = dec(zc2 / sf)
        show("clip(latent +-2)", stats(out_c2.astype(np.float32)))

    print(f"\n[save] diagnostic images -> {outdir}")


if __name__ == "__main__":
    main()
