# -*- coding: utf-8 -*-
"""Apply detail restoration (FSRCNN, optional RCAS) to a whole frames directory.

Why: the sliding-window relay leaves carried chunks about 40-56% below full-schedule frames
in high-frequency energy (Laplacian 164 -> 72 over 4 chunks), which reads as "frosted glass".
FSRCNN x4 -> resample back to source size restores that energy (measured 65.2 -> 108.5 on a
soft relay frame, i.e. up to the level of natively full-schedule frames) at ~0.8 s/frame,
with no visible halos. RCAS (FSR 1.0) is optional extra crispness.

Output layout matches the pipeline convention:
  <out>/frames/f000.png ...   plus   <out>/<tag>.gif   plus   <out>/sharpen.json

Usage:
  python post_sharpen.py --src <dir with frames/fNNN.png> --out <dir> \
      --mode fsrcnn --rcas 0.0
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
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scripts.lib.detail_restore import load_sr, rcas, lapvar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source frames dir (or its parent)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="fsrcnn", choices=["fsrcnn", "none"])
    ap.add_argument("--rcas", type=float, default=0.0, help="RCAS sharpness, 0 disables")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)

    # accept either the frames dir itself or its parent
    src = a.src
    if not glob.glob(os.path.join(src, "f*.png")):
        cand = os.path.join(src, "frames")
        if glob.glob(os.path.join(cand, "f*.png")):
            src = cand
    ps = sorted(glob.glob(os.path.join(src, "f*.png")),
                key=lambda p: int(re.search(r"f(\d+)\.png$", p).group(1)))
    if not ps:
        print(f"[err] no fNNN.png under {src}")
        return

    tag = a.tag or (os.path.basename(os.path.dirname(src.rstrip("\\/"))) or "frames")
    fdir = os.path.join(a.out, "frames")
    os.makedirs(fdir, exist_ok=True)

    sr = load_sr() if a.mode == "fsrcnn" else None
    print(f"[cfg] {len(ps)} frames  mode={a.mode}  rcas={a.rcas}  src={src}")
    print(f"[cfg] out={a.out}")

    laps_before, laps_after = [], []
    imgs = []
    t_all = time.time()
    for i, p in enumerate(ps):
        pil = Image.open(p).convert("RGB")
        laps_before.append(lapvar(pil))
        t = torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            if sr is not None:
                up = sr(t)
                t = F.interpolate(up, size=pil.size[::-1], mode="bicubic",
                                  align_corners=False).clamp(0, 1)
            if a.rcas > 0:
                t = rcas(t, sharpness=a.rcas)
        arr = (t.clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        laps_after.append(lapvar(arr))
        im = Image.fromarray(arr)
        im.save(os.path.join(fdir, f"f{i:03d}.png"))
        imgs.append(im)
        if (i + 1) % 10 == 0 or i == len(ps) - 1:
            print(f"  {i+1}/{len(ps)}  lap {laps_before[-1]:.1f} -> {laps_after[-1]:.1f}",
                  flush=True)

    dt = time.time() - t_all
    gif = os.path.join(a.out, f"{tag}_sharp.gif")
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=200, loop=0)

    mb = float(np.mean(laps_before))
    ma = float(np.mean(laps_after))
    print(f"\n{'='*66}")
    print(f"RESULT  {len(ps)} frames in {dt:.1f}s ({dt/len(ps):.2f}s/frame)")
    print(f"  Laplacian mean : {mb:.1f} -> {ma:.1f}  (x{ma/mb:.2f})")
    print(f"  per-frame gain : min x{min(np.array(laps_after)/np.array(laps_before)):.2f} "
          f"max x{max(np.array(laps_after)/np.array(laps_before)):.2f}")
    print(f"  first/last     : {laps_before[0]:.1f}->{laps_after[0]:.1f} / "
          f"{laps_before[-1]:.1f}->{laps_after[-1]:.1f}")
    print(f"[save] {fdir}\\f000..f{len(imgs)-1:03d}.png")
    print(f"[save] {gif}")

    with open(os.path.join(a.out, "sharpen.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(src=src, mode=a.mode, rcas=a.rcas, n=len(ps),
                       lap_before=laps_before, lap_after=laps_after,
                       lap_mean_before=mb, lap_mean_after=ma, seconds=dt), fh, indent=2)


if __name__ == "__main__":
    main()
