# -*- coding: utf-8 -*-
"""Final post-processing for the 512 anime-drama pipeline.

Frozen decision: FSRCNN is the detail restorer.
  * Real-ESRGAN x4plus anime 6B was rejected -- it VECTORISES/flattens existing texture
    (on an already-crisp character it *reduced* Laplacian to x0.87 and looked "plastic"),
    and it costs 17 s/frame vs FSRCNN's 0.24 s/frame.
  * FSRCNN behaves consistently regardless of input content (x1.37-1.67) and is ~70x faster.

Resample note: the x4 output is decimated back to source size with `area` (box filter),
which is the correct filter for exact 4x decimation. bicubic is avoided because it rings
badly on very sharp input (that mistake inflated an earlier ESRGAN measurement 2.8x).

Outputs (into --out):
    frames/f000.png ...      restored frame sequence
    <name>.gif               animation
    <name>.mp4               video (needs the `av` module; skipped with a warning if absent)

Usage:
  python post_process.py --src <dir with frames/> --out <dir> --name final_pipeline_512
"""
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

from detail_restore import load_sr  # noqa: E402


def lapvar(a):
    a = np.asarray(a).astype(np.float32)
    g = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    return float((4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
                  - g[1:-1, :-2] - g[1:-1, 2:]).var())


def write_mp4(frames, path, fps):
    """Encode with PyAV. Returns True on success."""
    try:
        import av
    except ImportError:
        print(f"[warn] `av` not available -> skipping {os.path.basename(path)} "
              f"(run this script with an interpreter that has PyAV)")
        return False
    from fractions import Fraction
    try:
        out = av.open(path, "w")
        st = out.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
    except Exception:
        out = av.open(path, "w")
        st = out.add_stream("mpeg4", rate=Fraction(fps).limit_denominator(1000))
    st.width, st.height = frames[0].shape[1], frames[0].shape[0]
    st.pix_fmt = "yuv420p"
    st.bit_rate = 16_000_000
    for fr in frames:
        for p in st.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
            out.mux(p)
    for p in st.encode():
        out.mux(p)
    out.close()
    print(f"[save] {path}  ({len(frames)} frames @ {fps}fps, "
          f"{len(frames)/fps:.2f}s)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="frames dir, or its parent")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="final_pipeline_512")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--no-esrgan-check", action="store_true",
                    help="skip the informative ESRGAN cross-check")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)

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

    fdir = os.path.join(a.out, "frames")
    os.makedirs(fdir, exist_ok=True)
    print(f"[cfg] {len(ps)} frames  src={src}")
    print(f"[cfg] out={a.out}  name={a.name}  fps={a.fps}")
    print(f"[cfg] restorer=FSRCNN x4 -> area decimate")

    sr = load_sr()
    laps_before, laps_after = [], []
    imgs = []
    t_all = time.time()
    for i, p in enumerate(ps):
        pil = Image.open(p).convert("RGB")
        laps_before.append(lapvar(pil))
        t = torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1
                                                            ).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            up = sr(t)
            dn = F.interpolate(up, size=pil.size[::-1], mode="area").clamp(0, 1)
        arr = (dn[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        laps_after.append(lapvar(arr))
        im = Image.fromarray(arr)
        im.save(os.path.join(fdir, f"f{i:03d}.png"))
        imgs.append(im)
        if (i + 1) % 10 == 0 or i == len(ps) - 1:
            print(f"  {i+1}/{len(ps)}  lap {laps_before[-1]:.1f} -> {laps_after[-1]:.1f}",
                  flush=True)

    dt = time.time() - t_all
    gif = os.path.join(a.out, f"{a.name}.gif")
    imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                 duration=int(round(1000 / a.fps)), loop=0)
    print(f"[save] {gif}  ({len(imgs)} frames @ {a.fps}fps)")
    mp4_ok = write_mp4([np.asarray(im) for im in imgs],
                       os.path.join(a.out, f"{a.name}.mp4"), a.fps)

    mb = float(np.mean(laps_before))
    ma = float(np.mean(laps_after))
    print(f"\n{'='*64}")
    print(f"RESULT  {len(ps)} frames in {dt:.1f}s ({dt/len(ps):.2f}s/frame)")
    print(f"  Laplacian mean : {mb:.1f} -> {ma:.1f}  (x{ma/mb:.2f})")
    print(f"  per-frame gain : x{min(np.array(laps_after)/np.array(laps_before)):.2f} "
          f".. x{max(np.array(laps_after)/np.array(laps_before)):.2f}")
    print(f"  frames dir     : {fdir}")

    with open(os.path.join(a.out, f"{a.name}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(src=src, restorer="fsrcnn", decimate="area", fps=a.fps,
                       n=len(ps), lap_before=laps_before, lap_after=laps_after,
                       lap_mean_before=mb, lap_mean_after=ma,
                       seconds=dt, mp4=mp4_ok), fh, indent=2)


if __name__ == "__main__":
    main()
