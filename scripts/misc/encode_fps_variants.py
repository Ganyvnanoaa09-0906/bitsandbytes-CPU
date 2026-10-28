# -*- coding: utf-8 -*-
"""Encode the 34-frame sliding-window result at several output frame rates so the
minimum acceptable rate for anime-drama content can be judged by eye.

Rationale: real frame interpolation (RIFE/IFNet) is unavailable in this environment, and
RAFT-large is far too slow on this CPU to synthesize ~86 in-between frames. The lever that
costs nothing is output frame rate: lower fps => a given clip lasts longer, so fewer
diffusion frames are needed to reach a target duration.

Also emits a "blend" variant per rate: a cheap linear frame blend (minterpolate-like
temporal smoothing) that does NOT synthesize new motion but softens the step between
frames. Useful to see whether blending rescues low rates or just smears.

Usage:
  python encode_fps_variants.py --src-dir <dir with _fNNN.png> --rates 8,6,5,4,3
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
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import argparse
import glob
import os
import re

import numpy as np
import av
from PIL import Image


def load_frames(src_dir):
    paths = sorted(glob.glob(os.path.join(src_dir, "*_f*.png")),
                   key=lambda p: int(re.search(r"_f(\d+)\.png$", p).group(1)))
    return [np.asarray(Image.open(p).convert("RGB")) for p in paths]


def encode(frames, path, fps):
    from fractions import Fraction
    # PyAV >= 16 requires an exact rational frame rate, not a float
    rate = Fraction(fps).limit_denominator(1000)
    out = av.open(path, "w")
    st = out.add_stream("mpeg4", rate=rate)
    st.width = frames[0].shape[1]
    st.height = frames[0].shape[0]
    st.pix_fmt = "yuv420p"
    st.bit_rate = 12_000_000
    for fr in frames:
        for p in st.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
            out.mux(p)
    for p in st.encode():
        out.mux(p)
    out.close()


def blend_frames(frames, factor):
    """Insert `factor-1` linearly blended frames between each pair (minterpolate-lite).

    This does NOT estimate motion; it cross-fades. On real motion it produces ghosting,
    which is exactly what we want to observe as the failure mode.
    """
    if factor <= 1:
        return list(frames)
    out = []
    for i in range(len(frames)):
        out.append(frames[i])
        if i + 1 < len(frames):
            a = frames[i].astype(np.float32)
            b = frames[i + 1].astype(np.float32)
            for k in range(1, factor):
                w = k / factor
                out.append(((a * (1 - w) + b * w)).round().astype(np.uint8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--rates", default="8,6,5,4,3")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--blend", action="store_true", default=True,
                    help="also emit a frame-blended variant at each rate")
    a = ap.parse_args()

    rates = [float(x) for x in a.rates.split(",") if x.strip()]
    outdir = a.out_dir or os.path.join(a.src_dir, "fps_variants")
    os.makedirs(outdir, exist_ok=True)

    frames = load_frames(a.src_dir)
    n = len(frames)
    print(f"[load] {n} frames from {a.src_dir}  size={frames[0].shape[1]}x{frames[0].shape[0]}")

    print(f"\n{'rate':>6} {'duration':>10} {'variant':>16} {'file':>40}")
    print("-" * 78)
    for r in rates:
        dur = n / r
        p = os.path.join(outdir, f"direct_{r:g}fps.mp4")
        encode(frames, p, r)
        print(f"{r:>6g} {dur:>9.2f}s {'direct':>16} {os.path.basename(p):>40}")

        if a.blend:
            # Blend by a fixed factor of 2 for every rate so the comparison is uniform:
            # the clip keeps its wall-clock duration only if we also raise the output rate,
            # but here we deliberately keep the SAME output rate so blending *lengthens*
            # the clip -- that is exactly what a viewer would see, and it exposes ghosting.
            factor = 2
            bf = blend_frames(frames, factor)
            pb = os.path.join(outdir, f"blend{ factor }x_{r:g}fps.mp4")
            encode(bf, pb, r)
            print(f"{r:>6g} {len(bf)/r:>9.2f}s "
                  f"{'blend%dx (same rate)' % factor:>16} {os.path.basename(pb):>40}")

    print(f"\n[save] all variants -> {outdir}")


if __name__ == "__main__":
    main()
