# -*- coding: utf-8 -*-
"""Quantify a depth-of-field prompt's effect: is the BACKGROUND blurred, and did the
SUBJECT stay sharp?

Compares two runs that differ only in prompt (same seeds/res/steps), so any difference is
attributable to the prompt.

Regions are defined geometrically because no face detector is available here:
  subject  = central box (x 25-75%, y 22-78%)   - should stay sharp
  bkg_ring = outer frame minus the subject box  - should lose detail if bokeh worked

For each region: Laplacian variance (high-frequency energy) and local contrast (p95-p5).
Also reports frame-level luminance stats to catch a prompt-induced exposure shift.

Usage:
  python dof_analyze.py --a <baseline frames dir> --b <dof frames dir> [--n 20]
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

import numpy as np
from PIL import Image


def luma(a):
    a = a.astype(np.float32)
    return 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]


def rlap(g):
    return float((4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
                  - g[1:-1, :-2] - g[1:-1, 2:]).var())


def rcontrast(g):
    return float(np.percentile(g, 95) - np.percentile(g, 5))


def region_masks(h, w):
    subj = np.zeros((h, w), bool)
    subj[int(h * 0.22):int(h * 0.78), int(w * 0.25):int(w * 0.75)] = True
    # erode the subject box a little so edge pixels don't leak into the background ring
    ring = ~subj
    return subj, ring


def frame_metrics(p):
    g = luma(np.asarray(Image.open(p).convert("RGB")))
    h, w = g.shape
    subj, ring = region_masks(h, w)
    out = {}
    for name, m in (("subject", subj), ("bkg", ring)):
        gg = g.copy()
        gg[~m] = np.nan
        # laplacian restricted to the region: compute on the mask-eroded area
        lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
        mm = m[1:-1, 1:-1]
        out[f"{name}_lap"] = float((lap[mm] ** 2).mean()) if mm.sum() else float("nan")
        out[f"{name}_contrast"] = rcontrast(g[m]) if m.sum() else float("nan")
        out[f"{name}_mean"] = float(g[m].mean()) if m.sum() else float("nan")
    out["frame_lap"] = rlap(g)
    out["frame_mean"] = float(g.mean())
    return out


def collect(d, n=None):
    ps = sorted(glob.glob(os.path.join(d, "*.png")),
                key=lambda p: int(re.search(r"f(\d+)\.png$", p).group(1)))
    if n:
        ps = ps[:n]
    return [frame_metrics(p) for p in ps], ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()

    fa, pa = collect(a.a, a.n)
    fb, pb = collect(a.b, a.n)
    n = min(len(fa), len(fb))
    print(f"[cmp] baseline {len(fa)} frames vs dof {len(fb)} frames -> using {n}\n")

    keys = ["subject_lap", "bkg_lap", "subject_contrast", "bkg_contrast", "frame_lap",
            "frame_mean"]
    print(f"{'metric':>18} {'baseline':>11} {'dof':>11} {'ratio':>8} {'verdict':>26}")
    print("-" * 80)
    for k in keys:
        va = float(np.nanmean([f[k] for f in fa[:n]]))
        vb = float(np.nanmean([f[k] for f in fb[:n]]))
        r = vb / va if va else float("nan")
        if k == "bkg_lap":
            v = "BACKGROUND BLURRED" if r < 0.6 else ("mild blur" if r < 0.85 else "no effect")
        elif k == "subject_lap":
            v = "subject kept sharp" if r > 0.8 else "subject LOST detail"
        else:
            v = ""
        print(f"{k:>18} {va:>11.2f} {vb:>11.2f} {r:>8.2f} {v:>26}")

    # per-frame background detail, to see whether blur is consistent or sporadic
    print(f"\nper-frame bkg_lap  baseline: " +
          " ".join(f"{f['bkg_lap']:.0f}" for f in fa[:n]))
    print(f"per-frame bkg_lap  dof     : " +
          " ".join(f"{f['bkg_lap']:.0f}" for f in fb[:n]))


if __name__ == "__main__":
    main()
