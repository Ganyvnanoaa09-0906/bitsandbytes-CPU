# -*- coding: utf-8 -*-
"""Quantify the "effect masking" phenomenon: how much of each frame is directional streaks.

Observation from the motion boundary sweep: as motion amplitude grows, the model increasingly
paints radial speed-lines / swirls that MASK broken anatomy. That is a measurable image
property: streak-dominated frames have a strongly peaked gradient-orientation distribution,
whereas ordinary frames spread energy over all orientations.

Metric:
  orient_entropy  Shannon entropy (bits) of the gradient-orientation histogram, weighted by
                  gradient magnitude. Max possible = log2(18 bins) = 4.17.
                  LOWER = more directionally concentrated = more streak/effect dominated.
  aniso           (max_bin - mean_bin) / mean_bin, a simple anisotropy ratio.
  streak_frac     fraction of gradient energy within +-15 deg of the dominant orientation
                  (restricted to strong-gradient pixels so flat areas do not dilute it).

Usage:
  python streak_metrics.py --root <dir with */frames> [--more <dir>]
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
import json
import os

import numpy as np
from PIL import Image

NBINS = 18


def luma(a):
    a = a.astype(np.float32)
    return 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]


def frame_streak(a, strong_pct=90):
    g = luma(a)
    H, W = g.shape
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag, strong_pct)
    m = mag >= max(thr, 1e-3)
    if m.sum() < 50:
        return dict(orient_entropy=float("nan"), aniso=float("nan"),
                    streak_frac=float("nan"), radial_align=float("nan"))

    # --- orientation histogram (captures PARALLEL streaks) ---
    ang = (np.arctan2(gy[m], gx[m]) + np.pi) % np.pi
    hist, _ = np.histogram(ang, bins=NBINS, range=(0, np.pi), weights=mag[m])
    p = hist / max(hist.sum(), 1e-9)
    nz = p[p > 0]
    ent = float(-(nz * np.log2(nz)).sum())
    dom = int(hist.argmax())
    centre = (dom + 0.5) * np.pi / NBINS
    d = np.abs((ang - centre + np.pi / 2) % np.pi - np.pi / 2)
    frac = float(mag[m][d <= np.deg2rad(15)].sum() / max(mag[m].sum(), 1e-9))

    # --- RADIAL alignment (captures SPOKE / speed-line bursts) ---
    # A radial spoke has its gradient PERPENDICULAR to the radius, i.e. the angle between
    # the gradient and the radial direction is ~90 deg, so |cos(delta)| -> 0.
    # For an isotropic image E[|cos|] = 2/pi = 0.637, so values well below that mean the
    # frame is dominated by radial structure.
    ys, xs = np.nonzero(m)
    rad = np.arctan2(ys - H / 2.0, xs - W / 2.0)
    grad = np.arctan2(gy[m], gx[m])
    delta = grad - rad
    rad_align = float(np.abs(np.cos(delta)).mean())

    return dict(orient_entropy=ent, aniso=float((hist.max() - hist.mean()) / max(hist.mean(), 1e-9)),
                streak_frac=frac, radial_align=rad_align)


def clip_streak(d):
    ps = sorted(glob.glob(os.path.join(d, "*.png")))
    recs = [frame_streak(np.asarray(Image.open(p).convert("RGB"))) for p in ps]
    return {k: float(np.nanmean([r[k] for r in recs]))
            for k in ("orient_entropy", "aniso", "streak_frac", "radial_align")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--more", default=None, help="additional root to also scan")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    roots = [a.root] + ([a.more] if a.more else [])
    res = {}
    print(f"{'clip':>14} {'orient_entropy':>15} {'aniso':>9} {'streak_frac':>12} {'radial_align':>13}")
    print("-" * 54)
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            fd = os.path.join(d, "frames")
            if not os.path.isdir(fd) or not glob.glob(os.path.join(fd, "*.png")):
                continue
            s = clip_streak(fd)
            res[os.path.basename(d)] = s
            print(f"{os.path.basename(d):>14} {s['orient_entropy']:>15.3f} "
                  f"{s['aniso']:>9.2f} {s['streak_frac']:>12.3f}")
    print(f"\n(max entropy = {np.log2(NBINS):.2f}; lower entropy / higher aniso = "
          f"more streak-dominated = more effect masking)")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[save] {a.out}")


if __name__ == "__main__":
    main()

