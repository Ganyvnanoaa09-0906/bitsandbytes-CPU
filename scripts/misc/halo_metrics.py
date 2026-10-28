# -*- coding: utf-8 -*-
"""Objective halo / naturalness metrics for detail restorers.

Needed because over-sharpening is easy to measure even without looking at the image:
halos are literally *overshoot* -- pixels pushed outside the local intensity range of the
source. Two metrics:

  overshoot%  fraction of pixels whose restored luma leaves the [min,max] of the source in a
              (2r+1) neighbourhood, by more than `tol`. A faithful restorer stays near 0;
              aggressive sharpening/hallucination pushes it up.

  flat_noise  gain in local variance measured ONLY inside source-flat regions (source local
              std < 2). Detail restoration should add structure at edges, not grain in flat
              areas; a big gain here means amplified noise / painting texture where there
              was none.

  edge_gain   Laplacian energy measured only in source-edge regions, to show whether the
              added energy is where it belongs.

Usage:
  python halo_metrics.py --src <frames dir> --variants "name=path,name=path" --picks 9,30
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
import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


def luma(a):
    a = a.astype(np.float32)
    return 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]


def box_min_max(g, r=2):
    """Local min/max via strided shifts (no scipy available)."""
    H, W = g.shape
    pad = np.pad(g, r, mode="edge")
    mn = np.full_like(g, np.inf)
    mx = np.full_like(g, -np.inf)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sl = pad[r + dy:r + dy + H, r + dx:r + dx + W]
            mn = np.minimum(mn, sl)
            mx = np.maximum(mx, sl)
    return mn, mx


def local_std(g, r=2):
    pad = np.pad(g, r, mode="edge")
    H, W = g.shape
    acc = np.zeros_like(g)
    acc2 = np.zeros_like(g)
    n = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sl = pad[r + dy:r + dy + H, r + dx:r + dx + W]
            acc += sl
            acc2 += sl * sl
            n += 1
    m = acc / n
    return np.sqrt(np.maximum(acc2 / n - m * m, 0))


def lap(g):
    return (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])


def metrics(src, res, tol=6.0, r=2):
    gs, gr = luma(src), luma(res)
    mn, mx = box_min_max(gs, r)
    over = ((gr > mx + tol) | (gr < mn - tol))
    sd = local_std(gs, r)
    flat = sd < 2.0
    edge = sd > 8.0

    def var_in(mask, g):
        if mask.sum() < 50:
            return float("nan")
        gg = g.copy()
        # local variance within the mask, approximated via masked laplacian energy
        l = lap(gg)
        m = mask[1:-1, 1:-1]
        return float((l[m] ** 2).mean()) if m.sum() > 20 else float("nan")

    return dict(
        overshoot_pct=float(100 * over.mean()),
        overshoot_max=float(np.abs(np.clip(gr - mx, None, 0)).max()
                            + np.abs(np.clip(gr - mn, 0, None)).max()),
        src_flat_pct=float(100 * flat.mean()),
        flat_src=var_in(flat, gs), flat_res=var_in(flat, gr),
        edge_src=var_in(edge, gs), edge_res=var_in(edge, gr),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--variants", required=True,
                    help='comma list of name=path, e.g. "esrgan=.../a.png,fsrcnn=.../b.png"')
    ap.add_argument("--picks", default="9,30")
    a = ap.parse_args()

    variants = {}
    for item in a.variants.split(","):
        if "=" in item:
            n, p = item.split("=", 1)
            variants[n.strip()] = p.strip()

    for pi in [int(x) for x in a.picks.split(",")]:
        sp = os.path.join(a.src, f"f{pi:03d}.png")
        src = np.asarray(Image.open(sp).convert("RGB"))
        print(f"\n=== frame f{pi:03d} ===")
        print(f"{'variant':>18} {'overshoot%':>11} {'maxovershoot':>13} "
              f"{'flat-src':>9} {'flat-res':>9} {'flat gain':>10} "
              f"{'edge gain':>10}")
        print("-" * 86)
        for name, p in variants.items():
            if not os.path.exists(p):
                print(f"{name:>18}  (missing {p})")
                continue
            res = np.asarray(Image.open(p).convert("RGB"))
            m = metrics(src, res)
            fg = (m["flat_res"] / m["flat_src"]) if m["flat_src"] else float("nan")
            eg = (m["edge_res"] / m["edge_src"]) if m["edge_src"] else float("nan")
            print(f"{name:>18} {m['overshoot_pct']:>11.2f} {m['overshoot_max']:>13.1f} "
                  f"{m['flat_src']:>9.2f} {m['flat_res']:>9.2f} {fg:>10.2f} {eg:>10.2f}")
        print("  (flat gain >> edge gain  => noise/over-sharpening in flat areas;"
              " overshoot% > ~1% => visible halos)")


if __name__ == "__main__":
    main()
