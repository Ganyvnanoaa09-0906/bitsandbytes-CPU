# -*- coding: utf-8 -*-
"""Motion-collapse boundary sweep.

Runs the FROZEN production pipeline (independent chunks, full 5-step schedule, DPM-Solver++
2M, SD1.5 VAE, 512) across escalating motion amplitudes, one shot per level, so the boundary
between "moves correctly" and "collapses" can be located.

Levels (character held constant; only the motion phrase changes):
  L1_micro     micro-expression         (upper body framing, matches production)
  L2_headturn  head turn
  L3_wave      waving / raised arm
  L4_runjump   running & jumping        (full body - heavy motion needs it)
  L5_fight     fighting / weapon swing  (full body)

Each level writes <out>/<level>/frames/*.png and reports per-level motion metrics:
  mad        mean frame-to-frame difference          -> motion amplitude
  mad_std    spread of those differences             -> jerkiness / instability
  lap        Laplacian variance (detail energy)      -> softness
  white%/green_px                                    -> artefact load
  ncc        mean normalised cross-correlation of consecutive frames
             (high = content MOVES; low = content is REPLACED -> collapse signature)

Usage:
  python motion_boundary.py --out D:\work\output\motion --levels L1_micro,L2_headturn,...
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
import argparse
import json
import os
import subprocess
import sys
import time

import numpy
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))

BASE = ("1girl, anime style, cute face, detailed eyes, beautiful detailed face, "
        "masterpiece, best quality, solo")

LEVELS = {
    "L1_micro":    f"{BASE}, upper body, gentle smile, subtle smile, blinking, micro expression",
    "L2_headturn": f"{BASE}, upper body, turning head to the side, looking away, profile view",
    "L3_wave":     f"{BASE}, upper body, waving hand, raised arm, greeting gesture, energetic",
    "L4_runjump":  ("1girl, anime style, cute face, detailed eyes, masterpiece, best quality, "
                    "solo, full body, running, jumping, dynamic action pose, energetic motion"),
    "L5_fight":    ("1girl, anime style, cute face, detailed eyes, masterpiece, best quality, "
                    "solo, full body, fighting stance, swinging sword, intense action, "
                    "motion blur, dynamic"),
}


def luma(a):
    a = a.astype(np.float32)
    return 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]


def lapvar(g):
    return float((4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
                  - g[1:-1, :-2] - g[1:-1, 2:]).var())


def ncc(g0, g1):
    """Normalised cross-correlation of consecutive luma frames."""
    a = g0 - g0.mean()
    b = g1 - g1.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-9 else float("nan")


def analyze(d):
    fs = sorted(os.listdir(d))
    fs = [f for f in fs if f.endswith(".png")]
    gs = []
    wh, gr = [], []
    for f in fs:
        a = np.asarray(Image.open(os.path.join(d, f)).convert("RGB")).astype(np.float32)
        g = luma(a)
        gs.append(g)
        wh.append(float(100 * (a > 245).mean()))
        gr.append(int(((a[:, :, 1] - np.maximum(a[:, :, 0], a[:, :, 2])) > 25).sum()))
    mads = [float(np.abs(gs[i + 1] - gs[i]).mean()) for i in range(len(gs) - 1)]
    nccs = [ncc(gs[i], gs[i + 1]) for i in range(len(gs) - 1)]
    return dict(n=len(gs), mad=float(np.mean(mads)), mad_std=float(np.std(mads)),
                mad_max=float(np.max(mads)), lap=float(np.mean([lapvar(g) for g in gs])),
                white=float(np.mean(wh)), green=int(np.sum(gr)),
                ncc=float(np.nanmean(nccs)),
                per_frame_mad=[round(m, 1) for m in mads])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--tag", default="mot")
    a = ap.parse_args()

    py = sys.executable
    drv = os.path.join(_HERE, "independent_chunks.py")
    sub = os.path.join(_HERE, "_run_one_motion.py")
    results = {}

    for lv in [x.strip() for x in a.levels.split(",") if x.strip()]:
        prompt = LEVELS[lv]
        odir = os.path.join(a.out, lv)
        print(f"\n{'='*70}\n[level] {lv}\n  prompt: {prompt}\n{'='*70}", flush=True)
        t0 = time.time()
        cmd = [py, "-u", drv, "--res", str(a.res), "--chunk", str(a.chunk), "--chunks", "1",
               "--steps", str(a.steps), "--tag", a.tag, "--prompt", prompt, "--outdir", odir]
        rc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="ignore")
        print(rc.stdout[-1500:] if rc.stdout else "", flush=True)
        if rc.returncode != 0:
            print(f"[warn] {lv} exit={rc.returncode}\n{rc.stderr[-800:]}", flush=True)
        fd = os.path.join(odir, "frames")
        if os.path.isdir(fd):
            m = analyze(fd)
            m["seconds"] = time.time() - t0
            results[lv] = m
            print(f"  -> mad={m['mad']:.2f} (max {m['mad_max']:.1f}) lap={m['lap']:.1f} "
                  f"ncc={m['ncc']:.3f} white={m['white']:.2f} green={m['green']} "
                  f"({m['seconds']:.0f}s)", flush=True)
        else:
            print(f"  [err] no frames dir for {lv}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "motion_boundary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'='*74}\nMOTION BOUNDARY TABLE\n{'='*74}")
    print(f"{'level':>12} {'mad':>7} {'mad_std':>8} {'mad_max':>8} {'lap':>8} "
          f"{'ncc':>7} {'white%':>7} {'green':>6} {'sec':>6}")
    for k, v in results.items():
        print(f"{k:>12} {v['mad']:>7.2f} {v['mad_std']:>8.2f} {v['mad_max']:>8.1f} "
              f"{v['lap']:>8.1f} {v['ncc']:>7.3f} {v['white']:>7.2f} {v['green']:>6} "
              f"{v['seconds']:>6.0f}")
    print("\n解读: ncc 高(>0.9)=内容在'移动'; ncc 低(<0.7)=内容被'替换'(崩坏特征);")
    print("      mad 高 + lap 低 = 崩坏(乱而糊); mad 高 + lap 稳 = 正常大动作")


if __name__ == "__main__":
    main()
