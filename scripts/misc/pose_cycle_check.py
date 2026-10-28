# -*- coding: utf-8 -*-
"""Geometry-only check of the pose cycle, no diffusion involved.

Answers two questions about `animate_pose`:
  1. does every frame get a DISTINCT skeleton? (v1 used sin(2*pi*i/n), which is zero at
     i=0 and i=n/2, so frames 0 and 5 of a 10-frame clip were byte-identical)
  2. are the limb segments rigid? (the v2 rewrite rotates elbow/knee first and then the
     forearm/shin about the *new* joint, so bone lengths must not change)

Usage:
  python pose_cycle_check.py --pose-src D:\\work\\output\\motion\\L4_runjump\\frames --pose-frame 5
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scripts.lib.controlnet_pose import animate_pose, extract_openpose18, get_pose_model

# (child, parent) pairs whose distance must stay constant
BONES = [(3, 2), (4, 3), (6, 5), (7, 6), (8, 1), (11, 1),
         (9, 8), (10, 9), (12, 11), (13, 12)]


def v1_pose(op, frame, n_frames, amp=1.0):
    """The original formula, kept here purely so the check can compare against it."""
    k = np.array(op, dtype=np.float32, copy=True)
    ph = 2 * np.pi * frame / max(n_frames, 1)
    sh = (k[2, :2] + k[5, :2]) / 2
    hip = (k[8, :2] + k[11, :2]) / 2
    torso = float(np.linalg.norm(sh - hip)) or 50.0
    swing = 0.55 * torso * amp
    for wrist, elbow, sh_i, s in ((4, 3, 2, 1), (7, 6, 5, -1)):
        if k[sh_i, 2] > 0:
            ang = 0.35 * np.sin(ph + (0 if s > 0 else np.pi)) * s
            for idx, fac in ((elbow, 0.5), (wrist, 1.0)):
                rel = k[idx, :2] - k[sh_i, :2]
                ca, sa = np.cos(ang * fac), np.sin(ang * fac)
                k[idx, :2] = k[sh_i, :2] + np.array([rel[0] * ca - rel[1] * sa,
                                                     rel[0] * sa + rel[1] * ca])
    for knee, ankle, hip_i, s in ((9, 10, 8, 1), (12, 13, 11, -1)):
        if k[hip_i, 2] > 0:
            ang = 0.30 * np.sin(ph + (0 if s > 0 else np.pi)) * s
            for idx, fac in ((knee, 0.5), (ankle, 1.0)):
                rel = k[idx, :2] - k[hip_i, :2]
                ca, sa = np.cos(ang * fac), np.sin(ang * fac)
                k[idx, :2] = k[hip_i, :2] + np.array([rel[0] * ca - rel[1] * sa,
                                                      rel[0] * sa + rel[1] * ca])
    if k[1, 2] > 0:
        off = 0.06 * swing * np.sin(ph)
        for idx in (0, 14, 15, 16, 17):
            k[idx, 0] += off
    return k


def seg_stats(poses):
    """Max frame-to-frame joint displacement and bone-length spread over a pose list."""
    steps = [float(np.abs(poses[i][:, :2] - poses[i - 1][:, :2]).max())
             for i in range(1, len(poses))]
    f0_fhalf = float(np.abs(poses[0][:, :2] - poses[len(poses) // 2][:, :2]).max())
    lens = {b: [] for b in BONES}
    for p in poses:
        for c, par in BONES:
            if p[c, 2] > 0 and p[par, 2] > 0:
                lens[(c, par)].append(float(np.linalg.norm(p[c, :2] - p[par, :2])))
    spread = {b: (max(v) - min(v)) for b, v in lens.items() if len(v) > 1}
    worst = max(spread.items(), key=lambda kv: kv[1]) if spread else (None, 0.0)
    return dict(step_min=min(steps), step_max=max(steps),
                f0_vs_fhalf=f0_fhalf,
                bone_spread_max=max(spread.values()) if spread else 0.0,
                worst_bone=worst[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-src", required=True)
    ap.add_argument("--pose-frame", type=int, default=5)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--amp", type=float, default=1.0)
    a = ap.parse_args()

    src = os.path.join(a.pose_src, f"f{a.pose_frame:03d}.png")
    got = extract_openpose18(get_pose_model(), src)
    if got is None:
        sys.exit(f"[err] no person detected in {src}")
    base, _ = got
    sh = (base[2, :2] + base[5, :2]) / 2
    hip = (base[8, :2] + base[11, :2]) / 2
    torso = float(np.linalg.norm(sh - hip))
    print(f"[pose] {os.path.basename(src)}  {int((base[:, 2] > 0).sum())}/18 joints, "
          f"torso={torso:.1f}px\n")

    for name, fn in (("v1 sin()", v1_pose), ("v3 cos+2nd", animate_pose)):
        poses = [fn(base, i, a.n, amp=a.amp) for i in range(a.n)]
        s = seg_stats(poses)
        dup = sum(1 for i in range(len(poses))
                  for j in range(i + 1, len(poses))
                  if np.array_equal(poses[i][:, :2], poses[j][:, :2]))
        print(f"{name}: consecutive-step {s['step_min']:.1f}..{s['step_max']:.1f}px  "
              f"| f0 vs f{a.n // 2} = {s['f0_vs_fhalf']:.1f}px  "
              f"| bone-length spread {s['bone_spread_max']:.4f}px "
              f"(worst bone {s['worst_bone']})  | identical pairs {dup}")
        if name.startswith("v3"):
            print("  per-frame wrist/knee y: " + " ".join(
                f"{p[4, 1]:.0f}/{p[9, 1]:.0f}" for p in poses))
    print("\n(identical pairs > 0 means some frames share the exact same skeleton; "
          "bone-length spread should be ~0 for both since v1 scaled both joints equally)")


if __name__ == "__main__":
    main()
