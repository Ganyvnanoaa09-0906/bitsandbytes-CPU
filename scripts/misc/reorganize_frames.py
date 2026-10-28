# -*- coding: utf-8 -*-
"""Reorganise existing diffusion output dirs into the frames/ subfolder layout.

Old layout:  <dir>/<tag>_f000.png, <tag>_f001.png, ... , <tag>.gif, <tag>.txt
New layout:  <dir>/frames/f000.png, f001.png, ...   +  <dir>/<tag>.gif, <tag>.txt

Idempotent: dirs already in the new layout are reported and skipped. Nothing is deleted;
files are moved only when the destination does not already exist.

Usage:
  python reorganize_frames.py --root D:\work\output\adiff_out            # dry run
  python reorganize_frames.py --root D:\work\output\adiff_out --apply
  python reorganize_frames.py --root D:\work\output\adiff_out --apply --only scan_fixed512
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
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--apply", action="store_true", help="actually move files (default: dry run)")
    ap.add_argument("--only", default=None, help="comma list of dir names to limit to")
    a = ap.parse_args()

    only = set(x.strip() for x in a.only.split(",")) if a.only else None
    dirs = [d for d in sorted(glob.glob(os.path.join(a.root, "*"))) if os.path.isdir(d)]

    total_moved = 0
    for d in dirs:
        name = os.path.basename(d)
        if only and name not in only:
            continue
        if name == "frames":
            continue
        pngs = sorted(glob.glob(os.path.join(d, "*_f*.png")))
        if not pngs:
            print(f"  [skip] {name}: no frame PNGs at top level")
            continue

        # A dir can hold unrelated diagnostics whose names also end in _fNNN.png
        # (e.g. vae_rt_counterfeit_f033.png). Renaming those to f033.png would collide with
        # the real frame, so only move files sharing the DOMINANT tag prefix.
        from collections import Counter
        prefixes = Counter(re.sub(r"_f\d+\.png$", "", os.path.basename(p)) for p in pngs
                           if re.search(r"_f\d+\.png$", os.path.basename(p)))
        if not prefixes:
            print(f"  [skip] {name}: no *_fNNN.png files")
            continue
        tag = prefixes.most_common(1)[0][0]
        keep = [p for p in pngs
                if re.sub(r"_f\d+\.png$", "", os.path.basename(p)) == tag]
        skipped = [p for p in pngs if p not in keep]

        fdir = os.path.join(d, "frames")
        existing = os.path.isdir(fdir)
        moved = 0
        planned = []
        for p in keep:
            m = re.search(r"_f(\d+)\.png$", os.path.basename(p))
            dst = os.path.join(fdir, f"f{int(m.group(1)):03d}.png")
            planned.append((p, dst))
            if os.path.exists(dst):
                continue
            moved += 1

        print(f"  {name}: tag='{tag}' -> {len(keep)} frame PNGs to frames/ "
              f"({moved} to move, {len(keep)-moved} already there"
              f"{', frames/ exists' if existing else ''})")
        if skipped:
            print(f"      left in place (different tag): "
                  f"{', '.join(os.path.basename(s) for s in skipped[:4])}"
                  f"{' ...' if len(skipped) > 4 else ''}")
        if a.apply:
            os.makedirs(fdir, exist_ok=True)
            for src, dst in planned:
                if os.path.exists(dst):
                    # keep the old copy only if identical-sized destination already exists
                    os.remove(src)
                else:
                    shutil.move(src, dst)
        total_moved += moved

    print(f"\n{'APPLIED' if a.apply else 'DRY RUN'}: {total_moved} files "
          f"{'moved' if a.apply else 'would be moved'}")
    if not a.apply:
        print("re-run with --apply to perform the moves")


if __name__ == "__main__":
    main()
