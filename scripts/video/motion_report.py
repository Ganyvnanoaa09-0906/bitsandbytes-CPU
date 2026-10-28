# -*- coding: utf-8 -*-
"""Analysis-only motion report: reuse motion_boundary.analyze over a root dir of clips.

No generation; just walk <root>/*/frames and print the motion table.

Usage:
  python motion_report.py --root D:\work\output\motion_mit
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
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scripts.lib.motion_boundary import analyze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = {}
    print(f"{'clip':>14} {'n':>3} {'mad':>7} {'mad_std':>8} {'mad_max':>8} "
          f"{'lap':>8} {'ncc':>7} {'white%':>7} {'green':>6}")
    print("-" * 78)
    for d in sorted(glob.glob(os.path.join(a.root, "*"))):
        fd = os.path.join(d, "frames")
        if not os.path.isdir(fd) or not glob.glob(os.path.join(fd, "*.png")):
            continue
        m = analyze(fd)
        res[os.path.basename(d)] = m
        print(f"{os.path.basename(d):>14} {m['n']:>3} {m['mad']:>7.2f} {m['mad_std']:>8.2f} "
              f"{m['mad_max']:>8.1f} {m['lap']:>8.1f} {m['ncc']:>7.3f} "
              f"{m['white']:>7.2f} {m['green']:>6}")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\n[save] {a.out}")


if __name__ == "__main__":
    main()
