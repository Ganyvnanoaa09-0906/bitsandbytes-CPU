# -*- coding: utf-8 -*-
"""Analysis-only motion report: reuse motion_boundary.analyze over a root dir of clips.

No generation; just walk <root>/*/frames and print the motion table.

Usage:
  python motion_report.py --root D:\work\output\motion_mit
"""
import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from motion_boundary import analyze  # noqa: E402


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
