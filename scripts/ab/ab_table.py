# -*- coding: utf-8 -*-
"""One combined table (motion + face) for a list of frames dirs.

`motion_report.py` and `face_consistency.py` both discover clips as `<root>/<clip>/frames`,
which does not fit the ControlNet runs (their frames live directly in `<root>/frames`).
This tool takes explicit frames dirs and prints both metric families in one pass.

Usage:
  python ab_table.py --dirs D:\\...\\ctl_L4\\off\\frames D:\\...\\ctl_L4\\on\\frames ^
                     --names L4_nocontrol L4_control
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
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scripts.lib.motion_boundary import analyze
from scripts.lib.face_consistency import analyze_dir, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--names", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    from torchvision.models.detection import (keypointrcnn_resnet50_fpn,
                                              KeypointRCNN_ResNet50_FPN_Weights)
    print("[load] keypointrcnn_resnet50_fpn", flush=True)
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()

    names = list(a.names) + [os.path.basename(os.path.dirname(d.rstrip("\\/")))
                             for _ in range(max(0, len(a.dirs) - len(a.names)))]
    import json
    res = {}
    print(f"\n{'clip':>16} {'n':>3} {'mad':>7} {'ncc':>7} {'lap':>8} {'white%':>7} "
          f"{'green':>6} {'det%':>5} {'face_conf':>10} {'face_jit':>9} {'body_jit':>9}")
    print("-" * 100)
    for name, d in zip(names, a.dirs):
        m = analyze(d)
        s = summarize(analyze_dir(model, d))
        res[name] = dict(motion=m, face=s, dir=d)
        print(f"{name:>16} {m['n']:>3} {m['mad']:>7.2f} {m['ncc']:>7.3f} {m['lap']:>8.1f} "
              f"{m['white']:>7.2f} {m['green']:>6} {100*s.get('det_rate', 0):>5.0f} "
              f"{s.get('face_conf', float('nan')):>10.2f} "
              f"{s.get('face_jitter', float('nan')):>9.2f} "
              f"{s.get('body_jitter', float('nan')):>9.2f}", flush=True)

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\n[save] {a.out}")


if __name__ == "__main__":
    main()
