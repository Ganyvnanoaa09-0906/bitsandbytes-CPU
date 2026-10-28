# -*- coding: utf-8 -*-
"""Face-consistency metric via COCO person keypoints (cheap stand-in for ArcFace).

Why: ArcFace ID-adapter (from the H3/ByteDance memo) needs insightface + onnxruntime + an
aligned-face pipeline, none of which are installed. torchvision already ships
keypointrcnn_resnet50_fpn (COCO 17 keypoints), whose first 5 keypoints are exactly the face
(nose, both eyes, both ears). Tracking those across frames gives an objective, model-free
measure of "does the face stay put and stay coherent", which is what the motion-collapse
boundary needs.

Per frame we record, for the highest-scoring person detection:
  kp[17]        keypoint coordinates (normalised to 0..1)
  kps[17]       per-keypoint confidence
  box           person bbox (normalised)

Aggregated per clip:
  det_rate      fraction of frames with a person detected (detection loss == collapse)
  face_conf     mean confidence of keypoints 0..4
  face_jitter   mean per-keypoint std across frames, in % of image size (instability)
  body_jitter   same for keypoints 0..10
  scale_drift   change in person bbox height from first to last frame (%)
  face_lap      Laplacian variance inside the face box (detail retained on the face)

Usage:
  python face_consistency.py --root <dir containing per-level frames dirs>
  python face_consistency.py --frames <one frames dir>
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
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
import re

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights


def lapvar(g):
    return float((4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
                  - g[1:-1, :-2] - g[1:-1, 2:]).var())


def luma(a):
    a = a.astype(np.float32)
    return 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]


def analyze_dir(model, d, thresh=0.5):
    ps = sorted(glob.glob(os.path.join(d, "*.png")),
                key=lambda p: int(re.search(r"f(\d+)\.png$", p).group(1)))
    recs = []
    for p in ps:
        arr = np.asarray(Image.open(p).convert("RGB"))
        H, W = arr.shape[:2]
        t = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            out = model([t])[0]
        scores = out["scores"].numpy()
        if len(scores) == 0 or scores.max() < thresh:
            recs.append(dict(path=os.path.basename(p), det=False))
            continue
        i = int(scores.argmax())
        kp = out["keypoints"][i].numpy()          # [17,3] x,y,vis
        ks = out["keypoints_scores"][i].numpy()   # [17]
        box = out["boxes"][i].numpy()
        # face box from the 5 face keypoints
        fx = kp[:5, 0] / W
        fy = kp[:5, 1] / H
        fw = max(fx.max() - fx.min(), 0.06) * W
        fh = max(fy.max() - fy.min(), 0.06) * H
        cx, cy = float(fx.mean()) * W, float(fy.mean()) * H
        x0 = int(max(0, cx - fw * 1.1)); x1 = int(min(W, cx + fw * 1.1))
        y0 = int(max(0, cy - fh * 1.4)); y1 = int(min(H, cy + fh * 1.8))
        g = luma(arr)
        fl = lapvar(g[y0:y1, x0:x1]) if (x1 - x0) > 8 and (y1 - y0) > 8 else float("nan")
        recs.append(dict(path=os.path.basename(p), det=True,
                         kp=(kp[:, :2] / np.array([W, H])).tolist(),
                         kps=ks.tolist(),
                         box=[float(box[1] / H), float(box[3] / H)],
                         score=float(scores.max()), face_lap=fl))
    return recs


def summarize(recs):
    det = [r for r in recs if r["det"]]
    if not det:
        return dict(det_rate=0.0, n=len(recs))
    kp = np.array([r["kp"] for r in det])          # [n,17,2]
    kps = np.array([r["kps"] for r in det])        # [n,17]
    h = np.array([r["box"][1] - r["box"][0] for r in det])
    return dict(
        n=len(recs), det_rate=len(det) / len(recs),
        face_conf=float(kps[:, :5].mean()),
        body_conf=float(kps[:, :11].mean()),
        face_jitter=float(kp[:, :5].std(0).mean() * 100),
        body_jitter=float(kp[:, :11].std(0).mean() * 100),
        face_lap=float(np.nanmean([r["face_lap"] for r in det])),
        scale_drift=float((h[-1] - h[0]) / max(h[0], 1e-6) * 100),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="parent dir with per-level frames/ dirs")
    ap.add_argument("--frames", default=None, help="a single frames dir")
    ap.add_argument("--out", default=None, help="where to write face_consistency.json")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    print("[load] keypointrcnn_resnet50_fpn (COCO)", flush=True)
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()

    targets = {}
    if a.frames:
        targets[os.path.basename(os.path.dirname(a.frames.rstrip("\\/"))) or "clip"] = a.frames
    if a.root:
        for d in sorted(glob.glob(os.path.join(a.root, "*"))):
            fd = os.path.join(d, "frames")
            if os.path.isdir(fd) and glob.glob(os.path.join(fd, "*.png")):
                targets.setdefault(os.path.basename(d), fd)
    if not targets:
        print("[err] nothing to analyse")
        return

    allres = {}
    print(f"\n{'clip':>12} {'det%':>6} {'face_conf':>10} {'face_jit%':>10} "
          f"{'body_jit%':>10} {'face_lap':>9} {'scale%':>8}")
    print("-" * 74)
    for name, d in targets.items():
        recs = analyze_dir(model, d)
        s = summarize(recs)
        allres[name] = dict(summary=s, frames=recs)
        print(f"{name:>12} {100*s.get('det_rate',0):>6.0f} {s.get('face_conf',float('nan')):>10.2f} "
              f"{s.get('face_jitter',float('nan')):>10.2f} "
              f"{s.get('body_jitter',float('nan')):>10.2f} "
              f"{s.get('face_lap',float('nan')):>9.1f} "
              f"{s.get('scale_drift',float('nan')):>8.1f}", flush=True)

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(allres, f, indent=2)
        print(f"\n[save] {a.out}")


if __name__ == "__main__":
    main()
