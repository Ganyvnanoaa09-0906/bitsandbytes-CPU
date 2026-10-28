# -*- coding: utf-8 -*-
"""dewatermark.py - 微博图去二维码/文字水印/logo 的轻量流水线。

自动检测是启发式（无 OpenCV 时的兜底）：
- 底边/角落的高亮、低饱和、边缘密集小区域；
- 支持手动 ROI 和手动 mask；
- 修复用最近邻填充 + 高斯模糊，或调用 cv2.inpaint（若已安装）。

推荐：装 opencv-contrib-python / pyzbar 后，把 detect_boxes 换成 QRCodeDetector + OCR。
"""
from __future__ import annotations
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, json, os
import numpy
from PIL import Image
from scipy import ndimage

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.uint8)

def save_rgb(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path, quality=95)

def _edge_mag(gray):
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    return np.sqrt(gx * gx + gy * gy)

def detect_boxes(arr, bottom_frac=0.22, max_area_frac=0.08, min_area=80):
    """启发式找底部/角落疑似水印区域，返回 [x0,y0,x1,y1] 列表。"""
    H, W = arr.shape[:2]
    gray = arr.mean(axis=2) / 255.0
    mx = gray.max(axis=2) if gray.ndim == 3 else gray
    sat = (arr.max(axis=2).astype(np.float32) - arr.min(axis=2).astype(np.float32)) / 255.0
    edge = _edge_mag(gray)
    # 候选：高亮或低饱和 + 边缘密集
    cand = ((gray > 0.72) | (sat < 0.10)) & (edge > np.percentile(edge, 90))
    # 限制在底部/顶部/左右角落
    mask = np.zeros((H, W), dtype=bool)
    mask[int(H * (1 - bottom_frac)):, :] = True
    mask[:int(H * bottom_frac * 0.5), :] = True
    ws = max(8, W // 8)
    mask[:, :ws] = True; mask[:, W - ws:] = True
    cand &= mask
    lab, n = ndimage.label(cand)
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        area = len(xs)
        if area < min_area or area > max_area_frac * H * W:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        if bw < 6 or bh < 6:
            continue
        if bw / max(bh, 1) > 12 or bh / max(bw, 1) > 12:
            continue
        boxes.append([int(x0), int(y0), int(x1), int(y1)])
    return boxes

def boxes_to_mask(shape, boxes):
    H, W = shape[:2]
    m = np.zeros((H, W), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        m[max(0, y0-2):min(H, y1+3), max(0, x0-2):min(W, x1+3)] = True
    return m

def inpaint_nearest(arr, mask):
    """最近邻填充 + 局部高斯模糊（无 cv2 兜底）。"""
    if not mask.any():
        return arr
    # 用距离变换找最近的非 mask 像素
    ind = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    out = arr[ind[0], ind[1]]
    # 只在 mask 区域做模糊，减少接缝
    blur = ndimage.gaussian_filter(out.astype(np.float32), sigma=1.6)
    out = out.astype(np.float32)
    for c in range(3):
        out[..., c][mask] = blur[..., c][mask]
    return np.clip(out, 0, 255).astype(np.uint8)
def process_one(path, out_dir, auto=True, manual_boxes=None, max_area_frac=0.08):
    arr = load_rgb(path)
    boxes = list(manual_boxes or [])
    if auto:
        boxes += detect_boxes(arr, max_area_frac=max_area_frac)
    if not boxes:
        return dict(src=path, boxes=[], removed=False, reason="no_watermark_detected")
    mask = boxes_to_mask(arr.shape, boxes)
    if mask.mean() > max_area_frac:
        return dict(src=path, boxes=boxes, removed=False, reason="mask_too_large")
    out = inpaint_nearest(arr, mask)
    name = os.path.splitext(os.path.basename(path))[0] + ".jpg"
    dst = os.path.join(out_dir, "clean", name)
    save_rgb(out, dst)
    return dict(src=path, dst=dst, boxes=boxes, removed=True, mask_ratio=float(mask.mean()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-auto", action="store_true")
    ap.add_argument("--max-area-frac", type=float, default=0.08)
    a = ap.parse_args()
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    files = []
    for dp, _, fns in os.walk(a.data):
        for fn in fns:
            if fn.lower().endswith(exts):
                files.append(os.path.join(dp, fn))
    os.makedirs(os.path.join(a.out, "clean"), exist_ok=True)
    with open(os.path.join(a.out, "dewatermark.jsonl"), "w", encoding="utf-8") as f:
        done = 0
        for p in sorted(files):
            try:
                rec = process_one(p, a.out, auto=not a.no_auto, max_area_frac=a.max_area_frac)
            except Exception as e:
                rec = dict(src=p, removed=False, reason=f"error:{type(e).__name__}")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done += 1
            if done % 50 == 0:
                print(f"[dewatermark] {done}/{len(files)}", flush=True)
    print(f"[dewatermark] done {done}/{len(files)}", flush=True)

if __name__ == "__main__":
    main()