# -*- coding: utf-8 -*-
"""clean_pipeline.py - 训练图清洗 / 单人双人分类 / 512 统一 / 水印标记。

用法:
  python clean_pipeline.py --data <图片目录> --out <输出目录> --res 512 --watermark
  python clean_pipeline.py --data <图片目录> --out <输出目录> --pose --pose-model keypointrcnn

输出:
  out/images/*.jpg       保留的图（统一 resize + 可选水印）
  out/poses/*.png        可选：OpenPose-18 骨架图
  out/metadata.jsonl     每条: image / pose / persons / scene / blur / watermark / keep
  out/rejected.jsonl     被过滤的图及原因
"""
from __future__ import annotations
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
import argparse, json, os, sys
from typing import Optional, Tuple
import numpy
import torch
import torch.nn.functional
from PIL import Image, ImageDraw
try:
    from scripts.lib.dewatermark import detect_boxes, boxes_to_mask, inpaint_nearest
except Exception:
    detect_boxes = boxes_to_mask = inpaint_nearest = None

def list_images(root: str):
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dp, fn))
    return sorted(out)

def load_rgb(path: str, max_side: Optional[int] = None) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR)
    return im

def to_tensor(im: Image.Image) -> torch.Tensor:
    a = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).contiguous()

def laplacian_var(x: torch.Tensor) -> float:
    g = (0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]).unsqueeze(0).unsqueeze(0)
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    return float(F.conv2d(g, k, padding=1).var().item())

def edge_density(x: torch.Tensor) -> float:
    g = (0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]).unsqueeze(0).unsqueeze(0)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    mag = torch.sqrt(F.conv2d(g, kx, padding=1) ** 2 + F.conv2d(g, ky, padding=1) ** 2 + 1e-8)
    return float((mag > 0.15).float().mean().item())
_pose_model = None

def count_persons(path: str, thresh: float = 0.6) -> int:
    """粗分类人数：优先 keypointrcnn（COCO 人体检测）；不可用时返回 -1。"""
    global _pose_model
    try:
        from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
        if _pose_model is None:
            _pose_model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).eval()
        arr = np.asarray(Image.open(path).convert("RGB"))
        t = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            out = _pose_model([t])[0]
        return int((out["scores"] > thresh).sum().item())
    except Exception:
        return -1

def classify_scene(x: torch.Tensor) -> Tuple[str, dict]:
    """启发式场景/主体分类；真实项目可换成 CLIP/ImageNet 分类器。"""
    r, g, b = x[0], x[1], x[2]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
    std = float(gray.std().item())
    ed = edge_density(x)
    skin = ((r > 0.45) & (g > 0.25) & (b > 0.15) & (r > g * 1.05) & (r > b * 1.15)).float().mean().item()
    green = (g > r * 1.10) & (g > b * 1.10)
    blue = (b > r * 1.05) & (b > g * 1.05)
    warm = (r > g * 1.08) & (r > b * 1.15)
    feat = dict(std=round(std, 4), edge=round(ed, 4), skin=round(float(skin), 4),
                green=round(float(green.float().mean().item()), 4),
                blue=round(float(blue.float().mean().item()), 4),
                warm=round(float(warm.float().mean().item()), 4))
    if std < 0.05 or ed < 0.02:
        return "plain", feat
    if skin > 0.08:
        return "portrait", feat
    if feat["green"] > 0.30 or feat["blue"] > 0.35:
        return "scenery", feat
    if feat["warm"] > 0.25 and ed > 0.08:
        return "snack", feat
    return "unknown", feat

def save_clean(im: Image.Image, out_path: str, res: int, watermark: bool = False, tag: str = ""):
    im2 = im.resize((res, res), Image.BILINEAR)
    if watermark:
        dr = ImageDraw.Draw(im2)
        txt = tag or "clean-pipeline"
        dr.rectangle([4, 4, 12 + 7 * len(txt), 20], fill=(0, 0, 0))
        dr.text((8, 6), txt, fill=(255, 255, 255))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im2.save(out_path, quality=95)
_pose_extractor = None

def extract_pose(path: str, out_pose: str, model_name: str = "keypointrcnn") -> Optional[str]:
    """DWPose/OpenPose 骨架提取占位：优先复用 controlnet_pose 的 keypointrcnn 实现。
    缺失 torchvision 权重或依赖时返回 None，不阻塞清洗流水线。
    """
    global _pose_extractor
    try:
        from scripts.lib import controlnet_pose
        if _pose_extractor is None:
            _pose_extractor = cp.get_pose_model()
        persons = cp.extract_openpose18_multi(_pose_extractor, path, thresh=0.4, max_persons=2)
        if not persons:
            return None
        W, H = persons[0][1]
        arr = cp.render_openpose_multi(persons, (W, H))
        Image.fromarray(arr).convert("RGB").resize((512, 512), Image.BILINEAR).save(out_pose)
        return out_pose
    except Exception as e:
        print(f"[pose] skip {os.path.basename(path)}: {type(e).__name__}", flush=True)
        return None
def process_one(path: str, out_images: str, out_poses: str, args) -> Optional[dict]:
    im = load_rgb(path, max_side=max(args.res, args.min_side))
    w, h = im.size
    x = to_tensor(im)
    if getattr(args, "dewatermark", False) and detect_boxes is not None:
        arr = np.asarray(im); boxes = detect_boxes(arr)
        if boxes:
            mask = boxes_to_mask(arr.shape, boxes)
            im = Image.fromarray(inpaint_nearest(arr, mask))
            x = to_tensor(im)
    blur = laplacian_var(x)
    scene, feat = classify_scene(x)
    n_person = count_persons(path) if args.persons else -1
    reasons = []
    if min(w, h) < args.min_side:
        reasons.append("too_small")
    if blur < args.blur_threshold:
        reasons.append("blur")
    if scene in ("plain", "scenery", "snack") and not args.keep_all:
        reasons.append(f"scene_{scene}")
    if args.require_person and n_person == 0:
        reasons.append("no_person")
    if n_person > args.max_person:
        reasons.append("too_many_person")
    rec = dict(src=path, scene=scene, blur=round(blur, 4), persons=n_person,
               watermark=bool(args.watermark), person_class=(
                   "single" if n_person == 1 else "double" if n_person == 2 else
                   "multi" if n_person > 2 else "unknown"),
               **feat)
    if reasons:
        rec["keep"] = False; rec["reasons"] = reasons
        return rec
    stem = os.path.splitext(os.path.basename(path))[0]
    img_out = os.path.join(out_images, stem + ".jpg")
    save_clean(im, img_out, args.res, watermark=args.watermark, tag=args.tag)
    rec["image"] = img_out; rec["keep"] = True
    if args.pose:
        pose_out = os.path.join(out_poses, stem + ".png")
        if extract_pose(img_out, pose_out, args.pose_model):
            rec["pose"] = pose_out
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="原始图片目录（可递归）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--min-side", type=int, default=512)
    ap.add_argument("--blur-threshold", type=float, default=0.003, help="Laplacian variance on 0..1 RGB; real photos usually >0.003")
    ap.add_argument("--watermark", action="store_true")
    ap.add_argument("--dewatermark", action="store_true", help="自动检测并去除二维码/文字水印（启发式兜底）")
    ap.add_argument("--tag", default="clean-pipeline")
    ap.add_argument("--pose", action="store_true", help="提取 DWPose/OpenPose 骨架")
    ap.add_argument("--pose-model", default="keypointrcnn")
    ap.add_argument("--persons", action="store_true", help="用 keypointrcnn 数人")
    ap.add_argument("--require-person", action="store_true")
    ap.add_argument("--max-person", type=int, default=2)
    ap.add_argument("--keep-all", action="store_true", help="不过滤风景/零食/纯色")
    args = ap.parse_args()
    imgs = list_images(args.data)
    out_images = os.path.join(args.out, "images")
    out_poses = os.path.join(args.out, "poses")
    os.makedirs(out_images, exist_ok=True)
    if args.pose: os.makedirs(out_poses, exist_ok=True)
    kept = rejected = 0
    with open(os.path.join(args.out, "metadata.jsonl"), "w", encoding="utf-8") as f_keep, \
         open(os.path.join(args.out, "rejected.jsonl"), "w", encoding="utf-8") as f_rej:
        for i, p in enumerate(imgs, 1):
            try:
                rec = process_one(p, out_images, out_poses, args)
            except Exception as e:
                rec = dict(src=p, keep=False, reasons=[f"error:{type(e).__name__}"])
            line = json.dumps(rec, ensure_ascii=False)
            if rec.get("keep"): f_keep.write(line + "\n"); kept += 1
            else: f_rej.write(line + "\n"); rejected += 1
            if i % 50 == 0: print(f"[clean] {i}/{len(imgs)} kept={kept} rejected={rejected}", flush=True)
    print(f"[clean] done total={len(imgs)} kept={kept} rejected={rejected} out={args.out}", flush=True)

if __name__ == "__main__":
    main()
