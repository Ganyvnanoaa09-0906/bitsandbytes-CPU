# -*- coding: utf-8 -*-
"""fast_clean.py - 批处理版训练图粗筛（模糊 / 纯色 / 截图 / 无人 / 风景过滤）。

目标：把 clean_pipeline.py 的 ~1.5 s/张 压到 <=0.2 s/张。

做法
----
1. **一次读一批**（默认 16 张），整批拼成一个 uint8 张量，只做一次 float 转换；
   所有图像指标用 torch 卷积在整批上并行算（不是逐张 for 循环）。
2. **人体检测换成轻量单阶段检测器** ssdlite320_mobilenet_v3_large（COCO person 类）。
   实测 CPU 62-76 ms/张；原来的 keypointrcnn_resnet50_fpn 要 2100 ms/张（批处理无收益）。
   可用 --detect 切换 / 关闭。
3. **去水印默认关闭**。原 `dewatermark.py` 依赖 scipy，本环境未安装，所以
   clean_pipeline 的 `--dewatermark` 一直是静默空转（import 失败 -> detect_boxes=None）。
   且最近邻 inpaint + 距离变换本身是逐张重活。需要时用 --dewatermark 显式打开。

用法
----
  python fast_clean.py --data D:\\work\\output --out <outdir>
  python fast_clean.py --data <dir> --out <dir> --detect none          # 最快，跳过人体检测
  python fast_clean.py --data <dir> --out <dir> --detect keypoint      # 高精度但 ~2s/张
  python fast_clean.py --data <dir> --out <dir> --dump-metrics         # 只出指标，便于调阈值

输出
----
  out/images/*.jpg        保留的图（统一 resize 到 --res）
  out/metadata.jsonl      保留记录
  out/rejected.jsonl      剔除记录 + 原因
  out/clean_report.json   分阶段耗时与统计
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
import argparse, json, os, sys, time

# torchvision 权重缓存重定向到工作区内（默认的 C:\Users\...\.cache 在沙箱里不可写）
_REPO = os.path.dirname(os.path.abspath(__file__))
if not os.environ.get("TORCH_HOME"):
    _th = os.path.join(_REPO, ".torch_home")
    if os.path.isdir(_th):
        os.environ["TORCH_HOME"] = _th

import numpy
import torch
import torch.nn.functional
from PIL import Image

EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
# 常见屏幕分辨率：疑似截图
SCREEN_RES = {(1920, 1080), (1080, 1920), (2560, 1440), (1440, 2560), (1366, 768),
              (1280, 720), (720, 1280), (3840, 2160), (1600, 900), (1024, 768),
              (2732, 2048), (2048, 2732), (1284, 2778), (1170, 2532)}
KX = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
KY = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
KL = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def list_images(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(EXTS):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def load_one(path, res, square_mode="crop", crop_bias=0.0, crop_region="full"):
    """解码 + 缩放，返回 (uint8 HWC ndarray, 原始宽高)。

    square_mode:
      crop    - 短边缩到 res 后裁剪成方形（默认；人像不会被拉变形）
      stretch - 直接拉成 res x res（会破坏长宽比，仅用于方图数据集）
      keep    - 只把长边缩到 res，保持原始长宽比、不裁剪

    crop_bias / crop_region（仅 square_mode=crop 生效）:
      crop_region='upper' 等价于 crop_bias=+0.18 —— 裁剪窗上移，取脸部/上半身，
                           用来绕开胸臀位置的马赛克
      crop_bias  正数=裁剪窗上移（避开底边水印），负数=下移。按图像高度比例计。
      （两者叠加后会被夹到合法范围内）

    JPEG 走 draft 做 DCT 域降采样，48MP 原图也能快速解码。
    """
    im = Image.open(path)
    ow, oh = im.size
    try:
        im.draft("RGB", (res, res))
    except Exception:
        pass
    im = im.convert("RGB")
    if square_mode == "keep":
        w, h = im.size
        if max(w, h) > res:
            s = res / max(w, h)
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    elif square_mode == "stretch":
        if im.size != (res, res):
            im = im.resize((res, res), Image.LANCZOS)
    else:  # crop
        w, h = im.size
        if w != h:
            s = res / min(w, h)
            nw, nh = max(res, int(w * s + 0.5)), max(res, int(h * s + 0.5))
            im = im.resize((nw, nh), Image.LANCZOS)
            bias = crop_bias + (0.18 if crop_region == "upper" else 0.0)
            l = (nw - res) // 2
            t = (nh - res) // 2 - int(round(bias * nh))
            l = max(0, min(l, nw - res))
            t = max(0, min(t, nh - res))
            im = im.crop((l, t, l + res, t + res))
        elif im.size != (res, res):
            im = im.resize((res, res), Image.LANCZOS)
    return np.asarray(im), (ow, oh)


def batch_tensor(arrays):
    """list[HWC uint8] -> (B,3,H,W) float32 in [0,1]，只做一次 float 转换。"""
    a = torch.from_numpy(np.stack(arrays))          # B,H,W,3 uint8
    return a.permute(0, 3, 1, 2).float().div_(255.0)


@torch.no_grad()
def batch_metrics(x):
    """整批并行计算所有廉价指标。x: (B,3,H,W) float32。返回 dict[str, tensor[B]]。"""
    B = x.shape[0]
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).unsqueeze(1)
    blur = F.conv2d(gray, KL, padding=1).var(dim=(1, 2, 3))
    gx = F.conv2d(gray, KX, padding=1)
    gy = F.conv2d(gray, KY, padding=1)
    edge = (torch.sqrt(gx * gx + gy * gy + 1e-8) > 0.15).float().mean(dim=(1, 2, 3))
    std = gray.std(dim=(1, 2, 3))
    skin = ((r > 0.45) & (g > 0.25) & (b > 0.15) & (r > g * 1.05) & (r > b * 1.15)).float().mean(dim=(1, 2))
    green = ((g > r * 1.10) & (g > b * 1.10)).float().mean(dim=(1, 2))
    warm = ((r > g * 1.08) & (r > b * 1.15)).float().mean(dim=(1, 2))
    blue = ((b > r * 1.05) & (b > g * 1.05)).float().mean(dim=(1, 2))
    sat = (x.max(dim=1).values - x.min(dim=1).values).mean(dim=(1, 2))

    # --- 截图/纯色启发式：5bit 量化后的颜色直方图 ---
    q = (x * 31.0 + 0.5).to(torch.int32)            # B,3,H,W -> 0..31
    code = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]  # B,H,W -> 0..32767
    off = (torch.arange(B, device=x.device).view(B, 1, 1) * 32768)
    hist = torch.bincount((code + off).reshape(-1).long(), minlength=B * 32768)
    hist = hist.view(B, 32768).float()
    n = float(x.shape[2] * x.shape[3])
    top1 = hist.max(dim=1).values / n                       # 最大纯色块占比
    nbins = (hist > 0).float().sum(dim=1) / n               # 颜色多样性

    # 外框 6% 边缘的均匀度（截图常有纯色边框/letterbox）
    k = max(1, int(x.shape[2] * 0.06))
    border = torch.cat([gray[:, :, :k, :].reshape(B, -1), gray[:, :, -k:, :].reshape(B, -1),
                        gray[:, :, :, :k].reshape(B, -1), gray[:, :, :, -k:].reshape(B, -1)], dim=1)
    border_std = border.std(dim=1)
    return dict(blur=blur, edge=edge, std=std, skin=skin, green=green,
                warm=warm, blue=blue, sat=sat, top1=top1, nbins=nbins,
                border_std=border_std)


def load_detector(kind, repo):
    """返回 (model, person_label_id) 或 (None, None)。"""
    if kind == "none":
        return None, None
    from torchvision.models.detection import (
        ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights)
    if kind == "keypoint":
        from torchvision.models.detection import (
            keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights)
        w = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        m = keypointrcnn_resnet50_fpn(weights=w).eval()
        return m, 1
    w = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    m = ssdlite320_mobilenet_v3_large(weights=w).eval()
    return m, 1  # COCO label 1 == person


@torch.no_grad()
def batch_persons(model, kind, imgs, score_thr=0.5):
    """整批人体检测，返回每张图的 person 数（-1 表示检测失败）。"""
    if model is None:
        return [-1] * len(imgs)
    try:
        outs = model(imgs)
    except Exception:
        return [-1] * len(imgs)
    res = []
    for o in outs:
        keep = (o["labels"] == 1) & (o["scores"] > score_thr)
        res.append(int(keep.sum().item()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='原始图片目录（递归）')
    ap.add_argument('--out', required=True, help='输出目录')
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--square-mode', default='crop', choices=['crop', 'stretch', 'keep'],
                    help='crop=短边缩到res后中心裁剪(默认，不拉变形) / stretch=拉伸成方形 / keep=保持长宽比')
    ap.add_argument('--crop-bias', type=float, default=0.0,
                    help='裁剪窗垂直偏移，正数=上移(避开底边水印)，按图像高度比例。如 0.12 = 上移 12%%')
    ap.add_argument('--crop-region', default='full', choices=['full', 'upper'],
                    help='upper = 裁剪窗上移(等价 crop-bias=0.18)，取脸部/上半身，绕开胸臀码区')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--detect', default='ssdlite', choices=['ssdlite', 'keypoint', 'none'],
                    help='人体检测器：ssdlite(默认,~70ms/张) / keypoint(~2s/张) / none')
    ap.add_argument('--detect-threshold', type=float, default=0.5)
    ap.add_argument('--blur-threshold', type=float, default=0.003, help='Laplacian 方差下限')
    ap.add_argument('--min-side', type=int, default=256)
    ap.add_argument('--plain-std', type=float, default=0.035, help='纯色图：灰度标准差下限')
    ap.add_argument('--plain-top1', type=float, default=0.80, help='纯色图：最大单色占比上限')
    ap.add_argument('--shot-top1', type=float, default=0.45, help='截图：最大单色占比下限')
    ap.add_argument('--shot-edge', type=float, default=0.045, help='截图：边缘密度上限')
    ap.add_argument('--require-person', action='store_true', help='剔除检测不到人的图')
    ap.add_argument('--max-person', type=int, default=0, help='>0 时剔除人数超过该值的图')
    ap.add_argument('--keep-all', action='store_true', help='不过滤纯色/截图')
    ap.add_argument('--scene-filter', action='store_true',
                    help='额外启用"风景/食物"过滤（clean_pipeline 遗留判据，默认关闭以免误杀动漫背景）')
    ap.add_argument('--dewatermark', action='store_true', help='显式开启去水印（需要 scipy，很慢）')
    ap.add_argument('--save', action='store_true', default=True)
    ap.add_argument('--no-save', dest='save', action='store_false')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--paths-from', default='',
                    help='只处理 jsonl 里列出的图（取 src/image 字段）；用于按更低阈值重跑被剔除的子集')
    ap.add_argument('--dump-metrics', action='store_true', help='额外输出全部指标到 metrics.jsonl')
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    repo = os.path.dirname(os.path.abspath(__file__))
    if a.paths_from:
        paths = []
        with open(a.paths_from, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                pth = r.get('src') or r.get('image')
                if pth and os.path.exists(pth):
                    paths.append(pth)
        print(f'[fast_clean] --paths-from: {len(paths)} paths', flush=True)
    else:
        paths = list_images(a.data)
    if a.limit:
        paths = paths[:a.limit]
    if not paths:
        raise SystemExit('no images found')
    out_images = os.path.join(a.out, "images")
    os.makedirs(out_images, exist_ok=True)

    dw = None
    if a.dewatermark:
        try:
            sys.path.insert(0, repo)
            from scripts.lib import dewatermark
            print('[fast_clean] dewatermark ENABLED (slow: scipy EDT per image)', flush=True)
        except Exception as e:
            dw = None
            print(f'[fast_clean] dewatermark UNAVAILABLE ({type(e).__name__}: {e}) -> skipped', flush=True)

    t_load = time.perf_counter()
    model, _ = load_detector(a.detect, repo)
    t_detect_load = time.perf_counter() - t_load

    S = dict(decode=0.0, metrics=0.0, detect=0.0, save=0.0, dewatermark=0.0)
    kept = rej = 0
    f_keep = open(os.path.join(a.out, "metadata.jsonl"), "w", encoding="utf-8")
    f_rej = open(os.path.join(a.out, "rejected.jsonl"), "w", encoding="utf-8")
    f_met = open(os.path.join(a.out, "metrics.jsonl"), "w", encoding="utf-8") if a.dump_metrics else None
    t_all = time.perf_counter()
    n_done = 0

    for s in range(0, len(paths), a.batch):
        chunk = paths[s:s + a.batch]
        t0 = time.perf_counter()
        arrays, sizes, ok = [], [], []
        for p in chunk:
            try:
                arr, sz = load_one(p, a.res, a.square_mode, a.crop_bias, a.crop_region)
                arrays.append(arr); sizes.append(sz); ok.append(True)
            except Exception:
                arrays.append(np.zeros((a.res, a.res, 3), np.uint8)); sizes.append((0, 0)); ok.append(False)
        t1 = time.perf_counter(); S['decode'] += t1 - t0

        x = batch_tensor(arrays)
        m = batch_metrics(x)
        t2 = time.perf_counter(); S['metrics'] += t2 - t1

        if model is not None:
            det_in = [x[i] for i in range(len(chunk))]
            persons = batch_persons(model, a.detect, det_in, a.detect_threshold)
        else:
            persons = [-1] * len(chunk)
        t3 = time.perf_counter(); S['detect'] += t3 - t2

        for i, p in enumerate(chunk):
            n_done += 1
            if not ok[i]:
                rec = dict(src=p, keep=False, reasons=['decode_error'])
            else:
                bw, bh = sizes[i]
                met = {k: float(v[i]) for k, v in m.items()}
                reasons = []
                if min(bw, bh) < a.min_side:
                    reasons.append('too_small')
                if met['blur'] < a.blur_threshold:
                    reasons.append('blur')
                if not a.keep_all:
                    if met['std'] < a.plain_std or met['top1'] > a.plain_top1:
                        reasons.append('plain')
                    is_screen_res = (bw, bh) in SCREEN_RES or (bh, bw) in SCREEN_RES
                    if (met['top1'] > a.shot_top1 and met['edge'] < a.shot_edge) or \
                       (is_screen_res and met['top1'] > 0.28 and met['edge'] < a.shot_edge):
                        reasons.append('screenshot')
                    # 风景/食物是 clean_pipeline 的遗留判据，不在"模糊+纯色+截图+无人"需求内，
                    # 且实测误杀动漫天空/草地背景（93 张），故默认关闭。
                    if a.scene_filter:
                        if met['green'] > 0.30 or met['blue'] > 0.35:
                            reasons.append('scenery')
                        if met['warm'] > 0.60 and met['skin'] < 0.02 and met['blur'] > 0.05:
                            reasons.append('food')
                if a.require_person and persons[i] == 0:
                    reasons.append('no_person')
                if a.max_person > 0 and persons[i] > a.max_person:
                    reasons.append('too_many_person')
                np_ = persons[i]
                met['persons'] = np_
                met['w'], met['h'] = bw, bh
                met['src'] = p
                if f_met is not None:
                    f_met.write(json.dumps(met, ensure_ascii=False) + "\n")
                rec = {k: round(v, 6) if isinstance(v, float) else v for k, v in met.items()}
                rec['src'] = p
                if reasons:
                    rec['keep'] = False; rec['reasons'] = reasons
                else:
                    # 保留相对目录结构：不同子目录存在大量同名帧（f000.png ...），
                    # 只用 basename 会互相覆盖。clean_pipeline 有这个 bug。
                    rel = os.path.relpath(p, a.data)
                    rel = os.path.splitext(rel)[0] + ".jpg"
                    rec['keep'] = True
                    if a.save:
                        t4 = time.perf_counter()
                        dst = os.path.join(out_images, rel)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        Image.fromarray(arrays[i]).save(dst, quality=95)
                        S['save'] += time.perf_counter() - t4
                        rec['image'] = dst

            (f_keep if rec.get('keep') else f_rej).write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec.get('keep'): kept += 1
            else: rej += 1

        if (s // a.batch) % 10 == 0:
            el = time.perf_counter() - t_all
            print(f'[fast_clean] {n_done}/{len(paths)} kept={kept} rej={rej} '
                  f'{el:.1f}s  {el/max(1,n_done)*1000:.0f} ms/img', flush=True)

    for f in (f_keep, f_rej, f_met):
        if f is not None: f.close()

    total = time.perf_counter() - t_all
    report = dict(total_s=round(total, 2), images=len(paths), kept=kept, rejected=rej,
                  ms_per_img=round(total / max(1, len(paths)) * 1000, 1),
                  stage_s={k: round(v, 2) for k, v in S.items()},
                  detect=a.detect, batch=a.batch, res=a.res)
    with open(os.path.join(a.out, "clean_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('[fast_clean] ' + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
