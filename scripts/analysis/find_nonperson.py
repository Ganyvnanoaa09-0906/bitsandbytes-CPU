# -*- coding: utf-8 -*-
"""find_nonperson.py - 挑出「一张人都没有」的图。

cos 图必然有人；一张人都没有的几乎必然是风景/物品/截图/纯色，
是「完全不是 cos 图」里最可靠、可解释的那一部分。

不自动删除，只产出一份候选清单 + 联系表，供人工确认。

用法:
  python find_nonperson.py --data <clean_nf/images> --out nonperson.jsonl --sheet nonperson.png
  python find_nonperson.py --data <dir> --limit 400        # 快速试跑
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, collections, json, os, random
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights)

IMG = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def list_images(root):
    out = []
    for dp, _, fns in os.walk(root):
        for f in fns:
            if f.lower().endswith(IMG):
                out.append(os.path.join(dp, f))
    return sorted(out)


def to_tensor(path, size=512):
    with Image.open(path) as im:
        im = im.convert('RGB')
        im.thumbnail((size, size))
        a = np.asarray(im)
    return torch.from_numpy(a).permute(2, 0, 1).float().div_(255.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='nonperson.jsonl')
    ap.add_argument('--sheet', default='nonperson.png')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--score', type=float, default=0.5)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--sheet-n', type=int, default=24)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    paths = list_images(a.data)
    if a.limit:
        paths = paths[:a.limit]
    print('待检测: %d 张' % len(paths))

    w = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=w).eval()

    dist = collections.Counter()
    zero = []
    rows = []
    with torch.no_grad():
        for i in range(0, len(paths), a.batch):
            chunk = paths[i:i + a.batch]
            tens = []
            ok = []
            for p in chunk:
                try:
                    tens.append(to_tensor(p)); ok.append(p)
                except Exception:
                    pass
            if not tens:
                continue
            try:
                outs = model(tens)
            except Exception as e:
                print('  批次失败:', e)
                continue
            for p, o in zip(ok, outs):
                k = (o['labels'] == 1) & (o['scores'] > a.score)
                n = int(k.sum().item())
                rows.append({'path': p, 'nperson': n,
                             'max_score': float(o['scores'][k].max()) if n else 0.0})
                dist[min(n, 4)] += 1
                if n == 0:
                    zero.append(p)
            if (i // a.batch) % 100 == 0 and i:
                print('  ...%d/%d  无人 %d' % (i, len(paths), len(zero)), flush=True)

    with open(a.out, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print()
    print('人数分布: 0人=%d  1人=%d  2人=%d  3人=%d  4人+=%d' % (
        dist[0], dist[1], dist[2], dist[3], dist[4]))
    print('无人图占比: %.2f%%' % (100.0 * dist[0] / max(1, len(rows))))
    print('清单 -> %s' % a.out)

    if zero:
        random.seed(0)
        samp = random.sample(zero, min(a.sheet_n, len(zero)))
        C, strip, cols = 240, 16, 6
        rr = (len(samp) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * C, rr * (C + strip)), (24, 24, 24))
        d = ImageDraw.Draw(sheet)
        for j, p in enumerate(samp):
            r_, c_ = divmod(j, cols)
            y0 = r_ * (C + strip)
            try:
                with Image.open(p) as im:
                    im = im.convert('RGB'); im.thumbnail((C, C))
                    sheet.paste(im, (c_ * C + (C - im.width) // 2, y0 + (C - im.height) // 2))
            except Exception:
                d.text((c_ * C + 4, y0 + 4), 'ERR', fill=(255, 80, 80))
            d.text((c_ * C + 3, y0 + C + 2), os.path.basename(p)[:30], fill=(200, 200, 200))
        sheet.save(a.sheet, quality=92)
        print('联系表 -> %s (%d 张抽样)' % (a.sheet, len(samp)))


if __name__ == '__main__':
    main()
