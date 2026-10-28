# -*- coding: utf-8 -*-
"""probe_new_data.py - 抽样拼成一张联系表，用于人工检查新数据的裁剪/水印/码区情况。

用法:
  python probe_new_data.py --groups "dl=D:\\a;tu=D:\\b:2" --out sheet.png
"""
from __future__ import annotations
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
import argparse, os, random
from PIL import Image, ImageDraw

EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def collect(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(EXTS):
                out.append(os.path.join(dp, fn))
    out.sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', required=True,
                    help='name=path[:每池张数] ，多项用 ; 分隔')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cell', type=int, default=320)
    ap.add_argument('--cols', type=int, default=6)
    ap.add_argument('--per-group', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    tiles = []
    for spec in a.groups.split(';'):
        spec = spec.strip()
        if not spec:
            continue
        name, path = spec.split('=', 1)
        n = a.per_group
        if ':' in path:
            head, tail = path.rsplit(':', 1)
            if tail.isdigit():
                path, n = head, int(tail)
        fs = collect(path)
        if not fs:
            print('  [空] %s' % name)
            continue
        pick = random.sample(fs, min(n, len(fs)))
        print('  %-10s 共 %5d 张, 抽样 %d' % (name, len(fs), len(pick)))
        for p in pick:
            tiles.append((name, p))

    if not tiles:
        print('没有可抽样的图片')
        return

    C, strip = a.cell, 18
    cols = a.cols
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * C, rows * (C + strip)), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    for i, (name, p) in enumerate(tiles):
        r, c = divmod(i, cols)
        y0 = r * (C + strip)
        try:
            with Image.open(p) as im:
                im = im.convert('RGB')
                im.thumbnail((C, C))
                sheet.paste(im, (c * C + (C - im.width) // 2, y0 + (C - im.height) // 2))
        except Exception:
            d.text((c * C + 4, y0 + 4), 'ERR', fill=(255, 80, 80))
        label = '%s %s' % (name, os.path.basename(p)[:30])
        d.text((c * C + 3, y0 + C + 4), label, fill=(200, 200, 200))
    sheet.save(a.out, quality=92)
    print('拼图 -> %s  (%d 张)' % (a.out, len(tiles)))


if __name__ == '__main__':
    main()
