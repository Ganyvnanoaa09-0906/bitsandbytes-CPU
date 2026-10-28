# -*- coding: utf-8 -*-
"""watermark_probe.py - 水印探测：叠加平均 + 角/中心边缘密度统计。

原理
----
同一来源的水印在每张图里**位置和形状固定**，而画面内容随机。
把 N 张图的同一区域叠加平均，随机内容会互相抵消（趋向均值灰），
固定的水印会保留下来 —— 从噪声里"浮"出来。

同时统计每个区域的边缘密度：水印是细笔画文字/logo，边缘密度显著高于
同位置的自然图像区域。

用法
----
  python watermark_probe.py --data <图片目录> --out <输出目录> [--n 800]
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, os, sys, random
import numpy as np
from PIL import Image

EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def list_images(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(EXTS):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=800)
    ap.add_argument('--patch', type=int, default=192, help='每个区域采样边长')
    a = ap.parse_args()

    paths = list_images(a.data)
    if not paths:
        raise SystemExit('no images')
    random.seed(0)
    sample = random.sample(paths, min(a.n, len(paths)))
    print(f'目录 {len(paths)} 张，抽样 {len(sample)} 张')

    P = a.patch
    # 四个角 + 中心（+ 上中、下中，水印也常横贯底边）
    regions = {
        '左上': (0, 0), '右上': (1, 0), '左下': (0, 1), '右下': (1, 1),
        '上中': (0.5, 0), '下中': (0.5, 1), '中心': (0.5, 0.5),
    }
    acc = {k: np.zeros((P, P, 3), np.float64) for k in regions}
    acc2 = {k: np.zeros((P, P, 3), np.float64) for k in regions}
    n_ok = 0
    # 每张图各区域的高频能量（边缘密度代理）
    energy = {k: [] for k in regions}

    for p in sample:
        try:
            im = Image.open(p).convert('RGB').resize((512, 512), Image.BILINEAR)
        except Exception:
            continue
        x = np.asarray(im, dtype=np.float32) / 255.0
        n_ok += 1
        for k, (fx, fy) in regions.items():
            cx = int(fx * (512 - P))
            cy = int(fy * (512 - P))
            patch = x[cy:cy + P, cx:cx + P]
            if patch.shape[:2] != (P, P):
                continue
            acc[k] += patch
            acc2[k] += patch * patch
            # 高频能量：与 3x3 均值的差
            g = patch.mean(axis=2)
            kx = np.abs(np.diff(g, axis=1)).mean()
            ky = np.abs(np.diff(g, axis=0)).mean()
            energy[k].append((kx + ky) / 2)

    if n_ok == 0:
        raise SystemExit('no readable images')

    os.makedirs(a.out, exist_ok=True)
    print(f'成功读取 {n_ok} 张\n')
    print(f'{"区域":<6} {"平均边缘能量":>12} {"标准差":>10}   {"与中心比":>10}')
    cen = float(np.mean(energy['中心'])) if energy['中心'] else 1.0
    for k in regions:
        v = np.array(energy[k])
        if v.size == 0:
            continue
        print(f'{k:<6} {v.mean():>12.5f} {v.std():>10.5f}   {v.mean()/max(cen,1e-9):>9.2f}x')

    # 保存各区域的平均图（水印会浮现）
    tiles = []
    for k in ('左上', '上中', '右上', '中心', '左下', '下中', '右下'):
        m = acc[k] / max(n_ok, 1)
        # 拉对比度，让弱水印可见
        lo, hi = np.percentile(m, 1), np.percentile(m, 99)
        vis = np.clip((m - lo) / max(hi - lo, 1e-6), 0, 1)
        tiles.append((k, vis))
    gap = 12
    H = P * 3 + gap * 2
    W = P * 3 + gap * 2
    canvas = np.ones((H, W, 3), np.float32) * 0.5
    pos = [(0, 0), (1, 0), (2, 0), (1, 1), (0, 2), (1, 2), (2, 2)]
    for (k, vis), (cx, cy) in zip(tiles, pos):
        y0 = cy * (P + gap); x0 = cx * (P + gap)
        canvas[y0:y0 + P, x0:x0 + P] = vis
    outp = os.path.join(a.out, 'corner_average.png')
    Image.fromarray((canvas * 255).astype(np.uint8)).save(outp)
    print(f'\n平均图（对比度拉伸）-> {outp}')

    # 逐像素"跨图一致性"：水印像素在不同图间差异极小但均值偏离背景
    stdmap = {}
    for k in ('左下', '右下', '下中'):
        m = acc[k] / max(n_ok, 1)
        v = acc2[k] / max(n_ok, 1) - m * m
        stdmap[k] = np.sqrt(np.clip(v, 0, None)).mean(axis=2)
        print(f'{k} 跨图标准差中位: {np.median(stdmap[k]):.4f}')
    return


if __name__ == '__main__':
    main()
