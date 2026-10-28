# -*- coding: utf-8 -*-
"""mosaic_mask.py - 定位马赛克区域，生成 VQ token 级 mask。

为什么用 mask 而不是删图
------------------------
1/3 的图带码、且集中压在胸/臀 —— 这个比例下模型必然学会"那个部位画马赛克"。
但删掉要损失 1/3 数据，inpaint 又是伪造。**正确做法是让码区不产生梯度**：
码区那些 token 完全不参与训练，马赛克就被彻底中和，而不是被稀释。

对 mask 来说，**误报无害**（只是少训几块干净像素），**漏报才致命**。
所以阈值故意调向高召回。

坐标对齐
--------
- 分析分辨率 512，定位网格 4x4 -> 每格 128px
- VQ token 网格 16x16（256px 输入，每个 token 16px）
- 512 上的 128px 格 == 256 上的 64px == **正好 4x4 个 token**
所以一个命中格直接映射成 4x4 的 token 掩码，不用插值。

用法
----
  # 标定阈值：看干净图的每格分数分布
  python mosaic_mask.py --calibrate --data <目录> --n 150 --res 512 --grid 4

  # 全量生成 mask
  python mosaic_mask.py --data <目录> --out masks.npz --res 512 --grid 4 --threshold 1.30

  # 出可视化（命中格画红框），用于人工确认
  python mosaic_mask.py --data <目录> --viz <viz目录> --res 512 --grid 4 --threshold 1.30 --viz-n 24
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, json, os, random, sys, time
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mosaic_detect_v2 import list_images, load_gray, score_patch, jpeg_roundtrip

JPEG_B = (8, 16, 24)   # JPEG 网格尺寸，排除（验证显示能显著降低误报）


def cell_scores(g, bsizes, grid):
    """把图切成 grid x grid 格，每格算一个马赛克分数。返回 (grid,grid) 数组。"""
    H, W = g.shape
    ph, pw = H // grid, W // grid
    out = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            p = g[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]
            s, _ = score_patch(p, bsizes, JPEG_B)
            out[i, j] = s
    return out


def cells_to_tokens(cell_mask, grid, tokens=16):
    """(grid,grid) 布尔 -> (tokens,tokens) 布尔。要求 tokens % grid == 0。"""
    k = tokens // grid
    return np.kron(cell_mask, np.ones((k, k), dtype=bool))


def calibrate(data, n, res, grid, bsizes):
    paths = list_images(data)
    random.seed(0)
    sample = random.sample(paths, min(n, len(paths)))
    allc = []
    t0 = time.perf_counter()
    for p in sample:
        try:
            g = jpeg_roundtrip(load_gray(p, res))
        except Exception:
            continue
        allc.append(cell_scores(g, bsizes, grid).ravel())
    c = np.concatenate(allc)
    print('标定：%d 张 x %d 格 = %d 个格子，耗时 %.1fs' % (
        len(allc), grid * grid, c.size, time.perf_counter() - t0))
    for q in (50, 75, 90, 95, 97, 99, 99.5, 99.9):
        print('  含 JPEG 往返: p%-5s = %.3f' % (q, np.percentile(c, q)))
    print('  max = %.3f  mean = %.3f' % (c.max(), c.mean()))
    print('\n建议阈值（干净格的误报率）:')
    for th in np.arange(1.1, 3.01, 0.1):
        fp = (c >= th).mean() * 100
        if fp <= 5.0:
            print('  th=%.2f -> 干净格误报 %.3f%%' % (th, fp))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='', help='图片目录；多个用 | 分隔')
    ap.add_argument('--paths', default='',
                    help='tokenize_dataset.py 产出的 <tokens>.paths.json —— '
                         '用它可保证 mask 与 token 严格同序（推荐）')
    ap.add_argument('--out', default='')
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--grid', type=int, default=4)
    ap.add_argument('--tokens', type=int, default=16)
    ap.add_argument('--threshold', type=float, default=1.30)
    ap.add_argument('--bmin', type=int, default=4)
    ap.add_argument('--bmax', type=int, default=28)
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--viz', default='')
    ap.add_argument('--viz-n', type=int, default=24)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    if a.tokens % a.grid:
        raise SystemExit('tokens(%d) 必须能被 grid(%d) 整除' % (a.tokens, a.grid))
    bsizes = list(range(a.bmin, a.bmax + 1))

    if a.calibrate:
        calibrate(list_images(a.data.split('|')[0]), a.n, a.res, a.grid, bsizes)
        return

    if a.paths:
        import json as _json
        paths = _json.load(open(a.paths, encoding='utf-8'))
        print('从路径清单读入 %d 条（与 token 同序）' % len(paths))
    else:
        roots = [x for x in a.data.split('|') if x.strip()]
        paths = []
        for r in roots:
            paths.extend(list_images(r))
        paths = sorted(set(paths))
    if a.limit:
        paths = paths[:a.limit]
    k = a.tokens // a.grid
    print('图片 %d 张 | 分析 %dpx | 定位 %dx%d 格 | 阈值 %.2f | 每格 %d 个 token' % (
        len(paths), a.res, a.grid, a.grid, a.threshold, k * k))

    masks, kept_paths, cellstats = [], [], []
    t0 = time.perf_counter()
    viz_imgs = []
    for idx, p in enumerate(paths):
        try:
            g = jpeg_roundtrip(load_gray(p, a.res))
            cs = cell_scores(g, bsizes, a.grid)
            cm = cs >= a.threshold
            masks.append(cells_to_tokens(cm, a.grid, a.tokens))
            kept_paths.append(p)
            cellstats.append(int(cm.sum()))
            # 可视化：命中比例高的优先展示
            if a.viz and len(viz_imgs) < 400:
                viz_imgs.append((float(cs.max()), p, cm.copy()))
        except Exception as e:
            masks.append(np.zeros((a.tokens, a.tokens), dtype=bool))
            kept_paths.append(p)
            cellstats.append(-1)
        if (idx + 1) % 500 == 0:
            print('  %d/%d %.1fs' % (idx + 1, len(paths), time.perf_counter() - t0), flush=True)

    M = np.stack(masks)
    cs_arr = np.array(cellstats)
    print('\n完成 %.1fs' % (time.perf_counter() - t0))
    print('  无命中格     : %d 张 (%.1f%%)' % ((cs_arr == 0).sum(), (cs_arr == 0).mean() * 100))
    print('  命中 1~3 格  : %d 张' % (((cs_arr >= 1) & (cs_arr <= 3)).sum()))
    print('  命中 >=4 格  : %d 张' % ((cs_arr >= 4).sum()))
    print('  token 被屏蔽比例: %.2f%%' % (M.mean() * 100))

    if a.out:
        np.savez_compressed(a.out, masks=M,
                            paths=np.array(kept_paths),
                            grid=a.grid, tokens=a.tokens, res=a.res,
                            threshold=a.threshold)
        print('-> %s  (masks %s)' % (a.out, M.shape))

    if a.viz and viz_imgs:
        os.makedirs(a.viz, exist_ok=True)
        viz_imgs.sort(key=lambda x: -x[0])
        sel = viz_imgs[:a.viz_n]
        cell = 300
        cols = 6
        rows = (len(sel) + cols - 1) // cols
        canvas = Image.new('RGB', (cols * cell, rows * (cell + 16)), (25, 25, 25))
        dr = ImageDraw.Draw(canvas)
        for i, (sc, p, cm) in enumerate(sel):
            r, c = divmod(i, cols)
            x, y = c * cell, r * (cell + 16)
            im = Image.open(p).convert('RGB').resize((cell - 6, cell - 6), Image.BILINEAR)
            canvas.paste(im, (x + 3, y + 16 + 3))
            kk = (cell - 6) / a.grid
            for gi in range(a.grid):
                for gj in range(a.grid):
                    if cm[gi, gj]:
                        dr.rectangle([x + 3 + gj * kk, y + 19 + gi * kk,
                                      x + 3 + (gj + 1) * kk, y + 19 + (gi + 1) * kk],
                                     outline=(255, 40, 40), width=3)
            dr.text((x + 4, y + 2), '%d sc=%.1f %s' % (i + 1, sc, os.path.basename(p)[:20]),
                    fill=(220, 220, 220))
        out = os.path.join(a.viz, 'mask_overlay.png')
        canvas.save(out)
        print('-> %s' % out)


if __name__ == '__main__':
    main()
