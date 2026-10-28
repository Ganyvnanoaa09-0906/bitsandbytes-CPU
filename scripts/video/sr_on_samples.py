# -*- coding: utf-8 -*-
"""sr_on_samples.py — 给 AR 出图接 Real-ESRGAN anime 6B ×4，和 bicubic 对比。

动机：分词器重建图（BEST_recon_Cmse/P）本身就软，说明"墙"在分词器不在 AR。
在动 AR 之前先量一下【后处理】能挽回多少观感 —— 这是最便宜的一档。

用法:
  python sr_on_samples.py --grid <4x4 采样网格.png> --out <对比图.png> --tiles 4,9
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, os, sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realesrgan_anime_test import load_realesrgan


def to_t(x):
    return torch.from_numpy(x).permute(2, 0, 1).float().div(255).unsqueeze(0)


def to_np(t):
    return (t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tiles', default='4,9', help='用哪几个 tile 做对比（0~15）')
    ap.add_argument('--crop', type=int, default=512, help='在 ×4 输出上取多大中心裁切')
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    im = Image.open(a.grid).convert('RGB')
    W, H = im.size
    side = 4
    tw, th = W // side, H // side
    print('[sr] 网格 %dx%d  tile %dx%d' % (W, H, tw, th))

    net = load_realesrgan()
    picks = [int(x) for x in a.tiles.split(',')]

    panels = []          # 每行: [bicubic, esrgan]
    for pi in picks:
        r, c = divmod(pi, side)
        tile = im.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
        x = to_t(np.asarray(tile))

        bic = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        with torch.no_grad():
            sr = net(x)
        sr = sr.clamp(0, 1)

        ch = a.crop
        def cc(t):
            _, _, hh, ww = t.shape
            y0 = max(0, (hh - ch) // 2); x0 = max(0, (ww - ch) // 2)
            return t[:, :, y0:y0 + ch, x0:x0 + ch]

        panels.append((cc(bic), cc(sr)))
        print('[sr] tile %d done' % pi)

    # 拼成  len(picks) 行 x 2 列
    rows = []
    for b, s in panels:
        rows.append(np.concatenate([to_np(b), np.full((a.crop, 4, 3), 255, np.uint8), to_np(s)], axis=1))
    out = np.concatenate(
        [x for pair in zip(rows, [np.full((4, rows[0].shape[1], 3), 255, np.uint8)] * len(rows))
         for x in pair][:-1], axis=0)
    Image.fromarray(out).save(a.out)
    print('[sr] -> %s   (左=bicubic ×4, 右=Real-ESRGAN anime 6B ×4, 100%% 裁切)  %dx%d'
          % (a.out, out.shape[1], out.shape[0]))


if __name__ == '__main__':
    main()
