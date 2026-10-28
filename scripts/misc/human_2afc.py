# -*- coding: utf-8 -*-
"""human_2afc.py — 二选一强制选择（2AFC）：比"24 张排序"容易得多，也更统计有效。

上一版让用户给 24 张同样抽象的图打分排序 —— 信噪比太低（用户原话："欣赏不来抽象风的油画"）。
但那句话本身是重要数据：**模型自由采样的输出全都离真实图像很远**，所以它们之间
的差异是"次品之间的差异"，绝对打分做不出来。

2AFC 设计：
  · 每行一对：一侧来自高通 D 的 top，另一侧来自 D 的 bottom（都取自同一 test split）
  · 左右顺序随机
  · 用户只需回答"哪一侧是更好的生成"（或"哪一侧更像一张真图/更不像坏图"）
  · 若用户显著偏向 D-top ⇒ D 与人的判断一致 ⇒ D 可作为大规模筛选代理
  · 若≈50% ⇒ D 在测与"人的观感"无关的东西 ⇒ 必须换判据
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
import argparse, json, os, sys
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.train_vqvae_gan import PatchDiscriminator
from scripts.lib.train_diag_disc import t_highpass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--disc', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pairs', type=int, default=12)
    ap.add_argument('--train-frac', type=float, default=0.65)
    ap.add_argument('--scale', type=int, default=1, help='每格放大倍数（256*scale）')
    ap.add_argument('--seed', type=int, default=202)
    a = ap.parse_args()

    z = torch.load(a.data, map_location='cpu', weights_only=False)
    free = z['free'].float()
    n = free.shape[0]
    nt = int(n * a.train_frac)
    te = free[nt:]                                    # 只用 test split，避免 D 见过
    print('[2afc] FREE test split %d 张' % te.shape[0])

    dk = torch.load(a.disc, map_location='cpu', weights_only=False)
    D = PatchDiscriminator(in_ch=dk.get('in_ch', 3), ndf=dk['ndf'], n_layers=3)
    D.load_state_dict(dk['state']); D.eval()
    dsize = dk.get('size', 128)

    imgs = te
    sc_in = imgs
    if imgs.shape[-1] != dsize:
        sc_in = F.interpolate(imgs, size=dsize, mode='bilinear', align_corners=False)
    with torch.no_grad():
        sc = D(t_highpass(sc_in)).mean(dim=(1, 2, 3))
    order = sc.argsort(descending=True)
    m = order.numel()
    k = min(a.pairs, m // 2)
    tops = order[:k].tolist()
    bots = order[-k:].tolist()
    print('[2afc] 分数范围: top %.3f .. bottom %.3f' % (sc[order[0]], sc[order[-1]]))

    g = torch.Generator().manual_seed(a.seed)
    flip = torch.rand(k, generator=g) < 0.5
    rows = []
    for i in range(k):
        L, R = (tops[i], bots[i]) if not flip[i] else (bots[i], tops[i])
        rows.append(dict(no=i + 1, L=int(L), R=int(R),
                         left_group='D-top' if not flip[i] else 'D-bottom',
                         dL=float(sc[L]), dR=float(sc[R])))

    S = 256 * a.scale
    PAD = 30
    # ★ 布局：每行放 cols 对（每对两格）。竖排 10 行会得到 1060x5420 的细长图，
    #   预览时被缩得看不清 —— 网格布局的宽高比更接近屏幕。
    cols = 2
    rowh = S + PAD
    nrow = (k + cols - 1) // cols
    pairw = 2 * S + 24
    canvas = Image.new('RGB', (cols * pairw + 12, nrow * rowh), (245, 245, 245))
    dr = ImageDraw.Draw(canvas)
    for i, r in enumerate(rows):
        gr, gc = divmod(i, cols)
        x0 = gc * pairw + 12
        y0 = gr * rowh + PAD
        dr.text((x0, y0 - 22), '#%d   L            R' % r['no'], fill=(0, 0, 0))
        for key, x in (('L', x0), ('R', x0 + S + 12)):
            t = ((imgs[r[key]].clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).numpy().astype('uint8')
            tile = Image.fromarray(t)
            if a.scale != 1:
                tile = tile.resize((S, S), Image.LANCZOS)
            canvas.paste(tile, (x, y0))
    canvas.save(a.out)
    with open(a.out.replace('.png', '_KEY.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(rows=rows), f, ensure_ascii=False, indent=1)
    print('[2afc] -> %s  (%dx%d)' % (a.out, canvas.width, canvas.height))
    print('[2afc] 答案键: %s' % a.out.replace('.png', '_KEY.json'))
    print('[2afc] 用户只需回答每行 L / R 哪一侧是"更好的生成"')


if __name__ == '__main__':
    main()
