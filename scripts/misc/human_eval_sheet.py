# -*- coding: utf-8 -*-
"""human_eval_sheet.py — 造一张【盲测对照表】，让"碳基高精度判别器"来验证高通 D。

为什么必须盲：
  我的高通 D 训到 100% 准确率，但它是否与【人的审美】一致，我无法自证 ——
  它可能只是学会了"高频占比"这个数值，而那已被证明会把噪声当"好"。
  所以：混入 D 的 top-8 / bottom-8 / 随机中间 8 张，打乱、编号、不透露档位。
  你只报编号 ⇒ 我算一致率。

产出一张 6x4 的编号对照表 + 一个不给人看的映射文件。
"""
import os, sys
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
import torch.nn.functional
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.train_vqvae_gan import PatchDiscriminator
from scripts.lib.train_diag_disc import t_highpass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--disc', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per-group', type=int, default=8)
    ap.add_argument('--seed', type=int, default=123)
    a = ap.parse_args()

    z = torch.load(a.data, map_location='cpu', weights_only=False)
    free = z['free'].float()
    n = free.shape[0]
    print('[sheet] FREE %s  step=%s' % (tuple(free.shape), z.get('step')))

    dk = torch.load(a.disc, map_location='cpu', weights_only=False)
    D = PatchDiscriminator(in_ch=dk.get('in_ch', 3), ndf=dk['ndf'], n_layers=3)
    D.load_state_dict(dk['state']); D.eval()
    dsize = dk.get('size', 128)

    sc = []
    for s in range(0, n, 64):
        img = free[s:s + 64]
        if img.shape[-1] != dsize:
            img = F.interpolate(img, size=dsize, mode='bilinear', align_corners=False)
        with torch.no_grad():
            sc.append(D(t_highpass(img)).mean(dim=(1, 2, 3)))
    sc = torch.cat(sc, 0)
    order = sc.argsort(descending=True)
    k = a.per_group

    top = order[:k].tolist()
    bot = order[-k:].tolist()
    g = torch.Generator().manual_seed(a.seed)
    mid_pool = order[k:n - k]
    mid = mid_pool[torch.randperm(mid_pool.numel(), generator=g)[:k]].tolist()

    idx = top + bot + mid
    labels = ['D-top'] * k + ['D-bottom'] * k + ['D-mid'] * k
    perm = torch.randperm(len(idx), generator=g).tolist()
    idx = [idx[i] for i in perm]
    labels = [labels[i] for i in perm]

    # 画表：每格 256，格上方留白写编号
    S, PAD = 256, 28
    cols = 6
    rows = (len(idx) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * S, rows * (S + PAD)), (245, 245, 245))
    dr = ImageDraw.Draw(canvas)
    for i, (im_i, lab) in enumerate(zip(idx, labels)):
        r, c = divmod(i, cols)
        t = ((free[im_i].clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).numpy().astype('uint8')
        tile = Image.fromarray(t).resize((S, S), Image.NEAREST)
        canvas.paste(tile, (c * S, r * (S + PAD) + PAD))
        dr.text((c * S + 8, r * (S + PAD) + 6), '#%d' % (i + 1), fill=(0, 0, 0))
    canvas.save(a.out)

    key = dict(order=[dict(no=i + 1, idx=im_i, group=lab, d_score=float(sc[im_i]))
                      for i, (im_i, lab) in enumerate(zip(idx, labels))])
    with open(a.out.replace('.png', '_KEY.json'), 'w', encoding='utf-8') as f:
        json.dump(key, f, ensure_ascii=False, indent=1)
    print('[sheet] -> %s  (%dx%d)' % (a.out, canvas.width, canvas.height))
    print('[sheet] 答案键已单独存盘（别给人看）：%s' % a.out.replace('.png', '_KEY.json'))
    print('[sheet] 混合：D-top %d / D-bottom %d / D-mid %d，已打乱' % (k, k, k))


if __name__ == '__main__':
    main()
