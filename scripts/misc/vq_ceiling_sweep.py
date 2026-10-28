# -*- coding: utf-8 -*-
"""vq_ceiling_sweep.py — 在【同一批图】上量各分词器的重建天花板 vs token 数。

动机（2026-09-19）：
  重看图显示 C_mse 的重建是"糊"，而换成 P（瓶颈注意力）也只是"稍锐"。
  糊的形状是【平滑模糊】而不是【块状量化】⇒ 怀疑限制不在码本大小，而在【信息预算】：
  256 个 token x 10 bit = 2560 bit，而一张 256x256 自然图在 30 dB 下大约需要 ~2e5 bit。
  如果这条成立，那么"加大码本"和"换更好的解码器"都是治标，
  真正要动的是 token 数量（或改用下游细节模型）。

本脚本给的是【果】：同一批图上，重建 PSNR 随 token 数怎么走。

用法:
  python vq_ceiling_sweep.py --paths <paths.json> --range 27658:29113 --n 64
"""
from __future__ import annotations
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
import argparse, json, os, sys, math, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_vqvae import vqvae_from_ckpt
from scripts.lib.ms_vq import msvqvae_from_ckpt

SIZE = 256
TF = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
                         transforms.Normalize([0.5] * 3, [0.5] * 3)])


def psnr(a, b):
    mse = float(F.mse_loss(a, b))
    return 10.0 * math.log10(4.0 / max(mse, 1e-12))     # 数据在 [-1,1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paths', default=r'D:\work\bitsandbytes-CPU\.autopilot\tokens_16x16_full.pt.paths.json')
    ap.add_argument('--range', default='27658:29113')
    ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    paths = json.load(open(a.paths, encoding='utf-8'))
    lo, hi = (int(x) for x in a.range.split(':'))
    sel = paths[lo:min(hi, len(paths))]
    if a.n and a.n < len(sel):
        step = len(sel) / a.n
        sel = [sel[int(i * step)] for i in range(a.n)]
    print('[sweep] %d 张验证图' % len(sel))

    xs = []
    for p in sel:
        with Image.open(p) as im:
            xs.append(TF(im.convert('RGB')))
    x = torch.stack(xs)
    print('[sweep] 输入张量', tuple(x.shape))

    CAND = [
        ('D_8x8mse      (8x8,  64 tok)', r'D:\work\cloud_results\vqvae_8x8.pt'),
        ('C_mse        (16x16, 256 tok)', r'D:\work\cloud_results\ckpt_C_mse.pt'),
        ('F_perc       (16x16, 256 tok)', r'D:\work\cloud_results\F_16x16_perc_last.pt'),
        ('P_attn       (16x16, 256 tok)', r'D:\work\cloud_results\P_attn_26040.pt'),
        ('V32_attn     (32x32,1024 tok)', r'D:\work\cloud_results\V32_attn_18045.pt'),
    ]
    if a.only:
        CAND = [c for c in CAND if a.only in c[0]]

    print()
    print('%-32s %8s %8s %8s' % ('tokenizer', 'tokens', 'PSNR', 'rel-L1'))
    print('-' * 62)
    rows = []
    for name, ck in CAND:
        if not os.path.exists(ck):
            print('%-32s  (缺文件)' % name); continue
        c = torch.load(ck, map_location='cpu', weights_only=False)
        if c.get('ms_vq'):
            model, _ = msvqvae_from_ckpt(c)
        else:
            model, _ = vqvae_from_ckpt(c)
        model.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(x), 8):
                xb = x[s:s + 8]
                if c.get('ms_vq'):
                    idx = model.encode_idx(xb)
                    rec = model.decode_idx(idx)
                else:
                    rec, idx, _ = model(xb)
                outs.append(rec.clamp(-1, 1))
        rec = torch.cat(outs, 0)
        nt = int(np.prod(idx.shape[1:])) if hasattr(idx, 'shape') else 0
        P = psnr(x, rec)
        rl1 = float((rec - x).abs().mean())
        print('%-32s %8d %8.2f %8.4f' % (name, nt, P, rl1))
        rows.append((nt, P))
    print()
    print('⇒ 看 PSNR 随 token 数的斜率：若每 4x token 只涨 ~2.5 dB，')
    print('   那么到 30 dB 需要 ~200x 的 token 数 ⇒ token 化路线本质上到不了高保真。')


if __name__ == '__main__':
    main()
