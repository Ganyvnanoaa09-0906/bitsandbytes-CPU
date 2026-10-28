# -*- coding: utf-8 -*-
"""diag_ms_scale.py — 多尺度残差链的量级诊断（找 rms 暴涨的真正来源）

已排除的假设（别再猜）：
  ① 死码重启抖动  —— 不是它，重启数已限流到 20/1024
  ② EMA 初值 1.0 追不上 —— 改成 per-batch RMS 后仍然暴涨
  ③ EMA(0.99) 时间常数太长 —— 同上

本脚本逐尺度打印【所有能想到的量级】，看哪一级第一个炸：
  f 的 RMS / 每级残差 r_k 的 RMS / 归一化因子 sk / 匹配到的码本行 L2 范数 /
  重建 q_k*sk 的 RMS / 累积 f_hat 的 RMS / 承诺损失
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import sys, glob, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
Image.MAX_IMAGE_PIXELS = None
from scripts.lib.small_vqvae import Encoder, Decoder
from scripts.lib.ms_vq import MultiScaleResidualVQ

SIZE, B, STEPS, C = 256, 4, 15, 64
PATCH = (1, 2, 4, 8, 16)


def rms(t):
    return t.detach().pow(2).mean().sqrt().item()


def load_batch(paths):
    out = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        im = im.resize((SIZE, SIZE), Image.BICUBIC)
        a = np.asarray(im, dtype='uint8').copy()
        out.append(torch.from_numpy(a).float().permute(2, 0, 1) / 127.5 - 1.0)
    return torch.stack(out)


files = glob.glob(r'D:\work\bitsandbytes-CPU\.autopilot\clean_nf\**\*.jpg', recursive=True)
random.seed(0)
bs = [[random.choice(files) for _ in range(B)] for _ in range(STEPS)]

torch.manual_seed(7)
enc = Encoder(ch=(64, 128, 256, 256), embed_dim=C, use_attn=False, two_3x3=False)
dec = Decoder(ch=(256, 256, 128, 64), embed_dim=C, use_attn=False, two_3x3=False)
cb = nn.Embedding(1024, C)
nn.init.normal_(cb.weight, std=0.5)
cb.weight.requires_grad_(False)
ms = MultiScaleResidualVQ(cb, patch_nums=PATCH)
ms.train()
opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=2e-4)

print('=' * 100)
print('多尺度残差链量级诊断   列=[f_rms | r0..r4 | sk0..sk4 | cbrow0..4 | qk0..4 | fhat | commit]')
print('=' * 100)
print('  %-4s %7s | %s | %s | %s | %s | %8s %8s' %
      ('step', 'f_rms', ' '.join('r%d' % i for i in range(5)),
       ' '.join('s%d ' % i for i in range(5)), ' '.join('cb%d' % i for i in range(5)),
       ' '.join('q%d ' % i for i in range(5)), 'fhat', 'commit'))

for step in range(STEPS):
    x = load_batch(bs[step])
    f = enc(x)
    H, W = f.shape[2], f.shape[3]
    f_hat = torch.zeros_like(f)
    r_list, s_list, cb_list, q_list = [], [], [], []
    for k, pn in enumerate(ms.patch_nums):
        f_k = F.interpolate(f, size=(pn, pn), mode='bilinear', align_corners=False)
        fh_k = F.interpolate(f_hat, size=(pn, pn), mode='bilinear', align_corners=False)
        r_k = f_k - fh_k
        sk = ms._rms_now(r_k)
        ms._update_rms(k, sk)
        idx = ms._match(r_k / sk)
        q_raw = ms.codebook(idx)
        q = q_raw.view(B, pn, pn, C).permute(0, 3, 1, 2) * sk
        r_list.append(rms(r_k)); s_list.append(float(sk))
        cb_list.append(q_raw.detach().norm(dim=1).mean().item())
        q_list.append(rms(q))
        f_hat = f_hat + F.interpolate(q, size=(H, W), mode='bilinear', align_corners=False)

    # 训练一步（复刻 smoke 的 loss）
    idxs, fh, vq = ms.f_to_idx(f)
    rec_img = dec(fh)
    l_rec = F.mse_loss(rec_img, x) + 0.5 * F.l1_loss(rec_img, x)
    loss = l_rec + vq
    opt.zero_grad(); loss.backward()
    for p in enc.parameters():
        if p.grad is not None:
            p.grad.nan_to_num_()
    torch.nn.utils.clip_grad_norm_([p for p in list(enc.parameters()) + list(dec.parameters())], 1.0)
    opt.step()

    if step < 8 or step % 3 == 0:
        print('  %-4d %7.3f | %s | %s | %s | %s | %8.3f %8.4f'
              % (step, rms(f),
                 ' '.join('%7.3f' % v for v in r_list),
                 ' '.join('%7.3f' % v for v in s_list),
                 ' '.join('%7.3f' % v for v in cb_list),
                 ' '.join('%7.3f' % v for v in q_list),
                 rms(f_hat), float(vq)))

print()
print('判读:')
print('  cb 行范数应从 1.0 收敛到 ~8.0（= sqrt(64)，即"归一化残差的聚簇均值"的模长）')
print('  qk（=cb行*sk）应 ≈ sk（每元素 RMS）。若 qk << sk 就是欠幅，若 qk >> sk 就是过幅')
print('  第一个开始指数增长的列 = 发散源头')
print('DONE')
