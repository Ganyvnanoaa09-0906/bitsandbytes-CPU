# -*- coding: utf-8 -*-
"""test_ms_dead.py — 量化死码重启的抖动强度（决定 ms_vq 能不能上云端）

动机：300 步 smoke test 里 PSNR 在 16.8~18.8 之间来回摆、不收敛。
假设：多尺度共享码本 + batch4 时总样本只有 4×341=1364，平均每码 1.33 个，
继承自单尺度 VQ 的绝对阈值 `dead = ema_count < 0.5` 会把几百个码判死，
于是每步都随机重置它们 ⇒ 码本永远在抖。

本脚本对比三种配置，只看两件事：
  ① 每步死码重启数（占码本比例）
  ② rec / PSNR 的波动幅度（后 1/3 步的标准差）

判据：重启比例应 < 2%，PSNR 标准差应显著下降。
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
import sys, os, glob, random, time, statistics
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

SIZE, B, STEPS = 256, 4, 90
C_EMB, PATCH = 64, (1, 2, 4, 8, 16)


def psnr(a, b):
    mse = (a - b).pow(2).mean().item()
    return 10 * np.log10(4.0 / max(mse, 1e-12))


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
print('=' * 84)
print('死码重启抖动量化   (%d 步/batch %d)' % (STEPS, B))
print('=' * 84)

# 三种配置：① 原样（绝对阈值 0.5）② 相对阈值 0.5 ③ 相对阈值 + 限流 2%
CFGS = [
    ('A 绝对阈值 0.5（继承单尺度 VQ）', dict(dead_thresh=0.5, dead_thresh_rel=False, max_restart_frac=0.0)),
    ('B 相对阈值 0.5',                  dict(dead_thresh=0.5, dead_thresh_rel=True,  max_restart_frac=0.0)),
    ('C 相对阈值 0.5 + 限流 2%',        dict(dead_thresh=0.5, dead_thresh_rel=True,  max_restart_frac=0.02)),
]

# 固定数据顺序，保证三种配置看到完全相同的 batch（避免时间漂移干扰）
random.seed(1)
BATCHES = [[random.choice(files) for _ in range(B)] for _ in range(STEPS)]

results = {}
for name, kw in CFGS:
    torch.manual_seed(7)
    random.seed(1)
    enc = Encoder(ch=(64, 128, 256, 256), embed_dim=C_EMB, use_attn=False, two_3x3=False)
    dec = Decoder(ch=(256, 256, 128, 64), embed_dim=C_EMB, use_attn=False, two_3x3=False)
    cb = nn.Embedding(1024, C_EMB)
    nn.init.normal_(cb.weight, std=0.5)
    cb.weight.requires_grad_(False)
    msvq = MultiScaleResidualVQ(cb, patch_nums=PATCH, **kw)
    msvq.train()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=3e-4)

    dead_hist, rec_hist, psnr_hist = [], [], []
    t0 = time.time()
    for step in range(STEPS):
        x = load_batch(BATCHES[step])
        f = enc(x)
        idxs, f_hat, vq_loss = msvq.f_to_idx(f)
        rec_img = dec(f_hat)
        l_rec = F.mse_loss(rec_img, x) + 0.5 * F.l1_loss(rec_img, x)
        loss = l_rec + vq_loss
        opt.zero_grad(); loss.backward()
        for p in enc.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_()
        opt.step()
        dead_hist.append(msvq.last_dead)
        rec_hist.append(float(l_rec.detach()))
        psnr_hist.append(psnr(rec_img.detach(), x))

    tail = slice(STEPS // 3, STEPS)          # 后 2/3 步看稳定性
    d = dead_hist
    results[name] = dict(
        dead_mean=statistics.mean(d), dead_max=max(d), dead_first=d[0],
        pct=d[0] / 1024 * 100,
        rec_mean=statistics.mean(rec_hist[tail]), rec_sd=statistics.stdev(rec_hist[tail]),
        psnr_mean=statistics.mean(psnr_hist[tail]), psnr_sd=statistics.stdev(psnr_hist[tail]),
        secs=time.time() - t0,
    )
    print()
    print('  [%s]' % name)
    print('    死码/步: 首步 %d (%.1f%%)  均值 %.1f  最大 %d'
          % (d[0], d[0] / 1024 * 100, statistics.mean(d), max(d)))
    print('    死码轨迹(每10步):', ' '.join('%d' % v for v in d[::10]))
    print('    后段 rec %.4f ± %.4f    PSNR %.2f ± %.2f   耗时 %.0fs'
          % (results[name]['rec_mean'], results[name]['rec_sd'],
             results[name]['psnr_mean'], results[name]['psnr_sd'], results[name]['secs']))

print()
print('=' * 84)
print('%-34s %8s %8s %10s %10s' % ('配置', '死码均值', '死码最大', 'rec 波动', 'PSNR 波动'))
print('-' * 84)
for name in results:
    r = results[name]
    print('%-34s %8.1f %8d %10.4f %10.3f'
          % (name, r['dead_mean'], r['dead_max'], r['rec_sd'], r['psnr_sd']))
print()
base = results[CFGS[0][0]]
for name in list(results)[1:]:
    r = results[name]
    print('  %s: PSNR 波动 %.3f -> %.3f (%.0f%%)   rec 波动 %.4f -> %.4f'
          % (name, base['psnr_sd'], r['psnr_sd'],
             100 * (r['psnr_sd'] / max(base['psnr_sd'], 1e-9) - 1),
             base['rec_sd'], r['rec_sd']))
print('DONE')
