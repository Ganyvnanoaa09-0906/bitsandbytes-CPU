# -*- coding: utf-8 -*-
"""verify_ar_loss.py — 查清 ar_swiglu 的 loss 到底是 1.00 还是 4.49

用【真实训练 token】算 loss，和报告数字对比。
若也对不上 ⇒ 报告里的 1.0024 有误（或那个数是别的东西）
若对得上 ⇒ 我上一个测试（用随机图编码的 token）有问题
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
import sys, os
import torch
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
print('=' * 78)
print('ar_swiglu loss 核验')
print('=' * 78)

from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2


def load_ar(ckpt):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfgd = ck['cfg']
    keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
    cfg = SmallImageConfigV2(**{k: v for k, v in cfgd.items() if k in keep})
    m = SmallARImageModelV2(cfg)
    m.load_state_dict(ck['model'])
    m.eval()
    return m, cfg, ck.get('step')


# ---------- 用真实训练 token ----------
print()
print('[1] 用 tokens_8x8.pt（真实的训练数据）')
P = r'D:\work\bitsandbytes-CPU\.autopilot\tokens_8x8.pt'
if not os.path.exists(P):
    print('  缺 %s' % P)
else:
    tok = torch.load(P, map_location='cpu')
    if hasattr(tok, 'long'):
        tok = tok.long()
    print('  shape %s  dtype %s' % (tuple(tok.shape), tok.dtype))
    m, cfg, step = load_ar(r'D:\work\cloud_results\swiglu_last.pt')
    print('  ckpt step=%s  total_tokens=%s  n_layer=%s  d_model=%s'
          % (step, cfg.total_tokens, cfg.n_layer, cfg.d_model))
    T = cfg.total_tokens
    tot = tok.numel() // T
    print('  样本数 %d' % tot)

    with torch.no_grad():
        outs = []
        for i in range(0, min(tot, 512), 32):
            b = tok[i:i + 32, :T]
            _, g = m.forward_logits(b)
            g = g[0] if isinstance(g, (tuple, list)) else g
            outs.append((g, b))
    # 汇总
    L = 0.0
    n = 0
    top1 = 0
    top5 = 0
    for g, b in outs:
        l = F.cross_entropy(g.reshape(-1, g.shape[-1]), b.reshape(-1), reduction='sum').item()
        L += l
        n += b.numel()
        top1 += (g.argmax(-1) == b).sum().item()
        t5 = g.topk(5, -1).indices
        top5 += (t5 == b.unsqueeze(-1)).any(-1).sum().item()
    print()
    print('  loss %.4f (ppl %.2f)   top-1 %.1f%%   top-5 %.1f%%'
          % (L / n, torch.exp(torch.tensor(L / n)).item(), 100 * top1 / n, 100 * top5 / n))
    print('  ※ 报告里写的训练 loss = 1.0024 (ppl 2.7)')

# ---------- 用随机图编码的 token（上一个测试的做法）----------
print()
print('[2] 用 128 张随机图的 VQ 编码 token（上一个测试的做法）')
import glob, random
import numpy as np
from PIL import Image
from scripts.lib.small_vqvae import vqvae_from_ckpt
Image.MAX_IMAGE_PIXELS = None

files = []
for pat in (r'D:\work\bitsandbytes-CPU\.autopilot\clean_nf\**\*.jpg',
            r'D:\work\bitsandbytes-CPU\.autopilot\clean_char2_1024\**\*.jpg'):
    files += glob.glob(pat, recursive=True)
random.seed(3)
paths = random.sample(files, 128)
imgs = []
for p in paths:
    im = Image.open(p).convert('RGB')
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    im = im.resize((256, 256), Image.BICUBIC)
    a = np.asarray(im, dtype='uint8').copy()
    imgs.append(torch.from_numpy(a).float().permute(2, 0, 1) / 127.5 - 1.0)
real = torch.stack(imgs)
vq, _ = vqvae_from_ckpt(r'D:\work\cloud_results\vqvae_8x8.pt')
vq.eval()
with torch.no_grad():
    _, idx, _ = vq(real)
tok2 = idx.view(idx.shape[0], -1).long()

m, cfg, _ = load_ar(r'D:\work\cloud_results\swiglu_last.pt')
with torch.no_grad():
    outs = []
    for i in range(0, 128, 32):
        b = tok2[i:i + 32]
        _, g = m.forward_logits(b)
        g = g[0] if isinstance(g, (tuple, list)) else g
        outs.append((g, b))
L = n = top1 = top5 = 0
for g, b in outs:
    L += F.cross_entropy(g.reshape(-1, g.shape[-1]), b.reshape(-1), reduction='sum').item()
    n += b.numel()
    top1 += (g.argmax(-1) == b).sum().item()
    t5 = g.topk(5, -1).indices
    top5 += (t5 == b.unsqueeze(-1)).any(-1).sum().item()
print('  loss %.4f (ppl %.2f)   top-1 %.1f%%   top-5 %.1f%%'
      % (L / n, torch.exp(torch.tensor(L / n)).item(), 100 * top1 / n, 100 * top5 / n))

print()
print('[3] 顺带检查：把 token 打乱后 loss 应该接近 ln(1024)=6.93')
perm = torch.randperm(tok2.shape[1])
shuf = tok2.clone()
for i in range(shuf.shape[0]):
    shuf[i] = shuf[i][perm]
with torch.no_grad():
    b = shuf[:32]
    _, g = m.forward_logits(b)
    g = g[0] if isinstance(g, (tuple, list)) else g
L = F.cross_entropy(g.reshape(-1, g.shape[-1]), b.reshape(-1)).item()
print('  打乱 token 的 loss %.4f (随机基线 ln(1024)=6.931)' % L)

print()
print('=' * 78)
print('DONE')
