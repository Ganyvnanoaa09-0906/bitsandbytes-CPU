# -*- coding: utf-8 -*-
"""diagnose_bits.py — 用 bit 分解诊断「逐位置 top-1 衰减」的真正原因

不需要重训：直接拿现有 flat head 的 logits，
比较 argmax 出来的 index 与真值 index 的【逐 bit 一致率】。

要回答的问题：
  · 衰减是"结构全错"（高位 bit 也错）还是"细节丢失"（只有低位 bit 错）？
  · 坍塌（hier 42~54%）时，错的是哪些 bit？
  · MSB（高位，决定粗略位置）vs LSB（低位，决定细节）

Infinity 论文的观察：MSB 准确率显著高于 LSB。
  ⇒ 若我们也是"MSB 高、LSB 低"，说明模型学到了粗结构但丢了细节
  ⇒ 若 MSB 也低，说明连结构都没学到（更严重）
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
import sys, os, glob, random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
Image.MAX_IMAGE_PIXELS = None
print('=' * 82)
print('bit 分解诊断：逐位置 top-1 衰减的根因')
print('=' * 82)

from scripts.lib.small_vqvae import vqvae_from_ckpt
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

N_BITS = 10


def load_ar(ckpt):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfgd = ck['cfg']
    keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
    cfg = SmallImageConfigV2(**{k: v for k, v in cfgd.items() if k in keep})
    m = SmallARImageModelV2(cfg)
    sd = ck['model']
    if any(k.startswith('loop_mods.') for k in sd):
        from scripts.lib.loop_transformer import LoopConfig
        mx = 4
        for k, v in sd.items():
            if k.endswith('step_enc.table.weight'):
                mx = v.shape[0]
                break
        m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                              loop_L=max(1, mx - 1), loop_L_max=max(1, mx - 1),
                              loop_random=False))
    m.load_state_dict(sd)
    m.eval()
    return m, cfg


def bit_acc(pred_idx, true_idx):
    """返回 (n_bits,) 的逐 bit 准确率 + exact"""
    w = (1 << torch.arange(N_BITS, dtype=torch.long))
    tb = ((true_idx.long().unsqueeze(-1) & w) > 0).long()
    pb = ((pred_idx.long().unsqueeze(-1) & w) > 0).long()
    correct = (pb == tb)
    acc = correct.float().mean(dim=tuple(range(correct.dim() - 1)))
    exact = correct.all(-1).float().mean().item()
    return acc, exact


# 数据（分布内）
files = glob.glob(r'D:\work\bitsandbytes-CPU\.autopilot\clean_nf\**\*.jpg', recursive=True)
random.seed(3)
paths = random.sample(files, 256)
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

CASES = [
    ('ar_swiglu T=64 (8x8)', r'D:\work\cloud_results\swiglu_last.pt',
     r'D:\work\cloud_results\vqvae_8x8.pt'),
    ('ar_flat T=256 (非hier)', r'D:\work\cloud_results\flat_12000.pt',
     r'D:\work\cloud_results\ckpt_C_mse.pt'),
    ('hier T=256 (hier)', r'D:\work\cloud_results\hier_ckpt_50000.pt',
     r'D:\work\cloud_results\ckpt_C_mse.pt'),
]

for label, ck, vqpath in CASES:
    if not os.path.exists(ck):
        print('\n%s: 缺文件' % label)
        continue
    print()
    print('=' * 82)
    print('[%s]' % label)
    m, cfg = load_ar(ck)
    vq, _ = vqvae_from_ckpt(vqpath)
    vq.eval()
    T = cfg.total_tokens
    side = int(round(T ** 0.5))

    toks = []
    with torch.no_grad():
        for i in range(0, len(real), 32):
            _, idx, _ = vq(real[i:i + 32])
            toks.append(idx.view(idx.shape[0], -1))
    tok = torch.cat(toks).long()

    outs = []
    with torch.no_grad():
        for i in range(0, len(tok), 32):
            _, g = m.forward_logits(tok[i:i + 32])
            g = g[0] if isinstance(g, (tuple, list)) else g
            outs.append(g)
    logits = torch.cat(outs)
    pred = logits.argmax(-1)

    acc, exact = bit_acc(pred, tok)
    print('  整体: top-1 %.1f%%   exact(全10bit对) %.2f%%'
          % ((pred == tok).float().mean().item() * 100, exact * 100))
    print()
    print('  逐 bit 准确率（bit0 = LSB，bit9 = MSB）:')
    print('    %-6s %-8s %s' % ('bit', '准确率', '说明'))
    print('    ' + '-' * 40)
    for b in range(N_BITS):
        note = ''
        if b == 0:
            note = 'LSB（最细）'
        elif b == N_BITS - 1:
            note = 'MSB（最粗）'
        print('    %-6d %6.1f%%  %s' % (b, acc[b].item() * 100, note))
    lo = acc[:N_BITS // 2].mean().item() * 100
    hi = acc[N_BITS // 2:].mean().item() * 100
    print()
    print('    低半位 %5.1f%%   高半位 %5.1f%%   ⇒ %s'
          % (lo, hi, 'MSB 更准（缺细节）' if hi > lo + 2 else
             ('LSB 更准（结构错）' if lo > hi + 2 else '基本持平')))

    # 逐位置（按序列位置 4 等分）的 bit 准确率
    print()
    print('  逐位置的 bit 准确率（看衰减来自哪些 bit）:')
    q = T // 4
    print('    %-14s %8s %8s %8s %8s' % ('位置', 'top-1', '低半位', '高半位', 'exact'))
    print('    ' + '-' * 54)
    for s in range(0, T, q):
        e = min(s + q, T)
        t1 = (pred[:, s:e] == tok[:, s:e]).float().mean().item() * 100
        a2, ex = bit_acc(pred[:, s:e], tok[:, s:e])
        print('    %-14s %7.1f%% %7.1f%% %7.1f%% %7.2f%%'
              % ('[%d,%d)' % (s, e), t1,
                 a2[:N_BITS // 2].mean().item() * 100,
                 a2[N_BITS // 2:].mean().item() * 100, ex * 100))

    # 2D 空间分布
    a2d = (pred == tok).float().view(-1, side, side).mean(0)
    print()
    print('  2D 空间 top-1（行均值，看空间衰减）:')
    print('    %s' % ' '.join('%.2f' % v for v in a2d.mean(1).tolist()[:16]))

print()
print('=' * 82)
print('判读:')
print('  高半位(MSB) 准 / 低半位(LSB) 差 ⇒ 模型学到粗结构，丢细节')
print('  两者都差 ⇒ 连结构都没学到')
print('  若 LSB 差是主因 ⇒ 换 bitwise 分类头（Infinity IVC）应能直接改善')
print('DONE')
