# -*- coding: utf-8 -*-
"""test_ar_infill2.py — AR 能不能做「条件续写」（修补能力）

核心问题：用户想用 AR 做脏区域更新（只重绘变化处）。
         这要求 AR 能【给定已生成的上下文，预测后续 token】。

测试：在真实 token 上做 teacher forcing，统计【逐位置】的 top-1/top-5 准确率。
  · 若后段位置的准确率高 ⇒ AR 会利用上下文 ⇒ 能做修补 ✅
  · 若所有位置准确率都很低 ⇒ 它只是"全局统计匹配器" ⇒ 不能修补 ❌

关键对照：训练 loss（T=64 是 1.00 / T=256 是 ~4.4）
  准确率 ≈ exp(-loss) 的量级
```
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
import sys, os, glob, random
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
Image.MAX_IMAGE_PIXELS = None
print('=' * 82)
print('AR 条件续写能力（决定能不能做脏区域修补）')
print('=' * 82)

from scripts.lib.small_vqvae import vqvae_from_ckpt

HERE = r'D:\work\cloud_results'
SIZE, N = 256, 128


def load_real(paths):
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


def load_ar(ckpt):
    from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
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


files = []
for pat in (r'D:\work\bitsandbytes-CPU\.autopilot\clean_nf\**\*.jpg',
            r'D:\work\bitsandbytes-CPU\.autopilot\clean_char2_1024\**\*.jpg'):
    files += glob.glob(pat, recursive=True)
random.seed(3)
real = load_real(random.sample(files, N))

CASES = [
    ('ar_swiglu  T=64  (8x8)', r'D:\work\cloud_results\swiglu_last.pt',
     r'D:\work\cloud_results\vqvae_8x8.pt'),
    ('ar_flat    T=256 (非hier)', r'D:\work\cloud_results\flat_12000.pt',
     r'D:\work\cloud_results\ckpt_C_mse.pt'),
    ('hier       T=256 (hier)', r'D:\work\cloud_results\hier_ckpt_50000.pt',
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
        for i in range(0, N, 16):
            _, idx, _ = vq(real[i:i + 16])
            toks.append(idx.view(idx.shape[0], -1))
    tok = torch.cat(toks).long()

    # teacher forcing: 模型内部会加 BOS，logits[:,i] 直接预测 tok[:,i]
    with torch.no_grad():
        outs = []
        for i in range(0, N, 16):
            _, g = m.forward_logits(tok[i:i + 16])
            g = g[0] if isinstance(g, (tuple, list)) else g
            outs.append(g)
        logits = torch.cat(outs)                     # (N, T, V)

    tgt = tok                                        # (N, T)
    top1 = (logits.argmax(-1) == tgt)                # (N, T) bool
    t5 = logits.topk(5, -1).indices
    top5 = (t5 == tgt.unsqueeze(-1)).any(-1)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           tgt.reshape(-1)).item()

    print('  整体: top-1 %.1f%%   top-5 %.1f%%   loss %.4f (ppl %.1f)'
          % (top1.float().mean().item() * 100,
             top5.float().mean().item() * 100, loss, np.exp(loss)))
    print()
    print('  逐位置准确率（按序列位置分 4 段）：')
    q = T // 4
    print('    %-16s %10s %10s' % ('位置区间', 'top-1', 'top-5'))
    print('    ' + '-' * 40)
    for s in range(0, T, q):
        e = min(s + q, T)
        print('    %-16s %9.1f%% %9.1f%%'
              % ('[%d, %d)' % (s, e),
                 top1[:, s:e].float().mean().item() * 100,
                 top5[:, s:e].float().mean().item() * 100))

    # 空间分布（如果 2D）
    t1_2d = top1.view(N, side, side).float().mean(0)
    print()
    print('  空间分布（哪些位置的 token 更好预测）：')
    print('    行均值: %s' % ' '.join('%.2f' % v for v in t1_2d.mean(1).tolist()))
    print('    列均值: %s' % ' '.join('%.2f' % v for v in t1_2d.mean(0).tolist()))

    print()
    print('  判读（能否做脏区域修补）：')
    tot = top1.float().mean().item()
    if tot > 0.5:
        print('    ✅ top-1 %.0f%% ⇒ AR 能较可靠地预测下一个 token ⇒ 【可以做修补】' % (tot * 100))
    elif tot > 0.2:
        print('    ⚠️ top-1 %.0f%% ⇒ 勉强可用，但修补质量有限' % (tot * 100))
    else:
        print('    ❌ top-1 %.0f%% ⇒ AR 无法可靠预测 ⇒ 【不能做修补】' % (tot * 100))

print()
print('=' * 82)
print('DONE')
