# -*- coding: utf-8 -*-
"""verify_ar_loss2.py — 修正分布后重测 AR 的修补能力

上一版的 bug：抽样时混入了 clean_char2_1024（1024px，与训练分布不同）
本版：只用 clean_nf（就是 tokens_8x8.pt 的来源）

关键问题：AR 在【分布内】的逐位置 top-1 准确率是多少？
  · 若 ~77% ⇒ 【能做脏区域修补】✅（用户的方案可行）
  · 若 ~48% ⇒ 勉强
  · 若 ~8%  ⇒ 不行
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
import sys, os, glob, random
import torch
import torch.nn.functional
import numpy
from PIL import Image

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
Image.MAX_IMAGE_PIXELS = None
print('=' * 80)
print('AR 修补能力（修正分布后）')
print('=' * 80)

from scripts.lib.small_vqvae import vqvae_from_ckpt
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2


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


def report(m, tok, name):
    T = tok.shape[1]
    side = int(round(T ** 0.5))
    with torch.no_grad():
        outs = []
        for i in range(0, tok.shape[0], 32):
            b = tok[i:i + 32]
            _, g = m.forward_logits(b)
            g = g[0] if isinstance(g, (tuple, list)) else g
            outs.append((g, b))
    L = n = top1 = top5 = 0
    t1_all = []
    for g, b in outs:
        L += F.cross_entropy(g.reshape(-1, g.shape[-1]), b.reshape(-1),
                             reduction='sum').item()
        n += b.numel()
        c = (g.argmax(-1) == b)
        top1 += c.sum().item()
        t1_all.append(c)
        t5 = g.topk(5, -1).indices
        top5 += (t5 == b.unsqueeze(-1)).any(-1).sum().item()
    print('  %-26s loss %.4f (ppl %6.2f)   top-1 %5.1f%%   top-5 %5.1f%%'
          % (name, L / n, np.exp(L / n), 100 * top1 / n, 100 * top5 / n))
    c = torch.cat(t1_all)
    q = T // 4
    seg = ['[%d,%d) %.1f%%' % (s, min(s + q, T), 100 * c[:, s:s + q].float().mean().item())
           for s in range(0, T, q)]
    print('    逐位置: %s' % '  '.join(seg))
    t1_2d = c.view(-1, side, side).float().mean(0)
    print('    行均值: %s' % ' '.join('%.2f' % v for v in t1_2d.mean(1).tolist()[:16]))
    return top1 / n


# ---------- 分布内：clean_nf（tokens_8x8 的来源）----------
print()
print('【分布内】clean_nf（512px，和 tokens_8x8.pt 同源）')
files = glob.glob(r'D:\work\bitsandbytes-CPU\.autopilot\clean_nf\**\*.jpg', recursive=True)
print('  可用 %d 张' % len(files))
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

VQ = {'8': (r'D:\work\cloud_results\vqvae_8x8.pt', 8),
      '16': (r'D:\work\cloud_results\ckpt_C_mse.pt', 16)}

print()
print('  --- 8x8（ar_swiglu）---')
vq8, _ = vqvae_from_ckpt(VQ['8'][0]); vq8.eval()
with torch.no_grad():
    _, idx, _ = vq8(real)
tok8 = idx.view(idx.shape[0], -1).long()
m8, _ = load_ar(r'D:\work\cloud_results\swiglu_last.pt')
a8 = report(m8, tok8, 'ar_swiglu (T=64)')

print()
print('  --- 16x16（ar_flat / hier）---')
vq16, _ = vqvae_from_ckpt(VQ['16'][0]); vq16.eval()
with torch.no_grad():
    _, idx, _ = vq16(real)
tok16 = idx.view(idx.shape[0], -1).long()
for nm, ck in (('ar_flat (非hier)', r'D:\work\cloud_results\flat_12000.pt'),
               ('hier (hier)', r'D:\work\cloud_results\hier_ckpt_50000.pt')):
    if os.path.exists(ck):
        mm, _ = load_ar(ck)
        report(mm, tok16, nm)

# ---------- 训练数据本身 ----------
print()
print('【对照】tokens_8x8.pt（训练数据本身，最容易）')
tok = torch.load(r'D:\work\bitsandbytes-CPU\.autopilot\tokens_8x8.pt',
                 map_location='cpu').long()
report(m8, tok[:512], 'tokens_8x8 训练集')

print()
print('=' * 80)
print('判读（用户想用 AR 做脏区域更新/修补）：')
print('  top-1 > 70% ⇒ ✅ 可行')
print('  top-1 40~70% ⇒ ⚠️ 勉强，修补会有明显错误')
print('  top-1 < 20% ⇒ ❌ 不可行')
print('DONE')
