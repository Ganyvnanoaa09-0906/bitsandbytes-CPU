"""Reference sheet for the memorisation test, and the token statistics that go with it.

Uses the decode path from sample_ar.py exactly -- codebook lookup, permute, vq.dec -- rather than
guessing an interface. An earlier attempt called the VQVAE's forward, which is the encoder and
expects images, and that is the second time today a check was written against an assumed interface
instead of the real one.

Two things are produced:

  * a sheet of the 100 subset images' reconstructions, which is what a successful fit should
    approach. 7.48 showed the tokenizer represents these faithfully, so the gap between model
    output and this sheet is entirely the model's.
  * token-level statistics over the subset: how distinct the 100 images are from each other. If
    they are nearly identical the test proves nothing, since memorising them would be trivial.

sample_ar.py already saves generated tokens next to each image, so the eventual comparison can be
done token by token, which is exact, rather than by pixels, which go through a decoder and a uint8
quantisation and can hide small differences.
"""
from __future__ import annotations
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import sys

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


torch.set_num_threads(1)
from scripts.lib.small_vqvae import vqvae_from_ckpt

SUB = r'D:\work\bitsandbytes-CPU\tokens_overfit100.pt'
VQ = r'D:\work\bitsandbytes-CPU\i5build\vqgan32\ckpt_last.pt'
OUT = r'D:\work\bitsandbytes-CPU\overfit100_reference.png'

sub = torch.load(SUB, map_location='cpu')
print('  子集: %s' % (tuple(sub.shape),))

# how distinct are the 100 images? if any two rows are near-identical the test is weak
originals = sub[0::2]                       # 100 rows, the unflipped ones
n = originals.shape[0]
agree = torch.zeros(n, n)
for i in range(n):
    agree[i] = (originals == originals[i]).float().mean(dim=1)
off = agree.clone()
off.fill_diagonal_(float('nan'))
print('  两两 token 一致率: 均值 %.4f · 最大 %.4f （随机基线 %.4f）'
      % (float(off[~off.isnan()].mean()), float(off[~off.isnan()].max()), 1 / 1024))
print('  ⇒ %s' % ('100 张图彼此足够不同,记忆测试有意义'
                  if float(off[~off.isnan()].max()) < 0.9 else
                  '有近乎重复的图,测试说服力下降'))

# reference sheet through the exact path sample_ar.py uses
vq, vk = vqvae_from_ckpt(VQ)
vq.eval()
side = 32
picks = list(range(0, 16))
idx = originals[picks][:, :side * side].view(-1, side, side)
with torch.no_grad():
    emb = vq.vq.codebook(idx)
    z = emb.permute(0, 3, 1, 2).contiguous()
    img = vq.dec(z)
print('  解码输出: %s  range [%.2f, %.2f]' % (tuple(img.shape), float(img.min()), float(img.max())))

from PIL import Image  # noqa: E402
arr = img.clamp(-1, 1).add(1).div(2).mul(255).byte().permute(0, 2, 3, 1).numpy()
w = h = img.shape[-1]
sheet = Image.new('RGB', (4 * w, 4 * h))
for i in range(16):
    sheet.paste(Image.fromarray(arr[i]), ((i % 4) * w, (i // 4) * h))
sheet.save(OUT)
print('  参考图: %s  (%dx%d)  —— 100 张子集中前 16 张的 VQ 重建' % (OUT, sheet.size[0], sheet.size[1]))
print()
print('  判据（事先固定）: 若过拟合 100 张后的采样结果【明显差于这张参考图】,')
print('                    则问题在架构或采样流程,而不在数据量。')
