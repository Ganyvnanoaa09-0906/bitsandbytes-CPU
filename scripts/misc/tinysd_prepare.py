"""Prepare a LoRA training set for Tiny-SD: cache latents and text embeddings.

Training a diffusion model on a CPU involves three components, and only one of them is what a training
loop should be measuring. The VAE encodes images to latents and the CLIP text encoder embeds captions,
both once per image and both frozen. Running them inside the loop would put their cost into every
measurement and make the training step's own profile unreadable. So they run once here and the results
are written to disk.

That is also what makes the loop honest: with the cache, one training step is exactly a UNet forward
pass, a backward pass and an optimizer update, and the probe in the next step measures nothing else.

The dataset is flat_real_300 -- 300 illustrations, 159 MB, which is the standard size for a LoRA and
deliberately not one of the larger themed sets here, because the point is a fast, repeatable loop rather
than a good model. Images are resized to 512 and centre-cropped, matching what the pipeline generates,
and the caption is a single constant since the set has no per-image text. A constant caption is a real
limitation and it is recorded rather than hidden: a LoRA trained on one caption learns a style, not a
concept conditioned on text.

Everything cached is written to disk and sized, so the next run can skip this entirely.
"""
from __future__ import annotations
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

import os
import sys
import time

import numpy as np
import torch
from PIL import Image

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


MODEL = r'D:\work\models\models\tiny-sd'
SRC = r'D:\work\sujvji\flat_real_300\images'
CACHE = r'D:\work\bitsandbytes-CPU\tinysd_cache'
N_IMAGES = 48
CAPTION = 'a flat vector illustration'

os.makedirs(CACHE, exist_ok=True)
torch.set_num_threads(6)

print()
print('  === 输入 ===')
files = sorted(f for f in os.listdir(SRC) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
print('    目录里 %d 张 · 本次用前 %d 张' % (len(files), min(N_IMAGES, len(files))))
files = files[:N_IMAGES]

print()
print('  === 加载 VAE 与 text_encoder（只跑一次）===')
from diffusers import AutoencoderKL, StableDiffusionPipeline  # noqa: E402
from transformers import CLIPTextModel, CLIPTokenizer  # noqa: E402

t0 = time.time()
vae = AutoencoderKL.from_pretrained(os.path.join(MODEL, 'vae'))
tok = CLIPTokenizer.from_pretrained(os.path.join(MODEL, 'tokenizer'))
te = CLIPTextModel.from_pretrained(os.path.join(MODEL, 'text_encoder'))
vae.eval(); te.eval()
print('    %.1f s' % (time.time() - t0))
n_vae = sum(p.numel() for p in vae.parameters())
n_te = sum(p.numel() for p in te.parameters())
print('    VAE %.1fM · text_encoder %.1fM' % (n_vae / 1e6, n_te / 1e6))

print()
print('  === 文本嵌入（恒定 caption）===')
with torch.no_grad():
    ids = tok(CAPTION, padding='max_length', max_length=77, truncation=True,
              return_tensors='pt').input_ids
    emb = te(ids)[0]
print('    %r ⇒ %s' % (CAPTION, tuple(emb.shape)))
torch.save(emb, os.path.join(CACHE, 'text_emb.pt'))

print()
print('  === VAE 编码（这是这一步的真正开销）===')
lat_list = []
t0 = time.time()
with torch.no_grad():
    for i, fn in enumerate(files):
        im = Image.open(os.path.join(SRC, fn)).convert('RGB')
        # 512 短边缩放 + 中心裁 ⇒ 与管线出图的尺寸口径一致
        w, h = im.size
        s = 512 / min(w, h)
        im = im.resize((max(512, int(round(w * s))), max(512, int(round(h * s)))), Image.LANCZOS)
        w, h = im.size
        left, top = (w - 512) // 2, (h - 512) // 2
        im = im.crop((left, top, left + 512, top + 512))
        arr = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        lat = vae.encode(arr.unsqueeze(0)).latent_dist.sample() * 0.18215
        lat_list.append(lat.squeeze(0).half())
        if (i + 1) % 12 == 0:
            el = time.time() - t0
            print('    %2d/%d · %.1f s · 每张 %.2f s' % (i + 1, len(files), el, el / (i + 1)))
el = time.time() - t0
lat = torch.stack(lat_list)
print('    ⇒ %s · 共 %.1f s · 每张 %.2f s' % (tuple(lat.shape), el, el / len(files)))
p = os.path.join(CACHE, 'latents.pt')
torch.save(lat, p)
print('    已存 %s（%.1f MB）' % (p, os.path.getsize(p) / 1e6))

print()
print('  === 缓存构成 ===')
print('    latents %s fp16 = %.1f MB' % (tuple(lat.shape), lat.numel() * 2 / 1e6))
print('    text_emb %s fp32 = %.1f MB' % (tuple(emb.shape), emb.numel() * 4 / 1e6))
print('    ⇒ ★ 训练循环里【只剩 UNet】—— 探针测到的才是训练本身的开销')
print()
print('  注:caption 是【常数】(这个集合没有逐图文本)')
print('     ⇒ ★ 这样训出来的 LoRA 学的是【风格】,不是【受文本控制的概念】')
print('     ⇒ 这条限制写下来,而不是藏起来')
print()
