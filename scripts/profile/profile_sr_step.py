# -*- coding: utf-8 -*-
"""profile_sr_step.py — 拆解 SR 训练步的成本

嫌疑：
  ① 图片加载 + resize（PIL，512 目标）
  ② VQ-VAE 编码+解码（冻结）
  ③ SR 网络前向+反向
  ④ VGG 感知损失（512×512 上跑 VGG16 前 16 层）
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import glob
import os
import random
import sys
import time

import numpy
import torch
import torch.nn
import torch.nn.functional
from PIL import Image

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_vqvae import vqvae_from_ckpt
from scripts.lib.sr_net import SRSmall

torch.set_num_threads(6)
Image.MAX_IMAGE_PIXELS = None
print('=' * 76)
print('SR 训练步成本拆解')
print('=' * 76)

files = glob.glob(r'D:\work\bitsandbytes-CPU\.autopilot\clean_char2_1024\**\*.jpg', recursive=True)
random.seed(0)
print('  图片 %d 张' % len(files))


def load(path, size):
    im = Image.open(path).convert('RGB')
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    im = im.resize((size, size), Image.BICUBIC)
    a = np.asarray(im, dtype='uint8').copy()
    return torch.from_numpy(a).float().permute(2, 0, 1) / 127.5 - 1.0


B = 4
TARGET = 512
print()


def timeit(name, fn, n=3, warm=1):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t = (time.perf_counter() - t0) / n
    print('  %-42s %8.1f ms' % (name, t * 1e3))
    return t


# ① 图片加载
paths = [random.choice(files) for _ in range(B)]
t_load = timeit('① 图片加载 %d 张 @%d' % (B, TARGET),
                lambda: [load(p, TARGET) for p in paths])

his = [load(p, TARGET) for p in paths]
hi_b = torch.stack(his)
lo_b = F.interpolate(hi_b, size=(256, 256), mode='bicubic', align_corners=False)

# ② VQ-VAE
vq, vk = vqvae_from_ckpt(r'D:\work\cloud_results\ckpt_C_mse.pt')
vq.eval()
for p in vq.parameters():
    p.requires_grad_(False)
with torch.no_grad():
    t_vq = timeit('② VQ-VAE 编码+解码 %d 张' % B, lambda: vq(lo_b))
    xr, _i, _v = vq(lo_b)

# ③ SR 前向
net = SRSmall(ch=32, n_res=4, scale=2)
net.train()
t_sr_f = timeit('③a SR 前向 %d 张（输出 %d）' % (B, 256 * 2), lambda: net(xr))

# ③b SR 反向
out = net(xr)
loss = F.l1_loss(out, hi_b)
opt = torch.optim.AdamW(net.parameters(), lr=2e-4)


def sr_bwd():
    o = net(xr)
    l = F.l1_loss(o, hi_b)
    opt.zero_grad()
    l.backward()
    opt.step()


t_sr_b = timeit('③b SR 前向+反向+优化 %d 张' % B, sr_bwd)

# ④ VGG 感知
try:
    from torchvision.models import vgg16
    vgg = vgg16(weights='DEFAULT').features[:16].eval()
    for p in vgg.parameters():
        p.requires_grad_(False)
    t_vgg = timeit('④a VGG 前向 @%d（输出侧）' % (256 * 2), lambda: vgg(out))
    # 下采样到 128 再算
    small = F.interpolate(out, size=(128, 128), mode='bilinear', align_corners=False)
    t_vgg2 = timeit('④b VGG 前向 @128（下采样后）', lambda: vgg(small))
except Exception as e:
    t_vgg = t_vgg2 = 0
    print('  VGG 不可用: %s' % str(e)[:50])

# 实测整步
print()
print('  实测整步（含全部）: %.2f s' % (3.57,))
print()
print('  各部分占比估算（一个完整训练步）:')
parts = [
    ('图片加载', t_load),
    ('VQ-VAE（no_grad）', t_vq),
    ('SR 前向+反向+优化', t_sr_b),
    ('VGG 感知 @512', 2 * t_vgg),
]
tot = sum(v for _, v in parts)
for n, v in parts:
    print('    %-24s %8.1f ms  %5.1f%%' % (n, v * 1e3, 100 * v / tot))
print('    %-24s %8.1f ms' % ('合计', tot * 1e3))
print()
print('  优化方向:')
if t_load > tot * 0.3:
    print('    ⚠️ 图片加载占比高 -> 预加载/缓存或用更快的解码')
if 2 * t_vgg > tot * 0.3:
    print('    ⚠️ VGG 感知占比高 -> 下采样后再算（@128 只要 %.0f ms）' % (t_vgg2 * 1e3))
print('    ✅ SR 反向是主要成本（%d 参数，%.0f ms）-> 这是必须的' % (
    sum(p.numel() for p in net.parameters()), t_sr_b * 1e3))
print('DONE')
