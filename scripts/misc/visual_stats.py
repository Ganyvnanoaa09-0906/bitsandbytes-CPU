# -*- coding: utf-8 -*-
"""visual_stats.py — 二阶指标：为什么 token 统计正常但图是色块

一阶（相邻相同率）已经证明"变化频率"是正常的。
本脚本测【二阶】—— 空间结构：
  ① 解码图的视觉统计量：Laplacian（边缘量）、色彩 std、梯度分布
  ② 2D 功率谱的径向分布（低频/中频/高频能量占比）
  ③ token 变化位置的空间分布（水平/垂直梯度方向）

对比：真实图重建 / ar_flat 生成 / hier 生成 / ar_swiglu 生成
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
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import sys, os, glob, random, time, math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
Image.MAX_IMAGE_PIXELS = None
print('=' * 84)
print('二阶指标：空间结构诊断')
print('=' * 84)

from scripts.lib.small_vqvae import vqvae_from_ckpt

N = 64
SIZE = 256


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


LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def visual_stats(img, name):
    """img: (N,3,H,W) in [-1,1]"""
    g = img.mean(1, keepdim=True)                       # 灰度
    lap = F.conv2d(g, LAP.to(g.dtype), padding=1)
    lap_var = lap.var().item()
    # 梯度
    gx = g[:, :, :, 1:] - g[:, :, :, :-1]
    gy = g[:, :, 1:, :] - g[:, :, :-1, :]
    gmag = (gx[:, :, :-1, :].pow(2) + gy[:, :, :, :-1].pow(2)).sqrt()
    # 功率谱（径向）
    f = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1)).abs().pow(2)
    H, W = g.shape[-2:]
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    r = ((yy - cy).float().pow(2) + (xx - cx).float().pow(2)).sqrt()
    rmax = r.max().item()
    low = f[:, :, r < rmax / 3].mean().item()
    mid = f[:, :, (r >= rmax / 3) & (r < 2 * rmax / 3)].mean().item()
    high = f[:, :, r >= 2 * rmax / 3].mean().item()
    tot = low + mid + high + 1e-12
    print('  %-26s Laplacian %8.5f   梯度均值 %7.5f  色彩std %6.4f'
          % (name, lap_var, gmag.mean().item(), img.std().item()))
    print('  %-26s 频谱: 低 %5.1f%%  中 %5.1f%%  高 %5.1f%%'
          % ('', 100 * low / tot, 100 * mid / tot, 100 * high / tot))
    return dict(lap=lap_var, grad=gmag.mean().item(), std=img.std().item(),
                low=low / tot, mid=mid / tot, high=high / tot)


# ---------- 真实图 ----------
files = []
for pat in (r'D:\work\bitsandbytes-CPU\.autopilot\clean_nf\**\*.jpg',
            r'D:\work\bitsandbytes-CPU\.autopilot\clean_char2_1024\**\*.jpg'):
    files += glob.glob(pat, recursive=True)
random.seed(11)
real = load_real(random.sample(files, N))

print()
print('[0] 真实图')
visual_stats(real, '真实 256x256')

# ---------- 分词器重建 ----------
print()
print('[1] 分词器重建（给定正确 token 的上限）')
vq16, _ = vqvae_from_ckpt(r'D:\work\cloud_results\ckpt_C_mse.pt')
vq16.eval()
with torch.no_grad():
    rec16, _, _ = vq16(real)
visual_stats(rec16, '16x16 重建')

vq8, _ = vqvae_from_ckpt(r'D:\work\cloud_results\vqvae_8x8.pt')
vq8.eval()
with torch.no_grad():
    rec8, _, _ = vq8(real)
visual_stats(rec8, '8x8 重建')


# ---------- AR 生成 ----------
def gen_and_decode(ckpt, out_name, L=0, seed=2024):
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
        use = L if L > 0 else max(1, mx - 1)
        m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                              loop_L=use, loop_L_max=max(1, mx - 1), loop_random=False))
    m.load_state_dict(sd)
    m.eval()
    side = int(round(cfg.total_tokens ** 0.5))
    toks = []
    with torch.no_grad():
        for i in range(0, N, 32):
            b = min(32, N - i)
            toks.append(m.generate(None, steps=cfg.total_tokens, top_k=1024,
                                   temperature=1.0, batch=b, seed=seed + i).long())
    toks = torch.cat(toks)
    vq = vq16 if side == 16 else vq8
    idx = toks[:, :side * side].view(-1, side, side)
    with torch.no_grad():
        emb = vq.vq.codebook(idx)
        z = emb.permute(0, 3, 1, 2).contiguous()
        img = torch.tanh(vq.dec(z))
    return img, toks, side


print()
print('[2] AR 生成')
for label, ck, L in (
    ('ar_flat  T=256', r'D:\work\cloud_results\flat_12000.pt', 0),
    ('hier      T=256', r'D:\work\cloud_results\hier_ckpt_50000.pt', 0),
    ('ar_swiglu T=64', r'D:\work\cloud_results\swiglu_last.pt', 0),
):
    if not os.path.exists(ck):
        print('  %-26s （缺文件）' % label)
        continue
    img, toks, side = gen_and_decode(ck, label, L)
    visual_stats(img, '%s (%dx%d)' % (label, side, side))

print()
print('=' * 84)
print('判读:')
print('  若生成图的 Laplacian / 高频能量 远低于重建图 ⇒ token 的空间结构错了（缺边缘）')
print('  若生成图的色彩 std 远低于重建图 ⇒ 色彩覆盖不足')
print('  若频谱形状不同 ⇒ 空间相关性的尺度不对')
print('DONE')
