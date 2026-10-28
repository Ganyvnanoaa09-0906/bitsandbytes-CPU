# -*- coding: utf-8 -*-
"""test_ms_vq_train.py — 多尺度残差量化能否【逐渐改善重建】（关键验证）

自检里的随机码本导致 RMSE 恒为 ~1.0（码本值为 ±0.001 ⇒ 量化输出 ≈ 0）。
本脚本用【真实数据】跑几十步训练，验证：
  ① rec loss 是否下降
  ② 逐尺度的重建 PSNR 是否【递进】（粗尺度 → 细尺度 逐步变好）
  ③ 各尺度的码本使用是否正常（没有全用一个码）
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
import sys, os, glob, random, time
import torch
import torch.nn
import torch.nn.functional
import numpy
from PIL import Image

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
Image.MAX_IMAGE_PIXELS = None
print('=' * 78)
print('多尺度残差量化：训练可行性验证')
print('=' * 78)

from scripts.lib.small_vqvae import Encoder, Decoder
from scripts.lib.ms_vq import MultiScaleResidualVQ

SIZE, B = 256, 4
C_EMB = 64
PATCH = (1, 2, 4, 8, 16)


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
print('  源图 %d 张' % len(files))

# 用 16x16 分词器的 ch（4 级下采样）
enc = Encoder(ch=(64, 128, 256, 256), embed_dim=C_EMB, use_attn=False, two_3x3=False)
dec = Decoder(ch=(256, 256, 128, 64), embed_dim=C_EMB, use_attn=False, two_3x3=False)

cb = nn.Embedding(1024, C_EMB)
# ⚠️ 关键：初始化值域要匹配 encoder 输出的尺度（否则见自检 [3] 的现象）
nn.init.normal_(cb.weight, std=0.5)
cb.weight.requires_grad_(False)
msvq = MultiScaleResidualVQ(cb, patch_nums=PATCH)

opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()) + [cb.weight], lr=3e-4)
print('  编码器 %.2fM  解码器 %.2fM  码本 1024x%d'
      % (sum(p.numel() for p in enc.parameters()) / 1e6,
         sum(p.numel() for p in dec.parameters()) / 1e6, C_EMB))
print('  总 token %d  尺度 %s' % (msvq.total, PATCH))

print()
print('  步   rec      vq     PSNR   逐尺度重建 PSNR')
print('  ' + '-' * 68)
t0 = time.time()
for step in range(300):
    ps = [random.choice(files) for _ in range(B)]
    x = load_batch(ps)
    f = enc(x)
    idxs, f_hat, vq_loss = msvq.f_to_idx(f)
    rec = dec(f_hat)

    # 损失：重建 + 承诺（承诺损失已含 beta 与逐尺度归一化，由 ms_vq 内部算好）
    l_rec = F.mse_loss(rec, x) + 0.5 * F.l1_loss(rec, x)
    l_vq = vq_loss
    loss = l_rec + l_vq

    opt.zero_grad(); loss.backward()
    for p in enc.parameters():
        if p.grad is not None:
            p.grad.nan_to_num_()
    opt.step()

    if step % 20 == 19 or step == 0:
        # 逐尺度重建 PSNR
        with torch.no_grad():
            per = []
            for k in range(1, len(PATCH) + 1):
                fh = msvq.idx_to_fhat(idxs, 16, 16, last_k=k)
                per.append(psnr(dec(fh), x))
        print('  %3d  %.4f  %.4f  %6.2f   %s'
              % (step + 1, l_rec.item(), l_vq.item(), psnr(rec, x),
                 ' '.join('%5.2f' % v for v in per)))

# 码本使用
msvq.eval()
with torch.no_grad():
    ps = [random.choice(files) for _ in range(B)]
    x = load_batch(ps)
    f = enc(x)
    idxs, _, _ = msvq.f_to_idx(f)
    print()
    print('  码本使用（每尺度，按【可用样本数】归一化 —— 不能除以 1024）:')
    for i, (n_uniq, cap, util, top, verdict) in enumerate(msvq.usage(idxs)):
        print('    尺度 %d (pn=%2d, %3d tok): %4d/%4d = %5.1f%%  最大占比 %5.1f%%  %s'
              % (i, PATCH[i], PATCH[i] * PATCH[i], n_uniq, cap, 100 * util,
                 100 * top, verdict))
    print('  逐尺度 RMS:', ' '.join('%.3f' % v for v in msvq.rms.tolist()))

print()
print('  耗时 %.1fs' % (time.time() - t0))
print()
print('=' * 78)
print('判读:')
print('  rec 应下降；逐尺度 PSNR 应【递进】（粗→细逐步变好）')
print('  ⇒ 若递进成立，则多尺度残差量化实现正确，可以上云端正式训')
print('DONE')
