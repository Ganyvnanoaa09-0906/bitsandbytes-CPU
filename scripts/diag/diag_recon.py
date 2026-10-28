# -*- coding: utf-8 -*-
"""diag_recon.py — 诊断一：tokenizer 重建上限。

如果 AR 生成的是"完美 token"，它能达到的最好结果就是本脚本的重建质量。
所以重建质量 = 生图质量的天花板，必须先把这条基线量出来。

输出：并排对比图（上排原图 / 下排重建）+ PSNR 统计。

用法:
  python diag_recon.py --vqvae <base>\\vqvae_ema_full2\\ckpt_last.pt --data <base>\\clean_nf\\images --n 12
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
import argparse, math, os, random, sys
import numpy
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_vqvae import VQVAE, vqvae_from_ckpt


def load(p, size):
    with Image.open(p) as im:
        im = im.convert('RGB').resize((size, size), Image.LANCZOS)
        a = np.asarray(im).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1)


def psnr(a, b):
    """a,b: torch [-1,1] CHW"""
    mse = float(((a - b) ** 2).mean())
    return 10.0 * math.log10(4.0 / max(mse, 1e-12))


def sharpness(t):
    """平均梯度幅值 —— GAN/感知损失的目标是更锐利的高频细节，
    PSNR 会因为像素值偏离而下降，但梯度能量会上升。

    GAN 模型的 PSNR 通常【低于】纯 MSE 模型，这是预期行为而不是变差，
    所以必须同时看这个指标，否则会误判。
    """
    gx = (t[:, 1:, :] - t[:, :-1, :]).abs().mean()
    gy = (t[:, :, 1:] - t[:, :, :-1]).abs().mean()
    return float((gx + gy) / 2)


def color_std(t):
    """每通道标准差均值 —— 糊掉的图会向均值收缩，std 变小。"""
    return float(t.reshape(3, -1).std(dim=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vqvae', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='diag_recon.png')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--threads', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    # 统一走 vqvae_from_ckpt：它会从 checkpoint 读全部架构字段
    # （ch / embed_dim / use_attn / two_3x3），避免硬编码导致 state_dict 对不上
    vq, vk = vqvae_from_ckpt(a.vqvae)
    ch = tuple(vk['ch']) if vk.get('ch') else (64, 128, 256, 256)
    down = 2 ** len(ch)
    g = a.size // down
    print('VQVAE 载入完成  ch=%s  embed_dim=%d  下采样 %dx  网格 %dx%d = %d token/图'
          % (ch, vq.enc.proj.out_channels, down, g, g, g * g))
    print('              瓶颈注意力=%s  ResBlock=%s  n_codes=%d'
          % ('开' if getattr(vq, 'use_attn', False) else '关',
             '两个3x3' if getattr(vq, 'two_3x3', False) else '3x3+1x1',
             vq.vq.codebook.num_embeddings))

    files = []
    for dp, _, fns in os.walk(a.data):
        for f in fns:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                files.append(os.path.join(dp, f))
    files.sort()
    random.seed(a.seed)
    samp = random.sample(files, min(a.n, len(files)))
    print('数据池 %d 张, 抽样 %d 张' % (len(files), len(samp)))

    C = 220
    strip = 16
    cols = len(samp)
    sheet = Image.new('RGB', (cols * C, 2 * (C + strip)), (24, 24, 24))
    d = ImageDraw.Draw(sheet)

    psnrs = []
    sharps = []
    stds = []
    sharp_o = []
    std_o = []
    with torch.no_grad():
        for i, p in enumerate(samp):
            x = load(p, a.size).unsqueeze(0)
            xr, idx, _ = vq(x)
            ps = psnr(x[0], xr[0])
            psnrs.append(ps)
            sharps.append(sharpness(xr[0]))
            stds.append(color_std(xr[0]))
            sharp_o.append(sharpness(x[0]))
            std_o.append(color_std(x[0]))

            def to_img(t):
                arr = ((t.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                return Image.fromarray(arr)
            orig = to_img(x[0]).resize((C, C), Image.NEAREST)
            rec = to_img(xr[0]).resize((C, C), Image.NEAREST)
            sheet.paste(orig, (i * C, 0))
            sheet.paste(rec, (i * C, C + strip))
            d.text((i * C + 4, C + 2), 'PSNR %.1f dB' % ps, fill=(200, 200, 200))
            # 码本使用
            uniq = int(idx.unique().numel())
            d.text((i * C + 4, 2 * C + strip), '', fill=(0, 0, 0))

    sheet.save(a.out, quality=92)
    psnrs.sort()
    mo = sum(sharp_o) / len(sharp_o)
    mr = sum(sharps) / len(sharps)
    so = sum(std_o) / len(std_o)
    sr = sum(stds) / len(stds)
    print()
    print('================ tokenizer 重建质量 ================')
    print('  抽样 %d 张' % len(psnrs))
    print('  PSNR          min %.2f   中位 %.2f   max %.2f  dB'
          % (psnrs[0], psnrs[len(psnrs) // 2], psnrs[-1]))
    print()
    print('  --- 感知指标（PSNR 对 GAN 模型会误导，必须一起看）---')
    print('  梯度幅值(锐度)  原图 %.5f  →  重建 %.5f   (保留 %.0f%%)'
          % (mo, mr, 100 * mr / mo if mo else 0))
    print('  色彩标准差      原图 %.5f  →  重建 %.5f   (保留 %.0f%%)'
          % (so, sr, 100 * sr / so if so else 0))
    print()
    print('  判读:')
    print('    PSNR 高 + 锐度低  =>  糊（纯 MSE 的典型结果）')
    print('    PSNR 低 + 锐度高  =>  有细节但像素偏离（GAN/感知损失的正常表现）')
    print('    PSNR 低 + 锐度也低 =>  真的没学好（不合格）')
    print()
    print('  参考: 好的离散 tokenizer 在 256×256 上 PSNR 通常 26~34 dB')
    print('        但 GAN 训练的 tokenizer PSNR 会明显低于此，看锐度保留率更准')
    print()
    print('  对比图 -> %s' % a.out)


if __name__ == '__main__':
    main()
