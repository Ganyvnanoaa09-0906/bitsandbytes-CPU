# -*- coding: utf-8 -*-
"""spectrum_gap.py — 量化"梵高风"：AR 出图到底缺了哪个频段的信息。

动机（2026-09-19）：
  用户一句"依旧梵高风"其实点中了要害 —— 构图/色调对、细节没有。
  但"缺细节"必须落到可度量的东西上，否则下游细节模型不知道该补什么。
  做法：径向平均功率谱（radially averaged power spectrum），
  对比【真实验证图】/【分词器重建（天花板）】/【AR 生成图】三条曲线，
  按低/中/高频段给出能量差（dB）。这个差就是"细节模型要补的量"。

用法:
  python spectrum_gap.py --sample D:\\work\\cloud_results\\SAMPLE_flat256_29500.png
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
import argparse, json, os, sys
import numpy
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def rad_spectrum(gray):
    """gray: (N,H,W) float32 in [0,1] -> 径向平均功率谱（归一化到低频=0 dB）"""
    N, H, W = gray.shape
    win = np.outer(np.hanning(H), np.hanning(W))[None]
    F = np.fft.fftshift(np.fft.fft2((gray - gray.mean(-1, keepdims=True).mean(-2, keepdims=True)) * win,
                                    axes=(-2, -1)), axes=(-2, -1))
    P = (np.abs(F) ** 2).mean(0)
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    nb = min(cy, cx)
    out = np.zeros(nb)
    for i in range(nb):
        m = r == i
        if m.any():
            out[i] = P[m].mean()
    return out


def bands(s, H):
    """按归一化频率分带：低/中/高"""
    nb = len(s)
    lo = s[1:max(2, nb // 8)].mean()
    mid = s[max(2, nb // 8):nb // 3].mean()
    hi = s[nb // 3:nb - 1].mean()
    return lo, mid, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', default=r'D:\work\cloud_results\SAMPLE_flat256_29500.png')
    ap.add_argument('--paths', default=r'.autopilot\tokens_16x16_full.pt.paths.json')
    ap.add_argument('--vqvae', default=r'D:\work\cloud_results\ckpt_C_mse.pt')
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    # 1) 真实验证图
    paths = json.load(open(a.paths, encoding='utf-8'))[27658:27658 + a.n]
    real = []
    for p in paths:
        with Image.open(p) as im:
            real.append(np.asarray(im.convert('L').resize((256, 256)), np.float32) / 255.0)
    real = np.stack(real)

    # 2) 分词器重建（天花板）
    from scripts.lib.small_vqvae import vqvae_from_ckpt
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    vk = torch.load(a.vqvae, map_location='cpu', weights_only=False)
    vq, _ = vqvae_from_ckpt(vk)
    vq.eval()
    xs = []
    for p in paths:
        with Image.open(p) as im:
            xs.append(tf(im.convert('RGB')))
    x = torch.stack(xs)
    with torch.no_grad():
        rec, _, _ = vq(x)
    rec = ((rec.clamp(-1, 1) + 1) / 2).mean(1).numpy()

    # 3) AR 生成图（把 4x4 网格切开）
    g = Image.open(a.sample).convert('L')
    W, H = g.size
    tw, th = W // 4, H // 4
    gen = []
    for r in range(4):
        for c in range(4):
            t = g.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th)).resize((256, 256))
            gen.append(np.asarray(t, np.float32) / 255.0)
    gen = np.stack(gen)

    names = ['真实图', '分词器重建(天花板)', 'AR 生成']
    datas = [real, rec, gen]
    print('%-20s %10s %10s %10s   %s' % ('', '低频', '中频', '高频', '相对真实的高频差'))
    print('-' * 76)
    base = None
    for nm, d in zip(names, datas):
        s = rad_spectrum(d)
        lo, mid, hi = bands(s, 256)
        if base is None:
            base = (lo, mid, hi)
        print('%-20s %10.2f %10.2f %10.2f   %+7.2f dB'
              % (nm, 10 * np.log10(lo / base[0]), 10 * np.log10(mid / base[1]),
                 10 * np.log10(hi / base[2]), 10 * np.log10(hi / base[2])))
    print()
    print('注：三条曲线都以【真实图的低频】为 0 dB 基准（相对比较）。')
    print('⇒ "分词器重建"与"AR 生成"的高频差，就是细节模型需要补的量；')
    print('   若两者接近，说明【是分词器在限制高频】，AR 已经不是瓶颈。')
    print('   若 AR 生成明显比重建更缺高频，说明【AR 采样本身在丢细节】。')


if __name__ == '__main__':
    main()
