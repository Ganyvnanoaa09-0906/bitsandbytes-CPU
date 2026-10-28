# -*- coding: utf-8 -*-
"""verify_latent_roundtrip.py — 最底层的对照：真实潜变量能不能解码回正常图像？

为什么必须先做这一步（2026-09-19）：
  扩散采样出来是【彩虹噪点】，我先后怀疑过采样器发散、clip 配置，
  但一直没有验证【最底层的一环】：`precompute_latents.py` 存下来的潜变量
  经 `dec()` 解码后，到底还是不是原来的图？
  如果这一环就是坏的，那么上面所有推理都是在错误的地基上做的。

做法：取验证集的真实 z_q -> dec() -> 存图；并与【VQVAE 直接 forward 的重建】对比。
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
import os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_vqvae import vqvae_from_ckpt

SIZE = 256
VQ = r'D:\work\cloud_results\ckpt_C_mse.pt'
LAT = r'.autopilot\latents_cmse.pt'
PATHS = r'.autopilot\tokens_16x16_full.pt.paths.json'


def grid(imgs):
    a = ((imgs.permute(0, 2, 3, 1).numpy() + 1) / 2 * 255).round().astype(np.uint8)
    return np.concatenate([np.concatenate([a[r * 4 + c] for c in range(4)], 1) for r in range(4)], 0)


def main():
    import json
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])

    vk = torch.load(VQ, map_location='cpu', weights_only=False)
    vq, _ = vqvae_from_ckpt(vk)
    vq.eval()

    L = torch.load(LAT, map_location='cpu', weights_only=False).float()
    idx = list(range(27658, 27658 + 16))
    zq = L[idx]
    print('[rt] 保存的潜变量: shape=%s  mean=%.4f std=%.4f  min=%.2f max=%.2f'
          % (tuple(zq.shape), float(zq.mean()), float(zq.std()), float(zq.min()), float(zq.max())))

    paths = json.load(open(PATHS, encoding='utf-8'))
    xs = []
    for i in idx:
        with Image.open(paths[i]) as im:
            xs.append(tf(im.convert('RGB')))
    x = torch.stack(xs)

    with torch.no_grad():
        # 1) 原图（参考）
        # 2) 直接 forward 的重建（VQVAE 自己的口径）
        xr, idx2, _ = vq(x)
        # 3) 只拿【保存下来的 z_q】喂解码器（扩散采样后要走的就是这条路）
        zq2 = vq.vq(vq.enc(x))[0]
        rec_saved = vq.dec(zq)
        rec_recomputed = vq.dec(zq2)

    print('[rt] 保存的 z_q 与重算的 z_q 是否一致: max|diff| = %.6f'
          % float((zq - zq2).abs().max()))
    for name, t in (('原图', x), ('VQVAE forward 重建', xr),
                    ('用保存的 z_q 解码', rec_saved), ('用重算的 z_q 解码', rec_recomputed)):
        import torch.nn.functional as F
        print('  %-22s PSNR=%.2f dB  range=[%.2f, %.2f]'
              % (name, 10 * np.log10(4.0 / max(float(F.mse_loss(t, x)), 1e-12)),
                 float(t.min()), float(t.max())))

    Image.fromarray(grid(x.clamp(-1, 1))).save(r'D:\work\cloud_results\RT_original.png')
    Image.fromarray(grid(rec_saved.clamp(-1, 1))).save(r'D:\work\cloud_results\RT_from_saved_zq.png')
    print('[rt] -> RT_original.png / RT_from_saved_zq.png')


if __name__ == '__main__':
    main()
