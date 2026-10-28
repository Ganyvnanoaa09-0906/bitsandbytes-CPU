# -*- coding: utf-8 -*-
"""precompute_latents.py — 把图片编码成 VQ-VAE 的【量化潜变量】z_q，供扩散模型训练。

为什么要用 z_q 而不是连续 z：
  decoder 是【喂 z_q 训出来的】。要跟 AR 做公平对比，扩散模型必须建模【同一个目标分布】：
    AR         : 离散 idx  ->  z_q = codebook[idx]  ->  dec
    扩散(本)    : 连续 z_q  --------------------------------->  dec
  用连续 z 会和 decoder 的训练分布错位，对比就不干净。

用法:
  python precompute_latents.py --ckpt <VQVAE.pt> --paths <paths.json> --out latents.pt
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
import argparse, json, os, sys, time
import numpy
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_vqvae import vqvae_from_ckpt

SIZE = 256
TF = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
                         transforms.Normalize([0.5] * 3, [0.5] * 3)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--paths', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--dtype', default='float16', choices=['float16', 'float32'])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    paths = json.load(open(a.paths, encoding='utf-8'))
    if a.limit:
        paths = paths[:a.limit]
    print('[lat] %d 张图' % len(paths), flush=True)

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    model, _ = vqvae_from_ckpt(ck)
    model.eval()
    print('[lat] VQVAE step=%s ch=%s n_codes=%s' % (
        ck.get('step'), ck.get('ch'), ck.get('n_codes')), flush=True)

    dt = torch.float16 if a.dtype == 'float16' else torch.float32
    outs = []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(paths), a.batch):
            ps = paths[s:s + a.batch]
            xs = []
            for p in ps:
                with Image.open(p) as im:
                    xs.append(TF(im.convert('RGB')))
            x = torch.stack(xs)
            z = model.enc(x)
            q = model.vq(z)[0]                      # 量化潜变量 = decoder 在训练时看到的
            outs.append(q.to(dt))
            if (s // a.batch) % 50 == 0:
                done = min(s + a.batch, len(paths))
                el = time.time() - t0
                print('[lat] %d/%d  %.0fs  %.1f img/s' % (
                    done, len(paths), el, done / max(el, 1e-9)), flush=True)

    L = torch.cat(outs, 0)
    torch.save(L, a.out)
    print('[lat] saved %s -> %s  %s  %.1f MB' % (
        tuple(L.shape), a.out, L.dtype, os.path.getsize(a.out) / 1e6), flush=True)
    # 统计（用于确认潜变量尺度，扩散的噪声日程要对得上）
    f = L.float()
    print('[lat] mean=%.4f std=%.4f min=%.3f max=%.3f' % (
        float(f.mean()), float(f.std()), float(f.min()), float(f.max())), flush=True)


if __name__ == '__main__':
    main()
