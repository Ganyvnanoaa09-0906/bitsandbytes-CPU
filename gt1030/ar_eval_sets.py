# -*- coding: utf-8 -*-
"""ar_eval_sets.py — 把 token 集合解成图片，供 VLM 裁判（lmm_describe_judge.py）使用。

产出目录结构（裁判脚本约定 real_i.png / gen_i.png）：
  <out>/real_0.png ...        ← 真实 token（从 tokens 文件取前 --n 行）
  <out>/gen_<tag>_i.png ...   ← 生成 token（每个 --gen 文件一组，tag 取文件名）

用法：
  python ar_eval_sets.py --tokens F:\\weibo_cos\\tokens_32x32_weibo.pt --n 16 \
      --gen F:\\weibo_cos\\_mgit_scr_b4\\samples.tokens.pt --gen F:\\weibo_cos\\_mgit_scr_b0\\samples.tokens.pt \
      --out F:\\weibo_cos\\ab_eval
"""
from __future__ import annotations

import argparse
import os

import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True, help="真实 token 文件（N×1024）")
    ap.add_argument("--vqvae", default=r"D:\work\cloud_salvage\ScPeP7\vqvae_V32.pt")
    ap.add_argument("--n", type=int, default=16, help="真实图数量")
    ap.add_argument("--gen", nargs="*", default=[], help="生成 token 文件（可多个）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from scripts.lib.small_vqvae import vqvae_from_ckpt
    vq, _ = vqvae_from_ckpt(a.vqvae)

    def decode(toks):
        side = int(round(toks.shape[1] ** 0.5))
        idx = toks[:, : side * side].view(-1, side, side)
        with torch.no_grad():
            emb = vq.vq.codebook(idx)
            z = emb.permute(0, 3, 1, 2).contiguous()
            img = vq.dec(z)
        out = []
        for i in range(img.shape[0]):
            t = ((img[i].clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).numpy().astype("uint8")
            out.append(Image.fromarray(t))
        return out

    real = torch.load(a.tokens, map_location="cpu", weights_only=False)
    if isinstance(real, dict):
        real = real.get("tokens", list(real.values())[0])
    real = real.long().view(real.shape[0], -1)[: a.n]
    for i, im in enumerate(decode(real)):
        im.save(os.path.join(a.out, f"real_{i}.png"))
    print(f"  真实图 {real.shape[0]} 张 → {a.out}")

    for p in a.gen:
        t = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(t, dict):
            t = t.get("tokens", list(t.values())[0])
        t = t.long().view(t.shape[0], -1)
        tag = os.path.basename(os.path.dirname(p)) or "gen"
        for i, im in enumerate(decode(t)):
            im.save(os.path.join(a.out, f"gen_{tag}_{i}.png"))
        print(f"  生成图 {t.shape[0]} 张（tag={tag}）→ {a.out}/gen_{tag}_*.png")


if __name__ == "__main__":
    main()
