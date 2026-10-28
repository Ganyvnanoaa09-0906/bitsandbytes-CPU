# -*- coding: utf-8 -*-
"""sd_refine_test.py — 把 AR 出图当 layout prior，用 SD img2img 补细节。

动机（2026-09-19 §10.18）：
  实测天花板：256 token 的分词器只能到 ~23 dB，而"加 token 数"每 4x 才 +3.9 dB。
  信息预算上 2560 bit vs 需求 ~2e5 bit，差 ~80x。
  ⇒ AR 的**构图/配色是准的**（重建图可证），缺的**只有高频细节**。
  而"补高频细节"正好是生成式细节模型擅长的 —— 项目里已有动漫底模 counterfeit-v3。

判据：strength 低（0.3~0.5）时应**保住 AR 的构图**、只补纹理；
      strength 高会把 AR 的内容冲掉（那就等于换成 SD 生成，不是"补细节"）。

用法:
  python sd_refine_test.py --grid <采样网格.png> --out <对比图.png> --tiles 4,9
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import argparse, os, sys, time
import numpy as np
import torch
from PIL import Image

MODEL = r'D:\work\textmodel\counterfeit-v3-diffusers'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tiles', default='4,9')
    ap.add_argument('--strengths', default='0.3,0.5')
    ap.add_argument('--steps', type=int, default=15)
    ap.add_argument('--prompt', default='1girl, anime illustration, detailed, best quality')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--zoom', type=int, default=2, help='每格放大倍数（方便看细节）')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from diffusers import StableDiffusionImg2ImgPipeline
    t0 = time.time()
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    print('[sd] 载入 %.0fs' % (time.time() - t0), flush=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    im = Image.open(a.grid).convert('RGB')
    W, H = im.size
    side = 4
    tw, th = W // side, H // side
    tiles = [int(x) for x in a.tiles.split(',')]
    strengths = [float(x) for x in a.strengths.split(',')]
    g = torch.Generator().manual_seed(0)

    rows = []
    for ti in tiles:
        r, c = divmod(ti, side)
        src = im.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
        panels = [src]
        for st in strengths:
            t1 = time.time()
            out = pipe(prompt=a.prompt, image=src, strength=st,
                       num_inference_steps=a.steps, guidance_scale=5.0,
                       generator=g).images[0]
            print('[sd] tile %d strength %.2f  %.0fs' % (ti, st, time.time() - t1), flush=True)
            panels.append(out)
        zoom = a.zoom
        panels = [p.resize((tw * zoom, th * zoom), Image.NEAREST) for p in panels]
        rows.append(np.concatenate([np.asarray(p) for p in panels], axis=1))

    sep = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
    parts = []
    for i, rw in enumerate(rows):
        if i:
            parts.append(sep)
        parts.append(rw)
    out = np.concatenate(parts, axis=0)
    Image.fromarray(out).save(a.out)
    print('[sd] -> %s  %dx%d' % (a.out, out.shape[1], out.shape[0]))
    print('[sd] 列顺序: [AR 原图] ' + ' '.join('[strength %.2f]' % s for s in strengths))


if __name__ == '__main__':
    main()
