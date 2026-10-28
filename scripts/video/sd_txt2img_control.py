# -*- coding: utf-8 -*-
"""sd_txt2img_control.py — 关键对照：SD 从噪声直接生成，是否就不输"AR 打底 + SD 精修"？

为什么必须做这个对照（报告 §10.19 待验证 #3）：
  我们的 AR 是 cond_mode='none'（无文本/类别条件）⇒ 它生成的也是"随机一张动漫图"。
  那么"AR 打底 + SD 精修"相对"SD 直接生成"到底多得了什么？
  如果纯 SD 不输，那么 AR 在这条链里就是多余的，整条路等于"换成 SD"。
  ⚠️ 若真如此，要说清楚：AR 的价值只可能来自【条件控制】或【数据集身份/风格先验】，
     而后者需要单独证明。

用法:
  python sd_txt2img_control.py --out <对照图.png> --n 8 --steps 18
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
import argparse, time
import numpy as np
import torch
from PIL import Image

MODEL = r'D:\work\textmodel\counterfeit-v3-diffusers'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--steps', type=int, default=18)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--prompt', default='1girl, anime illustration, detailed, best quality')
    ap.add_argument('--negative', default='lowres, bad anatomy, blurry, jpeg artifacts')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    imgs = []
    t0 = time.time()
    for i in range(a.n):
        g = torch.Generator().manual_seed(1000 + i)
        im = pipe(prompt=a.prompt, negative_prompt=a.negative,
                  num_inference_steps=a.steps, guidance_scale=5.0,
                  height=a.size, width=a.size, generator=g).images[0]
        imgs.append(np.asarray(im))
        print('[sd] %d/%d  %.0fs' % (i + 1, a.n, time.time() - t0), flush=True)

    cols = 4
    rows = (a.n + cols - 1) // cols
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            k = r * cols + c
            row.append(imgs[k] if k < len(imgs)
                       else np.full((a.size, a.size, 3), 255, np.uint8))
        grid.append(np.concatenate(row, axis=1))
    out = np.concatenate(grid, axis=0)
    Image.fromarray(out).save(a.out)
    print('[sd] -> %s  %dx%d  总 %.0fs' % (a.out, out.shape[1], out.shape[0], time.time() - t0))
    print('[sd] 对照口径与 sd_refine_grid 相同：同 prompt、同 seed 序列(1000+i)、同 steps')


if __name__ == '__main__':
    main()
