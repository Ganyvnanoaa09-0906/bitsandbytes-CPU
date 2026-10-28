# -*- coding: utf-8 -*-
"""sd_refine_sdxl.py — 用 SDXL 精修 AR 产出（写实向，对应用户"预期"那档质量）。

为什么换 SDXL：
  · SD1.5-base 拼装管线在 **512² txt2img** 下能出写实真人（assemble_sd15.py 实测），
    但在 **256² img2img** 下产出 hallucination —— 尺寸在训练分布之外是主因；
  · counterfeit-v3（动漫）在 256² 正常 ⇒ 动漫模型对低分辨率更鲁棒；
  · SDXL 是开放模型里写实能力最强的一档，且能在 1024² 工作。

成本预估（6 核 CPU）：
  SD1.5@512²/20 步 = 247 s ⇒ SDXL 参数约 3× ⇒ 512² 约 37 s/步。
  本脚本默认 512² + 8 有效步 ⇒ 约 300 s/格。**先用 --tiles 2 探路。**

--- 用法 ---
  python sd_refine_sdxl.py --grid <AR网格.png> --out <输出.png> --work 512 --tiles 2   # 探路
  python sd_refine_sdxl.py --grid <...> --out <...> --work 512                          # 全量
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
import argparse, os, time
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--root', default=r'D:\work\ms_cache\models\AI-ModelScope--stable-diffusion-xl-base-1.0\snapshots\v1.0.9')
    ap.add_argument('--strength', type=float, default=0.45)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--work', type=int, default=512, help='img2img 工作分辨率（SDXL 在 1024² 训，512 是折中）')
    ap.add_argument('--tiles', type=int, default=0, help='只跑前 N 格（0=全部 16 格）')
    ap.add_argument('--prompt', default=('a photograph of a real person, a woman wearing a '
                                         'detailed cosplay costume, long hair, detailed face, '
                                         'natural lighting, sharp focus, high quality photo'))
    ap.add_argument('--negative', default=('blurry, lowres, jpeg artifacts, bad anatomy, '
                                           'painting, illustration, cartoon, extra limbs'))
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from PIL import Image
    from diffusers import StableDiffusionXLImg2ImgPipeline
    from diffusers.utils import logging as dlog
    dlog.set_verbosity_error()

    t0 = time.perf_counter()
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(a.root, torch_dtype=torch.float32,
                                                            use_safetensors=True,
                                                            variant='fp16')
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    pipe.to('cpu')
    print('[sdxl] 装好 %.1fs  work=%d strength=%.2f steps=%d' % (
        time.perf_counter() - t0, a.work, a.strength, a.steps), flush=True)

    g = Image.open(a.grid).convert('RGB')
    W, H = g.size
    side = 4
    tw, th = W // side, H // side
    n = a.tiles if a.tiles else side * side
    out = Image.new('RGB', (tw * side, th * ((n + side - 1) // side)), (20, 20, 20))
    t0 = time.perf_counter()
    for i in range(n):
        r, c = divmod(i, side)
        tile = g.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
        src = tile.resize((a.work, a.work), Image.LANCZOS)
        with torch.inference_mode():
            im = pipe(prompt=a.prompt, negative_prompt=a.negative, image=src,
                      strength=a.strength, num_inference_steps=a.steps, guidance_scale=7.0,
                      generator=torch.Generator(device='cpu').manual_seed(1000 + i)).images[0]
        out.paste(im.resize((tw, th)), (c * tw, r * th))
        if (i + 1) % 2 == 0 or i + 1 == n:
            el = time.perf_counter() - t0
            print('  %d/%d  %.0f s/格  已用 %.1f 分钟' % (i + 1, n, el / (i + 1), el / 60), flush=True)
    out.save(a.out)
    print('[sdxl] -> %s  (%.0f s)' % (a.out, time.perf_counter() - t0))


if __name__ == '__main__':
    main()
