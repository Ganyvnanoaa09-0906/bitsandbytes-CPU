# -*- coding: utf-8 -*-
"""assemble_sd15.py — 用本地已有零件拼出【真正的 SD1.5】，并测它在真人/动漫两种提示下的表现。

为什么能白得一个模型：
  本机扫到两个残缺资产 ——
    · sd15_base/unet         : 原版 SD1.5 UNet（完整）
    · counterfeit-v3-diffusers: SD1.5 的动漫微调（unet+text_encoder+vae+tokenizer+scheduler 全套）
  而 counterfeit-v3 是 **SD1.5 的微调** ⇒ 它的 text_encoder / vae / tokenizer **就是 SD1.5 原版**。
  ⇒ `sd15_base/unet` + counterfeit 的 VAE/CLIP/tokenizer = **原版 SD1.5 管线**（无需下载）。

用途：本项目训练数据是【真人照 + 动漫插画】混合（caption 里两种风格标签都有）。
  · counterfeit-v3 → 高质量动漫人像（已验证出图很好）
  · 拼出来的 SD1.5 → 真人向（待本次验证）
  两个合起来才覆盖数据分布的两半。
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
import argparse, os, sys, time, json
import torch


def build_sd15(textmodel=r'D:\work\textmodel'):
    from diffusers import (StableDiffusionPipeline, UNet2DConditionModel,
                           AutoencoderKL, PNDMScheduler)
    from transformers import CLIPTextModel, CLIPTokenizer
    base = os.path.join(textmodel, 'sd15_base')
    cf = os.path.join(textmodel, 'counterfeit-v3-diffusers')
    unet = UNet2DConditionModel.from_pretrained(os.path.join(base, 'unet'),
                                                torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(os.path.join(cf, 'vae'), torch_dtype=torch.float32)
    te = CLIPTextModel.from_pretrained(os.path.join(cf, 'text_encoder'))
    tok = CLIPTokenizer.from_pretrained(os.path.join(cf, 'tokenizer'))
    sch = PNDMScheduler.from_pretrained(os.path.join(cf, 'scheduler'))
    pipe = StableDiffusionPipeline(vae=vae, text_encoder=te, tokenizer=tok, unet=unet,
                                   scheduler=sch, safety_checker=None,
                                   feature_extractor=None, requires_safety_checker=False)
    return pipe


PROMPTS = [
    ('real', "a photograph of a real person, a woman standing in a room, "
             "wearing a dress, looking at the viewer, detailed face, natural lighting, "
             "sharp focus, high quality photo"),
    ('anime', "1girl, solo, upper body, looking at viewer, detailed face, "
              "silver hair, school uniform, soft lighting, best quality"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--outdir', default=r'D:\work\cloud_results')
    ap.add_argument('--prompt-set', default='split',
                    help='split=真人/动漫各一张；collage=把真实 caption 的 tag 拼成提示词')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.outdir, exist_ok=True)

    print('[sd15] 拼装原版 SD1.5 ...', flush=True)
    t0 = time.perf_counter()
    pipe = build_sd15()
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    pipe.to('cpu')
    print('[sd15] 装好 %.1fs  threads=%d' % (time.perf_counter() - t0, a.threads), flush=True)

    from PIL import Image, ImageDraw
    from diffusers import DPMSolverMultistepScheduler
    imgs = []
    for tag, pr in PROMPTS:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        g = torch.Generator(device='cpu').manual_seed(a.seed)
        t1 = time.perf_counter()
        with torch.inference_mode():
            im = pipe(pr, num_inference_steps=a.steps, guidance_scale=7.5,
                      width=a.size, height=a.size, generator=g).images[0]
        dt = time.perf_counter() - t1
        p = os.path.join(a.outdir, 'SD15BASE_%s.png' % tag)
        im.save(p)
        imgs.append((tag, im))
        print('  [%s] %.1fs -> %s' % (tag, dt, p), flush=True)

    W = max(i.size[0] for _, i in imgs)
    H = max(i.size[1] for _, i in imgs)
    canvas = Image.new('RGB', (len(imgs) * W, H), (20, 20, 20))
    dr = ImageDraw.Draw(canvas)
    for i, (tag, im) in enumerate(imgs):
        canvas.paste(im, (i * W, 0))
        dr.text((i * W + 6, 6), tag, fill=(255, 255, 0))
    gp = os.path.join(a.outdir, 'SD15BASE_GRID.png')
    canvas.save(gp)
    print('[sd15] -> %s' % gp)


if __name__ == '__main__':
    main()
