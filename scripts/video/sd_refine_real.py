# -*- coding: utf-8 -*-
"""sd_refine_real.py — 用【拼装的原版 SD1.5（写实）】精修 AR 产出。

为什么值得单独做：
  · 现有精修用的是 `counterfeit-v3`（**动漫**微调），出来的全是动漫插画；
  · 而本项目训练数据是 **cos 真人照 + 动漫插画** 混合
    （caption 审计：`a photograph of a real person` 49% / `an anime illustration` 90%）；
  · 我们已用本地零件拼出了**原版 SD1.5**（`sd15_base/unet` + counterfeit 的 VAE/CLIP），
    实测能出**写实真人**。
  ⇒ 用它精修，才是与数据分布匹配的那一半。

流水线：AR@k1024 → 拼装 SD1.5 img2img → 输出
对照：prompt 用写实向自然语言（SD1.5 原版不认 danbooru tag 那套）。
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
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import argparse, os, time
import torch


def build_sd15(tm=r'D:\work\textmodel'):
    from diffusers import (StableDiffusionImg2ImgPipeline, UNet2DConditionModel,
                           AutoencoderKL, PNDMScheduler)
    from transformers import CLIPTextModel, CLIPTokenizer
    base = os.path.join(tm, 'sd15_base')
    cf = os.path.join(tm, 'counterfeit-v3-diffusers')
    unet = UNet2DConditionModel.from_pretrained(os.path.join(base, 'unet'),
                                                torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(os.path.join(cf, 'vae'), torch_dtype=torch.float32)
    te = CLIPTextModel.from_pretrained(os.path.join(cf, 'text_encoder'))
    tok = CLIPTokenizer.from_pretrained(os.path.join(cf, 'tokenizer'))
    sch = PNDMScheduler.from_pretrained(os.path.join(cf, 'scheduler'))
    return StableDiffusionImg2ImgPipeline(
        vae=vae, text_encoder=te, tokenizer=tok, unet=unet, scheduler=sch,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--strength', type=float, default=0.4)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--prompt',
                    default='a photograph of a real person, a woman wearing a cosplay '
                            'costume, detailed face, natural lighting, sharp focus, '
                            'high quality photo')
    ap.add_argument('--negative', default='blurry, lowres, jpeg artifacts, bad anatomy, painting')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--side', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from PIL import Image
    from diffusers import DPMSolverMultistepScheduler
    pipe = build_sd15()
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    pipe.to('cpu')

    g = Image.open(a.grid).convert('RGB')
    W, H = g.size
    tw, th = W // a.side, H // a.side
    print('[real] grid %dx%d  tile %dx%d  strength=%.2f steps=%d'
          % (W, H, tw, th, a.strength, a.steps), flush=True)

    out = Image.new('RGB', (W, H), (20, 20, 20))
    t0 = time.perf_counter()
    for r in range(a.side):
        for c in range(a.side):
            tile = g.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
            with torch.inference_mode():
                im = pipe(prompt=a.prompt, negative_prompt=a.negative, image=tile,
                          strength=a.strength, num_inference_steps=a.steps,
                          guidance_scale=7.5,
                          generator=torch.Generator(device='cpu').manual_seed(1000 + r * a.side + c)
                          ).images[0]
            out.paste(im.resize((tw, th)), (c * tw, r * th))
            n = r * a.side + c + 1
            if n % 4 == 0:
                print('  %d/%d  %.1f s/张' % (n, a.side * a.side,
                                             (time.perf_counter() - t0) / n), flush=True)
    out.save(a.out)
    print('[real] -> %s  (%.0f s)' % (a.out, time.perf_counter() - t0))


if __name__ == '__main__':
    main()
