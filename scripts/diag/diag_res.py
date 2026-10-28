# -*- coding: utf-8 -*-
"""diag_res.py — 检验"256² 是 SD1.5 的分布外尺寸"这一假设。

证据链：
  · `assemble_sd15.py` 的 txt2img 用 **512²**（默认参数）⇒ 出写实真人 ✅
  · `sd_refine_real.py` 的 img2img 用 **256²**（AR 格子尺寸）⇒ hallucination ❌
  · 同一个拼装管线、同一个 VAE/CLIP，只差尺寸
  · 已知常识：SD1.5 在 512² 上训练，256² 会产出混乱解剖/复制肢体

对照：counterfeit-v3（动漫模型）在 256² 上正常 ⇒ 动漫模型对低分辨率更鲁棒。

实验：同 2 格输入、同 seed、同 scheduler，只改 img2img 的工作分辨率。
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
import os, torch, time
from PIL import Image
from diffusers import (StableDiffusionImg2ImgPipeline, UNet2DConditionModel,
                       AutoencoderKL, DPMSolverMultistepScheduler)
from diffusers.utils import logging as dlog
from transformers import CLIPTextModel, CLIPTokenizer

torch.set_num_threads(int(os.environ.get('THREADS', '4')))
dlog.set_verbosity_error()

TM = r'D:\work\textmodel'
CF = os.path.join(TM, 'counterfeit-v3-diffusers')
GRID = r'D:\work\cloud_results\EXPO20000_K1024_FREE.png'
OUT = r'D:\work\cloud_results\DIAG_res'
os.makedirs(OUT, exist_ok=True)
PROMPT = ('a photograph of a real person, a woman wearing a cosplay costume, '
          'detailed face, natural lighting, sharp focus, high quality photo')
NEG = 'blurry, lowres, jpeg artifacts, bad anatomy'


def build():
    unet = UNet2DConditionModel.from_pretrained(os.path.join(TM, 'sd15_base', 'unet'),
                                                torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(os.path.join(CF, 'vae'), torch_dtype=torch.float32)
    te = CLIPTextModel.from_pretrained(os.path.join(CF, 'text_encoder'))
    tok = CLIPTokenizer.from_pretrained(os.path.join(CF, 'tokenizer'))
    sch = DPMSolverMultistepScheduler.from_pretrained(os.path.join(CF, 'scheduler'))
    pipe = StableDiffusionImg2ImgPipeline(
        vae=vae, text_encoder=te, tokenizer=tok, unet=unet, scheduler=sch,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    pipe.to('cpu')
    return pipe


g = Image.open(GRID).convert('RGB')
W, H = g.size
tw, th = W // 4, H // 4
tiles = [g.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th)) for r in range(1) for c in range(2)]

pipe = build()
for tag, work in [('256', 256), ('512', 512)]:
    row = Image.new('RGB', (tw * len(tiles), th), (20, 20, 20))
    t0 = time.perf_counter()
    for i, t in enumerate(tiles):
        src = t.resize((work, work), Image.LANCZOS) if work != tw else t
        with torch.inference_mode():
            im = pipe(prompt=PROMPT, negative_prompt=NEG, image=src, strength=0.40,
                      num_inference_steps=20, guidance_scale=7.5,
                      generator=torch.Generator(device='cpu').manual_seed(1000 + i)).images[0]
        row.paste(im.resize((tw, th)), (i * tw, 0))
    p = os.path.join(OUT, 'res%s.png' % tag)
    row.save(p)
    print('  %s² -> %s  (%.0f s, %.0f s/格)' % (tag, p, time.perf_counter() - t0,
                                                (time.perf_counter() - t0) / len(tiles)), flush=True)
