# -*- coding: utf-8 -*-
"""diag_refine.py — 隔离 sd_refine_real.py 出垃圾的原因。

对照事实：
  · counterfeit-v3 走 `from_pretrained(原生全套)` + img2img ⇒ 干净插画 ✅
  · 我拼装的 SD1.5 走手动拼装 + 我【替换了 scheduler】⇒ hallucination 垃圾 ❌
并且拼装本身没问题：`assemble_sd15.py` 的 txt2img 实测能出写实真人 ✅

⇒ 嫌疑落在三处：
   A) scheduler：`DPMSolverMultistepScheduler.from_config(PNDM.config)` 参数不匹配
   B) strength：0.40 + 20 步 ⇒ 只有 8 步有效，而 SD1.5-base 在低步数上更脆
   C) guidance/negative prompt

本脚本在【同一张输入 + 同一 seed】下只改一个变量。
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
import os, torch, time
from PIL import Image
from diffusers import (StableDiffusionImg2ImgPipeline, UNet2DConditionModel,
                       AutoencoderKL, PNDMScheduler, DPMSolverMultistepScheduler)
from diffusers.utils import logging as dlog
from transformers import CLIPTextModel, CLIPTokenizer

torch.set_num_threads(int(os.environ.get('THREADS', '4')))
dlog.set_verbosity_error()

TM = r'D:\work\textmodel'
CF = os.path.join(TM, 'counterfeit-v3-diffusers')
GRID = r'D:\work\cloud_results\EXPO20000_K1024_FREE.png'
OUT = r'D:\work\cloud_results\DIAG_refine'
os.makedirs(OUT, exist_ok=True)
PROMPT = ('a photograph of a real person, a woman wearing a cosplay costume, '
          'detailed face, natural lighting, sharp focus, high quality photo')
NEG = 'blurry, lowres, jpeg artifacts, bad anatomy, painting'


def build(sched_name):
    unet = UNet2DConditionModel.from_pretrained(os.path.join(TM, 'sd15_base', 'unet'),
                                                torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(os.path.join(CF, 'vae'), torch_dtype=torch.float32)
    te = CLIPTextModel.from_pretrained(os.path.join(CF, 'text_encoder'))
    tok = CLIPTokenizer.from_pretrained(os.path.join(CF, 'tokenizer'))
    # 关键：从 counterfeit 的【原生】scheduler 目录取配置（那是 SD1.5 原版配置）
    sd = os.path.join(CF, 'scheduler')
    sch = PNDMScheduler.from_pretrained(sd)
    if sched_name == 'dpm':
        sch = DPMSolverMultistepScheduler.from_pretrained(sd)   # ★ 用 from_pretrained，不是 from_config
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

CONFIGS = [
    ('A_pndm_s040', 'pndm', 0.40),
    ('B_dpm_s040', 'dpm', 0.40),
    ('C_dpm_s060', 'dpm', 0.60),
]

for name, sched, strength in CONFIGS:
    pipe = build(sched)
    print('  %s (sched=%s strength=%.2f)' % (name, sched, strength), flush=True)
    row = Image.new('RGB', (tw * len(tiles), th), (20, 20, 20))
    t0 = time.perf_counter()
    for i, t in enumerate(tiles):
        with torch.inference_mode():
            im = pipe(prompt=PROMPT, negative_prompt=NEG, image=t, strength=strength,
                      num_inference_steps=20, guidance_scale=7.5,
                      generator=torch.Generator(device='cpu').manual_seed(1000 + i)).images[0]
        row.paste(im.resize((tw, th)), (i * tw, 0))
    p = os.path.join(OUT, name + '.png')
    row.save(p)
    print('    -> %s  (%.0f s)' % (p, time.perf_counter() - t0), flush=True)

# 参考：输入长什么样
Image.new('RGB', (tw * len(tiles), th), (20, 20, 20))
ref = Image.new('RGB', (tw * len(tiles), th), (20, 20, 20))
for i, t in enumerate(tiles):
    ref.paste(t, (i * tw, 0))
ref.save(os.path.join(OUT, 'INPUT.png'))
print('  -> %s' % os.path.join(OUT, 'INPUT.png'))
