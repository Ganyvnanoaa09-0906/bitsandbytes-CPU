# -*- coding: utf-8 -*-
"""control_pure_sd.py — §10.19 那个对照的复现：纯 SD txt2img（不看 AR 输入）。

为什么要这个对照：
  精修图很好看，但必须排除"SD 只是无视输入、自己凭空画"这一解释。
  若纯 SD 也能出同样质量的图 ⇒ **AR 没贡献**，整条"AR 构图 + 扩散补细节"就不成立。
  §10.19 当时的结论：纯 SD 更锐但**塌成 8 张几乎相同的中景半身像**，
  而 AR 提供的是**构图多样性** —— 这是细节模型给不了的。
  现在 AR 的产出好得多（k=1024），需要重新确认这个分工还成不成立。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os, time
import torch

torch.set_num_threads(int(os.environ.get('THREADS', '6')))
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

M = r'D:\work\textmodel\counterfeit-v3-diffusers'
PROMPT = '1girl, cosplay costume, detailed face, high quality, best quality, sharp focus'
NEG = 'lowres, blurry, jpeg artifacts, bad anatomy'

p = StableDiffusionPipeline.from_pretrained(M, torch_dtype=torch.float32,
                                            safety_checker=None, requires_safety_checker=False)
p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config)
p.set_progress_bar_config(disable=True)
try:
    p.enable_attention_slicing()
except Exception:
    pass
p.to('cpu')

t0 = time.perf_counter()
ims = []
with torch.inference_mode():
    for i in range(16):
        im = p(PROMPT, negative_prompt=NEG, num_inference_steps=18, guidance_scale=7.0,
               width=256, height=256,
               generator=torch.Generator(device='cpu').manual_seed(2024 + i)).images[0]
        ims.append(im)
cv = Image.new('RGB', (4 * 256, 4 * 256), (20, 20, 20))
for i, im in enumerate(ims):
    r, c = divmod(i, 4)
    cv.paste(im, (c * 256, r * 256))
out = r'D:\work\cloud_results\CONTROL_pureSD_txt2img.png'
cv.save(out)
print('  pure SD txt2img -> %s  (%.0f s, %.1f s/张)'
      % (out, time.perf_counter() - t0, (time.perf_counter() - t0) / 16))
