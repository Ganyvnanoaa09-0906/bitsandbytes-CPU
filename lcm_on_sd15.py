# -*- coding: utf-8 -*-
"""lcm_on_sd15.py — 在**真正的 SD1.5** 上测 LCM LoRA（速度 + 画质）

为什么需要这个而不是直接改 lcm_speedup.py:
    lcm_speedup.py 跑的是 counterfeit-v3-diffusers，因为那是磁盘上唯一**完整**的
    diffusers pipeline。而 LCM LoRA 是为**原版 SD1.5** 训练的 ⇒ 拿微调模型当基座测
    速度可以（速度只取决于步数），但**画质结论不成立**。
    本仓库的 assemble_sd15.py 已经把原版 SD1.5 拼出来了（sd15_base/unet +
    counterfeit 的 VAE/CLIP/tokenizer），直接复用它的 build_sd15()，不重复造轮子。

判据（两条都要过）:
    1. 速度: LCM(4 步, cfg 1.5) 相对基线(20 步, cfg 7.5) 的 wall 时间比；
    2. 画质: **同一 seed 下**两种设置的输出都要非退化（finite、方差合理、非纯色），
       且 LCM 输出不应是噪声。⚠️ 这里**不做**"哪张更好看"的主观判断——
       那是人看的；本脚本只给客观有效性 + 落盘图片供人工对比。

oracle:
    · 同一 prompt / seed / 尺寸，只改 scheduler + 步数 + guidance；
    · 打印实际 scheduler 类名（防止没生效却以为生效）；
    · 图片落盘（含并排对照图），人工可直接看；
    · 用 torch.inference_mode()，与 assemble_sd15.py 的口径一致。
"""
from __future__ import annotations

import os
import sys
import time

import torch

REPO = r'D:\work\bitsandbytes-CPU'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'bitsandbytes'))
from assemble_sd15 import build_sd15  # noqa: E402  复用既有拼装函数

LCM = os.environ.get('LCM_DIR',
                     r'D:\work\ms_cache\models\latent-consistency--lcm-lora-sdv1-5')
RES = int(os.environ.get('RES', '512'))
STEPS_BASE = int(os.environ.get('STEPS_BASE', '20'))
STEPS_LCM = int(os.environ.get('STEPS_LCM', '4'))
SEED = int(os.environ.get('SEED', '777'))
OUTDIR = os.environ.get('OUTDIR', r'D:\work\cloud_results')

PROMPTS = [
    ('real', "a photograph of a real person, a woman standing in a room, "
             "wearing a dress, looking at the viewer, detailed face, natural lighting, "
             "sharp focus, high quality photo"),
    ('anime', "1girl, solo, upper body, looking at viewer, detailed face, "
              "silver hair, school uniform, soft lighting, best quality"),
]


def valid(im, tag):
    import numpy as np
    a = np.asarray(im, dtype=np.float32)
    ok = bool(np.isfinite(a).all()) and a.std() > 3.0
    print('      [%s] mean=%.1f std=%.1f -> %s' % (tag, a.mean(), a.std(),
                                                   'VALID' if ok else 'DEGENERATE'))
    return ok


def main():
    from diffusers import DPMSolverMultistepScheduler, LCMScheduler
    from PIL import Image, ImageDraw

    torch.set_num_threads(int(os.environ.get('THREADS', '6')))
    os.makedirs(OUTDIR, exist_ok=True)
    print('=' * 84)
    print('LCM LoRA on REAL SD1.5')
    print('=' * 84)
    print('res=%d base=%d步 lcm=%d步 seed=%d threads=%d'
          % (RES, STEPS_BASE, STEPS_LCM, SEED, torch.get_num_threads()))

    t0 = time.perf_counter()
    pipe = build_sd15()
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    print('[sd15] 拼装 %.1f s' % (time.perf_counter() - t0))

    rows = []
    for tag, pr in PROMPTS:
        # ---- 基线 ----
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        print('\n[%s] 基线 scheduler=%s' % (tag, type(pipe.scheduler).__name__))
        g = torch.Generator(device='cpu').manual_seed(SEED)
        t0 = time.perf_counter()
        with torch.inference_mode():
            im_a = pipe(pr, num_inference_steps=STEPS_BASE, guidance_scale=7.5,
                        width=RES, height=RES, generator=g).images[0]
        ta = time.perf_counter() - t0
        pa = os.path.join(OUTDIR, 'LCMAB_base_%s.png' % tag)
        im_a.save(pa)
        print('      %.1f s -> %s' % (ta, pa))
        oka = valid(im_a, 'base')

        # ---- LCM ----
        if not getattr(pipe, '_lcm_loaded', False):
            try:
                pipe.load_lora_weights(
                    LCM, weight_name='pytorch_lora_weights.safetensors',
                    local_files_only=True, adapter_name='lcm')
                pipe._lcm_loaded = True
                print('    LoRA 已加载 (adapter=lcm)')
            except Exception as e:
                print('    LoRA 加载失败: %s: %s' % (type(e).__name__, e))
                return 1
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        print('    LCM  scheduler=%s' % type(pipe.scheduler).__name__)
        g2 = torch.Generator(device='cpu').manual_seed(SEED)
        t0 = time.perf_counter()
        with torch.inference_mode():
            im_b = pipe(pr, num_inference_steps=STEPS_LCM, guidance_scale=1.5,
                        width=RES, height=RES, generator=g2).images[0]
        tb = time.perf_counter() - t0
        pb = os.path.join(OUTDIR, 'LCMAB_lcm_%s.png' % tag)
        im_b.save(pb)
        print('      %.1f s -> %s' % (tb, pb))
        okb = valid(im_b, 'lcm')

        rows.append((tag, ta, tb, oka, okb, im_a, im_b))

    # ---- 并排对照图 ----
    W = max(im.size[0] for r in rows for im in (r[5], r[6]))
    H = max(im.size[1] for r in rows for im in (r[5], r[6]))
    canvas = Image.new('RGB', (2 * W + 8, len(rows) * H), (20, 20, 20))
    dr = ImageDraw.Draw(canvas)
    for i, (tag, ta, tb, oka, okb, ia, ib) in enumerate(rows):
        canvas.paste(ia, (0, i * H))
        canvas.paste(ib, (W + 8, i * H))
        dr.text((6, i * H + 6), '%s BASE %d步 %.0fs' % (tag, STEPS_BASE, ta), fill=(255, 255, 0))
        dr.text((W + 14, i * H + 6), '%s LCM %d步 %.0fs' % (tag, STEPS_LCM, tb), fill=(0, 255, 255))
    gp = os.path.join(OUTDIR, 'LCMAB_grid.png')
    canvas.save(gp)

    print('\n[判据]')
    for tag, ta, tb, oka, okb, _, _ in rows:
        print('  %-6s 基线 %6.1fs  LCM %6.1fs  => %.2fx   有效性 base=%s lcm=%s'
              % (tag, ta, tb, ta / tb, oka, okb))
    print('\n对照图: %s   （人工看画质；脚本不给"哪张更好"的主观结论）' % gp)
    print('=' * 84)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
