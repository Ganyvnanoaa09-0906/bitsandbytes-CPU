# -*- coding: utf-8 -*-
"""lcm_speedup.py — LCM LoRA 在 CPU 上到底省多少（端到端，实测）

背景:
    仓库笔记 extract_pdfs.py:12 把 LCM/ADP/ADM 等扩散加速列为「速度已解决 ⇒ 低优先」。
    在 CPU 上这条判断**可能不成立**：§10.150 实测 SD1.5+AnimateDiff 单次 UNet 前向
    （512×512、16 帧）96.6 s，20 步 ⇒ 约 32 分钟/clip。而仓库里**已经有** LCM LoRA
    （D:\\work\\ms_cache\\models\\latent-consistency--lcm-lora-sdv1-5，834 张量，rank 64，
    维度匹配 SD1.5）。

判据:
    端到端 wall time 之比（同样输出尺寸/帧数），以及 LCM 输出是否**有效**
    （非全黑/非 NaN、方差合理）。速度提升若 ≥3× 且输出有效，这条就该从「低优先」摘出来。

oracle:
    · 两次运行用**同一 prompt、同一 seed、同一尺寸**，只改步数与 guidance；
    · 先做有效性检查再报速度（避免"快但坏"）；
    · 时间取 wall clock（端到端，含 VAE 解码），不是只测 UNet。
    · 显式打印实际使用的 scheduler 与 guidance，防止"以为在用 LCM 其实没生效"。

用法:
    python lcm_speedup.py                 # 256x256, 1 帧
    RES=512 FRAMES=8 python lcm_speedup.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

REPO = r'D:\work\bitsandbytes-CPU'
sys.path.insert(0, os.path.join(REPO, 'bitsandbytes'))

SD15 = os.environ.get('SD15_DIR', r'D:\work\textmodel\sd15_full')
LCM = os.environ.get('LCM_DIR',
                     r'D:\work\ms_cache\models\latent-consistency--lcm-lora-sdv1-5')
RES = int(os.environ.get('RES', '256'))
FRAMES = int(os.environ.get('FRAMES', '1'))
STEPS_BASE = int(os.environ.get('STEPS_BASE', '20'))
STEPS_LCM = int(os.environ.get('STEPS_LCM', '4'))
PROMPT = os.environ.get('PROMPT', 'a cinematic photo of a cat, detailed')
SEED = int(os.environ.get('SEED', '1234'))

torch.set_num_threads(int(os.environ.get('THREADS', '6')))


def stats(x, name):
    """有效性检查：非 NaN、非常数、方差合理。"""
    import numpy as np
    a = x.detach().float().cpu().numpy()
    finite = bool(np.isfinite(a).all())
    std = float(a.std())
    mean = float(a.mean())
    ok = finite and std > 1e-3
    print('    [%s] shape=%s mean=%.4f std=%.4f finite=%s -> %s'
          % (name, tuple(a.shape), mean, std, finite, 'VALID' if ok else 'INVALID'))
    return ok


def main():
    from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
    from diffusers import LCMScheduler, DPMSolverMultistepScheduler
    from transformers import CLIPTextModel, CLIPTokenizer

    print('=' * 84)
    print('LCM LoRA 端到端加速实测')
    print('=' * 84)
    print('sd15=%s' % SD15)
    print('lcm =%s' % LCM)
    print('res=%d frames=%d  base_steps=%d lcm_steps=%d  threads=%d'
          % (RES, FRAMES, STEPS_BASE, STEPS_LCM, torch.get_num_threads()))

    t0 = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        SD15, torch_dtype=torch.float32, safety_checker=None,
        requires_safety_checker=False, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    print('[load] %.1f s' % (time.perf_counter() - t0))

    g = torch.Generator().manual_seed(SEED)

    # ---- A: 基线（DPM-Solver，20 步，cfg 7.5）----
    print('\n[A] 基线')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    print('    scheduler=%s' % type(pipe.scheduler).__name__)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_a = pipe(PROMPT, num_inference_steps=STEPS_BASE, guidance_scale=7.5,
                     height=RES, width=RES, generator=g, num_frames=FRAMES
                     if hasattr(pipe, 'unet') and hasattr(pipe.unet, 'config')
                     and getattr(pipe.unet.config, 'in_channels', 4) == 4 else None
                     ).images
    ta = time.perf_counter() - t0
    print('    时间 %.2f s' % ta)

    # ---- B: LCM（LCMScheduler，4 步，cfg 1.5）----
    print('\n[B] LCM LoRA')
    try:
        pipe.load_lora_weights(LCM, local_files_only=True)
        print('    LoRA 已加载')
    except Exception as e:
        print('    LoRA 加载失败: %s: %s' % (type(e).__name__, e))
        return 1
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    print('    scheduler=%s' % type(pipe.scheduler).__name__)
    g2 = torch.Generator().manual_seed(SEED)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_b = pipe(PROMPT, num_inference_steps=STEPS_LCM, guidance_scale=1.5,
                     height=RES, width=RES, generator=g2).images
    tb = time.perf_counter() - t0
    print('    时间 %.2f s' % tb)

    print('\n[判据]')
    print('    基线 %.2f s (%d 步)  vs  LCM %.2f s (%d 步)  ⇒ %.2fx'
          % (ta, STEPS_BASE, tb, STEPS_LCM, ta / tb))
    print('    每步成本: 基线 %.3f s/步   LCM %.3f s/步  (应接近，说明只是步数少)'
          % (ta / STEPS_BASE, tb / STEPS_LCM))

    import numpy as np
    a = np.asarray(out_a[0], dtype=np.float32)
    b = np.asarray(out_b[0], dtype=np.float32)
    ok_a = a.std() > 1.0 and np.isfinite(a).all()
    ok_b = b.std() > 1.0 and np.isfinite(b).all()
    print('    输出有效性: 基线 %s (mean=%.1f std=%.1f)  LCM %s (mean=%.1f std=%.1f)'
          % ('OK' if ok_a else 'BAD', a.mean(), a.std(),
             'OK' if ok_b else 'BAD', b.mean(), b.std()))
    if ta / tb >= 3.0 and ok_b:
        print('\n    ⇒ LCM 有效且 ≥3x。CPU 上应把它从「低优先」提上来。')
    else:
        print('\n    ⇒ 未达 3x 或输出无效，需进一步查（打印的 scheduler 是否真是 LCM）。')
    print('=' * 84)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
