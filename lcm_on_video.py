# -*- coding: utf-8 -*-
"""lcm_on_video.py — LCM 用在**视频**上的实测（不是外推）

为什么必须实跑:
    lcm_on_sd15.py 在 512×512 单帧上实测 LCM 快 3.69–4.42×。把它外推到视频场景
    （§10.150 实测 16 帧单次前向 96.6 s，20 步 ≈ 32 分钟/clip）会得到"4 步 ≈ 8.6 分钟"。
    但那是**线性外推**，而 §10.148 的教训正是"斜率会漂移时不能外推"。
    视频相对图像多了时序 attention，其在 16 帧时的占比（§10.150：11.9%）与单帧不同
    ⇒ 每步成本未必与单帧一致。**必须在视频形状上实跑。**

复用:
    anime_adiff.py 已有 build()（组装 AnimateDiffPipeline，含 VAE 选择与三种 scheduler），
    本脚本直接复用，不重复造轮子。**注意**：它的 ADAPTER_V15 是 v1-5（非 -2），
    而 video_models.py 用的是 v1-5-2；这里保持与 anime_adiff.py 一致以便与你既有结果可比。

判据:
    1. 同一 prompt / seed / 帧数 / 分辨率下，20 步 vs LCM 4 步的 wall 时间比；
    2. LCM 输出视频**有效性**：帧数正确、帧间有变化（不是静止/纯色）、finite；
    3. 落盘 mp4 + 首帧对照图，供人工看。

oracle:
    · 打印实际 scheduler 类名（防止 LoRA/scheduler 没生效却以为生效）；
    · 同时报"每帧成本"，与 §10.150 的 96.6 s/前向（16 帧）对照，看是否一致；
    · 帧间差分均值 > 0 才算"真的动了"。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

REPO = r'D:\work\bitsandbytes-CPU'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'bitsandbytes'))
from anime_adiff import build, ANIME_PROMPT, ANIME_NEG  # noqa: E402  复用既有组装

LCM = os.environ.get('LCM_DIR',
                     r'D:\work\ms_cache\models\latent-consistency--lcm-lora-sdv1-5')
OUTDIR = os.environ.get('OUTDIR', r'D:\work\cloud_results')
RES = int(os.environ.get('RES', '256'))
FRAMES = int(os.environ.get('FRAMES', '8'))
STEPS_BASE = int(os.environ.get('STEPS_BASE', '20'))
STEPS_LCM = int(os.environ.get('STEPS_LCM', '4'))
SEED = int(os.environ.get('SEED', '1234'))


def frames_stats(frames, tag):
    a = np.stack([np.asarray(f, dtype=np.float32) for f in frames])   # (F,H,W,3)
    finite = bool(np.isfinite(a).all())
    std = float(a.std())
    # 帧间差分：真的在动才会有非零均值
    diff = float(np.abs(np.diff(a, axis=0)).mean()) if a.shape[0] > 1 else 0.0
    ok = finite and std > 3.0
    print('      [%s] frames=%d %s std=%.1f 帧间差分=%.2f %s'
          % (tag, a.shape[0], a.shape[1:], std, diff,
             'VALID' if (ok and diff > 0.1) else ('STATIC?' if ok else 'DEGENERATE')))
    return ok, diff


def save_video(frames, path, fps=8):
    try:
        from diffusers.utils import export_to_video
        export_to_video(frames, path, fps=fps)
        return True
    except Exception as e:
        print('      mp4 落盘失败(%s)，改为存帧 PNG' % type(e).__name__)
        base = os.path.splitext(path)[0]
        for i, f in enumerate(frames):
            f.save('%s_f%02d.png' % (base, i))
        return False


def main():
    from diffusers import LCMScheduler, DPMSolverMultistepScheduler
    from PIL import Image, ImageDraw

    torch.set_num_threads(int(os.environ.get('THREADS', '6')))
    os.makedirs(OUTDIR, exist_ok=True)
    print('=' * 86)
    print('LCM on VIDEO (AnimateDiff) — 实测，不外推')
    print('=' * 86)
    print('res=%d frames=%d base=%d步 lcm=%d步 seed=%d threads=%d'
          % (RES, FRAMES, STEPS_BASE, STEPS_LCM, SEED, torch.get_num_threads()))

    t0 = time.time()
    pipe = build(scheduler='dpm')
    pipe.set_progress_bar_config(disable=True)
    # ⚠️ 实测：enable_attention_slicing() 会让 VAE 解码慢 71%
    #    F=8 有 slicing 35.36 s (每帧 4.42 s)  vs  无 slicing 20.72 s (每帧 2.59 s)
    #    所以这里**明确关闭**，而不是跟着 anime_adiff.py 的示例默认打开。
    try:
        pipe.disable_attention_slicing()
        print('[opt] attention_slicing 已关闭（实测 VAE 快 1.71x）')
    except Exception:
        pass
    print('[build] %.1f s' % (time.time() - t0))

    # ---- 基线 ----
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    print('\n[BASE] scheduler=%s  steps=%d' % (type(pipe.scheduler).__name__, STEPS_BASE))
    g = torch.Generator(device='cpu').manual_seed(SEED)
    t0 = time.time()
    with torch.inference_mode():
        out = pipe(ANIME_PROMPT, negative_prompt=ANIME_NEG, num_frames=FRAMES,
                   num_inference_steps=STEPS_BASE, guidance_scale=7.5,
                   width=RES, height=RES, generator=g)
    tb = time.time() - t0
    # ⚠️ AnimateDiffPipeline 的 frames 是**嵌套 list**：外层 batch、内层才是帧。
    #    实测 type(out.frames)=list, len=1, out.frames[0] 是 list[PIL.Image]。
    #    直接按 out.frames 当帧列表用会在 8 帧规模上崩（VCRUNTIME140 0xc0000005）。
    fb = out.frames[0]
    okb, db = frames_stats(fb, 'base')
    vb = os.path.join(OUTDIR, 'LCMVID_base_%df_%dsteps.mp4' % (FRAMES, STEPS_BASE))
    save_video(fb, vb)
    print('      %.1f s  (每帧 %.2f s)  -> %s' % (tb, tb / FRAMES, vb))

    # ---- LCM ----
    print('\n[LCM] 加载 LoRA')
    try:
        pipe.load_lora_weights(LCM, weight_name='pytorch_lora_weights.safetensors',
                               local_files_only=True, adapter_name='lcm')
        print('      LoRA OK')
    except Exception as e:
        print('      LoRA 失败: %s: %s' % (type(e).__name__, e))
        return 1
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    print('      scheduler=%s  steps=%d' % (type(pipe.scheduler).__name__, STEPS_LCM))
    g2 = torch.Generator(device='cpu').manual_seed(SEED)
    t0 = time.time()
    with torch.inference_mode():
        out2 = pipe(ANIME_PROMPT, negative_prompt=ANIME_NEG, num_frames=FRAMES,
                    num_inference_steps=STEPS_LCM, guidance_scale=1.5,
                    width=RES, height=RES, generator=g2)
    tl = time.time() - t0
    fl = out2.frames[0]
    okl, dl = frames_stats(fl, 'lcm')
    vl = os.path.join(OUTDIR, 'LCMVID_lcm_%df_%dsteps.mp4' % (FRAMES, STEPS_LCM))
    save_video(fl, vl)
    print('      %.1f s  (每帧 %.2f s)  -> %s' % (tl, tl / FRAMES, vl))

    # ---- 首帧对照图 ----
    A, B = fb[0], fl[0]
    W, H = A.size
    canvas = Image.new('RGB', (2 * W + 8, H), (20, 20, 20))
    canvas.paste(A, (0, 0)); canvas.paste(B, (W + 8, 0))
    dr = ImageDraw.Draw(canvas)
    dr.text((6, 6), 'BASE %d steps %.0fs' % (STEPS_BASE, tb), fill=(255, 255, 0))
    dr.text((W + 14, 6), 'LCM %d steps %.0fs' % (STEPS_LCM, tl), fill=(0, 255, 255))
    gp = os.path.join(OUTDIR, 'LCMVID_grid.png')
    canvas.save(gp)

    print('\n[判据]')
    print('  基线 %6.1f s (%d步)  LCM %6.1f s (%d步)  => %.2fx'
          % (tb, STEPS_BASE, tl, STEPS_LCM, tb / tl))
    print('  每帧成本: %.2f s vs %.2f s ；每步成本: %.3f s vs %.3f s'
          % (tb / FRAMES, tl / FRAMES, tb / STEPS_BASE, tl / STEPS_LCM))
    print('  有效性: base=%s(帧间差 %.2f)  lcm=%s(帧间差 %.2f)' % (okb, db, okl, dl))
    # 与 §10.150 对照：16 帧 / latent 64 单次前向 96.6 s ⇒ 每帧 6.04 s
    print('\n  [与 §10.150 对照] 16帧/latent64 单次前向 96.6 s = 每帧 6.04 s')
    print('                     本次 res=%d/%d帧 的每帧成本见上；若量级差很多，')
    print('                     说明"每步成本与帧数无关"这个假设不成立。')
    print('  对照图: %s' % gp)
    print('=' * 86)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
