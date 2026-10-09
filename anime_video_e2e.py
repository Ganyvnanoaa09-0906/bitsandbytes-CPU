# -*- coding: utf-8 -*-
"""anime_video_e2e.py — AnimateDiff 端到端出片：生成 → 超分 → mp4

这是这台机器上**唯一能真出片**的路（Wan 那条线被 11.4 GB 的文本编码器挡住，
见 report §10.158）。

两阶段设计（重要，不是随手拆的）:
    阶段 1 生成：AnimateDiff 出低分辨率帧（本机可行包络内，§10.160）
    阶段 2 超分：逐帧 Real-ESRGAN 4× + 用 av 编码 mp4
    **必须分进程**：实测本机同时持有 AnimateDiff(5.3 GB) + ESRGAN 的中间张量会
    触发 0xC0000005（§10.155.3 记的"多实例并存"问题）。分阶段后每阶段内存独立。

判据（三条，缺一不可）:
    1. 帧数正确、每帧 finite、**帧间有变化**（不是静止/纯色）
    2. 超分输出尺寸 = 输入 × 4
    3. mp4 真的写成了（读回校验帧数与尺寸，不能只看文件存在）

oracle:
    · 帧间差分均值 > 0 才算"真的在动"（静止视频看起来正常但没用）
    · 生成/超分**分别计时**，因为两者的优化手段完全不同
    · mp4 用 av 编码，H.264 yuv420p（能直接被播放器/浏览器打开）

用法:
    python anime_video_e2e.py gen   --frames 16 --res 256 --steps 20
    python anime_video_e2e.py sr    --scale 4
    python anime_video_e2e.py all
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "bitsandbytes"))
torch.set_num_threads(int(os.environ.get("THREADS", "6")))

OUT_ROOT = os.environ.get("E2E_OUT", r"D:\work\cloud_results\anime_e2e")
LR_DIR = os.path.join(OUT_ROOT, "lr_frames")
SR_DIR = os.path.join(OUT_ROOT, "sr_frames")

PROMPT = ("1girl, solo, upper body, looking at viewer, detailed face, silver hair, "
          "school uniform, soft lighting, best quality, gentle smile")
NEG = ("lowres, bad anatomy, bad hands, extra digits, cropped, worst quality, "
       "low quality, bad face, deformed, blurry")


def frames_stats(frames, tag):
    a = np.stack([np.asarray(f, dtype=np.float32) for f in frames])
    finite = bool(np.isfinite(a).all())
    std = float(a.std())
    diff = float(np.abs(np.diff(a, axis=0)).mean()) if a.shape[0] > 1 else 0.0
    ok = finite and std > 3.0 and diff > 0.1
    print("  [%s] %d 帧 %s  std=%.1f  帧间差分=%.2f  %s"
          % (tag, a.shape[0], a.shape[1:], std, diff,
             "VALID" if ok else ("STATIC?" if finite and std > 3 else "DEGENERATE")))
    return ok, std, diff


# --------------------------------------------------------------------------
# 阶段 1：生成
# --------------------------------------------------------------------------
def stage_gen(frames, res, steps, seed, use_lcm, adapter, lora=""):
    from anime_adiff import build
    from PIL import Image

    os.makedirs(LR_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(LR_DIR, "*.png")):
        os.remove(f)

    print("[gen] 构建 AnimateDiff 管线（adapter=%s）" % os.path.basename(adapter))
    t0 = time.time()
    pipe = build(scheduler="dpm", adapter=adapter)
    pipe.set_progress_bar_config(disable=True)
    # 实测（report §10.155.3）：attention slicing 在这里是负优化，VAE 解码慢 1.71×
    try:
        pipe.disable_attention_slicing()
        print("[gen] attention_slicing 已关闭（实测 VAE 快 1.71×）")
    except Exception:
        pass
    print("[gen] 管线就绪 %.1f s" % (time.time() - t0))

    if use_lcm:
        from diffusers import LCMScheduler
        lcm = os.environ.get(
            "LCM_DIR",
            r"D:\work\ms_cache\models\latent-consistency--lcm-lora-sdv1-5")
        pipe.load_lora_weights(lcm, weight_name="pytorch_lora_weights.safetensors",
                               local_files_only=True, adapter_name="lcm")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        print("[gen] LCM LoRA 已加载，scheduler=%s" % type(pipe.scheduler).__name__)
    else:
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        print("[gen] scheduler=%s" % type(pipe.scheduler).__name__)

    if lora:
        print("[gen] 加载本 fork 训练的 LoRA: %s" % lora)
        before = len(getattr(pipe, "peft_config", {}) or {})
        pipe.load_lora_weights(os.path.dirname(lora),
                               weight_name=os.path.basename(lora),
                               local_files_only=True, adapter_name="fork")
        pipe.set_adapters(["fork"], adapter_weights=[1.0])
        print("[gen] LoRA 已挂载（adapter_name=fork, weight=1.0）"
              " peft_config %d -> %d"
              % (before, len(getattr(pipe, "peft_config", {}) or {})))
        # diffusers 在键全不匹配时【不报错】，只是什么都没加载 ✗
        # 所以这里打印 peft_config 的数量变化，让静默失败看得见

    g = torch.Generator(device="cpu").manual_seed(seed)
    gs = 1.5 if use_lcm else 7.5
    t0 = time.time()
    with torch.no_grad():
        out = pipe(prompt=PROMPT, negative_prompt=NEG, num_frames=frames,
                   num_inference_steps=steps, guidance_scale=gs,
                   height=res, width=res, generator=g)
    dt = time.time() - t0
    fr = out.frames[0]
    print("[gen] %dx%d x%df x%d步 cfg=%.1f = %.1f s (%.2f s/步, 每帧 %.2f s)"
          % (res, res, frames, steps, gs, dt, dt / steps, dt / frames))

    ok, std, diff = frames_stats(fr, "gen")
    imgs = [Image.fromarray(np.asarray(f)).convert("RGB") for f in fr]
    for i, im in enumerate(imgs):
        im.save(os.path.join(LR_DIR, "f%03d.png" % i))
    meta = dict(frames=frames, res=res, steps=steps, seed=seed, lcm=bool(use_lcm),
                prompt=PROMPT, negative=NEG, gen_seconds=dt,
                frame_diff=diff, std=std, adapter=os.path.basename(adapter))
    with open(os.path.join(OUT_ROOT, "gen_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print("[gen] %d 帧 -> %s" % (len(imgs), LR_DIR))
    del pipe
    gc.collect()
    return ok


# --------------------------------------------------------------------------
# 阶段 2：超分 + 编码
# --------------------------------------------------------------------------
def stage_sr(scale, fps, keep_frames):
    from PIL import Image
    from sr_cost import ANIME_6B, X4PLUS, load as load_rrdb

    os.makedirs(SR_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(SR_DIR, "*.png")):
        os.remove(f)

    src = sorted(glob.glob(os.path.join(LR_DIR, "*.png")))
    if not src:
        print("[sr] 没有输入帧，先跑 gen 阶段")
        return False
    print("[sr] 输入 %d 帧，放大 %d×" % (len(src), scale))

    # 6 块模型：实测 23 块在 256 输入下段错误（report §10.161），只用 6 块
    path = ANIME_6B if scale == 4 else X4PLUS
    nblk = 6 if scale == 4 else 23
    model, npar, miss, unexp = load_rrdb(path, nblk)
    print("[sr] %s  %.2fM 参数 missing=%d unexpected=%d"
          % (os.path.basename(path), npar / 1e6, miss, unexp))

    t0 = time.time()
    out_imgs = []
    for i, p in enumerate(src):
        im = Image.open(p).convert("RGB")
        x = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1)[None] / 255.0
        with torch.no_grad():
            y = model(x)
        y = y.clamp(0, 1)[0].permute(1, 2, 0).numpy()
        yi = Image.fromarray((y * 255).round().astype(np.uint8))
        out_imgs.append(yi)
        if keep_frames:
            yi.save(os.path.join(SR_DIR, "f%03d.png" % i))
        if (i + 1) % 4 == 0 or i == len(src) - 1:
            el = time.time() - t0
            print("  [sr] %d/%d  %.1f s  (每帧 %.2f s)"
                  % (i + 1, len(src), el, el / (i + 1)), flush=True)
    dt_sr = time.time() - t0

    print("[sr] 尺寸 %s -> %s  总 %.1f s (每帧 %.2f s)"
          % (src[0].split(os.sep)[-1] and Image.open(src[0]).size,
             out_imgs[0].size, dt_sr, dt_sr / len(src)))
    ok_shape = (out_imgs[0].size[0] == Image.open(src[0]).size[0] * scale)
    ok_motion, std, diff = frames_stats(out_imgs, "sr")

    # ---- 编码 mp4（用 av；本机没有 imageio/cv2）----
    os.makedirs(OUT_ROOT, exist_ok=True)
    mp4 = os.path.join(OUT_ROOT, "anime_final_%dpx_%df.mp4"
                       % (out_imgs[0].size[0], len(out_imgs)))
    try:
        import av
        container = av.open(mp4, mode="w")
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = out_imgs[0].size
        stream.pix_fmt = "yuv420p"
        for im in out_imgs:
            frame = av.VideoFrame.from_ndarray(np.asarray(im), format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()
        wrote = os.path.getsize(mp4)
        print("[encode] %s  %.2f MB  %dx%d @%dfps"
              % (mp4, wrote / 1e6, out_imgs[0].size[0], out_imgs[0].size[1], fps))
    except Exception as e:
        print("[encode] 失败: %s: %s" % (type(e).__name__, e))
        return False

    # ---- 读回校验：只看文件存在不算成功 ----
    try:
        import av
        c = av.open(mp4)
        st = c.streams.video[0]
        n = sum(1 for _ in c.decode(video=0))
        c.close()
        ok_read = (n == len(out_imgs))
        print("[verify] 读回 %d 帧（期望 %d），尺寸 %dx%d  %s"
              % (n, len(out_imgs), st.width, st.height,
                 "OK" if ok_read else "MISMATCH"))
    except Exception as e:
        print("[verify] 读回失败: %s" % e)
        ok_read = False

    print("\n[判据] 尺寸×%d=%s  动态=%s  mp4读回=%s  ⇒ %s"
          % (scale, ok_shape, ok_motion, ok_read,
             "PASS" if (ok_shape and ok_motion and ok_read) else "FAIL"))
    if not keep_frames:
        for f in glob.glob(os.path.join(SR_DIR, "*.png")):
            os.remove(f)
    return ok_shape and ok_motion and ok_read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "sr", "all"])
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lcm", action="store_true", help="用 LCM LoRA（4 步即可）")
    ap.add_argument("--lora", default="",
                    help="本 fork 训练出的 LoRA（convert_lora_ckpt.py 转过的 safetensors）")
    ap.add_argument("--adapter",
                    default=r"D:\work\textmodel\animatediff-motion-adapter-v1-5-2")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--keep-frames", action="store_true")
    a = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    print("=" * 84)
    print("AnimateDiff 端到端出片（生成 → 超分 → mp4）")
    print("=" * 84)
    print("输出目录: %s" % OUT_ROOT)
    t_all = time.time()
    ok = True
    if a.stage in ("gen", "all"):
        ok &= stage_gen(a.frames, a.res, a.steps, a.seed, a.lcm, a.adapter, a.lora)
    if a.stage in ("sr", "all"):
        ok &= stage_sr(a.scale, a.fps, a.keep_frames)
    print("\n总耗时 %.1f s   结果: %s" % (time.time() - t_all, "PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
