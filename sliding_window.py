# -*- coding: utf-8 -*-
"""sliding_window.py — AnimateDiff 滑窗接力生成（E2E 管线版，2026-10-23 port）。

本文件由 E2E 管线副本 port 回仓库（feat/video-e2e，起点 824995d0）。
原始实现的行为与数学未改，只加接口/修 bug；所有改动见下方 E2E-PATCH 列表。
滑窗接力：chunk 0 跑全 schedule，后续 chunk 用上一 chunk 的尾部 latent + 部分加噪接力，
只跑 schedule 尾部；最后把重叠区线性 ramp 混合。

E2E-PATCH 列表（详见 PENDING_REPO_PORT.md）：
  [P1] 加 `--outdir`：原件写死在 `anime_adiff.OUT_DIR`，管线需要 run 目录。
       输出布局：`<outdir>/frames/fNNN.png`、`<outdir>/video.gif`、`<outdir>/sample_strip.png`。
  [P2] 加 `--gen-json`：把每 chunk 的耗时/实跑步数/起始 timestep/latent std 写盘
       （latent std 是滑窗漂移的哨兵：原注释里 timestep 失配会让它每 chunk +32.4%）。
  [P3] **修复 `blend_overlap()` 空实现**：原件函数直接 `return frames`（什么都没做），
       真正的混合是内联在 main 里的另一段代码；现在函数实现 ramp 混合，并由 main 调用。
  [P4] 惰性 import：torch / anime_adiff 只在 main 里导入，`--help` 不需要 torch。
  [P5] 加 `--frames-dir`（默认 frames）与 `--limit-s`（单进程保险丝，超时在 chunk 边界退出）。

用法:
  python sliding_window_e2e.py --res 128 --chunk 8 --overlap 2 --chunks 4 --steps 6 \
      --outdir D:\\work\\video_mvp\\runs\\x
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

GUIDANCE = 7.5


def _reconfigure_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def blend_overlap(prev_frames, head_frames):
    """E2E-PATCH[P3]: 线性 ramp 混合重叠帧（原件此函数是 `return frames` 的空实现）。

    prev_frames: 上一 chunk 保留在 all_frames 尾部的那些帧（uint8 HxWx3）
    head_frames: 新 chunk 输出里对应重叠区的帧（uint8 HxWx3）
    返回: 混合后的帧列表（长度 = min(len, len)）

    权重：第 j 个重叠帧取 new 的比例 (j+1)/(keep+1)，与原件 main 内联逻辑完全一致
    （per-frame linear ramp，避免 seam）。
    """
    keep = min(len(prev_frames), len(head_frames))
    out = []
    for j in range(keep):
        w = (j + 1) / (keep + 1)
        blended = (prev_frames[j].astype(np.float32) * (1 - w)
                   + head_frames[j].astype(np.float32) * w)
        out.append(blended.round().astype(np.uint8))
    return out


def main():
    _reconfigure_stdout()
    ap = argparse.ArgumentParser(description="sliding-window AnimateDiff generation (E2E copy)")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=8, help="frames generated per chunk")
    ap.add_argument("--overlap", type=int, default=2, help="frames shared between chunks")
    ap.add_argument("--chunks", type=int, default=4, help="number of chunks")
    ap.add_argument("--steps", type=int, default=8, help="full-schedule steps for chunk 0")
    ap.add_argument("--carry-steps", type=int, default=3,
                    help="steps actually run for carried chunks (tail of the schedule)")
    ap.add_argument("--carry-noise", type=float, default=400.0,
                    help="timestep to noise carried latents to (lower = keep more structure)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--scheduler", default="dpm")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", default="slide")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--neg", default=None)
    # E2E-PATCH[P1]/[P5]
    ap.add_argument("--outdir", default="", help="run 目录（默认写 anime_adiff.OUT_DIR）")
    ap.add_argument("--frames-dir", default="frames")
    ap.add_argument("--gen-json", default="", help="per-chunk 计时/latent std 落盘")
    ap.add_argument("--limit-s", type=float, default=0.0, help=">0 时在 chunk 边界超时退出")
    # E2E-PATCH[P8]: adapter 与训练一致（v1-5-2；v1-5 无 mid_block 时序层）
    ap.add_argument("--adapter", default=r"D:\work\textmodel\animatediff-motion-adapter-v1-5-2")
    a = ap.parse_args()

    assert 0 <= a.overlap < a.chunk, "overlap must be smaller than chunk"

    # E2E-PATCH[P4]: 惰性 import
    import torch
    from PIL import Image
    from anime_adiff import build, ANIME_PROMPT, ANIME_NEG, OUT_DIR

    prompt = a.prompt if a.prompt is not None else ANIME_PROMPT
    neg = a.neg if a.neg is not None else ANIME_NEG
    outdir = a.outdir or OUT_DIR
    frames_dir = os.path.join(outdir, a.frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    torch.set_num_threads(a.threads)
    torch.backends.mkldnn.enabled = True

    print(f"[cfg] res={a.res} chunk={a.chunk} overlap={a.overlap} chunks={a.chunks} "
          f"steps={a.steps} carry_steps={a.carry_steps}", flush=True)
    print(f"[cfg] outdir={outdir}", flush=True)

    pipe = build(a.scheduler, adapter=a.adapter)
    pipe.set_progress_bar_config(disable=False)

    lat_h = a.res // pipe.vae_scale_factor
    gen = torch.Generator().manual_seed(a.seed)

    with torch.no_grad():
        pe, ne = pipe.encode_prompt(prompt, "cpu", 1, True, negative_prompt=neg)
    # E2E-PATCH[P6]: 按帧展开 context（UNetMotionModel 内部 B*F 折叠）。
    #   直接给 (2,77,768) 会报 `size of tensor a (2048) must match tensor b (256)`（VIDEO_GUIDE §3）。
    embeds = torch.cat([ne, pe]).repeat_interleave(a.chunk, dim=0).contiguous()

    sched = pipe.scheduler
    sched.set_timesteps(a.steps)
    full_ts = sched.timesteps

    all_frames = []
    prev_tail_latents = None
    t_start_all = time.time()
    chunk_log = []

    for ci in range(a.chunks):
        # E2E-PATCH[P5]: 时间保险丝只在 chunk 边界判断（不打断 chunk 内循环）
        if a.limit_s and (time.time() - t_start_all) > a.limit_s:
            print(f"[limit] {time.time()-t_start_all:.0f}s > limit {a.limit_s}s，"
                  f"在 chunk {ci} 边界退出", flush=True)
            break
        is_first = ci == 0
        sched.set_timesteps(a.steps)
        ts = sched.timesteps
        skip = 0
        if not is_first:
            below = [i for i, t in enumerate(ts) if float(t) <= a.carry_noise]
            skip = below[0] if below else max(0, len(ts) - 1)
            sched._step_index = skip
            sched.model_outputs = [None] * sched.config.solver_order
            sched._lower_order_nums = 0
        ts_run = ts[skip:]
        n_steps = len(ts_run)
        print(f"\n[chunk {ci}] frames={a.chunk} steps={n_steps} skip={skip} "
              f"timesteps={[int(t) for t in ts_run]}", flush=True)

        if is_first or prev_tail_latents is None:
            lat = torch.randn((1, pipe.unet.config.in_channels, a.chunk, lat_h, lat_h),
                              generator=gen, dtype=torch.float32)
        else:
            tail = prev_tail_latents
            reps = a.chunk - tail.shape[2]
            if reps > 0:
                pad = tail[:, :, -1:].repeat(1, 1, reps, 1, 1)
                lat = torch.cat([tail, pad], dim=2)
            else:
                lat = tail[:, :, :a.chunk]
            noise = torch.randn(lat.shape, generator=gen, dtype=torch.float32)
            # 必须噪到 solver 实际起始的 timestep（见原件注释：失配 ⇒ latent std 漂移 +32.4%）
            t0 = torch.as_tensor([float(ts[skip])])
            lat = sched.add_noise(lat, noise, t0)

        lat = lat * sched.init_noise_sigma
        lat_std_before = float(lat.std())

        t0 = time.time()
        with torch.no_grad():
            for i, t in enumerate(ts_run):
                model_in = torch.cat([lat] * 2)
                model_in = sched.scale_model_input(model_in, t)
                npr = pipe.unet(model_in, t, encoder_hidden_states=embeds).sample
                nu, nt = npr.chunk(2)
                npr = nu + GUIDANCE * (nt - nu)
                lat = sched.step(npr, t, lat,
                                 **pipe.prepare_extra_step_kwargs(gen, 0.0)).prev_sample
                print(f"    step {i+1}/{n_steps} t={int(t)} "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)

        with torch.no_grad():
            b, c, T, h, w = lat.shape
            flat = lat.permute(0, 2, 1, 3, 4).reshape(b * T, c, h, w)
            dec = pipe.vae.decode(flat / pipe.vae.config.scaling_factor, return_dict=False)[0]
            dec = (dec / 2 + 0.5).clamp(0, 1)
            arr = (dec.reshape(b, T, 3, dec.shape[-2], dec.shape[-1])
                      .permute(0, 1, 3, 4, 2)[0].cpu().numpy() * 255).round().astype(np.uint8)

        keep = max(1, a.overlap)
        prev_tail_latents = lat[:, :, -keep:].clone()

        if is_first:
            all_frames.extend(list(arr))
            print(f"  [chunk 0] kept {len(arr)} frames (total {len(all_frames)})", flush=True)
        else:
            new_frames = arr[keep:]
            # E2E-PATCH[P3]: 用真正实现的 blend_overlap（原件内联同样逻辑）
            blended_head = blend_overlap(all_frames[-keep:], arr[:keep])
            all_frames[-keep:] = blended_head
            all_frames.extend(list(new_frames))
            print(f"  [chunk {ci}] blended {len(blended_head)} overlap frames, "
                  f"added {len(new_frames)} (total {len(all_frames)})", flush=True)

        chunk_log.append(dict(chunk=ci, frames=int(len(arr)), steps=int(n_steps),
                              skip=int(skip), seconds=round(time.time() - t0, 2),
                              latent_std_before=round(lat_std_before, 4),
                              latent_std_after=round(float(lat.std()), 4)))

    dt = time.time() - t_start_all
    print(f"\n[gen] total {len(all_frames)} frames in {dt:.1f}s", flush=True)

    from PIL import Image
    tag = f"{a.tag}_{a.res}_{a.chunk}c{a.overlap}o{a.chunks}n_{a.steps}s"
    imgs = [Image.fromarray(f).convert("RGB") for f in all_frames]
    gif = os.path.join(outdir, "video.gif")
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=200, loop=0)
    for i, im in enumerate(imgs):
        im.save(os.path.join(frames_dir, f"f{i:03d}.png"))
    strip = os.path.join(outdir, "sample_strip.png")
    try:
        sw = min(len(imgs), 16)
        sheet = Image.new("RGB", (imgs[0].width * sw, imgs[0].height))
        for i in range(sw):
            sheet.paste(imgs[i], (i * imgs[0].width, 0))
        sheet.save(strip)
    except Exception as e:
        print(f"[warn] strip 生成失败: {e}")
    print(f"[save] {gif}\n[save] {frames_dir}\\f000..f{len(imgs)-1:03d}.png "
          f"({len(imgs)} frames)", flush=True)

    # E2E-PATCH[P2]: 落盘 per-chunk 计时/latent std
    gen_json = a.gen_json or os.path.join(outdir, "gen_meta.json")
    try:
        with open(gen_json, "w", encoding="utf-8") as fh:
            json.dump(dict(mode="sliding", res=a.res, chunk=a.chunk, overlap=a.overlap,
                           chunks_requested=a.chunks, steps=a.steps,
                           carry_steps=a.carry_steps, carry_noise=a.carry_noise,
                           seed=a.seed, seconds=dt, frames=len(all_frames),
                           per_chunk=chunk_log, gif=gif, frames_dir=frames_dir,
                           tag=tag), fh, indent=2, ensure_ascii=False)
        print(f"[save] {gen_json}")
    except Exception as e:
        print(f"[warn] gen_meta 写盘失败: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
