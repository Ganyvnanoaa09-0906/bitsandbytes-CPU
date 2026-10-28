# -*- coding: utf-8 -*-
"""sd_speed_matrix.py — 在【不下载任何东西】的前提下，把本地 SD 出图成本压下来。

背景：counterfeit-v3 实测 273 s/张（512²，20 步，6 线程）。要拿它当数据生成器，
      这个速度意味着 10000 张 = 758 小时，完全不可行。
      HF 不可达（LCM-LoRA 下不下来）⇒ 只能榨已有的：步数 / 分辨率 / 调度器 / 批处理。

测的矩阵（每格存图 + 报耗时）：
  调度器：DPM-Solver++（多步，SD1.5 上 8~12 步通常还可用）
  步数：4 / 8 / 12 / 20
  分辨率：512 / 384
  ⇒ 目的不是"最好看"，是找【可接受质量下的最低成本】。

⚠️ 纪律：单点计时不可信（本项目实测 12.7% 时间漂移）⇒ 每格跑 2 次取较快值，
   并记录线程数。
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
import argparse, os, sys, time, json
import torch

PROMPT = ("1girl, solo, upper body, looking at viewer, detailed face, blue eyes, "
          "silver hair, school uniform, soft lighting, best quality")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=r'D:\work\textmodel\counterfeit-v3-diffusers')
    ap.add_argument('--outdir', default=r'D:\work\cloud_results\sd_speed')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--seed', type=int, default=999)
    ap.add_argument('--reps', type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.outdir, exist_ok=True)

    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from PIL import Image

    t0 = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        a.model, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    pipe.to('cpu')
    print('[sd] 载入 %.1fs  threads=%d' % (time.perf_counter() - t0, a.threads), flush=True)

    rows, tiles = [], []
    for size in (512, 384):
        for steps in (20, 12, 8, 4):
            ts, img = [], None
            for r in range(a.reps):
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                g = torch.Generator(device='cpu').manual_seed(a.seed)
                t1 = time.perf_counter()
                with torch.inference_mode():
                    img = pipe(PROMPT, num_inference_steps=steps, guidance_scale=7.0,
                               width=size, height=size, generator=g).images[0]
                ts.append(time.perf_counter() - t1)
            best = min(ts)
            p = os.path.join(a.outdir, 's%d_st%02d.png' % (size, steps))
            img.save(p)
            tiles.append((size, steps, img))
            rows.append(dict(size=size, steps=steps, best_s=round(best, 1),
                             all_s=[round(x, 1) for x in ts],
                             s_per_img_1000=round(best * 1000 / 3600, 1)))
            print('  %dx%d  %2d 步: %6.1f s  (1000 张 = %5.1f h)'
                  % (size, size, steps, best, best * 1000 / 3600), flush=True)

    # 拼一张对照图（4 列 = 步数，2 行 = 分辨率）
    W = max(t[2].size[0] for t in tiles)
    H = max(t[2].size[1] for t in tiles)
    cols, rws = 4, 2
    canvas = Image.new('RGB', (cols * W, rws * H), (20, 20, 20))
    from PIL import ImageDraw
    dr = ImageDraw.Draw(canvas)
    for i, (size, steps, img) in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas.paste(img, (c * W, r * H))
        dr.text((c * W + 6, r * H + 6), '%dpx / %d steps' % (size, steps), fill=(255, 255, 0))
    gp = os.path.join(a.outdir, 'MATRIX.png')
    canvas.save(gp)
    json.dump(rows, open(os.path.join(a.outdir, 'matrix.json'), 'w'), indent=1)
    print('[sd] -> %s' % gp)
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == '__main__':
    main()
