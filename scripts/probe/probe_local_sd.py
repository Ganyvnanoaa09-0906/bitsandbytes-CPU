# -*- coding: utf-8 -*-
"""probe_local_sd.py — 实测本地现有的生图模型，看能不能出【可辨认的人】而不是色块/鬼图。

背景：本项目自研的 VQ-VAE + AR 自由采样产出的是"无主体的抽象色块"（已实测），
      所以需要一个能稳定出人的参照。本地扫到：
        · counterfeit-v3-diffusers  —— SD1.5 动漫微调（完整 diffusers 管线）
        · sd15_base                 —— SD1.5 原版
        · controlnet_openpose       —— 姿态控制（配合 SD 用）
      本脚本对每个可用模型实际跑一次 txt2img，存图 + 报耗时。

⚠️ 注意：本机无 GPU，全 CPU 推理，速度慢是正常的；先测 1 张探路。
"""
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
import argparse, os, sys, time
import torch

PROMPTS = [
    # counterfeit-v3 是动漫向模型，用 danbooru 风格的 tag 提示
    "1girl, solo, upper body, looking at viewer, detailed face, blue eyes, "
    "silver hair, school uniform, soft lighting, best quality",
    "1girl, solo, full body, standing, long hair, white dress, garden, "
    "daylight, detailed, best quality",
]


def grid(imgs, path):
    from PIL import Image
    n = len(imgs)
    W, H = imgs[0].size
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    canvas = Image.new('RGB', (cols * W, rows * H), (20, 20, 20))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas.paste(im, (c * W, r * H))
    canvas.save(path)
    return path


def run_diffusers(path, prompt, steps, seed, size, out):
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    t0 = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        path, torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    pipe.to('cpu')
    load_s = time.perf_counter() - t0

    g = torch.Generator(device='cpu').manual_seed(seed)
    t1 = time.perf_counter()
    with torch.no_grad():
        im = pipe(prompt, num_inference_steps=steps, guidance_scale=7.0,
                  width=size, height=size, generator=g).images[0]
    gen_s = time.perf_counter() - t1
    im.save(out)
    return load_s, gen_s, im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--outdir', default=r'D:\work\cloud_results')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    cands = [
        (r'D:\work\textmodel\counterfeit-v3-diffusers', 'counterfeit-v3'),
        (r'D:\work\textmodel\sd15_base', 'sd15_base'),
    ]
    os.makedirs(a.outdir, exist_ok=True)
    made = []
    for path, name in cands:
        if not os.path.exists(os.path.join(path, 'model_index.json')):
            print('[skip] %s 没有 model_index.json' % name)
            continue
        print('=== %s ===' % name, flush=True)
        ims = []
        for i, pr in enumerate(PROMPTS):
            out = os.path.join(a.outdir, 'LOCAL_SD_%s_%d.png' % (name, i))
            try:
                ls, gs, im = run_diffusers(path, pr, a.steps, a.seed + i, a.size, out)
                print('  [%d] 载入 %.1fs  生成 %.1fs (%dx%d, %d 步)  -> %s'
                      % (i, ls, gs, a.size, a.size, a.steps, os.path.basename(out)), flush=True)
                ims.append(im)
            except Exception as e:
                print('  [%d] FAILED: %s' % (i, str(e)[:200]), flush=True)
        if ims:
            made.append(grid(ims, os.path.join(a.outdir, 'LOCAL_SD_%s_GRID.png' % name)))
    print()
    for m in made:
        print('  -> %s' % m)


if __name__ == '__main__':
    main()
