# -*- coding: utf-8 -*-
"""gen_dataset.py — 用本地 SD 生成合成数据集，自带【精确无噪声】的标签。

为什么这件事同时解决两个问题（实测依据）：
  ① 内容：本项目训练数据是"真人照 + 动漫插画"混合（caption 里两种风格标签都有），
     两个本地模型正好各覆盖一半 —— counterfeit-v3 出动漫，拼装的 SD1.5 出写实真人。
  ② 标签：我们的 caption 只有 66 个 tag、信息上界仅 42 bit/图、
     **且自相矛盾**（"two people" 87% 与 "solo"/"one person" 79% 共存；
     "anime illustration" 90% 与 "photograph of a real person" 49% 共存 ⇒ 139% > 100%）。
     ⇒ 标签不可靠，条件信号被噪声压住。
     **而合成数据用的是【我们自己写的提示词】⇒ 标签精确、无矛盾、无噪声。**

⚠️ 提示词不用那 66 个 tag，而是【按属性组合合成】，保证每次组合都自洽。

产出：
  <outdir>/images/xxxxx.png      图像
  <outdir>/meta.jsonl            一行一条：{path, prompt, style, attrs, seed, steps, size, model}
  ⇒ 这份 meta 可以直接当完美 caption 用（比现有 captions_merged.jsonl 干净得多）
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
import argparse, itertools, json, os, random, sys, time
import torch

# ---- 提示词属性表（每一项内部互斥、彼此组合自洽）----
ANIME = dict(
    prefix="1girl, solo, {shot}, {hair}, {outfit}, {scene}, {extra}, detailed face, "
           "soft lighting, best quality",
    shot=["close-up portrait", "upper body", "full body, standing", "wide shot with scenery"],
    hair=["long silver hair", "black twintails", "short pink hair", "long black hair"],
    outfit=["a maid outfit", "a school uniform", "a white dress", "a bunny suit"],
    scene=["indoors, in a studio with plain background", "on a stage", "in a garden",
           "in a city street"],
    extra=["wearing cat ears", "wearing a hat", "holding a prop", "looking at the viewer"],
)
REAL = dict(
    prefix="a photograph of a real person, a woman, {shot}, {hair}, {outfit}, {scene}, "
           "{extra}, natural lighting, sharp focus, detailed face, high quality photo",
    shot=["close-up portrait", "half body shot", "full body shot", "wide shot with scenery"],
    hair=["curly brown hair", "long black hair", "short blonde hair", "tied-back dark hair"],
    outfit=["wearing a casual dress", "wearing a cosplay costume", "wearing a hoodie",
            "wearing a formal suit"],
    scene=["standing indoors by a window", "outdoors in a park", "in a photo studio",
           "on a city street"],
    extra=["looking at the camera", "smiling gently", "in a dynamic pose", "with hands on hips"],
)


def build_prompts(styles, n, seed):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        st = rnd.choice(styles)
        spec = ANIME if st == 'anime' else REAL
        attrs = {k: rnd.choice(spec[k]) for k in ('shot', 'hair', 'outfit', 'scene', 'extra')}
        out.append(dict(style=st, attrs=attrs, prompt=spec['prefix'].format(**attrs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--styles', default='anime,real')
    ap.add_argument('--steps', type=int, default=8, help='8 步是质量/成本甜点（实测：4 步起色偏）')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=20260920)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--guidance', type=float, default=7.0)
    ap.add_argument('--meta-append', action='store_true',
                    help='向已有 meta.jsonl 追加（用于分批续跑）')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    styles = [s.strip() for s in a.styles.split(',') if s.strip()]
    imgdir = os.path.join(a.outdir, 'images')
    os.makedirs(imgdir, exist_ok=True)
    metap = os.path.join(a.outdir, 'meta.jsonl')

    from diffusers import (StableDiffusionPipeline, UNet2DConditionModel,
                           AutoencoderKL, DPMSolverMultistepScheduler)
    from transformers import CLIPTextModel, CLIPTokenizer
    from PIL import Image

    tm = r'D:\work\textmodel'
    cf = os.path.join(tm, 'counterfeit-v3-diffusers')

    print('[gen] 载入两个模型 ...', flush=True)
    anime = StableDiffusionPipeline.from_pretrained(
        cf, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
    # ---- 用本地零件拼出原版 SD1.5（真人向）----
    unet = UNet2DConditionModel.from_pretrained(os.path.join(tm, 'sd15_base', 'unet'),
                                                torch_dtype=torch.float32)
    real = StableDiffusionPipeline(
        vae=AutoencoderKL.from_pretrained(os.path.join(cf, 'vae'), torch_dtype=torch.float32),
        text_encoder=CLIPTextModel.from_pretrained(os.path.join(cf, 'text_encoder')),
        tokenizer=CLIPTokenizer.from_pretrained(os.path.join(cf, 'tokenizer')),
        unet=unet,
        scheduler=DPMSolverMultistepScheduler.from_config(
            {**json.load(open(os.path.join(cf, 'scheduler', 'scheduler_config.json'),
                              encoding='utf-8'))}),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    for p in (anime, real):
        p.set_progress_bar_config(disable=True)
        try:
            p.enable_attention_slicing()
        except Exception:
            pass
        p.to('cpu')
    pipes = {'anime': anime, 'real': real}
    print('[gen] 就绪 threads=%d  steps=%d  size=%d' % (a.threads, a.steps, a.size), flush=True)

    prompts = build_prompts(styles, a.n, a.seed)
    rows, t0, done = [], time.perf_counter(), 0
    for i, pr in enumerate(prompts):
        outp = os.path.join(imgdir, '%06d_%s.png' % (i, pr['style']))
        if os.path.exists(outp):
            continue
        sd = a.seed + i
        g = torch.Generator(device='cpu').manual_seed(sd)
        with torch.inference_mode():
            im = pipes[pr['style']](pr['prompt'], num_inference_steps=a.steps,
                                    guidance_scale=a.guidance,
                                    width=a.size, height=a.size, generator=g).images[0]
        im.save(outp)
        rows.append(dict(path=outp.replace('\\', '/'), prompt=pr['prompt'],
                         style=pr['style'], attrs=pr['attrs'], seed=sd,
                         steps=a.steps, size=a.size,
                         model='counterfeit-v3' if pr['style'] == 'anime' else 'sd15_base'))
        done += 1
        if done % 4 == 0 or done == len(prompts):
            el = time.perf_counter() - t0
            print('  %d/%d  %.1f s/张  已用 %.0f s  预计剩余 %.0f s'
                  % (done, len(prompts), el / done, el, el / done * (len(prompts) - done)),
                  flush=True)

    mode = 'a' if a.meta_append else 'w'
    with open(metap, mode, encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('[gen] 写入 %d 条 -> %s' % (len(rows), metap))
    print('[gen] 图像目录 -> %s' % imgdir)


if __name__ == '__main__':
    main()
