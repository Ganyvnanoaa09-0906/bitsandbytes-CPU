# -*- coding: utf-8 -*-
"""sd_refine_grid.py — 把整个 4x4 AR 采样网格逐格过 SD img2img，输出精修后的网格。

这是"AR 当 layout prior + 细节模型"这条路的成图验证：
  列/行必须各自跟随【对应那一格】的构图，而不是全都收敛到同一张通用图 ——
  否则就不是"AR 控构图"，而是"SD 自己生成"。

用法:
  python sd_refine_grid.py --grid <AR采样网格.png> --out <精修网格.png> --strength 0.35
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse, os, sys, time

# ★★ 2026-09-25：必须在 import diffusers/peft 之前把【真正的 bitsandbytes 包】放上 sys.path。
#   背景（实测）：本项目的 bitsandbytes 是「仓库根 = D:\work\bitsandbytes-CPU\bitsandbytes」，
#   真正的 Python 包在它下面的 bitsandbytes\bitsandbytes\（有 __init__.py）。
#   而从 D:\work\bitsandbytes-CPU 这个 cwd 跑脚本时，Python 找到的是【仓库根那个没有 __init__.py
#   的目录】⇒ 拿到一个空的命名空间包（__file__=None、没有 nn、没有 optim）。
#   后果：peft 的 `is_bnb_4bit_available()` 里 `hasattr(bnb.nn, "Linear4bit")` 直接抛
#   AttributeError（peft 自身的写法问题：应先判 bnb.nn 是否存在），于是
#   `pipe.load_lora_weights()` 崩掉 —— 而这一步正是 LoRA 后训练增益 A/B 的必经之路。
_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bitsandbytes')
if os.path.isdir(os.path.join(_REPO_ROOT, 'bitsandbytes')) and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from PIL import Image

MODEL = r'D:\work\textmodel\counterfeit-v3-diffusers'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--strength', type=float, default=0.35)
    ap.add_argument('--steps', type=int, default=18)
    ap.add_argument('--prompt', default='1girl, anime illustration, detailed, best quality')
    ap.add_argument('--negative', default='lowres, bad anatomy, blurry, jpeg artifacts')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--side', type=int, default=4)
    ap.add_argument('--scale', type=int, default=1, help='输出每格放大倍数')
    # ★ 后训练增益的 A/B 入口（2026-09-25）：不传 --lora 就是报告 §10.35 那条基线
    #   （AR@k1024 + counterfeit-v3 精修 = 距真实 2.52 dB），传了就是微调后的版本。
    #   ⚠️ 报告 §10.37④ 的坑：LoRA 必须在【量化之前】fuse。这里全程 fp32、不量化，所以无此问题。
    ap.add_argument('--lora', default='', help='LoRA 适配器目录（如 _lora_cos15/final）；空=用底模')
    ap.add_argument('--lora-scale', type=float, default=1.0)
    # 换 VAE。动机：精修产出上反复出现【绿色伪影】，先前两次都把它归因于
    # "底模自己的 VAE"（报告 §6.8），但从来没真的换过一个 VAE 去验。
    # 本地就有原版 SD1.5 的 VAE（textmodel\sd15_full\vae）可作对照。
    ap.add_argument('--vae', default='', help='替换 VAE 目录；空=用底模自带的（基线路径不变）')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    if a.lora:
        print('[grid] 加载 LoRA: %s (scale=%.2f)' % (a.lora, a.lora_scale), flush=True)
        try:
            pipe.load_lora_weights(a.lora)
        except Exception as e:
            raise SystemExit('[grid] !! load_lora_weights 失败: %s: %s'
                             % (type(e).__name__, str(e)[:300]))
        try:
            pipe.fuse_lora(lora_scale=a.lora_scale)
            print('[grid] 已 fuse 进权重', flush=True)
        except Exception as e:
            print('[grid] fuse 失败(%s)，LoRA 以运行时适配器生效' % str(e)[:80], flush=True)
        # 生效性自检：权重里应当出现 lora 痕迹，否则就是"加载了但没接线"
        try:
            n_up = [k for k in pipe.unet.state_dict() if 'lora' in k.lower()]
            print('[grid] unet 里 lora 相关张量数 = %d %s'
                  % (len(n_up), '(已 fuse ⇒ 0 是正常的)' if not n_up else ''), flush=True)
        except Exception:
            pass

    if a.vae:
        from diffusers import AutoencoderKL
        print('[grid] 换 VAE: %s' % a.vae, flush=True)
        try:
            pipe.vae = AutoencoderKL.from_pretrained(a.vae, torch_dtype=torch.float32)
        except Exception as e:
            raise SystemExit('[grid] !! 换 VAE 失败: %s: %s' % (type(e).__name__, str(e)[:300]))

    im = Image.open(a.grid).convert('RGB')
    W, H = im.size
    side = a.side
    tw, th = W // side, H // side
    print('[grid] %dx%d  tile %dx%d  strength=%.2f steps=%d'
          % (W, H, tw, th, a.strength, a.steps), flush=True)

    out_tiles = []
    t0 = time.time()
    for i in range(side * side):
        r, c = divmod(i, side)
        src = im.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
        g = torch.Generator().manual_seed(1000 + i)      # 每格独立种子，避免同质
        res = pipe(prompt=a.prompt, negative_prompt=a.negative, image=src,
                   strength=a.strength, num_inference_steps=a.steps,
                   guidance_scale=5.0, generator=g).images[0]
        out_tiles.append(res)
        print('[grid] %2d/%d  %.0fs' % (i + 1, side * side, time.time() - t0), flush=True)

    sc = a.scale
    rows = []
    for r in range(side):
        row = [np.asarray(out_tiles[r * side + c].resize((tw * sc, th * sc), Image.LANCZOS))
               for c in range(side)]
        rows.append(np.concatenate(row, axis=1))
    out = np.concatenate(rows, axis=0)
    Image.fromarray(out).save(a.out)
    print('[grid] -> %s  %dx%d  总计 %.0fs' % (a.out, out.shape[1], out.shape[0], time.time() - t0))


if __name__ == '__main__':
    main()
