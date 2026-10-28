# -*- coding: utf-8 -*-
"""pipeline_ar_sdxl.py — 最终整合：AR 构图 + SDXL(4-bit) 写实精修。

两条线的合并（都由本轮实测支撑）：
  · AR@k1024  —— 自由采样距天花板仅 0.52 dB，且提供【构图多样性】
                 （纯 SD 对照实测：16 张塌成同一个正面特写）
  · SDXL 4-bit —— 768²×20 步产出写实 cos 人像，质量达参考图档次
                 内存 7 GB（fp32 要 14 GB，本机放不下）

用法：
  python pipeline_ar_sdxl.py --grid <AR网格.png> --out <输出.png> --tiles 4   # 探路
  python pipeline_ar_sdxl.py --grid <...> --out <...>                          # 全量
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
import argparse, gc, os, sys, time
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
B = r'D:\work\ms_cache\models\AI-ModelScope--stable-diffusion-xl-base-1.0\snapshots\v1.0.9'


def mem():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9, psutil.virtual_memory().available / 1e9
    except Exception:
        return -1.0, -1.0


def replace_linear(module, prefix=''):
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            q = Linear4bit(child.in_features, child.out_features,
                           bias=child.bias is not None, compute_dtype=torch.float32,
                           quant_type='nf4', compress_statistics=True)
            with torch.no_grad():
                q.weight = bnb.nn.Params4bit(child.weight.data.float(), requires_grad=False,
                                             quant_type='nf4', compress_statistics=True)
                q.weight = q.weight.to('cpu')
                if child.bias is not None:
                    q.bias.data.copy_(child.bias.data.float())
            setattr(module, name, q)
            n += 1
        else:
            n += replace_linear(child, prefix + '.' + name)
    return n


def upcast_rest(module):
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
    n = 0
    for m in module.modules():
        if isinstance(m, (Linear4bit, Linear8bitLt)):
            continue
        for p in list(m.parameters(recurse=False)):
            if p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.float()
                n += 1
        for b in list(m.buffers(recurse=False)):
            if b.dtype in (torch.float16, torch.bfloat16):
                b.data = b.data.float()
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', type=int, default=768, help='img2img 工作分辨率（SDXL 在 1024² 训）')
    ap.add_argument('--strength', type=float, default=0.5)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--tiles', type=int, default=0)
    ap.add_argument('--prompt', default=('a photograph of a real person, a woman wearing a '
                                         'detailed cosplay costume, long hair, detailed face, '
                                         'natural lighting, sharp focus, high quality photo'))
    ap.add_argument('--negative', default=('blurry, lowres, jpeg artifacts, bad anatomy, '
                                           'painting, illustration, cartoon, extra limbs'))
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from PIL import Image
    from diffusers import (StableDiffusionXLImg2ImgPipeline, UNet2DConditionModel,
                           AutoencoderKL, DPMSolverMultistepScheduler)
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from diffusers.utils import logging as dlog
    dlog.set_verbosity_error()

    V = dict(variant='fp16')
    print('=== ① UNet fp16 加载 -> 4bit 量化 -> conv 升 fp32 ===', flush=True)
    t0 = time.perf_counter()
    unet = UNet2DConditionModel.from_pretrained(os.path.join(B, 'unet'), torch_dtype=torch.float16,
                                                use_safetensors=True, **V)
    n = replace_linear(unet)
    gc.collect()
    nu = upcast_rest(unet)
    gc.collect()
    print('  量化 %d 层, 升级 %d 张量, %.1f s  RSS=%.2f GB 可用=%.2f GB'
          % (n, nu, time.perf_counter() - t0, *mem()), flush=True)

    print('=== ② 组 img2img 管线 ===', flush=True)
    pipe = StableDiffusionXLImg2ImgPipeline(
        vae=AutoencoderKL.from_pretrained(os.path.join(B, 'vae'), torch_dtype=torch.float32, **V),
        text_encoder=CLIPTextModel.from_pretrained(os.path.join(B, 'text_encoder'),
                                                   torch_dtype=torch.float32, **V),
        text_encoder_2=CLIPTextModelWithProjection.from_pretrained(
            os.path.join(B, 'text_encoder_2'), torch_dtype=torch.float32, **V),
        tokenizer=CLIPTokenizer.from_pretrained(os.path.join(B, 'tokenizer')),
        tokenizer_2=CLIPTokenizer.from_pretrained(os.path.join(B, 'tokenizer_2')),
        unet=unet,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(os.path.join(B, 'scheduler')),
    )
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    print('  就绪 RSS=%.2f GB' % mem()[0], flush=True)

    g = Image.open(a.grid).convert('RGB')
    W, H = g.size
    side = 4
    tw, th = W // side, H // side
    ntot = a.tiles if a.tiles else side * side
    rows = (ntot + side - 1) // side
    # ★ 按【工作分辨率】保存，不要缩回 256 —— 否则细节全丢，看图会误判（我已经因此误判过一次）
    ow = a.work
    out = Image.new('RGB', (ow * side, ow * rows), (20, 20, 20))
    t0 = time.perf_counter()
    for i in range(ntot):
        r, c = divmod(i, side)
        tile = g.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th))
        src = tile.resize((a.work, a.work), Image.LANCZOS)
        with torch.inference_mode():
            im = pipe(prompt=a.prompt, negative_prompt=a.negative, image=src,
                      strength=a.strength, num_inference_steps=a.steps, guidance_scale=7.0,
                      generator=torch.Generator(device='cpu').manual_seed(1000 + i)).images[0]
        out.paste(im, (c * ow, r * ow))
        if (i + 1) % 2 == 0 or i + 1 == ntot:
            el = time.perf_counter() - t0
            print('  %d/%d  %.0f s/格  已用 %.1f 分钟  RSS=%.2f GB'
                  % (i + 1, ntot, el / (i + 1), el / 60, mem()[0]), flush=True)
    out.save(a.out)
    print('=== -> %s  %s  (%.0f s) ===' % (a.out, out.size, time.perf_counter() - t0))


if __name__ == '__main__':
    main()
