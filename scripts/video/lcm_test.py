# -*- coding: utf-8 -*-
"""lcm_test.py — 用 LCM-LoRA 把 SDXL 的步数从 15 压到 4~6，量真实提速。

依据：profile 显示 80% 时间在 FeedForward 的纯计算，且 fp32/4bit 同速
     ⇒ 提速只能靠【减少前向次数】。LCM 把 15 步压到 4~6 步 ⇒ 理论 2.5~3.8×。

要点：
  · LCM-LoRA 必须配 `LCMScheduler`
  · 且 `guidance_scale` 要低（1.0~1.5），CFG 在少步下会毁图
  · LoRA 可 fuse 进权重（推理更快）—— 注意要在 4bit 量化【之前或之后】都能用，
    这里用 diffusers 的标准 load_lora_weights + fuse_lora
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
import argparse, gc, os, sys, time, glob
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
B = r'D:\work\ms_cache\models\AI-ModelScope--stable-diffusion-xl-base-1.0\snapshots\v1.0.9'
LCM = glob.glob(r'D:\work\ms_cache\models\AI-ModelScope--lcm-lora-sdxl\snapshots\*\pytorch_lora_weights.safetensors')


def replace_linear(module):
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
            n += replace_linear(child)
    return n


def upcast(module):
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
    for m in module.modules():
        if isinstance(m, (Linear4bit, Linear8bitLt)):
            continue
        for p in list(m.parameters(recurse=False)):
            if p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.float()
        for b in list(m.buffers(recurse=False)):
            if b.dtype in (torch.float16, torch.bfloat16):
                b.data = b.data.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=r'D:\work\cloud_results\LCM_TEST.png')
    ap.add_argument('--size', type=int, default=768)
    ap.add_argument('--steps-list', default='4,6,15')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--no-lora', action='store_true', help='对照：不加 LCM-LoRA')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from PIL import Image
    from diffusers import (StableDiffusionXLPipeline, UNet2DConditionModel,
                           AutoencoderKL, LCMScheduler, DPMSolverMultistepScheduler)
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from diffusers.utils import logging as dlog
    dlog.set_verbosity_error()

    print('=== 加载 UNet (fp16, 先不量化) ===', flush=True)
    V = dict(variant='fp16')
    unet = UNet2DConditionModel.from_pretrained(os.path.join(B, 'unet'),
                                                torch_dtype=torch.float16, **V)

    pipe = StableDiffusionXLPipeline(
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

    # ★ 顺序很关键：LoRA 必须先 fuse 进 fp16 的 Linear，**之后**才能量化。
    #   反过来的话 LoRA 会去改 4-bit 层的权重（打包成 uint8），报
    #   "The size of tensor a (204800) must match the size of tensor b (320)"。
    if not a.no_lora:
        assert LCM, 'LCM-LoRA 未找到'
        print('=== 载入并 fuse LCM-LoRA（在量化之前）===', flush=True)
        print('  %s' % LCM[0], flush=True)
        pipe.load_lora_weights(LCM[0], adapter_name='lcm')
        pipe.fuse_lora(adapter_names=['lcm'])
        from diffusers import LCMScheduler as _L
        pipe.scheduler = _L.from_config(pipe.scheduler.config)
        print('  scheduler=%s' % type(pipe.scheduler).__name__, flush=True)

    print('=== 量化 UNet -> 4bit（LoRA 已融进 fp16 权重）===', flush=True)
    n = replace_linear(unet); gc.collect(); upcast(unet); gc.collect()
    print('  %d 层 -> 4bit' % n, flush=True)

    PROMPT = ('a photograph of a real person, an ornate cosplay costume, long silver hair, '
              'close-up portrait, in a bamboo forest, detailed face, natural lighting, '
              'sharp focus, high quality photo')
    NEG = 'blurry, lowres, jpeg artifacts, bad anatomy, painting, illustration, cartoon'
    steps_list = [int(s) for s in a.steps_list.split(',')]
    ims, meta = [], []
    for st in steps_list:
        gs = 1.0 if not a.no_lora else 7.0
        t0 = time.perf_counter()
        with torch.inference_mode():
            im = pipe(prompt=PROMPT, negative_prompt=NEG, num_inference_steps=st,
                      guidance_scale=gs, width=a.size, height=a.size,
                      generator=torch.Generator(device='cpu').manual_seed(2024)).images[0]
        el = time.perf_counter() - t0
        ims.append(im)
        print('  steps=%-3d gs=%-4.1f  %.1f s  (%.1f s/步)' % (st, gs, el, el / st), flush=True)
        meta.append((st, el))
    w = a.size
    cv = Image.new('RGB', (w * len(ims), w), (20, 20, 20))
    for i, im in enumerate(ims):
        cv.paste(im, (i * w, 0))
    cv.save(a.out)
    print('=== -> %s ===' % a.out)
    print('  提速：15 步 %.1f s -> %d 步 %.1f s = %.2fx'
          % (meta[-1][1], meta[0][0], meta[0][1], meta[-1][1] / max(1e-9, meta[0][1])))


if __name__ == '__main__':
    main()
