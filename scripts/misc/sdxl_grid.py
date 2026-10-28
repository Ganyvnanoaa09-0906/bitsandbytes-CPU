# -*- coding: utf-8 -*-
"""sdxl_grid.py — 用 SDXL 4-bit 出一组【写实 cos 图】，用提示词换构图多样性。

结论链（都由本轮实测支撑）：
  · SDXL 4-bit txt2img @768² = 本机最好的写实质量（达用户参考图档次）
  · 但同一 prompt 出 16 张会塌成几乎同一构图（对照实验已证）
  · AR 能提供构图多样性，但它的空间结构太弱、会拖累 SDXL（整合实验已证）
  ⇒ 所以用【提示词多样性】换构图多样性 —— 同时每张图自带【精确标签】（合成数据路线）

产出：
  <out>.png        网格
  <out>.jsonl      每格一行：{index, prompt, seed, steps, size} —— 可直接当完美 caption
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
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import argparse, gc, json, os, sys, time
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
B = r'D:\work\ms_cache\models\AI-ModelScope--stable-diffusion-xl-base-1.0\snapshots\v1.0.9'

# 拆成可组合的属性，保证每次组合自洽（对应"自带精确标签"）
SHOTS = ['close-up portrait', 'upper body shot', 'full body shot', 'wide shot with scenery']
HAIRS = ['long silver hair', 'long black hair', 'short pink hair', 'twin-tail purple hair']
OUTFITS = ['an ornate cosplay costume', 'a gothic lolita dress', 'a futuristic bodysuit',
           'a traditional hanfu']
SCENES = ['in a sunlit room', 'in a bamboo forest', 'on a rooftop at dusk',
          'in a neon-lit street']
EXTRA = ['holding a prop weapon', 'with hands raised', 'looking at the camera',
         'in a dynamic pose']

NEG = ('blurry, lowres, jpeg artifacts, bad anatomy, painting, illustration, cartoon, '
       'extra limbs, deformed hands')


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


def upcast_rest(module):
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
    n = 0
    for m in module.modules():
        if isinstance(m, (Linear4bit, Linear8bitLt)):
            continue
        for p in list(m.parameters(recurse=False)):
            if p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.float(); n += 1
        for b in list(m.buffers(recurse=False)):
            if b.dtype in (torch.float16, torch.bfloat16):
                b.data = b.data.float(); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--size', type=int, default=768)
    ap.add_argument('--steps', type=int, default=15)
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--lcm', action='store_true',
                    help='挂 LCM-LoRA + LCMScheduler（guidance=1.0，无 CFG）⇒ 4~6 步可用')
    ap.add_argument('--lora', default=None, help='LCM-LoRA 路径（默认自动找 ms_cache）')
    ap.add_argument('--threads', type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    from PIL import Image
    from diffusers import (StableDiffusionXLPipeline, UNet2DConditionModel,
                           AutoencoderKL, DPMSolverMultistepScheduler)
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from diffusers.utils import logging as dlog
    dlog.set_verbosity_error()

    print('=== 量化 UNet -> 4bit ===', flush=True)
    t0 = time.perf_counter()
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

    # ★ LCM-LoRA 必须在【量化之前】fuse（否则会去改打包成 uint8 的 4-bit 权重而报错）
    if a.lcm:
        import glob as _g
        lp = a.lora or (_g.glob(r'D:\work\ms_cache\models\AI-ModelScope--lcm-lora-sdxl'
                                r'\snapshots\*\pytorch_lora_weights.safetensors') or [None])[0]
        assert lp, 'LCM-LoRA 未找到'
        print('=== fuse LCM-LoRA: %s ===' % lp, flush=True)
        pipe.load_lora_weights(lp, adapter_name='lcm')
        pipe.fuse_lora(adapter_names=['lcm'])
        from diffusers import LCMScheduler as _L
        pipe.scheduler = _L.from_config(pipe.scheduler.config)

    n = replace_linear(unet); gc.collect(); nu = upcast_rest(unet); gc.collect()
    print('  %d 层量化 / %d 张量升级, %.0f s' % (n, nu, time.perf_counter() - t0), flush=True)

    side = 2 if a.n <= 4 else 4
    out = Image.new('RGB', (a.size * side, a.size * ((a.n + side - 1) // side)), (20, 20, 20))
    meta, t0 = [], time.perf_counter()
    for i in range(a.n):
        pr = ('a photograph of a real person, a woman wearing %s, %s hair, %s, %s, %s, '
              'detailed face, natural lighting, sharp focus, high quality photo'
              % (OUTFITS[i % 4], HAIRS[i % 4].replace(' hair', ''),
                 SHOTS[(i // 1) % 4], SCENES[i % 4], EXTRA[(i * 3) % 4]))
        pr = ('a photograph of a real person, ' + OUTFITS[i % 4] + ', ' + HAIRS[i % 4] + ', '
              + SHOTS[(i * 3) % 4] + ', ' + SCENES[(i * 2 + 1) % 4] + ', ' + EXTRA[i % 4]
              + ', detailed face, natural lighting, sharp focus, high quality photo')
        sd = a.seed + i * 7
        gs = 1.0 if a.lcm else 7.0
        with torch.inference_mode():
            im = pipe(prompt=pr, negative_prompt=NEG, num_inference_steps=a.steps,
                      guidance_scale=gs, width=a.size, height=a.size,
                      generator=torch.Generator(device='cpu').manual_seed(sd)).images[0]
        r, c = divmod(i, side)
        out.paste(im, (c * a.size, r * a.size))
        meta.append(dict(index=i, path='%s#%d' % (a.out, i), prompt=pr,
                         seed=sd, steps=a.steps, size=a.size, model='SDXL-base-1.0-cpu4bit'))
        el = time.perf_counter() - t0
        print('  %d/%d  %.0f s/张  已用 %.1f 分钟' % (i + 1, a.n, el / (i + 1), el / 60), flush=True)
    out.save(a.out)
    with open(a.out.replace('.png', '.jsonl'), 'w', encoding='utf-8') as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
    print('=== -> %s  和 %s ===' % (a.out, a.out.replace('.png', '.jsonl')))


if __name__ == '__main__':
    main()
