"""Assemble official diffusers WanPipeline with converted components + cached prompt embeds."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import psutil
import torch
from PIL import Image

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
import wan_vae_profile as VP
from wan21_convert import load_wan_transformer
from wan21_4bit import quantize_linears


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['fp32', 'nf4'], required=True)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--frames', type=int, default=9)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--guidance', type=float, default=5.0)
    ap.add_argument('--shift', type=float, default=5.0)
    ap.add_argument('--outdir', default='')
    args = ap.parse_args()
    torch.set_num_threads(6)
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanPipeline

    text = torch.load(r'D:\work\textmodel\Wan2.1-T2V-1.3B\prompt_cat_hidden.pt',
                      map_location='cpu').float()
    neg = torch.load(r'D:\work\textmodel\Wan2.1-T2V-1.3B\prompt_empty_hidden.pt', map_location='cpu').float()
    vae = VP.load_decoder()
    transformer = load_wan_transformer()
    n = 0
    if args.backend == 'nf4':
        n = quantize_linears(transformer)
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.shift)
    pipe = WanPipeline(vae=vae, text_encoder=None, tokenizer=None,
                       transformer=transformer, scheduler=sched)
    generator = torch.Generator(device='cpu').manual_seed(1234)
    t0 = time.perf_counter()
    out = pipe(prompt_embeds=text, negative_prompt_embeds=neg,
               num_inference_steps=args.steps, height=args.size, width=args.size,
               num_frames=args.frames, guidance_scale=args.guidance,
               generator=generator, output_type='np')
    total = time.perf_counter() - t0
    frames = out.frames[0] if hasattr(out, 'frames') else out[0]
    if isinstance(frames, np.ndarray):
        vid = frames
    else:
        vid = np.asarray(frames)
    if vid.ndim == 4 and vid.shape[-1] == 3:
        if np.issubdtype(vid.dtype, np.floating) and float(np.nanmax(vid)) <= 1.5:
            vids = (vid * 255.0).clip(0, 255).round().astype(np.uint8)
        else:
            vids = vid.astype(np.uint8)
    else:
        vids = (vid * 255).clip(0, 255).astype(np.uint8)
    outdir = args.outdir or os.path.join(r'D:\work\research\bench_results',
                                         'wan21_official_%s_%dstep_%df' % (args.backend, args.steps, vids.shape[0]))
    os.makedirs(outdir, exist_ok=True)
    imgs = [Image.fromarray(vids[i]) for i in range(vids.shape[0])]
    for i, im in enumerate(imgs[:3]):
        im.save(os.path.join(outdir, 'frame_%03d.png' % i))
    imgs[0].save(os.path.join(outdir, 'anim.gif'), save_all=True, append_images=imgs[1:],
                 duration=1000 // 8, loop=0)
    stats = [(float((np.asarray(im.convert('L'), np.float32) / 255).std()),
              float((np.asarray(im.convert('L'), np.float32) / 255).std())) for im in imgs]
    res = dict(backend=args.backend, quantized=n, steps=args.steps, frames=len(imgs),
               total_s=total, rss_gb=rss_gb(), outdir=outdir,
               gif=os.path.join(outdir, 'anim.gif'),
               mean_std=float(np.mean([s[0] for s in stats])))
    print('RESULT ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
