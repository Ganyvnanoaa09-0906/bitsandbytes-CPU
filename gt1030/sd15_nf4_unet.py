"""SD1.5 UNet 4-bit NF4 CPU inference: one denoise+VAE step vs fp32 baseline.

Usage:
  python sd15_nf4_unet.py --backend fp32 --size 512 --steps 1
  python sd15_nf4_unet.py --backend nf4  --size 512 --steps 1
"""
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

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def replace_linear_4bit(module, prefix=''):
    import bitsandbytes as bnb
    n = 0
    for name, child in list(module.named_children()):
        full = f'{prefix}.{name}' if prefix else name
        if isinstance(child, torch.nn.Linear):
            q = bnb.nn.Linear4bit(child.in_features, child.out_features,
                                  bias=child.bias is not None,
                                  compute_dtype=torch.float32,
                                  quant_type='nf4', compress_statistics=True)
            with torch.no_grad():
                q.weight = bnb.nn.Params4bit(child.weight.data.clone(),
                                             requires_grad=False,
                                             quant_type='nf4',
                                             compress_statistics=True)
                q.weight = q.weight.to('cpu')
                if child.bias is not None:
                    q.bias.data.copy_(child.bias.data)
            setattr(module, name, q)
            n += 1
        else:
            n += replace_linear_4bit(child, full)
    return n


def laplacian_std(img):
    a = np.asarray(img.convert('L'), dtype=np.float32) / 255.0
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1])
    return float(lap.std()), float(a.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=r'D:\work\textmodel\_sd15_combined')
    ap.add_argument('--backend', choices=['fp32', 'nf4'], required=True)
    ap.add_argument('--steps', type=int, default=1)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--prompt', default='a photo of a cat sitting on a chair')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    torch.set_num_threads(6)

    from diffusers import StableDiffusionPipeline
    t0 = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32,
                                                   safety_checker=None,
                                                   requires_safety_checker=False,
                                                   local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    load = time.perf_counter() - t0
    n = 0
    if args.backend == 'nf4':
        n = replace_linear_4bit(pipe.unet)
    log = '[%s] loaded %.1fs | quantized %d linears | rss %.2fGB'
    print(log % (args.backend, load, n, rss_gb()), flush=True)

    gen = torch.Generator(device='cpu').manual_seed(args.seed)
    t0 = time.perf_counter()
    image = pipe(prompt=args.prompt, num_inference_steps=args.steps, height=args.size,
                 width=args.size, generator=gen).images[0]
    run = time.perf_counter() - t0
    lap, std = laplacian_std(image)
    out = args.out or os.path.join(r'D:\work\research\bench_results',
                                   'sd15_unet_%s_%d_%dsteps.png' % (args.backend, args.size, args.steps))
    image.save(out)
    res = dict(backend=args.backend, size=args.size, steps=args.steps, load_s=load,
               run_s=run, s_per_step=run / args.steps, rss_gb=rss_gb(),
               lap_std=lap, gray_std=std, image=out, quantized_linears=n)
    print('RESULT ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
