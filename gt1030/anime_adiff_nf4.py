"""AnimateDiff motion adapter + NF4 UNet one-step video smoke (CPU)."""
from __future__ import annotations
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
from scripts.lib import anime_adiff
from sd15_nf4_unet import replace_linear_4bit



def replace_unet_nf4(module, skip_sub='motion_modules', prefix=''):
    import bitsandbytes as bnb
    n = 0
    for name, child in list(module.named_children()):
        full = f'{prefix}.{name}' if prefix else name
        if skip_sub in full:
            continue
        if isinstance(child, torch.nn.Linear):
            q = bnb.nn.Linear4bit(child.in_features, child.out_features,
                                  bias=child.bias is not None,
                                  compute_dtype=torch.float32,
                                  quant_type='nf4', compress_statistics=True)
            with torch.no_grad():
                q.weight = bnb.nn.Params4bit(child.weight.data.clone(), requires_grad=False,
                                             quant_type='nf4', compress_statistics=True)
                q.weight = q.weight.to('cpu')
                if child.bias is not None:
                    q.bias.data.copy_(child.bias.data)
            setattr(module, name, q)
            n += 1
        else:
            n += replace_unet_nf4(child, skip_sub, full)
    return n

def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def frame_stats(frames):
    a = np.stack([np.asarray(f.convert('L'), dtype=np.float32) / 255.0 for f in frames])
    lap = (a[:, :-2, 1:-1] + a[:, 2:, 1:-1] + a[:, 1:-1, :-2] + a[:, 1:-1, 2:]
           - 4 * a[:, 1:-1, 1:-1])
    return float(a.std()), float(lap.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['fp32', 'nf4'], required=True)
    ap.add_argument('--res', type=int, default=128)
    ap.add_argument('--frames', type=int, default=16)
    ap.add_argument('--steps', type=int, default=1)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    torch.set_num_threads(6)
    t0 = time.perf_counter()
    pipe = A.build('dpm')
    load = time.perf_counter() - t0
    n = 0
    if args.backend == 'nf4':
        n = replace_unet_nf4(pipe.unet)
    print('[%s] built %.1fs quantized %d rss %.2fGB' % (args.backend, load, n, rss_gb()), flush=True)
    g = torch.Generator().manual_seed(args.seed)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = pipe(prompt=A.ANIME_PROMPT, negative_prompt=A.ANIME_NEG,
                   num_frames=args.frames, num_inference_steps=args.steps,
                   guidance_scale=7.5, height=args.res, width=args.res, generator=g)
    run = time.perf_counter() - t0
    frames = [Image.fromarray(np.asarray(f)).convert('RGB') for f in out.frames[0]]
    std, lap = frame_stats(frames)
    outdir = args.out or os.path.join(r'D:\work\research\bench_results',
                                      'anime_nf4_%s_%d_%df_%ds' % (args.backend, args.res, args.frames, args.steps))
    os.makedirs(outdir, exist_ok=True)
    frames[0].save(os.path.join(outdir, 'frame0.png'))
    res = dict(backend=args.backend, res=args.res, frames=args.frames, steps=args.steps,
               load_s=load, run_s=run, s_per_step=run / args.steps, rss_gb=rss_gb(),
               quantized=n, frame_std=std, lap_std=lap, out=outdir)
    print('RESULT ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
