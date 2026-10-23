"""Wan2.1 true-prompt multi-step video smoke: save frames/GIF/MP4 + stats."""
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


def frame_stats(a):
    x = a.astype(np.float32) / 255.0
    lap = (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:] - 4 * x[1:-1, 1:-1])
    return float(x.std()), float(lap.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['fp32', 'nf4'], required=True)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--frames', type=int, default=9, help='latent T; VAE output ~(T-1)*4+1')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--outdir', default='')
    args = ap.parse_args()
    torch.set_num_threads(6)
    text = torch.load(r'D:\work\textmodel\Wan2.1-T2V-1.3B\prompt_cat_hidden.pt',
                      map_location='cpu').float()
    torch.manual_seed(1234)
    latent_t = max(1, (args.frames - 1) // 4 + 1)
    hw = args.size // 8
    latent = torch.randn(1, 16, latent_t, hw, hw)
    model = load_wan_transformer()
    n = 0
    if args.backend == 'nf4':
        n = quantize_linears(model)
    vae = VP.load_decoder()
    from diffusers import FlowMatchEulerDiscreteScheduler
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=5.0)
    sched.set_timesteps(args.steps)
    t0 = time.perf_counter()
    lat = latent
    step_times = []
    with torch.no_grad():
        for t in sched.timesteps:
            s = time.perf_counter()
            pred = model(lat, timestep=t.unsqueeze(0), encoder_hidden_states=text,
                         return_dict=False)[0]
            lat = sched.step(pred, t, lat).prev_sample
            step_times.append(time.perf_counter() - s)
    fwd = time.perf_counter() - t0
    z = int(vae.config.z_dim)
    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1)
    std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1)
    t0 = time.perf_counter()
    with torch.no_grad():
        video = vae.decode(lat / std + mean, return_dict=False)[0]
    dec = time.perf_counter() - t0
    vid = ((video[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 3, 0).numpy()
    outdir = args.outdir or os.path.join(r'D:\work\research\bench_results',
                                         'wan21_video_%s_%dstep_%dframes' % (args.backend, args.steps, args.frames))
    os.makedirs(outdir, exist_ok=True)
    imgs = [Image.fromarray(vid[i]) for i in range(vid.shape[0])]
    for i, im in enumerate(imgs):
        im.save(os.path.join(outdir, 'frame_%03d.png' % i))
    imgs[0].save(os.path.join(outdir, 'anim.gif'), save_all=True, append_images=imgs[1:], duration=125, loop=0)
    try:
        import imageio.v2 as imageio
        imageio.mimsave(os.path.join(outdir, 'anim.mp4'), vid, fps=8)
    except Exception as e:
        print('mp4 skipped:', type(e).__name__, str(e)[:120], flush=True)
    stats = [frame_stats(vid[i]) for i in range(vid.shape[0])]
    res = dict(backend=args.backend, steps=args.steps, latent=tuple(latent.shape),
               out=tuple(video.shape), quantized=n, fwd_s=fwd, step_s=float(np.mean(step_times)),
               vae_s=dec, rss_gb=rss_gb(),
               frame_std=float(np.mean([s[0] for s in stats])),
               lap_std=float(np.mean([s[1] for s in stats])),
               outdir=outdir, gif=os.path.join(outdir, 'anim.gif'))
    print('RESULT ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
