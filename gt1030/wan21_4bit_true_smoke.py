"""Real-prompt Wan T2V 1-step: fp32 vs NF4 DiT, same latent/text/seed; save PNG+stats."""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from scripts.wan import wan_vae_profile as VP
from wan21_convert import load_wan_transformer
from wan21_4bit import quantize_linears


def frame_stats(arr):
    a = arr.astype(np.float32) / 255.0
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1])
    return float(a.std()), float(lap.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['fp32', 'nf4'], required=True)
    ap.add_argument('--steps', type=int, default=1)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    torch.set_num_threads(6)
    torch.manual_seed(0)
    text = torch.load(r'D:\work\textmodel\Wan2.1-T2V-1.3B\prompt_cat_hidden.pt',
                      map_location='cpu').float()
    torch.manual_seed(1234)
    latent = torch.randn(1, 16, 1, 32, 32)
    model = load_wan_transformer()
    n = 0
    if args.backend == 'nf4':
        n = quantize_linears(model)
    vae = VP.load_decoder()
    from diffusers import FlowMatchEulerDiscreteScheduler
    sched = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=5.0)
    sched.set_timesteps(args.steps)
    fwd = 0.0
    latents = latent
    with torch.no_grad():
        for t in sched.timesteps:
            t0 = time.perf_counter()
            noise_pred = model(latents, timestep=t.unsqueeze(0),
                               encoder_hidden_states=text, return_dict=False)[0]
            latents = sched.step(noise_pred, t, latents).prev_sample
            fwd += time.perf_counter() - t0
    out = latents
    z_dim = int(vae.config.z_dim)
    lat_mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
    lat_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1)
    out = out / lat_std + lat_mean
    t0 = time.perf_counter()
    with torch.no_grad():
        frames = vae.decode(out, return_dict=False)[0]
    dec = time.perf_counter() - t0
    img = ((frames[0, :, 0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).byte().numpy()
    outpath = args.out or (r'D:\work\research\bench_results\wan21_true_%s.png' % args.backend)
    Image.fromarray(img).save(outpath)
    std, lap = frame_stats(img)
    res = dict(backend=args.backend, quantized=n, fwd_s=fwd, vae_s=dec,
               out=tuple(out.shape), frames=tuple(frames.shape), finite=bool(torch.isfinite(frames).all()),
               std=std, lap=lap, path=outpath)
    print('RESULT ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
