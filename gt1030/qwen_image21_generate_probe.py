"""qwen_image21_generate_probe.py — end-to-end Qwen-Image-2.1 4-bit generation on CPU.

Loads the custom NF4 transformer (qwen4bit_loader), the 3D VAE and the flow-matching
scheduler; uses random prompt embeddings for now (the 8B Qwen3VL text-encoder loader is
the next step), so the image content is meaningless but the whole pipeline is exercised:
latent init -> N transformer steps -> VAE decode -> PNG.

Usage:
  python gt1030/qwen_image21_generate_probe.py --size 128 --steps 4
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from qwen4bit_loader import load_transformer

FULL = r'D:\work\qwen_env\Qwen-Image-2.1'
VAE_SCALE = 16


def calculate_shift(seq, base_seq=256, max_seq=8192, base_shift=0.5, max_shift=0.9):
    m = (max_shift - base_shift) / (max_seq - base_seq)
    b = base_shift - m * base_seq
    return seq * m + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--steps', type=int, default=4)
    ap.add_argument('--text-len', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='')
    ap.add_argument('--prompt-embeds', default='')
    args = ap.parse_args()
    torch.set_num_threads(6)
    torch.manual_seed(args.seed)

    print('1) loading 4-bit transformer...', flush=True)
    transformer = load_transformer()

    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.models import AutoencoderKLQwenImage21
    print('2) loading scheduler + VAE...', flush=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        FULL, subfolder='scheduler', local_files_only=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(
        FULL, subfolder='vae', torch_dtype=torch.float32, local_files_only=True)
    vae.eval()
    import psutil
    print('   VAE loaded | RSS %.2f GB' % (psutil.Process().memory_info().rss / 1e9), flush=True)

    # ---- latents ---------------------------------------------------------
    h_lat = w_lat = 2 * (args.size // (VAE_SCALE * 2))
    channels = vae.config.z_dim
    latents = torch.randn(1, 1, channels, h_lat, w_lat, dtype=torch.float32)
    latents = latents.view(1, channels, h_lat * w_lat).transpose(1, 2)  # (1, seq, 64)
    seq = latents.shape[1]
    print('size=%d | latent %dx%dx%d | seq=%d' % (args.size, channels, h_lat, w_lat, seq), flush=True)

    context_in = json.load(open(os.path.join(FULL, 'transformer', 'config.json')))['context_in_dim']
    if args.prompt_embeds:
        d = torch.load(args.prompt_embeds, map_location='cpu')
        prompt_embeds = d['prompt_embeds'].float()
        prompt_embeds_mask = d['prompt_embeds_mask']
        base_img_mask = d['image_pad_mask'].bool()
        print('prompt:', d.get('prompt', '?')[:80], flush=True)
    else:
        prompt_embeds = torch.randn(1, args.text_len, context_in, dtype=torch.float32)
        prompt_embeds_mask = None
        base_img_mask = torch.zeros(1, args.text_len, dtype=torch.bool)
    img_mask = torch.cat([base_img_mask, torch.ones(1, seq // 4, dtype=torch.bool)], dim=1)
    img_shapes = [[(1, h_lat, w_lat)]]

    sigmas = np.linspace(1.0, 1.0 / args.steps, args.steps)
    mu = calculate_shift(seq)
    scheduler.set_timesteps(sigmas=sigmas, mu=mu, device='cpu')
    timesteps = scheduler.timesteps
    print('3) denoising %d steps | mu=%.3f | t0=%.2f' % (len(timesteps), mu, float(timesteps[0])), flush=True)

    t_start = time.perf_counter()
    for i, t in enumerate(timesteps):
        t0 = time.perf_counter()
        ts = t.expand(1).to(latents.dtype) / 1000.0
        with torch.no_grad():
            noise_pred = transformer(
                hidden_states=latents, timestep=ts, encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_embeds_mask, img_shapes=img_shapes, img_mask=img_mask,
                return_dict=False)[0]
        noise_pred = noise_pred[:, -seq:]
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        print('  step %d/%d | %.1f s | RSS %.2f GB'
              % (i + 1, len(timesteps), time.perf_counter() - t0,
                 psutil.Process().memory_info().rss / 1e9), flush=True)
    print('denoise total %.1f s' % (time.perf_counter() - t_start), flush=True)

    # ---- VAE decode ------------------------------------------------------
    z = latents.transpose(1, 2).reshape(1, channels, 1, h_lat, w_lat)
    mean = torch.tensor(vae.config.latents_mean).view(1, channels, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, channels, 1, 1, 1)
    z = z * std + mean
    t0 = time.perf_counter()
    with torch.no_grad():
        image = vae.decode(z, return_dict=False)[0][:, :, 0]
        if image.shape[1] > 3:
            image = image[:, :3]
    image = (image / 2 + 0.5).clamp(0, 1)
    print('4) VAE decode %.1f s | image %s' % (time.perf_counter() - t0, tuple(image.shape)), flush=True)
    arr = (image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    from PIL import Image
    out = args.out or os.path.join(r'D:\work\research\bench_results',
                                   'qwen_image21_prompt_%d_%dsteps.png' % (args.size, args.steps))
    Image.fromarray(arr).save(out)
    print('saved', out)


if __name__ == '__main__':
    raise SystemExit(main())
