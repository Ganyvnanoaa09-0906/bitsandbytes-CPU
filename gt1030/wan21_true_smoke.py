"""Stage 2: real-prompt Wan T2V smoke (DiT + VAE only, text hidden from stage 1)."""
from __future__ import annotations

import json
import sys
import time

import psutil
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
import wan_vae_profile as VP
from wan21_convert import load_wan_transformer


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def main():
    torch.set_num_threads(6)
    text = torch.load(r'D:\work\textmodel\Wan2.1-T2V-1.3B\prompt_cat_hidden.pt',
                      map_location='cpu').float()
    t0 = time.perf_counter(); model = load_wan_transformer(); load_dit = time.perf_counter() - t0
    t0 = time.perf_counter(); vae = VP.load_decoder(); load_vae = time.perf_counter() - t0
    latent = torch.randn(1, 16, 1, 32, 32)
    ts = torch.tensor([500.0])
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(latent, timestep=ts, encoder_hidden_states=text, return_dict=False)[0]
    fwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    with torch.no_grad():
        frames = vae.decode(out, return_dict=False)[0]
    dec = time.perf_counter() - t0
    res = dict(load_dit_s=load_dit, load_vae_s=load_vae, fwd_s=fwd, vae_s=dec,
               out=tuple(out.shape), frames=tuple(frames.shape),
               finite=bool(torch.isfinite(frames).all()),
               mean=float(frames.mean()), std=float(frames.std()), rss_gb=rss_gb(),
               prompt='a cat walking on grass')
    print('RESULT ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
