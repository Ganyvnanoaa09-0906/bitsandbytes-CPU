"""Wan VAE RMSNorm CPU optimization: F.normalize -> x*rsqrt(sum(x^2))."""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.wan import wan_vae_profile as P
from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm

torch.set_num_threads(int(os.environ.get('THREADS', '6')))


def fast_forward(self, x):
    if self.channel_first:
        ss = x.pow(2).sum(dim=1, keepdim=True)
    else:
        ss = x.pow(2).sum(dim=-1, keepdim=True)
    inv = torch.rsqrt(ss.clamp_min(1e-24))
    return (x * inv) * self.scale * self.gamma + self.bias


def main():
    model = P.load_decoder()
    latent = torch.randn(1, 16, 5, 8, 8)
    torch.manual_seed(0)
    with torch.no_grad():
        s0 = time.perf_counter(); out0 = model.decode(latent, return_dict=False)[0]; t_orig = time.perf_counter() - s0
    print('original: %.1f ms' % (t_orig * 1e3), flush=True)
    orig = WanRMS_norm.forward
    WanRMS_norm.forward = fast_forward
    with torch.no_grad():
        s1 = time.perf_counter(); out1 = model.decode(latent, return_dict=False)[0]; t_fast = time.perf_counter() - s1
    print('fast rms: %.1f ms  speedup %.2fx  maxdiff %.2e'
          % (t_fast * 1e3, t_orig / t_fast, float((out0 - out1).abs().max())), flush=True)
    WanRMS_norm.forward = orig


if __name__ == '__main__':
    main()
