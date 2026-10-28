"""Wan2.1 T2V minimal smoke: converted transformer + mapped VAE decoder (no T5)."""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import os
import sys
import time

import psutil
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from scripts.wan import wan_vae_profile as VP
from wan21_convert import load_wan_transformer


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def main():
    torch.set_num_threads(6)
    t0 = time.perf_counter()
    model = load_wan_transformer()
    load_dit = time.perf_counter() - t0
    t0 = time.perf_counter()
    vae = VP.load_decoder()
    load_vae = time.perf_counter() - t0
    print('loaded dit %.1fs vae %.1fs rss %.2fGB' % (load_dit, load_vae, rss_gb()), flush=True)

    B, C, T, H, W = 1, 16, 1, 32, 32
    latent = torch.randn(B, C, T, H, W)
    text = torch.randn(B, 512, 4096)
    ts = torch.tensor([500.0])
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(latent, timestep=ts, encoder_hidden_states=text, return_dict=False)[0]
    fwd = time.perf_counter() - t0
    print('dit forward %.1fs shape %s finite %s rss %.2fGB'
          % (fwd, tuple(out.shape), bool(torch.isfinite(out).all()), rss_gb()), flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        frames = vae.decode(out, return_dict=False)[0]
    dec = time.perf_counter() - t0
    print('vae decode %.1fs shape %s finite %s mean %.4f std %.4f rss %.2fGB'
          % (dec, tuple(frames.shape), bool(torch.isfinite(frames).all()),
             float(frames.mean()), float(frames.std()), rss_gb()), flush=True)
    res = dict(dit_load_s=load_dit, vae_load_s=load_vae, dit_fwd_s=fwd,
               vae_decode_s=dec, latent=tuple(latent.shape), out=tuple(out.shape),
               frames=tuple(frames.shape), rss_gb=rss_gb())
    print('RESULT ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
