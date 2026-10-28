"""Final Wan VAE decode best CPU config: cl3d convs + Conv2d channels_last + fast RMS."""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import os
import statistics
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import wan_vae_profile as P
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanRMS_norm

torch.set_num_threads(int(os.environ.get('THREADS', '6')))


def fast_rms(self, x):
    ss = x.pow(2).sum(dim=1, keepdim=True)
    return (x * torch.rsqrt(ss.clamp_min(1e-24))) * self.scale * self.gamma + self.bias


def main():
    model = P.load_decoder()
    latent = torch.randn(1, 16, 5, 8, 8)
    with torch.no_grad():
        ref = model.decode(latent, return_dict=False)[0]
    base = []
    with torch.no_grad():
        for _ in range(3):
            t = time.perf_counter(); model.decode(latent, return_dict=False); base.append(time.perf_counter() - t)
    for m in model.decoder.modules():
        if isinstance(m, WanCausalConv3d):
            m.to(memory_format=torch.channels_last_3d)
        elif isinstance(m, torch.nn.Conv2d):
            m.to(memory_format=torch.channels_last)
    WanRMS_norm.forward = fast_rms
    x = latent.contiguous(memory_format=torch.channels_last_3d)
    with torch.no_grad():
        out = model.decode(x, return_dict=False)[0]
    err = float((out - ref).abs().max())
    times = []
    with torch.no_grad():
        for _ in range(5):
            t = time.perf_counter(); model.decode(x, return_dict=False); times.append(time.perf_counter() - t)
    b = statistics.median(base); o = statistics.median(times)
    print('baseline median %.1fms (%s)' % (b * 1e3, ['%.0f' % (t * 1e3) for t in base]))
    print('best     median %.1fms (%s)  speedup %.2fx  maxdiff %.2e'
          % (o * 1e3, ['%.0f' % (t * 1e3) for t in times], b / o, err))


if __name__ == '__main__':
    main()
