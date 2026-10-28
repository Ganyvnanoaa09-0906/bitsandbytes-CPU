"""Try torch.compile on Wan VAE decode (CPU inductor), with cl3d + Conv2d cl + fast RMS."""
from __future__ import annotations

import os
import statistics
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.wan import wan_vae_profile as P
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanRMS_norm

torch.set_num_threads(int(os.environ.get('THREADS', '6')))


def fast_rms(self, x):
    ss = x.pow(2).sum(dim=1, keepdim=True)
    return (x * torch.rsqrt(ss.clamp_min(1e-24))) * self.scale * self.gamma + self.bias


def main():
    model = P.load_decoder()
    latent = torch.randn(1, 16, 5, 8, 8)
    for m in model.decoder.modules():
        if isinstance(m, WanCausalConv3d):
            m.to(memory_format=torch.channels_last_3d)
        elif isinstance(m, torch.nn.Conv2d):
            m.to(memory_format=torch.channels_last)
    WanRMS_norm.forward = fast_rms
    x = latent.contiguous(memory_format=torch.channels_last_3d)
    with torch.no_grad():
        ref = model.decode(x, return_dict=False)[0]
    base = []
    with torch.no_grad():
        for _ in range(3):
            t = time.perf_counter(); model.decode(x, return_dict=False); base.append(time.perf_counter() - t)
    print('eager median %.1fms' % (statistics.median(base) * 1e3), flush=True)
    try:
        cm = torch.compile(model.decode, mode='reduce-overhead', fullgraph=False)
        with torch.no_grad():
            t = time.perf_counter(); out = cm(x, return_dict=False)[0]; first = time.perf_counter() - t
        print('compile first call %.1fs' % first, flush=True)
        err = float((out - ref).abs().max())
        ts = []
        with torch.no_grad():
            for _ in range(3):
                t = time.perf_counter(); cm(x, return_dict=False); ts.append(time.perf_counter() - t)
        print('compiled median %.1fms  speedup %.2fx  maxdiff %.2e'
              % (statistics.median(ts) * 1e3, statistics.median(base) / statistics.median(ts), err), flush=True)
    except Exception as e:
        print('compile failed:', type(e).__name__, str(e)[:400], flush=True)


if __name__ == '__main__':
    main()
