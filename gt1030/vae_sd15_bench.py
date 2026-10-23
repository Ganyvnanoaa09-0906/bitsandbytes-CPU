"""SD1.5 2D VAE decode: channels_last + inference_mode benchmark for video e2e chain."""
from __future__ import annotations

import os
import statistics
import sys
import time

import torch

sys.path.insert(0, r'D:\work\video_mvp\e2e')
os.environ.setdefault('ADIFF_VAE', r'D:\work\textmodel\sd15_full\vae')


def main():
    from diffusers import AutoencoderKL
    torch.set_num_threads(int(os.environ.get('THREADS', '6')))
    vae = AutoencoderKL.from_pretrained(os.environ['ADIFF_VAE'], torch_dtype=torch.float32,
                                        local_files_only=True).eval()
    lat = torch.randn(16, 4, 16, 16)  # e2e: b=1,T=16,latent 16x16
    with torch.inference_mode():
        ref = vae.decode(lat, return_dict=False)[0]
    def bench(tag):
        ts = []
        with torch.inference_mode():
            for _ in range(3):
                t = time.perf_counter(); out = vae.decode(lat, return_dict=False)[0]; ts.append(time.perf_counter() - t)
        print('%-18s median %.1f ms  maxdiff %.2e' % (tag, statistics.median(ts) * 1e3,
                                                     float((out - ref).abs().max())), flush=True)
    bench('NCHW')
    vae.to(memory_format=torch.channels_last)
    lat_cl = lat.contiguous(memory_format=torch.channels_last)
    bench('channels_last')


if __name__ == '__main__':
    main()
