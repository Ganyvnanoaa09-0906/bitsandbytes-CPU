"""probe_qwen_vae_hybrid512.py — 512^2 Qwen VAE hybrid (large convs on iGPU)."""
from __future__ import annotations

import time

import torch
from diffusers.models import AutoencoderKLQwenImage21

import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from qwen_vae_hybrid_conv import HybridCausalConv3d, select_large_convs  # noqa: E402
from opencl_fp16_ops import OpenCLFP16  # noqa: E402

FULL = r'D:\work\qwen_env\Qwen-Image-2.1'


def main():
    torch.set_num_threads(6)
    rt = OpenCLFP16.get()
    z = torch.randn(1, 64, 1, 32, 32)
    base = AutoencoderKLQwenImage21.from_pretrained(
        FULL, subfolder='vae', torch_dtype=torch.float32, local_files_only=True).eval()
    with torch.no_grad():
        t0 = time.perf_counter(); ref = base.decode(z, return_dict=False)[0]
        cpu_s = time.perf_counter() - t0
    print('CPU 512 VAE %.2f s | out %s' % (cpu_s, tuple(ref.shape)), flush=True)
    for k in (3, 6, 12):
        vae = AutoencoderKLQwenImage21.from_pretrained(
            FULL, subfolder='vae', torch_dtype=torch.float32, local_files_only=True).eval()
        for name, mod in select_large_convs(vae, k=k):
            parts = name.split('.')
            par = vae
            for p in parts[:-1]:
                par = getattr(par, p)
            setattr(par, parts[-1], HybridCausalConv3d(mod, rt))
        with torch.no_grad():
            t0 = time.perf_counter(); out = vae.decode(z, return_dict=False)[0]
            hy_s = time.perf_counter() - t0
        rel = float((out - ref).norm() / ref.norm())
        print('hybrid k=%2d | %.2f s | %.2fx | rel %.2e'
              % (k, hy_s, cpu_s / hy_s, rel), flush=True)
        del vae


if __name__ == '__main__':
    main()
