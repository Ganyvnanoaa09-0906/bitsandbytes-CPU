"""Video training linear offload prototype: CPU torch.mm vs iGPU Hgemm.

Profile (S128/T16/rank8, step-only): aten::mm+addmm 10.25s / 25.87s (40%),
mkldnn_convolution+backward 9.79s (38%), attention ~1.2s (5%).
This measures representative UNet linear shapes at M=B*T*H*W=4096.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.hybrid_gpu as G

SHAPES = [
    ('block320 qkv', 4096, 960, 320),
    ('block320 proj', 4096, 320, 320),
    ('cross768 qkv', 4096, 2304, 768),
    ('cross768 proj', 4096, 768, 768),
    ('block1280 qkv', 4096, 3840, 1280),
    ('block1280 ffn', 4096, 5120, 1280),
    ('block640 ffn', 4096, 2560, 640),
]


def main():
    torch.set_num_threads(5)
    print('%-18s %8s %10s %10s %10s %10s %10s %10s' % ('shape', 'GF', 'CPU ms', 'iGPU f16 ms', 'f16 GF', 'iGPU f32 ms', 'f32 GF', 'f32 err'))
    for name, m, n, k in SHAPES:
        a = torch.randn(m, k)
        b = torch.randn(k, n)
        # CPU
        torch.mm(a, b)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); c = torch.mm(a, b); ts.append(time.perf_counter() - t0)
        cpu_ms = sorted(ts)[1] * 1e3
        # iGPU Hgemm
        ah = a.numpy().astype(np.float16); bh = b.numpy().astype(np.float16)
        out = G.hgemm(ah, bh)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); out = G.hgemm(ah, bh); ts.append(time.perf_counter() - t0)
        gpu_ms = sorted(ts)[1] * 1e3
        gf = 2.0 * m * n * k / 1e9
        err = float(np.max(np.abs(out.astype(np.float32) - (a.numpy() @ b.numpy()))))
        af = a.numpy().astype(np.float32); bf = b.numpy().astype(np.float32)
        out32 = G.sgemm(af, bf)
        t0 = time.perf_counter(); out32 = G.sgemm(af, bf); s32 = time.perf_counter() - t0
        err32 = float(np.max(np.abs(out32 - (a.numpy() @ b.numpy()))))
        print('%-18s %8.2f %10.2f %10.2f %10.1f %10.2f %10.1f  %.1e' %
              (name, gf, cpu_ms, gpu_ms, gf / (gpu_ms / 1e3),
               s32 * 1e3, gf / s32, err32))


if __name__ == '__main__':
    main()
