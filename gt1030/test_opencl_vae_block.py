"""test_opencl_vae_block.py — fp16 VAE-decoder-style block (ResNet + upsample + conv).

VAE decoder pieces after the mid block: ResNetBlock2D -> nearest-2x upsample -> Conv3x3.
Checks parity and timing against the CPU for a 512-channel 32->64 block.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from opencl_fp16_ops import OpenCLFP16  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


def main():
    rt = OpenCLFP16.get()
    torch.set_num_threads(6)
    cin = cout = 512
    h = w = 32
    conv1 = torch.nn.Conv2d(cin, cout, 3, padding=1, bias=False).eval()
    conv2 = torch.nn.Conv2d(cout, cout, 3, padding=1, bias=False).eval()
    up_conv = torch.nn.Conv2d(cout, cout, 3, padding=1, bias=False).eval()
    gn1 = torch.nn.GroupNorm(32, cin).eval()
    gn2 = torch.nn.GroupNorm(32, cout).eval()
    x = torch.randn(1, cin, h, w)

    with torch.no_grad():
        h1 = conv1(F.silu(gn1(x)))
        h2 = conv2(F.silu(gn2(h1)))
        y = h2 + x
        up = F.interpolate(y, scale_factor=2, mode='nearest')
        ref = up_conv(up).numpy()

    def hf(t):
        return t.detach().numpy().astype(np.float16)

    xb = rt.to_gpu(hf(x))
    w1 = rt.to_gpu(hf(conv1.weight))
    w2 = rt.to_gpu(hf(conv2.weight))
    wu = rt.to_gpu(hf(up_conv.weight))
    g1 = rt.to_gpu(hf(gn1.weight))
    b1 = rt.to_gpu(hf(gn1.bias))
    g2 = rt.to_gpu(hf(gn2.weight))
    b2 = rt.to_gpu(hf(gn2.bias))

    def gpu_forward():
        a = rt.group_norm_silu_buf(xb, (1, cin, h, w), g1, b1, groups=32)
        c, csh = rt.conv2d_h(a, w1, (1, cin, h, w), (cout, cin, 3, 3))
        a.release()
        b = rt.group_norm_silu_buf(c, csh, g2, b2, groups=32)
        c.release()
        d, dsh = rt.conv2d_h(b, w2, csh, (cout, cout, 3, 3))
        b.release()
        y = rt.add_buf(d, xb, int(np.prod(dsh)))
        d.release()
        u, ush = rt.upsample2x_buf(y, dsh)
        y.release()
        o, osh = rt.conv2d_h(u, wu, ush, (cout, cout, 3, 3))
        u.release()
        return o, osh

    ob, osh = gpu_forward()
    got = rt.to_host(ob, osh).astype(np.float32)
    ob.release()
    err = rel(got, ref)

    with torch.no_grad():
        for _ in range(2):
            h1 = conv1(F.silu(gn1(x)))
            h2 = conv2(F.silu(gn2(h1)))
            up = F.interpolate(h2 + x, scale_factor=2, mode='nearest')
            _ = up_conv(up)
        t0 = time.perf_counter()
        for _ in range(5):
            h1 = conv1(F.silu(gn1(x)))
            h2 = conv2(F.silu(gn2(h1)))
            up = F.interpolate(h2 + x, scale_factor=2, mode='nearest')
            _ = up_conv(up)
        cpu_s = (time.perf_counter() - t0) / 5
    gpu_forward()
    rt.queue.finish()
    t0 = time.perf_counter()
    for _ in range(5):
        o, sh = gpu_forward()
        o.release()
    rt.queue.finish()
    gpu_s = (time.perf_counter() - t0) / 5
    print('VAE up-block 512ch 32->64 | parity relL2 %.4f | CPU %.1f ms | GPU %.1f ms | '
          'GPU/CPU %.2fx' % (err, cpu_s * 1e3, gpu_s * 1e3, gpu_s / cpu_s))
    for b in (xb, w1, w2, wu, g1, b1, g2, b2):
        b.release()


if __name__ == '__main__':
    main()
