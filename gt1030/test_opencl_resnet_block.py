"""test_opencl_resnet_block.py — fp16 device-resident SD ResNetBlock2D parity + speed.

The UNet's basic unit is: GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Conv3x3
(+ optional 1x1 shortcut conv) -> add.  Everything stays on the iGPU (weights, activations,
intermediate buffers); only the input/output cross the bus for this parity test.
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


def make_block(cin, cout, h, w, groups=32):
    conv1 = torch.nn.Conv2d(cin, cout, 3, padding=1, bias=False).eval()
    conv2 = torch.nn.Conv2d(cout, cout, 3, padding=1, bias=False).eval()
    gn1 = torch.nn.GroupNorm(groups, cin).eval()
    gn2 = torch.nn.GroupNorm(groups, cout).eval()
    shortcut = (torch.nn.Conv2d(cin, cout, 1, bias=False).eval()
                if cin != cout else None)
    x = torch.randn(1, cin, h, w)
    with torch.no_grad():
        h1 = conv1(F.silu(gn1(x)))
        h2 = conv2(F.silu(gn2(h1)))
        s = shortcut(x) if shortcut is not None else x
        ref = h2 + s
    return conv1, conv2, gn1, gn2, shortcut, x, ref


def half(t):
    return t.detach().numpy().astype(np.float16)


def cpu_forward(x, conv1, conv2, gn1, gn2, shortcut):
    h1 = conv1(F.silu(gn1(x)))
    h2 = conv2(F.silu(gn2(h1)))
    s = shortcut(x) if shortcut is not None else x
    return h2 + s


def main():
    rt = OpenCLFP16.get()
    torch.set_num_threads(6)
    for cin, cout, h in [(320, 320, 64), (320, 640, 32), (640, 640, 32),
                         (640, 1280, 16), (1280, 1280, 16), (1280, 1280, 8)]:
        conv1, conv2, gn1, gn2, shortcut, x, ref = make_block(cin, cout, h, h)
        xb = rt.to_gpu(half(x))
        w1 = rt.to_gpu(half(conv1.weight))
        w2 = rt.to_gpu(half(conv2.weight))
        gamma1 = rt.to_gpu(half(gn1.weight))
        beta1 = rt.to_gpu(half(gn1.bias))
        gamma2 = rt.to_gpu(half(gn2.weight))
        beta2 = rt.to_gpu(half(gn2.bias))
        ws = rt.to_gpu(half(shortcut.weight)) if shortcut is not None else None

        def gpu_forward():
            # ResnetBlock2D order: GN -> SiLU -> Conv1 -> GN -> SiLU -> Conv2 (+ shortcut)
            a = rt.group_norm_silu_buf(xb, (1, cin, h, h), gamma1, beta1, groups=32)
            b, bsh = rt.conv2d_h(a, w1, (1, cin, h, h), (cout, cin, 3, 3))
            a.release()
            h2 = rt.group_norm_silu_buf(b, bsh, gamma2, beta2, groups=32)
            b.release()
            c, csh = rt.conv2d_h(h2, w2, bsh, (cout, cout, 3, 3))
            h2.release()
            if ws is not None:
                s, ssh = rt.conv2d_h(xb, ws, (1, cin, h, h), (cout, cin, 1, 1), pad=0)
            else:
                s, ssh = xb, (1, cin, h, h)
            y = rt.add_buf(c, s, int(np.prod(csh)))
            c.release()
            if ws is not None:
                s.release()
            return y, csh

        yb, yshape = gpu_forward()
        got = rt.to_host(yb, yshape).astype(np.float32)
        # first parity pass
        err = rel(got, ref.numpy())
        for b in (yb,):
            b.release()
        # timing
        with torch.no_grad():
            for _ in range(2):
                cpu_forward(x, conv1, conv2, gn1, gn2, shortcut)
            t0 = time.perf_counter()
            for _ in range(5):
                cpu_forward(x, conv1, conv2, gn1, gn2, shortcut)
            cpu_s = (time.perf_counter() - t0) / 5
        gpu_forward()
        rt.queue.finish()
        t0 = time.perf_counter()
        for _ in range(5):
            y, sh = gpu_forward()
            y.release()
        rt.queue.finish()
        gpu_s = (time.perf_counter() - t0) / 5
        print('resnet %d->%d @%d | parity relL2 %.4f | CPU %.1f ms | GPU %.1f ms | '
              'GPU/CPU %.2fx' % (cin, cout, h, err, cpu_s * 1e3, gpu_s * 1e3, gpu_s / cpu_s))
        for b in (xb, w1, w2, gamma1, beta1, gamma2, beta2):
            b.release()
        if ws is not None:
            ws.release()


if __name__ == '__main__':
    main()
