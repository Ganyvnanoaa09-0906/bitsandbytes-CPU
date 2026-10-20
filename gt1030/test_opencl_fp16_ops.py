"""test_opencl_fp16_ops.py — correctness + resident benchmark for the fp16 OpenCL ops.

Checks the fp16 conv2d (im2col + CLBlast Hgemm) and fused GroupNorm+SiLU against CPU
references, then benchmarks a Conv-GN-SiLU block with all buffers resident on the iGPU.
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


def test_conv(rt):
    for cin, h, w, k, stride, pad, dil, n in [
            (320, 64, 64, 3, 1, 1, 1, 1), (640, 32, 32, 3, 1, 1, 2, 1),
            (64, 32, 32, 3, 2, 1, 1, 2), (320, 32, 32, 1, 1, 0, 1, 1)]:
        x = np.random.randn(n, cin, h, w).astype(np.float32)
        conv = torch.nn.Conv2d(cin, cin, k, stride=stride, padding=pad, dilation=dil).eval()
        wt = conv.weight.detach().numpy().astype(np.float32)
        xh, wh = x.astype(np.float16), wt.astype(np.float16)
        xb, wb = rt.to_gpu(xh), rt.to_gpu(wh)
        out_buf, oshape = rt.conv2d_h(xb, wb, xh.shape, wh.shape, stride, pad, dil)
        got = rt.to_host(out_buf, oshape).astype(np.float32)
        ref = F.conv2d(torch.from_numpy(xh.astype(np.float32)),
                       torch.from_numpy(wh.astype(np.float32)),
                       stride=stride, padding=pad, dilation=dil).numpy()
        print('conv n=%d c=%d h=%d k=%d s=%d p=%d d=%d -> %s | relL2 %.4f'
              % (n, cin, h, k, stride, pad, dil, oshape, rel(got, ref)))
        for b in (xb, wb, out_buf):
            b.release()


def test_gn(rt):
    for n, c, h, w, g in [(2, 320, 16, 16, 32), (1, 640, 8, 8, 32)]:
        x = np.random.randn(n, c, h, w).astype(np.float32)
        gamma = np.random.randn(c).astype(np.float32)
        beta = np.random.randn(c).astype(np.float32)
        xh, gh, bh = x.astype(np.float16), gamma.astype(np.float16), beta.astype(np.float16)
        xb, gb, bb = rt.to_gpu(xh), rt.to_gpu(gh), rt.to_gpu(bh)
        yb = rt.group_norm_silu_buf(xb, xh.shape, gb, bb, groups=g)
        got = rt.to_host(yb, xh.shape).astype(np.float32)
        xt = torch.from_numpy(xh.astype(np.float32))
        ref = F.silu(F.group_norm(xt, g, torch.from_numpy(gh.astype(np.float32)),
                                  torch.from_numpy(bh.astype(np.float32)), 1e-6)).numpy()
        print('gn  n=%d c=%d h=%d g=%d | relL2 %.4f' % (n, c, h, g, rel(got, ref)))
        for b in (xb, gb, bb, yb):
            b.release()


def bench_block(rt):
    torch.set_num_threads(6)
    for cin, h, w in [(320, 64, 64), (640, 32, 32), (1280, 16, 16)]:
        conv = torch.nn.Conv2d(cin, cin, 3, padding=1).eval()
        gn = torch.nn.GroupNorm(32, cin).eval()
        with torch.no_grad():
            x = torch.randn(1, cin, h, w)
            for _ in range(2):
                y = F.silu(gn(conv(x)))
            t0 = time.perf_counter()
            for _ in range(5):
                y = F.silu(gn(conv(x)))
            cpu_s = (time.perf_counter() - t0) / 5

        xb = rt.to_gpu(x.numpy().astype(np.float16))
        wb = rt.to_gpu(conv.weight.detach().numpy().astype(np.float16))
        gb = rt.to_gpu(gn.weight.detach().numpy().astype(np.float16))
        bb = rt.to_gpu(gn.bias.detach().numpy().astype(np.float16))

        def run():
            o, oshape = rt.conv2d_h(xb, wb, xh_shape, wb_shape)
            n = rt.group_norm_silu_buf(o, oshape, gb, bb, groups=32)
            o.release()
            n.release()
        xh_shape = tuple(x.shape)
        wb_shape = tuple(conv.weight.shape)
        run()
        rt.queue.finish()
        t0 = time.perf_counter()
        for _ in range(5):
            run()
        rt.queue.finish()
        gpu_s = (time.perf_counter() - t0) / 5
        print('block c=%d h=%d | CPU %.1f ms | GPU %.1f ms | GPU/CPU %.2fx'
              % (cin, h, cpu_s * 1e3, gpu_s * 1e3, gpu_s / cpu_s))
        for b in (xb, wb, gb, bb):
            b.release()


if __name__ == '__main__':
    rt = OpenCLFP16.get()
    print('device:', rt.device_name, '| max alloc %.0f MB' % (rt.max_alloc / 1e6))
    test_conv(rt)
    test_gn(rt)
    bench_block(rt)
