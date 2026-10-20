"""test_opencl_fp16_attention.py — fp16 device-resident multi-head attention parity."""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from opencl_fp16_ops import OpenCLFP16  # noqa: E402


def cpu_attention(x, ctx, wq, wk, wv, wo, heads):
    x = torch.from_numpy(x).float()
    ctx = torch.from_numpy(ctx).float()
    wq = torch.from_numpy(wq).float(); wk = torch.from_numpy(wk).float()
    wv = torch.from_numpy(wv).float(); wo = torch.from_numpy(wo).float()
    B, S, C = x.shape
    Skv = ctx.shape[1]
    D = C // heads
    q = x @ wq.T
    k = ctx @ wk.T
    v = ctx @ wv.T
    q = q.reshape(B, S, heads, D).permute(0, 2, 1, 3)
    k = k.reshape(B, Skv, heads, D).permute(0, 2, 1, 3)
    v = v.reshape(B, Skv, heads, D).permute(0, 2, 1, 3)
    scores = (q @ k.transpose(-2, -1)) / np.sqrt(D)
    probs = torch.softmax(scores, dim=-1)
    o = (probs @ v).permute(0, 2, 1, 3).reshape(B, S, C)
    return (o @ wo.T).numpy()


def main():
    rt = OpenCLFP16.get()
    rng = np.random.default_rng(0)
    for B, S, Skv, C, heads in [(1, 64, 64, 320, 8), (2, 256, 256, 320, 8),
                                (1, 77, 77, 640, 8), (2, 64, 77, 320, 8)]:
        x = rng.standard_normal((B, S, C)).astype(np.float32)
        ctx = rng.standard_normal((B, Skv, C)).astype(np.float32)
        wq, wk, wv, wo = [rng.standard_normal((C, C)).astype(np.float32) / np.sqrt(C)
                          for _ in range(4)]
        ref = cpu_attention(x, ctx, wq, wk, wv, wo, heads)
        xb = rt.to_gpu(x.astype(np.float16))
        cb = rt.to_gpu(ctx.astype(np.float16))
        wq_b = rt.to_gpu(wq.T.copy().astype(np.float16))
        wk_b = rt.to_gpu(wk.T.copy().astype(np.float16))
        wv_b = rt.to_gpu(wv.T.copy().astype(np.float16))
        wo_b = rt.to_gpu(wo.T.copy().astype(np.float16))
        ob = rt.attention_h(xb, cb, B, S, Skv, C, heads, wq_b, wk_b, wv_b, wo_b)
        got = rt.to_host(ob, (B, S, C)).astype(np.float32)
        rel = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-9))
        print('attn B=%d S=%d Skv=%d C=%d H=%d | relL2 %.4f' % (B, S, Skv, C, heads, rel))
        for b in (xb, cb, wq_b, wk_b, wv_b, wo_b, ob):
            b.release()



    # benchmark: self-attention at SD-like sequence lengths
    import time
    torch.set_num_threads(6)
    for B, S, C, heads in [(2, 256, 320, 8), (2, 1024, 320, 8), (1, 1024, 640, 8)]:
        x = rng.standard_normal((B, S, C)).astype(np.float32)
        w = [rng.standard_normal((C, C)).astype(np.float32) / np.sqrt(C) for _ in range(4)]
        xt = torch.from_numpy(x)
        with torch.no_grad():
            for _ in range(2):
                cpu_attention(x, x, *w, heads)
            t0 = time.perf_counter()
            for _ in range(3):
                cpu_attention(x, x, *w, heads)
            cpu_s = (time.perf_counter() - t0) / 3
        xb = rt.to_gpu(x.astype(np.float16))
        wbs = [rt.to_gpu(wi.T.copy().astype(np.float16)) for wi in w]
        def run():
            o = rt.attention_h(xb, xb, B, S, S, C, heads, *wbs)
            o.release()
        run(); rt.queue.finish()
        t0 = time.perf_counter()
        for _ in range(3):
            run()
        rt.queue.finish()
        gpu_s = (time.perf_counter() - t0) / 3
        print('bench B=%d S=%d C=%d | CPU %.1f ms | GPU %.1f ms | GPU/CPU %.2fx'
              % (B, S, C, cpu_s * 1e3, gpu_s * 1e3, gpu_s / cpu_s))
        xb.release()
        for b in wbs:
            b.release()


if __name__ == '__main__':
    main()
