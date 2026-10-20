"""test_opencl_fp16_transformer.py — full fp16 UNet BasicTransformerBlock parity/speed."""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from opencl_fp16_ops import OpenCLFP16  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


def cpu_attention(x, wq, wk, wv, wo, heads):
    B, S, C = x.shape
    D = C // heads
    q = (x @ wq.T).reshape(B, S, heads, D).permute(0, 2, 1, 3)
    k = (x @ wk.T).reshape(B, S, heads, D).permute(0, 2, 1, 3)
    v = (x @ wv.T).reshape(B, S, heads, D).permute(0, 2, 1, 3)
    probs = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(D), dim=-1)
    o = (probs @ v).permute(0, 2, 1, 3).reshape(B, S, C)
    return o @ wo.T


def cpu_block(x, ln1, ln2, wq, wk, wv, wo, w1, w2, heads):
    B, S, C = x.shape
    n1 = ln1(x) if callable(ln1) else torch.nn.functional.layer_norm(x, (C,), ln1[0], ln1[1], 1e-6)
    a = cpu_attention(n1, wq, wk, wv, wo, heads)
    x1 = x + a
    n2 = torch.nn.functional.layer_norm(x1, (C,), ln2[0], ln2[1], 1e-6).reshape(B * S, C)
    h = n2 @ w1
    inner = h.shape[-1] // 2
    a_, gate = h[:, :inner], h[:, inner:]
    g = torch.nn.functional.gelu(a_) * gate
    f = (g @ w2).reshape(B, S, C)
    return x1 + f


def main():
    rt = OpenCLFP16.get()
    torch.set_num_threads(6)
    rng = np.random.default_rng(0)
    for B, S, C, heads in [(2, 256, 320, 8), (1, 576, 320, 8), (1, 1024, 640, 8)]:
        x = rng.standard_normal((B, S, C)).astype(np.float32)
        ln1g, ln1b = rng.standard_normal(C).astype(np.float32), rng.standard_normal(C).astype(np.float32)
        ln2g, ln2b = rng.standard_normal(C).astype(np.float32), rng.standard_normal(C).astype(np.float32)
        wq, wk, wv, wo = [(rng.standard_normal((C, C)) / np.sqrt(C)).astype(np.float32) for _ in range(4)]
        w1 = (rng.standard_normal((C, 8 * C)) / np.sqrt(C)).astype(np.float32)
        w2 = (rng.standard_normal((4 * C, C)) / np.sqrt(4 * C)).astype(np.float32)

        xt = torch.from_numpy(x)
        ref = cpu_block(xt, (torch.from_numpy(ln1g), torch.from_numpy(ln1b)),
                        (torch.from_numpy(ln2g), torch.from_numpy(ln2b)),
                        torch.from_numpy(wq), torch.from_numpy(wk),
                        torch.from_numpy(wv), torch.from_numpy(wo),
                        torch.from_numpy(w1), torch.from_numpy(w2), heads).numpy()

        hf = lambda a: a.astype(np.float16)
        xb = rt.to_gpu(hf(x))
        g1 = rt.to_gpu(hf(ln1g)); b1 = rt.to_gpu(hf(ln1b))
        g2 = rt.to_gpu(hf(ln2g)); b2 = rt.to_gpu(hf(ln2b))
        wqb = rt.to_gpu(hf(wq.T.copy())); wkb = rt.to_gpu(hf(wk.T.copy()))
        wvb = rt.to_gpu(hf(wv.T.copy())); wob = rt.to_gpu(hf(wo.T.copy()))
        w1b = rt.to_gpu(hf(w1)); w2b = rt.to_gpu(hf(w2))

        def gpu_block():
            return rt.transformer_block_h(xb, B, S, C, heads, g1, b1, wqb, wkb, wvb, wob,
                                          g2, b2, w1b, w2b)
        ob = gpu_block()
        got = rt.to_host(ob, (B, S, C)).astype(np.float32)
        ob.release()
        err = rel(got, ref)

        with torch.no_grad():
            for _ in range(1):
                cpu_block(xt, (torch.from_numpy(ln1g), torch.from_numpy(ln1b)),
                          (torch.from_numpy(ln2g), torch.from_numpy(ln2b)),
                          torch.from_numpy(wq), torch.from_numpy(wk),
                          torch.from_numpy(wv), torch.from_numpy(wo),
                          torch.from_numpy(w1), torch.from_numpy(w2), heads)
            t0 = time.perf_counter()
            for _ in range(2):
                cpu_block(xt, (torch.from_numpy(ln1g), torch.from_numpy(ln1b)),
                          (torch.from_numpy(ln2g), torch.from_numpy(ln2b)),
                          torch.from_numpy(wq), torch.from_numpy(wk),
                          torch.from_numpy(wv), torch.from_numpy(wo),
                          torch.from_numpy(w1), torch.from_numpy(w2), heads)
            cpu_s = (time.perf_counter() - t0) / 2
        gpu_block(); rt.queue.finish()
        t0 = time.perf_counter()
        for _ in range(2):
            ob = gpu_block(); ob.release()
        rt.queue.finish()
        gpu_s = (time.perf_counter() - t0) / 2
        print('transformer B=%d S=%d C=%d | parity %.4f | CPU %.1f ms | GPU %.1f ms | '
              'GPU/CPU %.2fx' % (B, S, C, err, cpu_s * 1e3, gpu_s * 1e3, gpu_s / cpu_s))
        for t in (xb, g1, b1, g2, b2, wqb, wkb, wvb, wob, w1b, w2b):
            t.release()


if __name__ == '__main__':
    main()
