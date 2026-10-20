"""test_opencl_fp16_ffn.py — fp16 LayerNorm / GEGLU / Transformer-FFN parity + speed."""
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
    rng = np.random.default_rng(0)

    # LayerNorm parity
    rows, C = 1024, 320
    x = rng.standard_normal((rows, C)).astype(np.float32)
    g = rng.standard_normal(C).astype(np.float32)
    b = rng.standard_normal(C).astype(np.float32)
    xb = rt.to_gpu(x.astype(np.float16))
    gb = rt.to_gpu(g.astype(np.float16))
    bb = rt.to_gpu(b.astype(np.float16))
    yb = rt.layer_norm_buf(xb, rows, C, gb, bb)
    got = rt.to_host(yb, (rows, C)).astype(np.float32)
    ref = F.layer_norm(torch.from_numpy(x.astype(np.float16).astype(np.float32)), (C,),
                       torch.from_numpy(g), torch.from_numpy(b), 1e-6).numpy()
    print('layer_norm rows=%d C=%d | relL2 %.4f' % (rows, C, rel(got, ref)))
    for t in (xb, gb, bb, yb):
        t.release()

    # GEGLU parity (exact GELU reference)
    rows, cols = 1024, 640
    z = rng.standard_normal((rows, 2 * cols)).astype(np.float32)
    zb = rt.to_gpu(z.astype(np.float16))
    gb = rt.geglu_buf(zb, rows, cols)
    got = rt.to_host(gb, (rows, cols)).astype(np.float32)
    zt = torch.from_numpy(z.astype(np.float16).astype(np.float32))
    a, gate = zt.chunk(2, dim=-1)
    ref = (F.gelu(a) * gate).numpy()
    print('geglu rows=%d cols=%d | relL2 %.4f' % (rows, cols, rel(got, ref)))
    for t in (zb, gb):
        t.release()

    # FFN block: x -> linear(8C) -> GEGLU -> linear(C) -> + residual
    for rows, C in [(4096, 320), (1024, 640)]:
        inner = 4 * C
        w1 = (rng.standard_normal((C, 8 * C)) / np.sqrt(C)).astype(np.float32)
        w2 = (rng.standard_normal((inner, C)) / np.sqrt(inner)).astype(np.float32)
        x = rng.standard_normal((rows, C)).astype(np.float32)

        def cpu_fwd():
            h = x @ w1
            a, gate = h[:, :inner], h[:, inner:]
            return torch.from_numpy(a * (0.5 * (1 + np.tanh(0.7978845608 *
                                                             (a + 0.044715 * a ** 3)))) * gate) @ torch.from_numpy(w2) + torch.from_numpy(x)
        ref = cpu_fwd().numpy()

        xb = rt.to_gpu(x.astype(np.float16))
        w1b = rt.to_gpu(w1.astype(np.float16))
        w2b = rt.to_gpu(w2.astype(np.float16))

        def gpu_fwd():
            h, _ = rt.linear_h(xb, w1b, rows, C, 8 * C)
            g = rt.geglu_buf(h, rows, inner)
            h.release()
            o, _ = rt.linear_h(g, w2b, rows, inner, C)
            g.release()
            y = rt.add_buf(o, xb, rows * C)
            o.release()
            return y
        yb = gpu_fwd()
        got = rt.to_host(yb, (rows, C)).astype(np.float32)
        yb.release()
        err = rel(got, ref)

        with torch.no_grad():
            for _ in range(2):
                cpu_fwd()
            t0 = time.perf_counter()
            for _ in range(3):
                cpu_fwd()
            cpu_s = (time.perf_counter() - t0) / 3
        gpu_fwd(); rt.queue.finish()
        t0 = time.perf_counter()
        for _ in range(3):
            yb = gpu_fwd(); yb.release()
        rt.queue.finish()
        gpu_s = (time.perf_counter() - t0) / 3
        print('ffn rows=%d C=%d | parity %.4f | CPU %.1f ms | GPU %.1f ms | GPU/CPU %.2fx'
              % (rows, C, err, cpu_s * 1e3, gpu_s * 1e3, gpu_s / cpu_s))
        for t in (xb, w1b, w2b):
            t.release()


if __name__ == '__main__':
    main()
