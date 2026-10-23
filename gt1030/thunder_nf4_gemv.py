"""NF4 fused AVX2 GEMV single-layer benchmark (M=1) vs dequant+materialize."""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
import bitsandbytes as bnb
import bitsandbytes.functional as BF

SHAPES = [
    ('qkv', 4096, 2048),
    ('o', 2048, 2048),
    ('gate', 6144, 2048),
    ('up', 6144, 2048),
    ('down', 2048, 6144),
    ('lm_head', 151936, 2048),
]


def bench(fn, iters=5):
    fn()
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)


def run(name, N, K):
    torch.manual_seed(0)
    W = (torch.randn(N, K) * 0.02).to(torch.float16)
    A = (torch.randn(1, K) * 0.1).to(torch.bfloat16)
    q, state = BF.quantize_4bit(W, blocksize=64, quant_type='nf4',
                                compress_statistics=False, quant_storage=torch.uint8)
    wb = q.numel() + state.absmax.numel() * 4
    with torch.no_grad():
        def fused():
            return BF.gemv_4bit(A, q, state=state)
        def material():
            w = BF.dequantize_4bit(q, state).float()
            return torch.nn.functional.linear(A.float(), w)
        y1 = fused(); y2 = material()
    f = bench(fused); m = bench(material)
    err = float((y1.float() - y2).abs().max())
    rel = float((y1.float() - y2).norm() / y2.norm())
    res = dict(shape=name, N=N, K=K, weight_bytes=wb, weight_gb=wb / 1e9,
               fused_ms=f, material_ms=m, speedup=m / f,
               fused_gbps=(wb / 1e9) / (f / 1000.0),
               material_gbps=(wb / 1e9) / (m / 1000.0),
               absmax=err, relL2=rel)
    print('RESULT ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    for s in SHAPES:
        run(*s)
