"""Benchmark AVX2 2bit fused GEMV (uniform LUT) vs materialized dequant."""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import ctypes
import json
import sys
import time

import numpy
import torch

sys.path.insert(0, r'D:\work\bnb-quant')
from scripts.lib.quant2bit import quantize, dequantize, UNIFORM2

LIB = ctypes.CDLL(r'D:\work\bitsandbytes-CPU\gt1030\q2bit_gemv.dll')
LIB.q2bit_gemv.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
LIB.q2bit_gemv.restype = None

SHAPES = [('qkv', 4096, 2048), ('o', 2048, 2048), ('gate', 6144, 2048),
          ('up', 6144, 2048), ('down', 2048, 6144), ('lm_head', 151936, 2048)]


def bench(fn, iters=5):
    fn()
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)


def run(name, N, K):
    torch.manual_seed(0)
    W = (torch.randn(N, K) * 0.02).to(torch.float16)
    A = (torch.randn(1, K) * 0.1).float()
    packed, absmax, n, pad, shape = quantize(W, table=UNIFORM2, block=64)
    packed = packed.contiguous(); absmax = absmax.contiguous()
    out = torch.empty(N, dtype=torch.float32)
    wb = packed.numel() + absmax.numel() * 4

    def fused():
        LIB.q2bit_gemv(A.numpy().ctypes.data, packed.numpy().ctypes.data,
                       absmax.numpy().ctypes.data, out.numpy().ctypes.data, N, K)
        return out

    def material():
        w = dequantize(packed, absmax, n, pad, shape, table=UNIFORM2).float()
        return torch.nn.functional.linear(A, w)

    y1 = fused().clone(); y2 = material()
    f = bench(fused); m = bench(material)
    res = dict(shape=name, N=N, K=K, weight_bytes=wb, weight_gb=wb / 1e9,
               fused_ms=f, material_ms=m, speedup=m / f,
               fused_gbps=(wb / 1e9) / (f / 1000.0),
               absmax_err=float((y1 - y2).abs().max()),
               relL2=float((y1 - y2).norm() / y2.norm()))
    print('RESULT ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    for s in SHAPES:
        run(*s)
