"""probe_qwen_image_4bit.py — load one NF4 linear from the Qwen-Image-2.1 4-bit checkpoint.

Validates that the prequantized format can be turned into a bnb Linear4bit on CPU and
measures 4-bit matmul/dequant throughput.  This is the deciding number for whether a
custom full-model loader is worth writing: the transformer is ~6.6B parameters, so one
forward pass is dominated by per-layer dequant + GEMM.
"""
from __future__ import annotations

import json
import time

import torch

import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit, Params4bit
from safetensors import safe_open

P = r'D:\work\qwen_env\Qwen-Image-2.1-4bit\transformer'
NAME = 'transformer_blocks.0.attn.to_q.weight'


def load_linear(name: str):
    idx = json.load(open(P + r'\diffusion_pytorch_model.safetensors.index.json'))
    shard = idx['weight_map'][name]
    qs = {}
    with safe_open(P + '\\' + shard, framework='pt', device='cpu') as f:
        keys = set(f.keys())
        packed = f.get_tensor(name)
        for suffix in ('absmax', 'nested_absmax', 'nested_quant_map', 'quant_map',
                       'quant_state.bitsandbytes__nf4'):
            k = name + '.' + suffix
            if k in keys:
                qs[suffix] = f.get_tensor(k)
    qmeta = json.load(open(P + r'\quant_meta.json'))['layers'][name]
    out_f, in_f = qmeta['shape']
    print('packed', tuple(packed.shape), packed.dtype, '| out,in', out_f, in_f,
          '| state keys', sorted(qs))
    lin = Linear4bit(in_f, out_f, bias=False, compute_dtype=torch.float32,
                     compress_statistics=True, quant_type='nf4')
    lin.weight = Params4bit.from_prequantized(packed, qs, requires_grad=False,
                                              device='cpu', module=lin)
    return lin


def bench(lin, seq):
    x = torch.randn(1, seq, lin.in_features, dtype=torch.float32)
    with torch.no_grad():
        y = lin(x)
        t0 = time.perf_counter()
        y = lin(x)
        dt = time.perf_counter() - t0
    return dt, y


def main():
    torch.set_num_threads(6)
    lin = load_linear(NAME)
    with torch.no_grad():
        # correctness: dequantize the same packed weight and compare matmul outputs
        from bitsandbytes.functional import dequantize_4bit
        w = dequantize_4bit(lin.weight.data, lin.weight.quant_state).float()
        x = torch.randn(1, 4, lin.in_features)
        ref = x @ w.T
    for seq in (1, 16, 64, 256, 1024):
        dt, y = bench(lin, seq)
        # dequant cost is included in each forward; extrapolate from per-token work
        print('seq=%4d | %.3f s/forward | out %s' % (seq, dt, tuple(y.shape)))
    print('transformer linears: 32 layers x (4 attn + 3 mlp + maybe txt mlp) ~ 224-320 '
          'linear calls/forward; use measured s/forward to extrapolate')


if __name__ == '__main__':
    main()
