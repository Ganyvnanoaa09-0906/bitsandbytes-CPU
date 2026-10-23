"""Wan2.1 converted DiT: NF4 Linear replacement, dummy forward vs fp32, RSS/time."""
from __future__ import annotations

import json
import os
import sys
import time

import psutil
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from wan21_convert import load_wan_transformer

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
import bitsandbytes as bnb


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def quantize_linears(module, prefix=''):
    n = 0
    for name, child in list(module.named_children()):
        full = f'{prefix}.{name}' if prefix else name
        if isinstance(child, torch.nn.Linear) and 'condition_embedder' not in full and 'proj_out' not in full:
            q = bnb.nn.Linear4bit(child.in_features, child.out_features,
                                  bias=child.bias is not None,
                                  compute_dtype=torch.float32,
                                  quant_type='nf4', compress_statistics=True)
            with torch.no_grad():
                q.weight = bnb.nn.Params4bit(child.weight.data.clone(), requires_grad=False,
                                             quant_type='nf4', compress_statistics=True)
                q.weight = q.weight.to('cpu')
                if child.bias is not None:
                    q.bias.data.copy_(child.bias.data)
            setattr(module, name, q)
            n += 1
        else:
            n += quantize_linears(child, full)
    return n


def main():
    torch.set_num_threads(6)
    t0 = time.perf_counter()
    model = load_wan_transformer()
    load_s = time.perf_counter() - t0
    B, C, T, H, W = 1, 16, 1, 32, 32
    latent = torch.randn(B, C, T, H, W)
    text = torch.randn(B, 512, 4096)
    ts = torch.tensor([500.0])
    with torch.no_grad():
        t0 = time.perf_counter()
        ref = model(latent, timestep=ts, encoder_hidden_states=text, return_dict=False)[0]
        fp32_s = time.perf_counter() - t0
    rss_fp32 = rss_gb()
    t0 = time.perf_counter()
    n = quantize_linears(model)
    q_s = time.perf_counter() - t0
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model(latent, timestep=ts, encoder_hidden_states=text, return_dict=False)[0]
        nf4_s = time.perf_counter() - t0
    err = float((out - ref).abs().max())
    rel = float((out - ref).norm() / ref.norm())
    res = dict(linears=n, load_s=load_s, quant_s=q_s, fp32_fwd_s=fp32_s, nf4_fwd_s=nf4_s,
               rss_fp32_gb=rss_fp32, rss_nf4_gb=rss_gb(),
               maxerr=err, relL2=rel, finite=bool(torch.isfinite(out).all()),
               shape=list(out.shape))
    print('RESULT ' + json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
