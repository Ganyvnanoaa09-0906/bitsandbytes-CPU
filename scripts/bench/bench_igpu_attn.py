# -*- coding: utf-8 -*-
"""bench_igpu_attn.py — 核显的关键问题：慢的是 F.sdpa，还是整个注意力？

文档已确立：DML 的 F.sdpa 在 S=256/512/1024 是 4.2/27.2/158.1 ms，
而 CPU 只要 1.4/5.8/22.9 ms（3~7× 慢）。原因是 multi_head_attention 专用内核
需要新驱动，本机 27.20.11032 不支持。

假设：显式 matmul + F.softmax 能绕开那个内核，直接用 DML 的快 GEMM。

同时按 seq 扫描 —— 因为我们的 8x8 方案是 seq=64，正好在核显有利区间。

必须在 .venv_dml 里跑。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_directml as dml
    HAS_DML = True
except Exception:
    HAS_DML = False

torch.set_num_threads(6)
CPU = torch.device('cpu')
DEV = dml.device() if HAS_DML else CPU


def bench(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def attn_sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def attn_explicit(q, k, v):
    """显式 matmul + softmax，绕开 sdpa 专用内核"""
    S = q.shape[-2]
    scale = 1.0 / (q.shape[-1] ** 0.5)
    score = (q @ k.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), diagonal=1)
    score = score.masked_fill(mask, float('-inf'))
    p = F.softmax(score, dim=-1)
    return p @ v


print('=' * 74, flush=True)
print('核显注意力：F.sdpa  vs  显式 matmul   (B=8, heads=8, hd=64)', flush=True)
print('DirectML:', HAS_DML, flush=True)
print('=' * 74, flush=True)

B, NH, HD = 8, 8, 64

for tag, dev in (('CPU', CPU), ('DML', DEV)):
    print()
    print('--- %s ---' % tag, flush=True)
    print('  %-8s %12s %12s %12s %12s' % ('seq', 'sdpa fwd', 'explicit fwd', 'sdpa 训练', 'explicit 训练'), flush=True)
    for S in (64, 128, 256):
        try:
            q = torch.randn(B, NH, S, HD, device=dev)
            k = torch.randn(B, NH, S, HD, device=dev)
            v = torch.randn(B, NH, S, HD, device=dev)
            q2 = q.clone().requires_grad_(True)
            k2 = k.clone().requires_grad_(True)
            v2 = v.clone().requires_grad_(True)
            # 显式路径需要额外的 requires_grad 副本
            def fwd_s():
                with torch.no_grad():
                    return attn_sdpa(q, k, v)

            def fwd_e():
                with torch.no_grad():
                    return attn_explicit(q, k, v)

            def tr_s():
                o = attn_sdpa(q2, k2, v2)
                o.pow(2).mean().backward()

            q3 = q.clone().requires_grad_(True)
            k3 = k.clone().requires_grad_(True)
            v3 = v.clone().requires_grad_(True)

            def tr_e():
                o = attn_explicit(q3, k3, v3)
                o.pow(2).mean().backward()

            a = bench(fwd_s) * 1e3
            b = bench(fwd_e) * 1e3
            c = bench(tr_s) * 1e3
            d = bench(tr_e) * 1e3
            print('  %-8d %12.3f %12.3f %12.3f %12.3f' % (S, a, b, c, d), flush=True)
            del q, k, v, q2, k2, v2, q3, k3, v3
        except Exception as e:
            print('  %-8d 失败: %s' % (S, type(e).__name__ + ': ' + str(e)[:60]), flush=True)

print()
print('--- 结论判据 ---', flush=True)
print('  若 DML explicit << DML sdpa  →  改显式 matmul 就能绕开慢内核', flush=True)
print('  若 DML 在 seq=64 已经接近或超过 CPU  →  8x8 方案下核显可行', flush=True)
print()
print('DONE', flush=True)
