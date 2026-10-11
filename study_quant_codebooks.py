"""study_quant_codebooks.py -- which codebook wins at 3 and 2 bits, on real weights?

NF4's 16 levels come from the normal distribution's quantiles, normalised to [-1,1],
which is information-theoretically optimal for normally distributed inputs. That
argument gets weaker as levels are removed: with 8 or 4 levels the estimator's
resolution is coarse enough that the exact spacing matters more, and real weight
matrices are not exactly normal -- they are roughly normal with outliers and
per-block variation, which is why the format quantises per block with an absmax.

So the question is empirical. Candidates, all evaluated against the same weights:

  nf      normal quantiles at 2^k levels, normalised       (the NF4 construction)
  fp      exponential spacing, i.e. the FP4 idea           (fine near zero)
  uniform equal spacing over [-1,1]                        (the naive baseline)
  kmeans  fitted to the actual weights                     (data driven)

Metric: relative quantisation error ||w - dequant(w)|| / ||w||, computed per block
of 64 with an absmax scale, exactly as the kernels do it. MSE is a proxy for what
matters, so this is only the first cut -- the format that wins here still has to be
checked against downstream loss before anything is concluded.

Weights come from checkpoints already on disk, so no model needs training.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')


def normal_quantiles(levels):
    """Levels from the normal quantiles at midpoints, normalised to [-1, 1]."""
    from math import erf, sqrt
    from statistics import NormalDist
    nd = NormalDist()
    n = 1 << levels
    q = np.array([nd.inv_cdf((i + 0.5) / n) for i in range(n)], dtype=np.float64)
    q = q / np.abs(q).max()
    # force exact endpoints so the scale is used fully
    q[0], q[-1] = -1.0, 1.0
    q[n // 2 - 1] = 0.0 if n // 2 - 1 >= 0 else q[n // 2 - 1]
    return np.sort(q).astype(np.float32)


def fp_levels(levels):
    """Exponential spacing: the FP4 idea, generalised to 2^levels values."""
    n = 1 << levels
    half = n // 2
    exps = np.arange(half, dtype=np.float64)
    mags = np.exp2(-exps * (4.0 / max(half - 1, 1)))     # 1.0 down to 2^-4
    neg = -mags[::-1]
    vals = np.concatenate([neg, mags[1:] if half > 1 else mags])
    vals = vals[np.abs(vals) <= 1.0]
    while len(vals) < n:
        vals = np.concatenate([[vals[0]], vals])
    vals = np.unique(np.round(vals[:n], 6))
    return vals.astype(np.float32)


def uniform_levels(levels):
    n = 1 << levels
    return np.linspace(-1.0, 1.0, n).astype(np.float32)


def kmeans_levels(w, levels, iters=25, seed=0):
    """1-D k-means over the block-normalised weights."""
    n = 1 << levels
    x = np.clip(w / (np.abs(w).max() or 1.0), -1, 1).astype(np.float32)
    rng = np.random.default_rng(seed)
    c = np.sort(rng.choice(x, size=min(n, len(x)), replace=False)).astype(np.float64)
    for _ in range(iters):
        idx = np.abs(x[:, None] - c[None, :]).argmin(axis=1)
        for i in range(n):
            m = idx == i
            if m.any():
                c[i] = x[m].mean()
    c = np.sort(np.clip(c, -1, 1))
    c[0], c[-1] = -1.0, 1.0
    return c.astype(np.float32)


def quantize_error(w, code, block=64):
    """Per-block absmax quantisation, exactly as the kernels do it."""
    w = w.reshape(-1).astype(np.float32)
    pad = (-len(w)) % block
    if pad:
        w = np.concatenate([w, np.zeros(pad, dtype=np.float32)])
    wb = w.reshape(-1, block)
    scale = np.abs(wb).max(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    x = wb / scale                                          # in [-1, 1]
    idx = np.abs(x[:, :, None] - code[None, None, :]).argmin(axis=2)
    dq = code[idx] * scale
    num = float(np.linalg.norm(wb - dq))
    den = float(np.linalg.norm(wb)) or 1.0
    return num / den


def load_weights(paths, limit_mb=24):
    """Pull float tensors out of checkpoints until the budget is filled."""
    import torch
    ws = []
    total = 0
    for p in paths:
        try:
            ck = torch.load(p, map_location='cpu', weights_only=False)
        except Exception:
            continue
        sd = ck.get('model', ck) if isinstance(ck, dict) else {}
        if not isinstance(sd, dict):
            continue
        for k, v in sd.items():
            if not torch.is_tensor(v) or v.dtype != torch.float32 or v.ndim < 2:
                continue
            a = v.detach().numpy().reshape(-1)
            if a.size < 4096:
                continue
            ws.append(a[:min(a.size, 1 << 20)])
            total += ws[-1].nbytes
            if total > limit_mb * 2**20:
                return ws
    return ws


CAND = sorted(glob.glob(r'D:\work\bitsandbytes-CPU\i5build\ar32_run1\ckpt_3000.pt') +
              glob.glob(r'D:\work\bitsandbytes-CPU\i5build\vqgan32\ckpt_last.pt') +
              glob.glob(r'D:\work\bitsandbytes-CPU\i5build\loop_ar3\ckpt.pt\ckpt_last.pt'))
ws = load_weights(CAND)
if not ws:
    print('  ✗ 没找到可用的权重')
    raise SystemExit(1)
tot = sum(w.size for w in ws)
print('权重来源: %d 个张量, 合计 %.1fM 参数' % (len(ws), tot / 1e6))
print()

for levels in (4, 3, 2):
    print('=== %d bit（%d 级）===' % (levels, 1 << levels))
    codes = {
        'nf     正态分位数': normal_quantiles(levels),
        'fp     指数间隔': fp_levels(levels),
        'uniform 等距': uniform_levels(levels),
        'kmeans 数据驱动': kmeans_levels(np.concatenate(ws)[:200000], levels),
    }
    print('  %-18s %-9s %10s %10s' % ('码表', '级数', '相对误差', '相对最好'))
    print('  ' + '-' * 52)
    errs = {}
    for name, c in codes.items():
        e = np.mean([quantize_error(w, c) for w in ws])
        errs[name] = e
    best = min(errs.values())
    for name, e in sorted(errs.items(), key=lambda kv: kv[1]):
        print('  %-18s %-9d %10.5f %9.2fx' % (name, len(codes[name]), e, e / best))
    print('  码表值: nf=%s' % np.round(codes['nf     正态分位数'], 3))
    print()
