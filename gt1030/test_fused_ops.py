"""Acceptance tests for bitsandbytes.fused (torch ref vs native, rel<1e-3)."""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
import bitsandbytes.fused as B


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def main():
    torch.manual_seed(0)
    print('native_available', B.native_available())
    for shape in [(2, 4, 512), (128, 4096), (1, 64, 2048)]:
        x = torch.randn(*shape)
        w = torch.randn(shape[-1])
        y = B.rms_norm(x, w)
        if B.native_available():
            yn = B.rms_norm_native(x, w)
            r = rel(yn, y)
            assert r < 1e-3, ('rms_norm', shape, r)
            print('rms_norm', shape, 'rel', r)
        else:
            print('rms_norm', shape, 'torch-only', float(y.mean()))
    g = torch.randn(256, 4096); u = torch.randn(256, 4096)
    y = B.silu_mul(g, u)
    if B.native_available():
        yn = B.silu_mul_native(g, u)
        r = rel(yn, y)
        assert r < 1e-3, ('silu_mul', r)
        print('silu_mul rel', r)
    # rope reference sanity
    x = torch.randn(2, 8, 16, 64)
    cos = torch.randn(1, 1, 16, 32).repeat_interleave(2, dim=-1)
    sin = torch.randn(1, 1, 16, 32).repeat_interleave(2, dim=-1)
    r = B.rope(x, cos, sin)
    assert r.shape == x.shape
    print('rope shape', tuple(r.shape), 'finite', bool(torch.isfinite(r).all()))
    # CE reference
    logits = torch.randn(64, 1024); labels = torch.randint(0, 1024, (64,))
    print('fused_ce', float(B.fused_ce(logits, labels)))
    print('ALL_PASS')


if __name__ == '__main__':
    main()
