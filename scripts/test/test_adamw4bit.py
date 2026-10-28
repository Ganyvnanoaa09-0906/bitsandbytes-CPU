# -*- coding: utf-8 -*-
"""test_adamw4bit.py — tests for the restored 4-bit AdamW optimizer (dev8).

Covers: state size (~1.031 bytes/param), native vs pure-torch trajectory parity,
agreement with fp32 torch AdamW, and the CLI-documented usage (state_bytes).
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'bitsandbytes'))

from bitsandbytes.optim import AdamW4bit  # noqa: E402
from bitsandbytes.optim import adamw4bit as m  # noqa: E402


def test_state_size():
    p = torch.nn.Parameter(torch.randn(4096))
    opt = AdamW4bit([p], lr=1e-3)
    p.grad = torch.randn_like(p)
    opt.step()
    bpp = opt.state_bytes() / p.numel()
    assert 1.0 <= bpp <= 1.10, bpp
    assert opt.state_bytes() == 4224
    print("[PASS] state_bytes = %d (%.3f B/param, fp32 AdamW = 8)" % (opt.state_bytes(), bpp))


def _trajectory(force_torch: bool):
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(1000))
    opt = AdamW4bit([p], lr=1e-3, weight_decay=0.01)
    m._set_force_torch(force_torch)
    try:
        for s in range(10):
            torch.manual_seed(100 + s)
            p.grad = torch.randn_like(p) * 0.02
            opt.step()
    finally:
        m._set_force_torch(False)
    return p.detach().clone()


def test_native_parity():
    if not m.native_available():
        print("[SKIP] native 4-bit kernel not in the loaded library (torch fallback build)")
        return
    fast = _trajectory(False)
    slow = _trajectory(True)
    diff = float((fast - slow).abs().max())
    assert diff < 2e-4, diff
    assert torch.allclose(fast, slow, atol=2e-4, rtol=1e-5)
    print("[PASS] native 4-bit kernel == torch fallback over 10 steps (maxdiff %.1e)" % diff)


def test_vs_fp32_adamw():
    torch.manual_seed(2)
    p_ref = torch.nn.Parameter(torch.randn(1024))
    p_4b = torch.nn.Parameter(p_ref.detach().clone())
    ref = torch.optim.AdamW([p_ref], lr=1e-3, weight_decay=0.01)
    got = AdamW4bit([p_4b], lr=1e-3, weight_decay=0.01)
    grads = []
    for s in range(20):
        torch.manual_seed(200 + s)
        g = torch.randn(1024) * 0.02
        grads.append(g)
    for g in grads:
        p_ref.grad = g.clone()
        p_4b.grad = g.clone()
        ref.step()
        got.step()
    rel = float((p_ref.detach() - p_4b.detach()).norm() / p_ref.detach().norm())
    assert rel < 0.01, rel
    print("[PASS] 4-bit AdamW tracks fp32 AdamW over 20 steps (relL2 %.2e)" % rel)


def main():
    print("native 4-bit kernel:", m.native_available())
    test_state_size()
    test_native_parity()
    test_vs_fp32_adamw()
    print("\nADAMW4BIT TESTS PASSED")


if __name__ == "__main__":
    main()
