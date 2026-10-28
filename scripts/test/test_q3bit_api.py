# -*- coding: utf-8 -*-
"""test_q3bit_api.py — tests for the public bitsandbytes.q3bit API (dev8).

Covers: pack/unpack round-trip, quantise/dequantise layout + error bounds,
native-vs-torch dequant parity, gemm parity, QuantLinear3bit forward/backward
(frozen base, LoRA grads, input grad), no_grad fused path, convert_linears_3bit,
and the forced pure-torch fallback (the Windows-wheel situation).
"""
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
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'bitsandbytes'))

from bitsandbytes import q3bit  # noqa: E402


def test_pack_roundtrip():
    torch.manual_seed(0)
    codes = torch.randint(0, 8, (5, 64), dtype=torch.uint8)
    packed = q3bit.pack_3bit(codes)
    back = q3bit.unpack_3bit(packed, 64)
    assert torch.equal(codes, back), "pack/unpack mismatch"
    assert packed.shape == (5, 24), packed.shape
    print("[PASS] pack_3bit/unpack_3bit round-trip (5x64 -> 5x24 bytes)")


def test_quantize_errors():
    torch.manual_seed(0)
    for shape in ((64, 64), (100, 37)):
        w = torch.randn(*shape)
        for bits, fn_q, fn_d, limit in ((3, q3bit.quantize_3bit, q3bit.dequantize_3bit, 0.30),
                                        (2, q3bit.quantize_2bit, q3bit.dequantize_2bit, 0.60)):
            packed, absmax = fn_q(w)
            back = fn_d(packed, absmax, w.shape)
            assert back.shape == w.shape, (bits, shape, back.shape)
            rel = float((w - back).norm() / w.norm())
            assert rel < limit, (bits, shape, rel, limit)
        print("[PASS] quantize/dequantize shape=%s error bounds" % (shape,))


def test_native_parity():
    if not q3bit.native_available():
        print("[SKIP] native q3bit kernels not in the loaded library (fallback-only build)")
        return
    torch.manual_seed(1)
    w = torch.randn(128, 128)
    packed, absmax = q3bit.quantize_3bit(w)
    fast = q3bit.dequantize_3bit(packed, absmax, w.shape)
    q3bit._set_force_torch(True)
    try:
        slow = q3bit.dequantize_3bit(packed, absmax, w.shape)
    finally:
        q3bit._set_force_torch(False)
    diff = float((fast - slow).abs().max())
    # same arithmetic, but the C kernel and torch may round the final ULP differently
    assert diff < 1e-5, diff
    print("[PASS] native 3-bit dequant == torch fallback (maxdiff %.1e)" % diff)


def test_gemm():
    torch.manual_seed(2)
    N, K, M = 256, 192, 8
    w = torch.randn(N, K)
    packed, absmax = q3bit.quantize_3bit(w)
    x = torch.randn(M, K)
    ref = x @ q3bit.dequantize_3bit(packed, absmax, (N, K)).t()
    out = q3bit.gemm_3bit(x, packed, absmax)
    reldiff = float((out - ref).abs().max() / (ref.abs().max() + 1e-9))
    assert reldiff < 1e-5, reldiff
    # transposed variant
    dy = torch.randn(M, N)
    ref_t = dy @ q3bit.dequantize_3bit(packed, absmax, (N, K))
    out_t = q3bit.gemm_3bit_transposed(dy, packed, absmax)
    reldiff_t = float((out_t - ref_t).abs().max() / (ref_t.abs().max() + 1e-9))
    assert reldiff_t < 1e-5, reldiff_t
    print("[PASS] gemm_3bit / gemm_3bit_transposed vs torch ref (%.1e, %.1e)"
          % (reldiff, reldiff_t))


def test_quant_linear():
    torch.manual_seed(3)
    lin = torch.nn.Linear(128, 96, bias=True)
    ql = q3bit.QuantLinear3bit.from_linear(lin, lora_r=4)
    x = torch.randn(3, 128)
    y = ql(x)
    assert y.shape == (3, 96), y.shape
    # reference computed in no_grad from the same (dequantised) weights + LoRA
    with torch.no_grad():
        w = ql.weight()
        ref = torch.nn.functional.linear(x, w, ql.bias) + \
            (x @ ql.lora_A.t()) @ ql.lora_B.t() * ql.scale
        y2 = ql(x)
    assert torch.allclose(y, ref, atol=1e-5, rtol=1e-4), float((y - ref).abs().max())
    assert torch.allclose(y, y2, atol=1e-5, rtol=1e-4), float((y - y2).abs().max())
    # backward: LoRA B starts at zero, so dL/dB != 0 and dL/dA == 0 until B moves
    ql.zero_grad(set_to_none=True)
    y.sum().backward()
    assert ql.lora_B.grad is not None and ql.lora_B.grad.abs().sum() > 0
    assert ql.lora_A.grad is not None
    with torch.no_grad():
        ql.lora_B.normal_(0.0, 0.02)
    ql.zero_grad(set_to_none=True)
    ql(x).sum().backward()
    assert ql.lora_A.grad is not None and ql.lora_A.grad.abs().sum() > 0
    assert ql.lora_B.grad is not None and ql.lora_B.grad.abs().sum() > 0
    # only the two LoRA tensors are trainable; bias is frozen
    trainable = {n for n, p in ql.named_parameters() if p.requires_grad}
    assert trainable == {"lora_A", "lora_B"}, trainable
    print("[PASS] QuantLinear3bit forward/backward (LoRA grads, frozen base, no_grad path)")


def test_convert_model():
    torch.manual_seed(4)

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(64, 64)
            self.b = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU())
            self.lm_head = torch.nn.Linear(32, 128, bias=False)

        def forward(self, x):
            return self.lm_head(self.b(self.a(x)))

    m = Toy()
    n = q3bit.convert_linears_3bit(m)
    assert n == 2, n
    assert isinstance(m.a, q3bit.QuantLinear3bit)
    assert isinstance(m.b[0], q3bit.QuantLinear3bit)
    assert isinstance(m.lm_head, torch.nn.Linear), "lm_head must stay fp32"
    out = m(torch.randn(2, 64))
    assert out.shape == (2, 128)
    print("[PASS] convert_linears_3bit: 2 layers replaced, lm_head kept fp32")


def test_forced_fallback():
    torch.manual_seed(5)
    w = torch.randn(96, 64)
    packed, absmax = q3bit.quantize_3bit(w)
    ref = q3bit.dequantize_3bit(packed, absmax, w.shape)
    lin = torch.nn.Linear(64, 96)
    with torch.no_grad():
        lin.weight.copy_(w)
    ql = q3bit.QuantLinear3bit.from_linear(lin)
    q3bit._set_force_torch(True)
    try:
        assert not q3bit.native_available()
        back = q3bit.dequantize_3bit(packed, absmax, w.shape)
        out = q3bit.gemm_3bit(torch.randn(2, 64), packed, absmax)
        y = ql(torch.randn(2, 64))
    finally:
        q3bit._set_force_torch(False)
    assert torch.allclose(back, ref, atol=1e-5, rtol=0)
    assert out.shape == (2, 96) and y.shape == (2, 96)
    print("[PASS] forced pure-torch fallback (dequant/gemm/QuantLinear3bit) works")


def main():
    q3bit._set_force_torch(False)
    print("native q3bit kernels:", q3bit.native_available())
    test_pack_roundtrip()
    test_quantize_errors()
    test_native_parity()
    test_gemm()
    test_quant_linear()
    test_convert_model()
    test_forced_fallback()
    print("\nQ3BIT API TESTS PASSED")


if __name__ == "__main__":
    main()
