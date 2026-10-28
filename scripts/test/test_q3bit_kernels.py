# -*- coding: utf-8 -*-
"""test_q3bit_kernels.py — unit tests for the 2/3-bit quantization prototype.

Covers: pack/unpack code round-trip, quantize->dequantize error bounds, and (when the local
prototype DLLs q3bit_kernel.dll / q3bit_gemm.dll are present) bit-exactness of the AVX2
dequant kernel and fused GEMM vs the torch reference. The kernel parts are skipped with a
clear message on machines without the prototype DLLs.
"""
from __future__ import annotations
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import ctypes
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scripts.lib.q3bit_proto import dequantize_blocks, pack_codes, quantize_blocks, quantize_blocks_8bit, unpack_codes

DLL_DIR = HERE


def test_pack_roundtrip():
    torch.manual_seed(0)
    for bits, block in ((2, 64), (3, 64)):
        codes = torch.randint(0, (1 << bits), (7, block), dtype=torch.uint8)
        packed = pack_codes(codes, bits)
        back = unpack_codes(packed, bits, block)
        assert torch.equal(codes, back), (bits, "code round-trip mismatch")
        # expected bytes per block: 2-bit -> block/4, 3-bit -> block/8*3
        expect = block // 4 if bits == 2 else block // 8 * 3
        assert packed.shape == (7, expect), (bits, packed.shape, expect)
    print("[PASS] pack/unpack code round-trip bit-exact (2-bit, 3-bit)")


def test_error_bounds():
    torch.manual_seed(0)
    w = torch.randn(1 << 20)
    for bits, limit in ((2, 0.60), (3, 0.35)):
        packed, absmax, shape, n = quantize_blocks(w, bits, 64)
        back = dequantize_blocks(packed, absmax, shape, n, bits, 64)
        rel = float((w - back).norm() / w.norm())
        assert rel < limit, (bits, rel, limit)
        print("[PASS] %d-bit relL2=%.4f (< %.2f)" % (bits, rel, limit))
    q, absmax, shape, n = quantize_blocks_8bit(w, 64)
    back8 = (q.view(-1).float() * (absmax / 127.5).repeat_interleave(64)[:n]
             - absmax.repeat_interleave(64)[:n])
    rel8 = float((w - back8).norm() / w.norm())
    assert rel8 < 0.01, rel8
    print("[PASS] 8-bit relL2=%.4f (< 0.01)" % rel8)


def _load_kernel():
    for name in ('q3bit_kernel.dll', 'q3bit_kernel.so'):
        path = os.path.join(DLL_DIR, name)
        if os.path.exists(path):
            lib = ctypes.CDLL(path)
            for fn in ('bnb_dequant_2bit', 'bnb_dequant_3bit'):
                getattr(lib, fn).argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_longlong,
                                             ctypes.c_int]
                getattr(lib, fn).restype = None
            return lib
    return None


def _load_gemm():
    for name in ('q3bit_gemm.dll', 'q3bit_gemm.so'):
        path = os.path.join(DLL_DIR, name)
        if os.path.exists(path):
            lib = ctypes.CDLL(path)
            lib.q3bit_gemm.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int]
            lib.q3bit_gemm.restype = ctypes.c_int
            return lib
    return None


def test_dequant_kernel():
    lib = _load_kernel()
    if lib is None:
        print("[SKIP] dequant kernel DLL not present (prototype artifacts only)")
        return
    torch.manual_seed(1)
    for bits in (2, 3):
        w = torch.randn(64, 64)
        packed, absmax, shape, n = quantize_blocks(w, bits, 64)
        ref = dequantize_blocks(packed, absmax, shape, n, bits, 64).reshape(-1)
        out = torch.empty_like(ref)
        fn = lib.bnb_dequant_3bit if bits == 3 else lib.bnb_dequant_2bit
        fn(packed.contiguous().data_ptr(), absmax.contiguous().data_ptr(),
           out.data_ptr(), absmax.numel(), 64)
        assert torch.equal(out, ref), (bits, "kernel vs torch maxdiff",
                                       float((out - ref).abs().max()))
        print("[PASS] %d-bit AVX2 dequant kernel bit-exact vs torch" % bits)


def test_fused_gemm():
    lib = _load_gemm()
    if lib is None:
        print("[SKIP] fused GEMM DLL not present (prototype artifacts only)")
        return
    torch.manual_seed(2)
    N, K, M = 512, 256, 8
    w = torch.randn(N, K)
    packed, absmax, shape, n = quantize_blocks(w, 3, 64)
    ref_w = dequantize_blocks(packed, absmax, shape, n, 3, 64)
    a = torch.randn(M, K)
    ref = a @ ref_w.t()
    out = torch.empty(M, N, dtype=torch.float32)
    rc = lib.q3bit_gemm(a.data_ptr(), M, packed.data_ptr(), absmax.data_ptr(),
                        out.data_ptr(), N, K, 64, 2, 32)
    assert rc == 0, rc
    reldiff = float((out - ref).abs().max() / (ref.abs().max() + 1e-9))
    assert reldiff < 1e-5, reldiff
    print("[PASS] fused 3-bit GEMM vs torch ref reldiff=%.2e" % reldiff)


def main():
    test_pack_roundtrip()
    test_error_bounds()
    test_dequant_kernel()
    test_fused_gemm()
    print("\nQ3BIT KERNEL TESTS PASSED")


if __name__ == "__main__":
    main()
