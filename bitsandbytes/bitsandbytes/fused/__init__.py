"""CPU fused operators (Liger-style minimal package).

Public API:
  rms_norm(x, weight, eps=1e-6)        # torch reference (autograd-friendly)
  silu_mul(gate, up)                  # torch reference (autograd-friendly)
  rope(x, cos, sin)                   # torch reference (autograd-friendly)
  fused_ce(logits, labels)            # torch reference, mean cross entropy
  native_available()                  # zig AVX2 kernels loaded?
  rms_norm_native(x, weight, eps)     # forward-only AVX2
  silu_mul_native(gate, up)           # forward-only AVX2

The torch versions are the contract; native versions must match rel<1e-3.
"""
from __future__ import annotations

from ._ref import fused_ce, rms_norm, rope, silu_mul
from ._native import native_available, rms_norm_native, silu_mul_native

__all__ = ['rms_norm', 'silu_mul', 'rope', 'fused_ce',
           'native_available', 'rms_norm_native', 'silu_mul_native']
