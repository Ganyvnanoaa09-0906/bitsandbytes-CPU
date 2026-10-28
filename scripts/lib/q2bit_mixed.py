"""Experimental 2-bit frozen base + LoRA (pure-torch dequant; no fused 2-bit GEMM).

Reuses bitsandbytes.q3bit pack/dequant layouts. block must be a multiple of 4.
This is an evaluation/training prototype, not a productised API.
"""
from __future__ import annotations
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from bitsandbytes.q3bit import quantize_2bit, dequantize_2bit


class QuantLinear2bit(nn.Module):
    def __init__(self, in_features, out_features, bias=True, block=64,
                 lora_r=8, lora_alpha=16, dtype=torch.float32, device=None):
        super().__init__()
        if block % 4 != 0:
            raise ValueError('block must be a multiple of 4 for 2-bit')
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.block = int(block)
        self.register_buffer('packed', None)
        self.register_buffer('absmax', None)
        self.bias = nn.Parameter(torch.zeros(self.out_features, dtype=dtype, device=device),
                                 requires_grad=False) if bias else None
        self.lora_A = nn.Parameter(torch.empty(lora_r, self.in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, lora_r, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.scale = lora_alpha / lora_r

    @classmethod
    def from_linear(cls, linear, block=64, lora_r=8, lora_alpha=16):
        m = cls(linear.in_features, linear.out_features, bias=linear.bias is not None,
                block=block, lora_r=lora_r, lora_alpha=lora_alpha,
                dtype=linear.weight.dtype, device=linear.weight.device)
        packed, absmax = quantize_2bit(linear.weight.detach(), block)
        m.packed, m.absmax = packed, absmax
        if linear.bias is not None:
            with torch.no_grad():
                m.bias.copy_(linear.bias.detach())
        return m

    def weight(self):
        return dequantize_2bit(self.packed, self.absmax,
                               (self.out_features, self.in_features), block=self.block)

    def forward(self, x):
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        base = F.linear(x2, self.weight().to(x.dtype), self.bias)
        lora = (x2 @ self.lora_A.t()) @ self.lora_B.t() * self.scale
        return (base + lora).reshape(*lead, self.out_features)


def convert_linears_2bit(model, block=64, lora_r=8, lora_alpha=16,
                         skip_names=('lm_head',), name_prefix=''):
    n = 0
    for name, child in list(model.named_children()):
        full = f'{name_prefix}.{name}' if name_prefix else name
        if isinstance(child, nn.Linear):
            if name in skip_names:
                continue
            setattr(model, name, QuantLinear2bit.from_linear(
                child, block=block, lora_r=lora_r, lora_alpha=lora_alpha))
            n += 1
        else:
            n += convert_linears_2bit(child, block=block, lora_r=lora_r,
                                      lora_alpha=lora_alpha, skip_names=skip_names,
                                      name_prefix=full)
    return n
