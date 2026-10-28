# -*- coding: utf-8 -*-
"""video_quant.py — bnb-cpu 5D/视频张量量化推理原型。

实测结论（见 TECH_REPORT.md）：视频大 M 场景下 dequant+MKL 明显快于 fused 8bit GEMM。
因此本模块默认推荐：
  - Linear: dequant_linear_nd（8bit 存储 + 反量化 + F.linear / MKL）
  - Conv2d: dequant_conv2d_nd（8bit 存储 + 反量化 + F.conv2d / MKL-DNN）
fused_linear_nd 仅作为省 fp32 权重临时张量的备选。
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
import torch
import torch.nn.functional as F
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise, fused_dequant_linear_8bit

DEFAULT_LINEAR_BACKEND = "dequant_mkl"
DEFAULT_CONV_BACKEND = "dequant_mkldnn"
__all__ = [
    "linear8_code", "to_channels_last",
    "quantize_linear_weight_8bit", "quantize_conv2d_weight_8bit",
    "dequant_linear_nd", "fused_linear_nd", "linear_nd",
    "dequant_conv2d_nd",
]

def linear8_code(device="cpu"):
    return torch.arange(256, dtype=torch.float32, device=device) * (2.0 / 255.0) - 1.0

def to_channels_last(x, in_features):
    if x.shape[-1] == in_features:
        return x
    if x.dim() >= 3 and x.shape[1] == in_features:
        dims = [0] + list(range(2, x.dim())) + [1]
        return x.permute(*dims).contiguous()
    raise ValueError(f"in_features={in_features} not found in shape={tuple(x.shape)}")

def quantize_linear_weight_8bit(weight, blocksize=64):
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D [N,K], got {tuple(weight.shape)}")
    K = weight.shape[1]
    if K % blocksize:
        raise ValueError(f"K={K} must be divisible by blocksize={blocksize}")
    code = linear8_code(weight.device)
    q, st = quantize_blockwise(weight.contiguous(), code=code, blocksize=blocksize)
    return q, st.absmax, code

def quantize_conv2d_weight_8bit(weight, blocksize=64):
    if weight.dim() != 4:
        raise ValueError(f"Conv2d weight must be 4D [O,I,kH,kW], got {tuple(weight.shape)}")
    code = linear8_code(weight.device)
    q, st = quantize_blockwise(weight.contiguous(), code=code, blocksize=blocksize)
    return q, st.absmax, code, weight.shape

def dequant_linear_nd(x, qweight, absmax, code, blocksize=64, channels_last=None):
    K = qweight.shape[1]
    if channels_last is False or (channels_last is None and x.shape[-1] != K):
        x = to_channels_last(x, K)
    w = dequantize_blockwise(qweight, absmax=absmax, code=code, blocksize=blocksize).reshape(qweight.shape)
    return F.linear(x, w)

def fused_linear_nd(x, qweight, absmax, blocksize=64, channels_last=None):
    K = qweight.shape[1]
    if channels_last is False or (channels_last is None and x.shape[-1] != K):
        x = to_channels_last(x, K)
    return fused_dequant_linear_8bit(x, qweight, absmax, blocksize)

def linear_nd(x, qweight, absmax, code, blocksize=64, channels_last=None):
    """默认推荐：dequant + MKL，视频大 M 下远快于 fused。"""
    return dequant_linear_nd(x, qweight, absmax, code, blocksize=blocksize, channels_last=channels_last)

def dequant_conv2d_nd(x, qweight, absmax, code, weight_shape, blocksize=64,
                      stride=1, padding=0, dilation=1, groups=1):
    """默认推荐：Conv2d 权重 8bit 存储 -> 反量化 -> F.conv2d(MKL-DNN)。"""
    w = dequantize_blockwise(qweight, absmax=absmax, code=code, blocksize=blocksize).reshape(weight_shape)
    return F.conv2d(x, w, stride=stride, padding=padding, dilation=dilation, groups=groups)