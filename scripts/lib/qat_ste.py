# -*- coding: utf-8 -*-
"""qat_ste.py — 直接在训练里做低比特量化（STE），不打包、不解包。

支持三种模式（2026-11-17 扩展，对应"上游 NF4/混合精度"的思路）：
  · uniform : 均匀网格，bits 位；
  · nf      : 正态分位网格（NF 风格，非均匀），bits 位；
  · mixed   : **大部分块 2-bit，少数"特殊值"块保持高 bit**（按块 |w| 的 absmax 选 top hi_frac）。

用法：
    from scripts.lib.qat_ste import wrap_qat
    wrap_qat(model, bits=2, block=64, mode='mixed', hi_bits=4, hi_frac=0.1)
环境变量（train_ar_v2 用）：
    AR_QAT_BITS=2|3   AR_QAT_MODE=uniform|nf|mixed   AR_QAT_HI_BITS=4   AR_QAT_HI_FRAC=0.1
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# NF4 的 16 级（bitsandbytes 公开值），NF2 用正态分位推出 4 级
NF4_LEVELS = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
              -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
              0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
              0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
              0.7229568362236023, 1.0]


def _nf_levels(bits: int) -> list:
    if bits == 4:
        return NF4_LEVELS
    n = 1 << bits
    # 用正态分位 (i+0.5)/n 作为非均匀网格，最后按最大绝对值归一到 ±1
    import statistics
    qs = []
    for i in range(n):
        p = (i + 0.5) / n
        # 逆误差函数近似
        qs.append(math.sqrt(2.0) * _erfinv(2 * p - 1))
    m = max(abs(q) for q in qs) or 1.0
    return [q / m for q in qs]


def _erfinv(x: float) -> float:
    """Winitzki 近似（足够做量化网格）。"""
    a = 0.147
    ln = math.log(1 - x * x) if abs(x) < 1 else -1e9
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(max(0.0, math.sqrt(t * t - ln / a) - t)), x)


def _uniform_levels(bits: int) -> list:
    n = 1 << bits
    half = n / 2.0
    return [(i - half + 0.5) / half for i in range(n)]     # -1+1/half ... 1-1/half


def _quant_block(blk: torch.Tensor, levels: list) -> torch.Tensor:
    """把 (..., block) 的块按给定 levels（归一到 ±1）量化再反量化。"""
    lv = torch.tensor(levels, dtype=blk.dtype, device=blk.device)
    scale = blk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    x = blk / scale                                          # -> [-1, 1]
    # 找最近 level（levels 小，直接 broadcast 比较）
    d = (x.unsqueeze(-1) - lv).abs()
    idx = d.argmin(dim=-1)
    return lv[idx] * scale


def _qat(w: torch.Tensor, bits: int, block: int, mode: str,
         hi_bits: int = 4, hi_frac: float = 0.1) -> torch.Tensor:
    if bits <= 0:
        return w
    orig = w.shape
    wf = w.reshape(-1, orig[-1])
    rows, cols = wf.shape
    pad = (-cols) % block
    if pad:
        wf = F.pad(wf, (0, pad))
    blk = wf.reshape(rows, -1, block)

    if mode == "nf":
        dq = _quant_block(blk, _nf_levels(bits))
    elif mode == "mixed32":
        lo = _quant_block(blk, _uniform_levels(bits))
        bmax = blk.abs().amax(dim=-1)
        k = max(1, int(round(hi_frac * bmax.numel())))
        thr = bmax.flatten().kthvalue(bmax.numel() - k + 1).values
        mask = (bmax >= thr).unsqueeze(-1)
        dq = torch.where(mask, blk, lo)          # 少数块原样保留（fp32）
    elif mode == "mixed":
        lo = _quant_block(blk, _uniform_levels(bits))
        hi = _quant_block(blk, _uniform_levels(hi_bits))
        # 按块 absmax 选 top hi_frac 用高比特
        bmax = blk.abs().amax(dim=-1)                        # (rows, nblk)
        k = max(1, int(round(hi_frac * bmax.numel())))
        thr = bmax.flatten().kthvalue(bmax.numel() - k + 1).values
        mask = (bmax >= thr).unsqueeze(-1)
        dq = torch.where(mask, hi, lo)
    else:
        dq = _quant_block(blk, _uniform_levels(bits))

    dq = dq.reshape(rows, -1)
    if pad:
        dq = dq[:, :cols]
    dq = dq.reshape(orig)
    return w + (dq - w).detach()


class QATLinear(nn.Module):
    """nn.Linear 的 QAT 版本：前向量化权重，反向 STE。"""

    def __init__(self, lin: nn.Linear, bits: int, block: int, mode: str = "uniform",
                 hi_bits: int = 4, hi_frac: float = 0.1):
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.weight = lin.weight
        self.bias = lin.bias
        self.bits = bits
        self.block = block
        self.mode = mode
        self.hi_bits = hi_bits
        self.hi_frac = hi_frac

    def forward(self, x):
        w = _qat(self.weight, self.bits, self.block, self.mode, self.hi_bits, self.hi_frac)
        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, qat_bits={self.bits}, "
                f"mode={self.mode}, block={self.block}")


def wrap_qat(model: nn.Module, bits: int, block: int = 64, skip_names=("lm_head",),
             mode: str = "uniform", hi_bits: int = 4, hi_frac: float = 0.1) -> int:
    if bits <= 0:
        return 0
    n = 0
    for name, child in list(model.named_children()):
        if any(s and s in name for s in skip_names):
            continue
        if isinstance(child, nn.Linear):
            setattr(model, name, QATLinear(child, bits, block, mode, hi_bits, hi_frac))
            n += 1
        else:
            n += wrap_qat(child, bits, block, skip_names, mode, hi_bits, hi_frac)
    return n
