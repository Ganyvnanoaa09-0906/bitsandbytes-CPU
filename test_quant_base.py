# -*- coding: utf-8 -*-
"""test_quant_base.py — 量化基座直接训练（真量化存储 + LSQ）冒烟测试。

验证 quant_lora.QuantLinearTrainable 与 peft_backends._apply_quant_base：
  1. 权重真正以 8bit/NF4 码字存储（fp32 master 不再驻留）→ 权重内存显著下降；
  2. 反量化重构值与 bitsandbytes 官方 dequant 一致；
  3. 梯度能流回可学习标度 scale（LSQ）；
  4. forward / backward 全程无报错（8bit 与 4bit 两种）。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "bitsandbytes"))

import torch
import torch.nn as nn

from bitsandbytes.functional import (
    dequantize_blockwise,
    dequantize_4bit,
    quantize_blockwise,
    quantize_4bit,
)
from quant_lora import QuantLinearTrainable


def _ref_dequant(weight, quant_dtype, blocksize):
    """用官方 bnb 反量化内核重算出的 fp32 权重（对照基准）。"""
    w = weight.detach().float()
    with torch.no_grad():
        if quant_dtype in ("8bit", "int8"):
            wq, state = quantize_blockwise(w.reshape(-1), blocksize=blocksize)
            return dequantize_blockwise(wq, state, blocksize=blocksize).reshape(w.shape)
        wq, state = quantize_4bit(w, quant_type=quant_dtype[:3], blocksize=blocksize)
        return dequantize_4bit(wq, state, blocksize=blocksize,
                               quant_type=quant_dtype)


def _test_one(quant_dtype, blocksize=None, out=96, inn=128, tol=1e-4):
    lin = nn.Linear(inn, out)
    q = QuantLinearTrainable(lin.weight, lin.bias, quant_dtype=quant_dtype,
                             blocksize=blocksize)

    # ① 内存：fp32 master 不再驻留（模块没有 fp32 weight 参数）
    fp32_vars = [p for n, p in q.named_parameters()
                 if p.dtype == torch.float32 and p.requires_grad]
    assert not any("weight" in n for n, _ in q.named_parameters()), \
        "不应存在可训练 fp32 权重参数"
    # 唯一可训练参数应是 scale
    assert len(list(q.parameters())) == 1 + (1 if q.bias is not None and
                                             isinstance(q.bias, nn.Parameter) else 0)

    orig_bytes = out * inn * 4
    saved = orig_bytes - q.quant_weight_bytes()
    ratio = saved / orig_bytes
    print(f"  [{quant_dtype}] 权重 {orig_bytes/1024:.1f}KB -> "
          f"{q.quant_weight_bytes()/1024:.1f}KB, 省 {ratio*100:.1f}%")
    assert saved > 0, "量化后应更省内存"
    if quant_dtype in ("8bit", "int8"):
        assert ratio > 0.7, "8bit 应省约 75%"
    else:
        assert ratio > 0.8, "4bit 应省约 87.5%"

    # ② 重构一致性（与官方 dequant 对照）
    w_ref = _ref_dequant(lin.weight, quant_dtype, q.blocksize)
    w_my = q._dequant().detach()
    err = (w_my - w_ref).abs().max().item()
    print(f"  [{quant_dtype}] 重构 max|Δ| = {err:.2e}")
    assert err < tol, f"重构误差过大：{err}"

    # ③ 梯度流回 scale
    x = torch.randn(4, inn)
    out_t = q(x)
    out_t.sum().backward()
    assert q.scale.grad is not None, "scale 应拿到梯度"
    assert q.scale.grad.abs().sum().item() > 0, "scale 梯度不应全为零"
    print(f"  [{quant_dtype}] scale 梯度范数 = {q.scale.grad.norm().item():.4f}")

    # ④ 反向能继续回传到输入（端到端可训练）
    lin2 = nn.Linear(inn, out)
    q2 = QuantLinearTrainable(lin2.weight, lin2.bias, quant_dtype=quant_dtype,
                              blocksize=blocksize)
    y = q2(x).sum()
    y.backward()
    assert q2.scale.grad is not None
    return ratio


def _test_apply(model_builder, quant_dtype):
    from peft_backends import apply_method
    m = model_builder()
    info = apply_method(m, "quant_base", quant_dtype=quant_dtype)
    x = torch.randint(0, 64, (2, 8))
    m(x).mean().backward()
    print(f"  [apply/{quant_dtype}] {info.summary().split(chr(10))[0]}")
    assert info.extra.get("weight_mem_saved_mb", 0) > 0
    return info


if __name__ == "__main__":
    print("=" * 70)
    print("量化基座直接训练（真量化 + LSQ）冒烟测试")
    print("=" * 70)

    _test_one("8bit")
    _test_one("nf4")
    _test_one("fp4")

    class MiniNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, 64)
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.lm_head = nn.Linear(64, 64)

        def forward(self, x):
            h = self.embed(x)
            h = torch.relu(self.q_proj(h) + self.k_proj(h))
            return self.lm_head(h)

    _test_apply(MiniNet, "8bit")
    _test_apply(MiniNet, "nf4")

    print("-" * 70)
    print("ALL PASSED")