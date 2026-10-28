# -*- coding: utf-8 -*-
"""bitwise_head.py — Infinity 风格的 bitwise 分类头（arXiv 2412.04431）

论文数字（Infinity IVC）：
  常规 1024/65536 类 softmax → d=log2(V) 个 bit 的独立 BCE
  参数 124M → 0.65M（省 99.95%），FID 反而 4.49 → 3.45
  理由：索引标签在 near-zero 维度上梯度是噪声，轻微扰动整个 index 跳变；
        bit 标签只变一两个 bit，其他维度的监督仍清晰。

对我们的意义：
  ① head 从 512×1024=524K 降到 512×10=5.1K（省 99%）
  ② 【诊断能力】逐 bit 准确率（分 MSB/LSB）可以区分
     "逐位置 top-1 衰减"到底是位置问题还是低位 bit 问题
     —— 这是我们当前最需要的诊断

用法：
    head = BitwiseHead(d_model=512, n_bits=10)      # 10 bit = 1024 类
    logits = head(x)                                 # (..., 10) 原始 logits
    loss = bitwise_bce(logits, target_index)         # 标量
    idx = bitwise_sample(logits, temperature=1.0)    # (...,) 0..1023
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


class BitwiseHead(nn.Module):
    """把 V 类分类换成 n_bits 个独立二分类（V = 2^n_bits）。"""

    def __init__(self, d_model, n_bits=10):
        super().__init__()
        self.d = d_model
        self.n_bits = n_bits
        self.vocab = 1 << n_bits              # 2^n_bits
        self.lin = nn.Linear(d_model, n_bits, bias=True)
        # 初始化：bias 让初始 p(bit=1) ≈ 0.5，权重小
        nn.init.normal_(self.lin.weight, std=0.02)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        """返回 (..., n_bits) 的原始 logits（未过 sigmoid）。"""
        return self.lin(x)

    def logits_from_index(self, idx):
        """给定期望的 index，返回"完美"的 bit logits（用于对齐检查）。"""
        bits = self.index_to_bits(idx)
        return (bits.float() * 2 - 1) * 20.0     # ±20 → sigmoid 后接近 0/1

    def index_to_bits(self, idx):
        """(...,) int64 → (..., n_bits) int64，bit i 是 idx 的第 i 位（LSB 在前）。"""
        idx = idx.long()
        shifts = torch.arange(self.n_bits, device=idx.device, dtype=torch.long)
        return (idx.unsqueeze(-1) >> shifts) & 1

    def bits_to_index(self, bits):
        """(..., n_bits) → (...,) int64。"""
        weights = (1 << torch.arange(self.n_bits, device=bits.device,
                                     dtype=torch.long))
        return (bits.long() * weights).sum(-1)


def bitwise_bce(logits, target_index, reduction='mean', loss_weights=None):
    """logits: (..., n_bits)    target_index: (...,) int64

    loss = 各 bit 的 BCE 之和（Infinity 用独立的 10 个 BCE）
    loss_weights: (n_bits,) 每 bit 权重。论文发现 MSB 准确率显著高于 LSB，
                  可选地给低位更高权重（默认全 1）。
    """
    n_bits = logits.shape[-1]
    weights = (1 << torch.arange(n_bits, device=target_index.device,
                                 dtype=torch.long))
    bits = ((target_index.long().unsqueeze(-1) & weights) > 0).float()
    loss = F.binary_cross_entropy_with_logits(logits, bits, reduction='none')
    if loss_weights is not None:
        loss = loss * loss_weights.to(loss.dtype).view(*([1] * (loss.dim() - 1)), n_bits)
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


@torch.no_grad()
def bitwise_sample(logits, temperature=1.0, greedy=False):
    """逐 bit 采样后拼回 index。logits: (..., n_bits) → (...,) int64"""
    p = torch.sigmoid(logits / max(temperature, 1e-6))
    if greedy:
        bits = (p > 0.5).long()
    else:
        bits = (torch.rand_like(p) < p).long()
    n_bits = logits.shape[-1]
    w = (1 << torch.arange(n_bits, device=logits.device, dtype=torch.long))
    return (bits * w).sum(-1)


@torch.no_grad()
def bitwise_stats(logits, target_index):
    """逐 bit 准确率 + per-bit 校准信息（诊断用）。

    返回 dict:
      acc_per_bit: (n_bits,)   每个 bit 的准确率
      acc_msb / acc_lsb: 高半位/低半位的平均准确率
      exact: 整个 index 全对的比率
    """
    n_bits = logits.shape[-1]
    w = (1 << torch.arange(n_bits, device=target_index.device, dtype=torch.long))
    tb = ((target_index.long().unsqueeze(-1) & w) > 0).long()
    pb = (logits > 0).long()
    correct = (pb == tb)                              # (..., n_bits)
    acc = correct.float().mean(dim=tuple(range(correct.dim() - 1)))   # (n_bits,)
    exact = correct.all(-1).float().mean().item()
    half = n_bits // 2
    return {
        'acc_per_bit': acc.cpu(),
        'acc_low': acc[:half].mean().item(),          # 低位（LSB 侧）
        'acc_high': acc[half:].mean().item(),         # 高位（MSB 侧）
        'exact': exact,
    }


if __name__ == '__main__':
    torch.manual_seed(0)
    print('=' * 70)
    print('BitwiseHead 自检')
    print('=' * 70)
    h = BitwiseHead(64, n_bits=10)
    x = torch.randn(4, 256, 64)
    lg = h(x)
    print('  logits 形状:', tuple(lg.shape), '(应为 (4,256,10))')
    print('  参数量: %d  (对比 FlatHead 64x1024=%d)' % (
        sum(p.numel() for p in h.parameters()), 64 * 1024))

    idx = torch.randint(0, 1024, (4, 256))
    # bits ↔ index 往返
    bits = h.index_to_bits(idx)
    back = h.bits_to_index(bits)
    print('  index→bits→index 往返一致:', bool((back == idx).all()))

    loss = bitwise_bce(lg, idx)
    print('  随机 logits 的 BCE: %.4f (理论 ln2≈0.693)' % loss.item())

    # 完美 logits 的 loss 应接近 0
    perfect = h.logits_from_index(idx)
    print('  完美 logits 的 BCE: %.6f (应≈0)' % bitwise_bce(perfect, idx).item())

    st = bitwise_stats(perfect, idx)
    print('  完美 logits 的逐 bit 准确率: 全 1 →', float(st['acc_per_bit'].min()))
    st2 = bitwise_stats(lg, idx)
    print('  随机 logits 的准确率: low %.3f  high %.3f  exact %.4f'
          % (st2['acc_low'], st2['acc_high'], st2['exact']))

    s = bitwise_sample(lg, temperature=1.0)
    print('  采样 index 形状:', tuple(s.shape), ' 范围 [%d, %d]' % (s.min(), s.max()))
    print()
    print('DONE')
